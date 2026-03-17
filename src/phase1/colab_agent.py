import json
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright
from transformers import AutoModelForCausalLM, AutoTokenizer


# Colab notes:
# 1) pip install -r requirements-colab.txt
# 2) python -m playwright install chromium
# 3) export MODEL_ID="microsoft/phi-2" (or another local model)
# 4) python src/phase1/colab_agent.py


@dataclass
class AgentConfig:
    model_id: str = "microsoft/phi-2"
    max_steps: int = 8
    max_new_tokens: int = 140
    temperature: float = 0.2
    top_p: float = 0.9
    context_char_limit: int = 3000
    logs_dir: str = "data/logs"
    screenshots_dir: str = "data/screenshots"


@dataclass
class AgentAction:
    action: str
    target: str = ""
    action_input: str = ""
    reason: str = ""


class JsonlLogger:
    def __init__(self, logs_dir: str, run_id: str) -> None:
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.logs_dir / f"{run_id}.jsonl"

    def log(self, event: Dict[str, Any]) -> None:
        payload = {"ts": datetime.utcnow().isoformat(), **event}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


class LocalReasoner:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    def decide(self, goal: str, observation: Dict[str, Any]) -> AgentAction:
        prompt = self._build_prompt(goal, observation)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(output[0], skip_special_tokens=True)
        candidate = generated[len(prompt):].strip() if generated.startswith(prompt) else generated
        return self._parse_action(candidate)

    def _build_prompt(self, goal: str, observation: Dict[str, Any]) -> str:
        state = json.dumps(observation, ensure_ascii=False)
        if len(state) > self.config.context_char_limit:
            state = state[: self.config.context_char_limit] + "..."

        return (
            "You are a web navigation agent.\n"
            "Goal: " + goal + "\n"
            "Current web state (JSON):\n" + state + "\n\n"
            "Return exactly one JSON object with keys: action, target, action_input, reason.\n"
            "Allowed action values: click, type, scroll, wait, finish.\n"
            "Rules:\n"
            "- Use click when target matches a visible link/button text.\n"
            "- Use type with action_input for text fields.\n"
            "- Use finish only when the goal is completed.\n"
            "- Keep reason short.\n\n"
            "JSON:\n"
        )

    def _parse_action(self, text: str) -> AgentAction:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return AgentAction(action="wait", reason="No valid JSON from model")

        try:
            payload = json.loads(match.group(0))
            action = str(payload.get("action", "wait")).strip().lower()
            if action not in {"click", "type", "scroll", "wait", "finish"}:
                action = "wait"
            return AgentAction(
                action=action,
                target=str(payload.get("target", "")).strip(),
                action_input=str(payload.get("action_input", "")).strip(),
                reason=str(payload.get("reason", "")).strip(),
            )
        except json.JSONDecodeError:
            return AgentAction(action="wait", reason="JSON parse failed")


class BrowserController:
    def __init__(self, screenshots_dir: str) -> None:
        self.screenshots_dir = Path(screenshots_dir)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)

    def run_task(
        self,
        reasoner: LocalReasoner,
        logger: JsonlLogger,
        start_url: str,
        goal: str,
        max_steps: int,
        task_id: str,
    ) -> Dict[str, Any]:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--disable-dev-shm-usage", "--no-sandbox", "--disable-gpu"],
            )
            context = browser.new_context(viewport={"width": 1280, "height": 720})
            page = context.new_page()

            page.goto(start_url, wait_until="domcontentloaded", timeout=30000)

            final = {
                "task_id": task_id,
                "goal": goal,
                "start_url": start_url,
                "completed": False,
                "steps": 0,
                "final_answer": "",
                "last_url": page.url,
            }

            for step in range(1, max_steps + 1):
                observation = self._observe(page, task_id, step)
                action = reasoner.decide(goal, observation)
                result = self._execute_action(page, action)

                logger.log(
                    {
                        "type": "step",
                        "task_id": task_id,
                        "step": step,
                        "observation": observation,
                        "action": asdict(action),
                        "result": result,
                    }
                )

                final["steps"] = step
                final["last_url"] = page.url

                if action.action == "finish":
                    final["completed"] = True
                    final["final_answer"] = action.action_input
                    break

            browser.close()

        logger.log({"type": "task_summary", **final})
        return final

    def _observe(self, page, task_id: str, step: int) -> Dict[str, Any]:
        text = page.inner_text("body")
        text = " ".join(text.split())[:2000]

        links = page.eval_on_selector_all(
            "a",
            "els => els.slice(0, 15).map(e => (e.innerText || '').trim()).filter(Boolean)",
        )
        buttons = page.eval_on_selector_all(
            "button",
            "els => els.slice(0, 15).map(e => (e.innerText || '').trim()).filter(Boolean)",
        )

        screenshot_path = self.screenshots_dir / f"{task_id}_step{step}_{int(time.time())}.png"
        page.screenshot(path=str(screenshot_path), full_page=False)

        return {
            "url": page.url,
            "title": page.title(),
            "visible_text": text,
            "links": links,
            "buttons": buttons,
            "screenshot_path": str(screenshot_path),
        }

    def _execute_action(self, page, action: AgentAction) -> Dict[str, Any]:
        try:
            if action.action == "click":
                self._click_by_text(page, action.target)
                return {"ok": True, "detail": f"clicked target={action.target}"}

            if action.action == "type":
                selector = "input, textarea"
                page.locator(selector).first.fill(action.action_input, timeout=3000)
                page.keyboard.press("Enter")
                return {"ok": True, "detail": "typed into first input"}

            if action.action == "scroll":
                page.mouse.wheel(0, 900)
                return {"ok": True, "detail": "scrolled down"}

            if action.action == "wait":
                page.wait_for_timeout(1200)
                return {"ok": True, "detail": "waited"}

            if action.action == "finish":
                return {"ok": True, "detail": "finished"}

            return {"ok": False, "detail": f"unsupported action={action.action}"}

        except PlaywrightTimeoutError as e:
            return {"ok": False, "detail": f"timeout: {e}"}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "detail": f"error: {e}"}

    @staticmethod
    def _click_by_text(page, target: str) -> None:
        if not target:
            raise ValueError("click action requires target")

        try:
            page.get_by_role("link", name=target, exact=False).first.click(timeout=3000)
            return
        except Exception:
            pass

        try:
            page.get_by_role("button", name=target, exact=False).first.click(timeout=3000)
            return
        except Exception:
            pass

        page.locator(f"text={target}").first.click(timeout=3000)


def load_tasks(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("tasks file must be a JSON list")
    return data


def run_batch(config: AgentConfig, tasks_path: str) -> List[Dict[str, Any]]:
    run_id = datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
    logger = JsonlLogger(config.logs_dir, run_id)
    reasoner = LocalReasoner(config)
    controller = BrowserController(config.screenshots_dir)

    logger.log(
        {
            "type": "run_start",
            "run_id": run_id,
            "model_id": config.model_id,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
    )

    summaries = []
    for task in load_tasks(tasks_path):
        summary = controller.run_task(
            reasoner=reasoner,
            logger=logger,
            start_url=task["start_url"],
            goal=task["goal"],
            max_steps=config.max_steps,
            task_id=task["id"],
        )
        summaries.append(summary)

    logger.log({"type": "run_end", "run_id": run_id, "task_count": len(summaries)})
    return summaries


def main() -> None:
    config = AgentConfig(
        model_id=os.getenv("MODEL_ID", "microsoft/phi-2"),
        max_steps=int(os.getenv("MAX_STEPS", "8")),
    )

    tasks_path = os.getenv("TASKS_PATH", "configs/tasks_phase1.json")
    summaries = run_batch(config, tasks_path)

    print("=== Phase 1 run summary ===")
    completed = 0
    for item in summaries:
        completed += 1 if item["completed"] else 0
        print(json.dumps(item, ensure_ascii=False))

    success_rate = completed / max(1, len(summaries))
    print(f"Success rate: {success_rate:.2%}")


if __name__ == "__main__":
    main()
