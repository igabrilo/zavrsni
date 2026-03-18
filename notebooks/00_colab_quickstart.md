# Colab Quickstart (Phase 1 Agent)

## 1) Runtime i repo

```bash
# U Colabu prvo uključi GPU: Runtime -> Change runtime type -> T4/L4/A100
!git clone https://github.com/igabrilo/zavrsni.git
%cd zavrsni
```

## 2) Instalacija ovisnosti

```bash
!pip install -q -r requirements-colab.txt
!apt-get update -y && apt-get install -y libatk1.0-0 libatk-bridge2.0-0 libgtk-3-0 libnss3 libnspr4 libcups2 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libxkbcommon0 libasound2 libatspi2.0-0
!python -m playwright install --with-deps chromium
```

## 3) Varijable

```python
import os

os.environ["MODEL_ID"] = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# Opcionalno za vece limite i brze preuzimanje:
# os.environ["HF_TOKEN"] = "tvoj_hf_token"
os.environ["MAX_STEPS"] = "8"
os.environ["TASKS_PATH"] = "configs/tasks_phase1.json"
```

## 4) Pokretanje agenta

```bash
!python src/phase1/colab_agent.py
```

## 5) Pregled rezultata

```bash
!ls -lah data/logs
!ls -lah data/screenshots | head
```

## 6) Mijenjanje taskova

Uredi datoteku `configs/tasks_phase1.json` i ponovno pokreni korak 4.
