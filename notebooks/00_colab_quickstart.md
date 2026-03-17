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
!python -m playwright install chromium
```

## 3) Varijable

```python
import os

os.environ["MODEL_ID"] = "microsoft/phi-2"   # ili drugi lokalni model
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
