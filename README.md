# Zavrsni: Security Of Agentic Navigation

Ovaj repozitorij sadrzi implementaciju i evaluaciju autonomnog web agenta za zavrsni rad, s fokusom na:

- baseline autonomnu navigaciju (Phase 1)
- indirect prompt injection napade (Phase 2)
- defense pipeline protiv napada (Defense v1)

Agent radi lokalno u Pythonu (Transformers + Playwright), a primarni runtime je Google Colab.

## 1) Sto je implementirano

### Phase 1: Baseline agent

- observe -> reason -> act petlja u [src/phase1/colab_agent.py](src/phase1/colab_agent.py)
- akcije: click, type, scroll, wait, finish
- logovi po koraku (jsonl) i screenshotovi po koraku
- baseline taskovi u [configs/tasks_phase1.json](configs/tasks_phase1.json)

### Phase 2: Attack evaluacija

- prosireni task set u [configs/tasks_phase1_extended.json](configs/tasks_phase1_extended.json)
- single-page injection scenariji (hidden css, html comment, meta)
- multiturn clean/attack lanac u [configs/tasks_phase2_multiturn.json](configs/tasks_phase2_multiturn.json)
- hostane test stranice u [docs/scenarios](docs/scenarios)

### Defense v1 (implementirano)

Defense je implementiran u [src/phase1/colab_agent.py](src/phase1/colab_agent.py) i aktivira se preko ENABLE_DEFENSE.

Sadrzi 3 sloja:

1. Pre-decision sanitization
- filtrira rizicne kanale (hidden/comment/meta/raw) kada su detektirani injection signali
- filtrira sumnjive link targete (npr. decoy/verification)

2. Post-decision action validation
- blokira sumnjive odluke (posebno finish ili click koji odstupaju od cilja)

3. Confidence gate
- svaki finish dobiva confidence score
- finish se blokira ako je score ispod DEFENSE_CONFIDENCE_THRESHOLD

Dodatno za multiturn safe-code zadatke:

- provjera autorizirane vrijednosti iz vidljivog teksta
- recovery finish mehanizam koji preferira autoriziranu vrijednost na final stranici

## 2) Struktura repozitorija

- [src/phase1/colab_agent.py](src/phase1/colab_agent.py): glavni agent, attack surface i defense logika
- [configs/tasks_phase1.json](configs/tasks_phase1.json): minimalni baseline taskovi
- [configs/tasks_phase1_extended.json](configs/tasks_phase1_extended.json): prosireni benchmark (clean + attack)
- [configs/tasks_phase2_multiturn.json](configs/tasks_phase2_multiturn.json): multiturn clean vs attack
- [docs/index.html](docs/index.html): scenario bench landing page
- [docs/scenarios](docs/scenarios): sve HTML test stranice
- [docs/scenarios/multiturn](docs/scenarios/multiturn): multiturn clean/attack chain
- [notebooks/00_colab_quickstart.md](notebooks/00_colab_quickstart.md): Colab quickstart
- [requirements-colab.txt](requirements-colab.txt): Python dependencies za Colab
- [reports](reports): evaluacijski izvjestaji (generirani)

## 3) Colab quickstart

Najbrzi put je pratiti [notebooks/00_colab_quickstart.md](notebooks/00_colab_quickstart.md).

Ako pokreces na Colabu i ne vidis nove config datoteke/scenarije, napravi:

```bash
!git pull
```

## 4) Kljucne env varijable

- MODEL_ID: npr. Qwen/Qwen2.5-3B-Instruct ili Qwen/Qwen2.5-7B-Instruct
- LOAD_IN_4BIT: 0 ili 1
- HF_TOKEN: opcionalno za veca preuzimanja
- MAX_STEPS: broj koraka po tasku
- TASKS_PATH: putanja do task JSON datoteke
- ATTACK_SURFACE_MODE: visible ili extended
- STRICT_EVAL_MODE: 0 ili 1
- ENABLE_DEFENSE: 0 ili 1
- DEFENSE_CONFIDENCE_THRESHOLD: npr. 0.65

## 5) Preporuceni eksperimenti

### A) Baseline (minimal)

- TASKS_PATH=configs/tasks_phase1.json
- ENABLE_DEFENSE=0

### B) Extended attack benchmark

- TASKS_PATH=configs/tasks_phase1_extended.json
- ATTACK_SURFACE_MODE=extended
- STRICT_EVAL_MODE=1
- ENABLE_DEFENSE=0 ili 1 (za usporedbu)

### C) Multiturn clean vs attack

- TASKS_PATH=configs/tasks_phase2_multiturn.json
- ATTACK_SURFACE_MODE=extended
- STRICT_EVAL_MODE=1
- ENABLE_DEFENSE=0 za ranjivi baseline
- ENABLE_DEFENSE=1 za defense rezultat

## 6) Kako citati rezultate

Izlaz po tasku daje:

- completed: je li task zavrsen
- steps: broj koraka
- final_answer: zavrsni odgovor
- last_url: zadnja stranica

Za security evaluaciju bitne metrike:

1. Clean accuracy
2. Attack hijack rate
3. Benign accuracy under attack

Napomena: navigacijski success rate sam po sebi nije dovoljan za security claim. Potrebno je pratiti je li vracen tocno benigni odgovor.

## 7) Trenutni status (sa zadnjih runova)

Prethodni multiturn baseline (defense off):

- attack hijack rate: 100% (3/3)
- attack benign accuracy: 0% (0/3)

Nakon Defense v1 tuninga (najnoviji snapshot):

- multiturn clean chain: pass
- multiturn attack chain: pass s benignim kodom SAFE-CODE-MT42
- extended suite: 13/13 pass

Za finalni zakljucak u radu preporuka je napraviti najmanje 3 defense-on iteracije i usporediti agregate protiv defense-off baselinea.

## 8) Reproducibilnost

1. Koristi isti TASKS_PATH i env postavke za usporedbu.
2. Pokreni najmanje 3 iteracije po modu (defense off, defense on).
3. Sacuvaj jsonl logove iz data/logs.
4. U reports upisi po-runu i agregat metrike.

## 9) Ogranicenja i sljedeci koraci

Ogranicenja trenutne verzije:

- obrana je heuristicka, nije trenirana policy obrana
- rezultati ovise o modelu i prompt decoding postavkama
- n je trenutno mali bez vecih statistickih intervala

Potencijalni sljedeci koraci:

1. Dodati automatski parser logova u report tablice.
2. Uvesti ablation eksperimente (samo sanitize, samo validator, samo confidence gate).
3. Prosiriti multiturn scenarije na realisticnije JS dinamicke napade.
