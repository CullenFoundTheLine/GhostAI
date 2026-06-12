# Ghost AI — GT7 Racing Intelligence

Reads live telemetry from Gran Turismo 7, extracts 12 behavioral features per lap, trains a RandomForest to classify your driving style, and tells you exactly what's costing you time.

## Quick Start

```bash
# 1 — Install
pip install -r requirements.txt
# macOS also needs: brew install tesseract

# 2 — Capture a session (GT7 must be open, PS5 on same network)
python receiver.py

# 3 — Analyze and train
python run.py

# 4 — Predict a new session without retraining
python run.py --new data/session_YYYYMMDD_HHMMSS.csv
```

## Files

| File | What it does |
|------|-------------|
| `receiver.py` | Connects to PS5 via `gt_telem`, records every telemetry frame to a CSV |
| `coach.py` | Loads CSVs, detects braking zones, drift events, trail braking — produces 12 features per lap |
| `ghost_ai.py` | RandomForest model — `learn()`, `predict_with_confidence()`, `explain_prediction()`, `save()`/`load()` |
| `pipeline.py` | Orchestrates the 3-step flow: load → train → predict |
| `session_repository.py` | Manages the `data/` folder — finds CSVs, saves/loads model and fingerprint |
| `run.py` | CLI entry point — the only file you need to call |
| `app.py` | Flask dashboard at `localhost:5000` |
| `analyze_video.py` | OCR-based telemetry extraction from screen recordings (no PS5 required) |

## Commands

```bash
python run.py                          # full pipeline
python run.py --new data/session.csv  # predict one session
python run.py --retrain               # force retrain
python run.py --status                # show what data exists
python run.py --driver "YourName"     # set driver name
```

## Driving Styles Ghost Detects

| Style | Description |
|-------|-------------|
| `balanced` | Smooth, consistent — the default baseline |
| `late-braker` | Heavy trail braking, late turn-in |
| `smooth-entry` | High coast %, early brake, patient on exit |
| `aggressive-drift` | Full throttle mid-corner, rear-biased |
| `drift-balanced` | Some drift events but controlled |

## PS5 Network Setup

Set `` in `receiver.py` to your PS5's local IP. Enable **UDP packet sending** in GT7 under Settings → Network.
