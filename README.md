# Ghost AI — GT7 Racing Intelligence

Reads live telemetry from Gran Turismo 7, extracts 12 behavioral features per lap, trains a RandomForest to classify your driving style, and tells you exactly what's costing you time.

---

## Quick Start

```bash
# 1 — Install dependencies
pip install -r requirements.txt
# macOS also needs:
brew install tesseract

# 2 — Set your PS5 IP in telemetry/receiver.py, then capture a session
#     (GT7 must be open, PS5 on same network, UDP telemetry enabled in GT7 settings)
python telemetry/receiver.py

# 3 — Analyze and train
python run.py

# 4 — Predict a new session without retraining
python run.py --predict data/session_YYYYMMDD_HHMMSS.csv

# 5 — Launch the web dashboard
python api/app.py   # → localhost:5000
```

---

## Project Structure

```
ghostAI/
├── run.py                      # CLI entry point — start here
├── pipeline.py                 # Orchestrates load → train → predict
├── coach.py                    # Lap analysis: extracts 12 features per lap
├── ghost_ai.py                 # RandomForest model: learn, predict, explain
├── session_repository.py       # File manager for data/ folder
├── requirements.txt
│
├── analyzer/                   # Modular analysis components
│   ├── behavior.py             # Behavioral pattern detection
│   ├── braking.py              # Braking zone detection
│   ├── drift.py                # Drift event detection
│   ├── features.py             # Feature extraction helpers
│   ├── goal_evaluator.py       # Sub-goal evaluation (brake_pressure > 200 rule)
│   └── lap.py                  # Lap segmentation
│
├── api/                        # Flask web dashboard
│   ├── app.py                  # Flask app — run for localhost:5000
│   └── routes.py               # API routes
│
├── auth/
│   └── login.py                # Auth (future: multi-driver support)
│
├── feedback/
│   └── ghost_feedback.py       # Feedback generation (wired into pipeline)
│
├── storage/                    # Database layer
│   ├── database.py             # SQLite connection (ghost.db)
│   ├── models.py               # Session and lap models
│   └── repository.py          # DB read/write layer
│
├── telemetry/                  # PS5 / GT7 connection
│   ├── receiver.py             # Live GT7 telemetry → CSV
│   ├── parser.py               # Telemetry frame parsing
│   └── session.py              # Session management
│
├── templates/                  # Flask HTML templates
├── data/                       # Session CSVs and trained model (git-ignored)
│   ├── session_YYYYMMDD_HHMMSS.csv
│   ├── ghost_model.pkl
│   └── driver_fingerprint.json
│
├── analyze_video.py            # OCR telemetry from screen recordings
├── .env                        # PS5 IP and secrets (git-ignored)
└── .gitignore
```

---

## Commands

```bash
python3 run.py                              # Full pipeline (default)
python3 run.py --predict data/session.csv  # Predict one new session
python3 run.py --retrain                   # Force delete model and retrain
python3 run.py --status                    # Show data and model status
python3 run.py --driver "YourName"         # Set driver name
```

---

## How It Works

```
telemetry/receiver.py
    ↓ PS5 UDP packets → CSV rows
coach.py / analyzer/
    ↓ 12 behavioral features per lap
ghost_ai.py
    ↓ RandomForest: learn() → predict_with_confidence() → explain_prediction()
pipeline.py
    ↓ Orchestrates steps 1-3, saves model, prints report
analyzer/goal_evaluator.py
    ↓ brake_pressure > 200 → save to learning set
    ↓ Sub-goals: Entry / Developing / Optimal per zone type
api/app.py
    ↓ Dashboard at localhost:5000
```

---

## Driving Styles Ghost Detects

| Style | What it means |
|-------|---------------|
| `aggressive-drift` | Full throttle mid-corner, rear bias, low apex speed |
| `late-braker` | Heavy trail braking, late turn-in |
| `smooth-entry` | High coast %, early brake, patient on exit |
| `drift-balanced` | Some drift events but controlled |
| `balanced` | Smooth, consistent — the default baseline |

---

## Goal Evaluator — Braking Sub-Goals

The `GoalEvaluator` in `analyzer/goal_evaluator.py` applies the core learning rule:

```
brake_pressure > 200  →  save event to learning set
brake_pressure ≤ 200  →  discard
```

Each saved event is also evaluated against graduated sub-goals:

| Zone Type | Sub-Goal | Min Pressure | Min Duration |
|-----------|----------|-------------|--------------|
| Heavy brake | Entry | 150 | 0.3s |
| Heavy brake | Developing | 200 | 0.5s |
| Heavy brake | Optimal | 230 | 0.8s |
| Trail brake | Developing | 50 | 0.2s |
| Trail brake | Structured | 100 | 0.4s |
| Chicane | Entry | 120 | 0.2s |
| Chicane | Optimal | 200 | 0.3s |

---

## PS5 Network Setup

1. In GT7: **Settings → Network → UDP Packet Sending → On**
2. Note your PS5's local IP (Settings → Network → View Connection Status)
3. Set that IP in `telemetry/receiver.py`:
   ```python
   client = TurismoClient(ps_ip="192.x.x.x")
   ```
4. Run `python telemetry/receiver.py` while in a race or time trial

---

## No PS5? Use Video

`analyze_video.py` extracts telemetry from screen recordings using OCR. Useful for offline analysis or when GT7 UDP isn't available.

```bash
python analyze_video.py your_recording.mp4
python analyze_video.py --live    # webcam / capture card
```

---

## Driver Fingerprint

After training, Ghost saves a `data/driver_fingerprint.json`:

```json
{
  "driver": "Cullen",
  "total_laps": 18,
  "dominant_style": "aggressive-drift",
  "style_breakdown": {
    "aggressive-drift": 11,
    "late-braker": 4,
    "drift-balanced": 2,
    "smooth-entry": 1
  }
}
```

The fingerprint is what the prediction agent will use once it's built.

---

## Status: What's Working vs In Progress

| Component | Status |
|-----------|--------|
| `telemetry/receiver.py` | ✅ Working |
| `coach.py` + feature extraction | ✅ Working |
| `ghost_ai.py` RandomForest | ✅ Working |
| `pipeline.py` full flow | ✅ Working |
| `run.py` CLI | ✅ Working |
| `analyzer/goal_evaluator.py` | ✅ Built — wiring in progress |
| `api/app.py` dashboard | 🔧 Connected, expanding |
| `storage/` database layer | 🔧 Built, wiring in progress |
| `feedback/` module | 🔧 Built, wiring in progress |
| Prediction agent | 📋 Next — built on top of learning set |
