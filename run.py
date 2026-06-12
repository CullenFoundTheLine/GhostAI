# run.py
# THE ENTRY POINT — the single command you run
#
# This is the only file you need to call from the terminal.
# It decides what to do based on the command you give it.
#
# Commands:
#   python3 run.py                        # full pipeline: load all sessions → train → predict → report
#   python3 run.py --new data/session.csv # predict one new session without retraining
#   python3 run.py --retrain              # force retrain even if model exists
#   python3 run.py --status               # show what data and models exist
#   python3 run.py --driver "Cullen"      # set driver name
#
# How it connects to everything:
#
#   run.py
#     → creates GhostPipeline (pipeline.py)
#         → uses SessionRepository (session_repository.py) to find/load CSV files
#         → uses analyze_file (coach.py) to extract 12 features per lap
#         → uses GhostAI (ghost_ai.py) to train RandomForest and predict
#         → saves model + fingerprint back through SessionRepository
#         → prints the report

import argparse
import sys
import os

# Make sure Python can find all the project files
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import GhostPipeline
from session_repository import SessionRepository


def main():
    parser = argparse.ArgumentParser(
        description="Ghost AI Racing Intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                            Run full pipeline on all sessions
  python run.py --new data/session.csv     Predict one new session
  python run.py --retrain                  Force retrain the model
  python run.py --status                   Show data and model status
  python run.py --driver "Cullen"          Set driver name
        """
    )

    parser.add_argument(
        "--driver", "-d",
        default="Cullen",
        help="Driver name (default: Cullen)"
    )
    parser.add_argument(
        "--new", "-n",
        metavar="CSV_PATH",
        help="Predict on a new session CSV without retraining"
    )
    parser.add_argument(
        "--retrain", "-r",
        action="store_true",
        help="Force retrain the model even if one is already saved"
    )
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Show what session files and models exist, then exit"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Path to the data directory (default: data/)"
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # STATUS — just show what exists and exit
    # ------------------------------------------------------------------
    if args.status:
        repo = SessionRepository(args.data_dir)
        repo.status()

        fp = repo.load_fingerprint()
        if fp:
            print(f"[Ghost AI] Driver fingerprint on file:")
            print(f"  Driver:   {fp.get('driver')}")
            print(f"  Laps:     {fp.get('total_laps')}")
            print(f"  Style:    {fp.get('dominant_style', '').upper()}")
            print(f"  Breakdown:{fp.get('style_breakdown')}")
        else:
            print("[Ghost AI] No driver fingerprint saved yet.")
        return

    # ------------------------------------------------------------------
    # CREATE PIPELINE
    # ------------------------------------------------------------------
    pipeline = GhostPipeline(
        driver_name=args.driver,
        data_dir=args.data_dir
    )

    # ------------------------------------------------------------------
    # PREDICT NEW — run inference on one new session
    # ------------------------------------------------------------------
    if args.new:
        if not os.path.exists(args.new):
            print(f"[Ghost AI] File not found: {args.new}")
            sys.exit(1)

        # Load existing model if available, otherwise train first
        repo = SessionRepository(args.data_dir)
        if not repo.model_exists():
            print("[Ghost AI] No trained model found. Running full pipeline first...")
            pipeline.load_and_analyze()
            trained = pipeline.train()
            if not trained:
                print("[Ghost AI] Could not train. Drive more sessions.")
                sys.exit(1)

        pipeline.predict_new(args.new)
        return

    # ------------------------------------------------------------------
    # RETRAIN — delete saved model and retrain from scratch
    # ------------------------------------------------------------------
    if args.retrain:
        model_path = f"{args.data_dir}/ghost_model.pkl"
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"[Ghost AI] Deleted saved model. Retraining...")

    # ------------------------------------------------------------------
    # FULL PIPELINE — the default command
    # ------------------------------------------------------------------
    pipeline.run()


if __name__ == "__main__":
    main()
