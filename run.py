# run.py
# GhostAI — the only file you need to call
#
# Usage:
#   python3 run.py                              # full pipeline
#   python3 run.py --status                    # show data and model status
#   python3 run.py --retrain                   # delete model and retrain
#   python3 run.py --predict data/session.csv  # predict one session
#   python3 run.py --coach data/session.csv    # full coaching report
#   python3 run.py --driver "YourName"         # set driver name

from pipeline import GhostPipeline
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="GhostAI Racing Intelligence")
    parser.add_argument("--driver",   default="Cullen", help="Driver name")
    parser.add_argument("--data-dir", default="data",   help="Data directory")
    parser.add_argument("--predict",  metavar="CSV",    help="Predict a new session CSV")
    parser.add_argument("--coach",    metavar="CSV",    help="Full coaching report for a session")
    parser.add_argument("--status",   action="store_true", help="Show model and data status")
    parser.add_argument("--retrain",  action="store_true", help="Delete model and retrain")
    args = parser.parse_args()

    ghost = GhostPipeline(driver_name=args.driver, data_dir=args.data_dir)

    if args.status:
        ghost.repo.status()
        if ghost._try_load_model():
            ghost.ghost.status()
        return

    if args.retrain:
        ghost.repo.delete_model()
        ghost.run()
        return

    if args.predict:
        if not os.path.exists(args.predict):
            print(f"[Ghost AI] File not found: {args.predict}")
            return
        ghost.predict_new(args.predict)
        return

    if args.coach:
        if not os.path.exists(args.coach):
            print(f"[Ghost AI] File not found: {args.coach}")
            return
        ghost.run_coaching_report(args.coach)
        return

    # Default: full pipeline
    ghost.run()


if __name__ == "__main__":
    main()
