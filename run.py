# run.py
# GhostAI CLI entry point — the only file you need to call
#
# Usage:
#   python run.py                              # full pipeline (default)
#   python run.py --status                    # show what data exists
#   python run.py --retrain                   # delete model and retrain
#   python run.py --predict data/session.csv  # predict one new session
#   python run.py --driver "YourName"         # set driver name
 
from pipeline import GhostPipeline
import argparse
import os
 
 
def main():
    parser = argparse.ArgumentParser(description="GhostAI Racing Intelligence")
    parser.add_argument("--driver",  default="Cullen",  help="Driver name")
    parser.add_argument("--data-dir", default="data",   help="Data directory")
    parser.add_argument("--predict", metavar="CSV",     help="Predict a new session CSV")
    parser.add_argument("--status",  action="store_true", help="Show model and data status")
    parser.add_argument("--retrain", action="store_true", help="Delete saved model and retrain")
    args = parser.parse_args()
 
    ghost = GhostPipeline(driver_name=args.driver, data_dir=args.data_dir)
 
    if args.status:
        # SessionRepository has the full status printer
        ghost.repo.status()
        # Also print GhostAI model internals if a model is loaded
        if ghost._try_load_model():
            ghost.ghost.status()
        return
 
    if args.retrain:
        # Delete the saved model so pipeline.run() retrains from scratch
        ghost.repo.delete_model()
        ghost.run()
        return
 
    if args.predict:
        if not os.path.exists(args.predict):
            print(f"[Ghost AI] File not found: {args.predict}")
            return
        ghost.predict_new(args.predict)
        return
 
    # Default: full pipeline
    ghost.run()
 
 
if __name__ == "__main__":
    main()
 