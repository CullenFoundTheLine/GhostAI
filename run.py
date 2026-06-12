from pipeline import GhostPipeline
import argparse


def main():

    parser = argparse.ArgumentParser(
        description="GhostAI Racing Intelligence"
    )

    parser.add_argument("--driver", default="Cullen")
    parser.add_argument("--predict")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--retrain", action="store_true")

    args = parser.parse_args()

    ghost = GhostPipeline(
        driver_name=args.driver
    )

    if args.status:
        ghost.status()
        return

    if args.predict:
        ghost.predict_session(args.predict)
        return

    if args.retrain:
        ghost.retrain()
        return

    ghost.run()


if __name__ == "__main__":
    main()