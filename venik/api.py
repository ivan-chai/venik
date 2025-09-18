import argparse
import os
import sys
import yaml

from mlflow.tracking import MlflowClient


def parse_arguments():
    parser = argparse.ArgumentParser("Fetch information from MLFlow api")
    parser.add_argument("command", choices=["parameters", "metrics"])
    parser.add_argument("run_id", help="The ID of the run")
    args = parser.parse_args()
    return args


def main(args):
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", None)
    if tracking_uri is None:
        raise RuntimeError("Need MLFLOW_TRACKING_URI environment variable")
    client = MlflowClient(tracking_uri)
    run = client.get_run(args.run_id)
    if args.command == "parameters":
        result = run.data.params
    elif args.command == "metrics":
        result = run.data.metrics
    else:
        assert False
    result = {k: v for k, v in sorted(result.items())}
    yaml.safe_dump(result, sys.stdout)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
