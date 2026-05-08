import argparse
import datetime
import os
import re

import numpy as np
from mlflow.tracking import MlflowClient


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Download MLflow metric timeseries for runs matching a regexp"
    )
    parser.add_argument("run_pattern", help="Regular expression to match run names")
    parser.add_argument("metric_pattern", help="Regular expression to match metric names")
    parser.add_argument(
        "target", nargs="?", default=".", help="Destination folder (default: current directory)"
    )
    return parser.parse_args()


def main(args):
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", None)
    if tracking_uri is None:
        raise RuntimeError("Need MLFLOW_TRACKING_URI environment variable")

    client = MlflowClient(tracking_uri)
    run_pattern = re.compile(args.run_pattern)
    metric_pattern = re.compile(args.metric_pattern)

    experiments = client.search_experiments()
    for experiment in experiments:
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        for run in runs:
            run_name = run.info.run_name or run.info.run_id
            if not run_pattern.search(run_name):
                continue

            metric_keys = [k for k in run.data.metrics if metric_pattern.search(k)]
            if not metric_keys:
                print(f"No metrics for {run_name}")
                continue

            print(run_name)

            run_id = run.info.run_id
            run_dir = os.path.join(args.target, experiment.name, run_name)
            os.makedirs(run_dir, exist_ok=True)

            info = run.info
            data = {}
            if info.start_time:
                data["start_time"] = datetime.datetime.fromtimestamp(
                    info.start_time / 1000, tz=datetime.timezone.utc
                ).isoformat()
            if info.end_time:
                data["end_time"] = datetime.datetime.fromtimestamp(
                    info.end_time / 1000, tz=datetime.timezone.utc
                ).isoformat()
            for key in metric_keys:
                history = client.get_metric_history(run_id, key)
                history.sort(key=lambda m: m.step)
                data[key] = np.array([m.value for m in history])

            out_path = os.path.join(run_dir, f"{run_id}.npy")
            np.save(out_path, data)
            print(out_path)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
