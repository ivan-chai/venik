import argparse
import datetime
import os
import re
import yaml

from mlflow.tracking import MlflowClient


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract parameters and metrics from MLflow runs matching a regexp"
    )
    parser.add_argument("pattern", help="Regular expression to match run names")
    parser.add_argument(
        "target", nargs="?", default=".", help="Target folder (default: current directory)"
    )
    return parser.parse_args()


def main(args):
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", None)
    if tracking_uri is None:
        raise RuntimeError("Need MLFLOW_TRACKING_URI environment variable")

    client = MlflowClient(tracking_uri)
    pattern = re.compile(args.pattern)

    experiments = client.search_experiments()
    for experiment in experiments:
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        for run in runs:
            run_name = run.info.run_name or run.info.run_id
            if not pattern.search(run_name):
                continue

            experiment_name = experiment.name.lstrip("/")
            run_dir = os.path.join(args.target, experiment_name, run_name)
            os.makedirs(run_dir, exist_ok=True)

            run_id = run.info.run_id

            params = {k: v for k, v in sorted(run.data.params.items())}
            params_path = os.path.join(run_dir, f"parameters-{run_id}.yaml")
            with open(params_path, "w") as f:
                yaml.safe_dump(params, f)

            metrics = {k: v for k, v in sorted(run.data.metrics.items())}
            metrics_path = os.path.join(run_dir, f"metrics-{run_id}.yaml")
            with open(metrics_path, "w") as f:
                yaml.safe_dump(metrics, f)

            info = run.info
            meta = {
                "run_id": run_id,
                "run_name": run_name,
                "experiment_id": info.experiment_id,
                "experiment_name": experiment.name,
                "status": info.status,
                "user_id": info.user_id,
            }
            if info.start_time:
                meta["start_time"] = datetime.datetime.fromtimestamp(
                    info.start_time / 1000, tz=datetime.timezone.utc
                ).isoformat()
            if info.end_time:
                meta["end_time"] = datetime.datetime.fromtimestamp(
                    info.end_time / 1000, tz=datetime.timezone.utc
                ).isoformat()
            meta_path = os.path.join(run_dir, f"meta-{run_id}.yaml")
            with open(meta_path, "w") as f:
                yaml.safe_dump(meta, f)

            print(f"{run_dir}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
