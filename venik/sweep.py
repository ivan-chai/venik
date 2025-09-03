import argparse
import optuna
import os
import random
import string
import yaml
from mlflow.tracking import MlflowClient

from .utils import SweepDB, get_optuna_storage, ParameterSampler


def parse_arguments():
    parser = argparse.ArgumentParser("Start sweep. Use environment variables for locating MLflow and Optuna.")
    parser.add_argument("config", help="Config path")
    args = parser.parse_args()
    return args


def init_sweep(args):
    with open(args.config, "r") as fp:
        config = yaml.safe_load(fp)

    for key in ["project", "name", "run_cap", "command", "method", "metric", "parameters"]:
        if key not in config:
            raise ValueError(f"Missing {key} field")

    # Check config.
    assert config["run_cap"] >= 0
    assert config["method"] in ["bayes"]
    assert config["metric"]["goal"] in ["maximize", "minimize"]
    sampler = ParameterSampler(parameters=config["parameters"])

    suffix = "".join([random.choice(string.ascii_letters) for _ in range(6)])
    sweep_id = config["project"] + "-" + config["name"] + "-" + suffix

    # Create a parent MLflow run.
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", None)
    if tracking_uri is None:
        raise RuntimeError("Need MLFLOW_TRACKING_URI environment variable")
    client = MlflowClient(tracking_uri)

    experiment_name = config["project"]
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    parent_run = client.create_run(experiment_id=experiment_id,
                                   run_name="sweep-" + sweep_id)
    config["_parent_mlflow_run_id_"] = parent_run.info.run_id

    try:
        # Create sweep.
        db = SweepDB()
        if sweep_id in db.get_sweeps_list():
            raise RuntimeError(f"Duplicate sweep: {sweep_id}")
        db.add_sweep(sweep_id, config)
        try:
            # Create study.
            storage = get_optuna_storage()
            study = optuna.create_study(study_name=sweep_id, storage=storage,
                                        direction=config["metric"]["goal"])
        except Exception:
            db.del_sweep(sweep_id)
            raise
    except Exception:
        client.delete_run(parent_run.info.run_id)
        raise
    print(f"Created sweep with ID: {sweep_id}")


if __name__ == "__main__":
    args = parse_arguments()
    init_sweep(args)
