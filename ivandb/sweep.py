import argparse
import mlflow
import optuna
import os
import random
import string
import yaml

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

    db = SweepDB()
    if sweep_id in db.get_sweeps_list():
        raise RuntimeError(f"Duplicate sweep: {sweep_id}")
    db.add_sweep(sweep_id, config)
    try:
        # Generate Sweep ID.
        storage = get_optuna_storage()
        study = optuna.create_study(study_name=sweep_id, storage=storage,
                                    direction=config["metric"]["goal"])
    except Exception:
        db.del_sweep(sweep_id)
        raise
    print(f"Created sweep with ID: {sweep_id}")


if __name__ == "__main__":
    args = parse_arguments()
    init_sweep(args)
