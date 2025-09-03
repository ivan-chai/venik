import argparse
import json
import optuna
import os
import string
import tempfile
from mlflow.tracking import MlflowClient
import subprocess as sp

from .utils import SweepDB, get_optuna_storage, ParameterSampler


def parse_arguments():
    parser = argparse.ArgumentParser("Start sweep")
    parser.add_argument("sweep_id", help="Config path")
    parser.add_argument("--count", type=int, help="The total amount of runs")
    parser.add_argument("-a", "--args", nargs="*", help="Extra parameters for the worker")
    args = parser.parse_args()
    return args


class Agent:
    def __init__(self, sweep_id, cmd_args=None):
        self.sweep_id = sweep_id
        self.config = SweepDB().get_sweep_config(sweep_id)
        assert "run_cap" in self.config
        self.sampler = ParameterSampler(self.config["parameters"])
        self.cmd_args = cmd_args

    @property
    def default_count(self):
        return self.config["run_cap"]

    def __call__(self, trial):
        # Sample parameters.
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", None)
        if tracking_uri is None:
            raise RuntimeError("Need MLFLOW_TRACKING_URI environment variable")
        client = MlflowClient(tracking_uri)
        args = self.sampler.sample(trial)

        # Construct command.
        cmd = []
        env = {}
        for token in self.config["command"]:
            if token == "${env}":
                env.update(os.environ)
            elif token == "${args_no_hyphens}":
                for name, value in args.items():
                    cmd.append(f"{name}={value}")
            elif token.startswith("${") and token.endswith("}"):
                raise ValueError(f"Unknown token: {token}")
            else:
                cmd.append(token)
        if self.cmd_args is not None:
            cmd = cmd + self.cmd_args

        with tempfile.NamedTemporaryFile("r") as fp_info:
            # Setup MLflow environment.
            env.update({
                "MLFLOW_INFO_FILE": fp_info.name,
                "MLFLOW_PARENT_RUN_ID": self.config["_parent_mlflow_run_id_"],
                "MLFLOW_EXPERIMENT_NAME": self.config["project"],
                "MLFLOW_TAGS": env.get("MLFLOW_TAGS", "") + ";" + f"sweep_id={self.sweep_id};sweep_index={trial.number}"
            })
            print("Environment:", env)

            # Run.
            print("Run:", cmd)
            result = sp.run(cmd, env=env)
            if result.returncode != 0:
                raise RuntimeError(f"Subprocess failed with exit code: {result.returncode}.")
            run_id = json.load(fp_info)["run_id"]

        # Extract metric.
        metric_name = self.config["metric"]["name"]
        run = client.get_run(run_id)
        metrics = run.data.metrics
        metric = metrics[metric_name]
        return metric


def main(args):
    storage = get_optuna_storage()
    study = optuna.load_study(study_name=args.sweep_id, storage=storage)

    agent = Agent(args.sweep_id, cmd_args=args.args)
    count = args.count if args.count is not None else agent.default_count
    study.optimize(agent, n_trials=count)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
