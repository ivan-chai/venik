import os
import mlflow
import json
from pytorch_lightning.loggers import MLFlowLogger as MLFlowLoggerPL
from pytorch_lightning.loggers.mlflow import _get_resolve_tags
from mlflow.tracking import MlflowClient


class MLFlowLogger(MLFlowLoggerPL):
    """Logger extension that uses environment variables.

    List of environment variables:
    - MLFLOW_INFO_FILE
    - MLFLOW_TRACKING_URI
    - MLFLOW_TRACKING_USERNAME
    - MLFLOW_TRACKING_PASSWORD
    - MLFLOW_EXPERIMENT_NAME
    - MLFLOW_RUN_NAME
    - MLFLOW_TAGS
    - MLFLOW_PARENT_RUN_ID
    """

    def __init__(self, *,
                 experiment_name=None,
                 run_name=None,
                 project=None,  # Alias for experiment_name.
                 name=None,  # Alias for run_name.
                 tracking_uri=None,
                 tags=None,
                 run_id=None,
                 **kwargs):
        if tracking_uri is None:
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", None)
        run_name = run_name or name
        if run_name is None:
            run_name = os.environ.get("MLFLOW_RUN_NAME", None)
        tags = tags or {}
        if "MLFLOW_TAGS" in os.environ:
            for pair in os.environ["MLFLOW_TAGS"].split(";"):
                pair = pair.strip()
                if pair:
                    name, value = pair.split("=")
                    tags[name] = value

        parent_run_id = os.environ.get("MLFLOW_PARENT_RUN_ID", None)
        if (parent_run_id is not None) and (run_id is None):
            mlflow.set_tracking_uri(tracking_uri)
            client = MlflowClient()
            experiment_id = client.get_run(parent_run_id).info.experiment_id
            experiment_name = client.get_experiment(experiment_id).name

            if run_name is not None:
                from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME

                if MLFLOW_RUN_NAME in tags:
                    log.warning(
                        f"The tag {MLFLOW_RUN_NAME} is found in tags. The value will be overridden by {run_name}."
                    )
                tags[MLFLOW_RUN_NAME] = run_name

            resolve_tags = _get_resolve_tags()
            run = mlflow.start_run(experiment_id=experiment_id,
                                   run_name=run_name,
                                   nested=True,
                                   parent_run_id=parent_run_id,
                                   tags=resolve_tags(tags))
            run_id = run.info.run_id
        else:
            experiment_name = experiment_name or project or os.environ.get("MLFLOW_EXPERIMENT_NAME", "lightning_logs")
        super().__init__(experiment_name=experiment_name,
                         run_name=run_name,
                         tracking_uri=tracking_uri,
                         tags=tags,
                         run_id=run_id,
                         **kwargs)

        if "MLFLOW_INFO_FILE" in os.environ:
            # Create run and log run_id.
            client = self.experiment
            with open(os.environ["MLFLOW_INFO_FILE"], "w") as fp:
                json.dump({"run_id": self._run_id}, fp)
