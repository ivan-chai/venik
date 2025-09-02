import os
import json
from pytorch_lightning.loggers import MLFlowLogger as MLFlowLoggerPL


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
    """

    def __init__(self, *,
                 experiment_name=None,
                 run_name=None,
                 project=None,  # Alias for experiment_name.
                 name=None,  # Alias for run_name.
                 tracking_uri=None,
                 tags=None,
                 **kwargs):
        experiment_name = experiment_name or project
        run_name = run_name or name
        if experiment_name is None:
            experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "lightning_logs")
        if tracking_uri is None:
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", None)
        if run_name is None:
            run_name = os.environ.get("MLFLOW_RUN_NAME", None)
        if (tags is None) and ("MLFLOW_TAGS" in os.environ):
            tags = {}
            for pair in os.environ["MLFLOW_TAGS"].split(";"):
                pair = pair.strip()
                if pair:
                    name, value = pair.split("=")
                    tags[name] = value

        super().__init__(experiment_name=experiment_name,
                         run_name=run_name,
                         tracking_uri=tracking_uri,
                         tags=tags,
                         **kwargs)

        if "MLFLOW_INFO_FILE" in os.environ:
            # Create run and log run_id.
            client = self.experiment
            with open(os.environ["MLFLOW_INFO_FILE"], "w") as fp:
                json.dump({"run_id": self._run_id}, fp)
