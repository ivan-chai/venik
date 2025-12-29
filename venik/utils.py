import certifi
import optuna
import re
import os
import json
import sqlalchemy as sa
from urllib.parse import quote_plus
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool


OPTUNA_DB = "Optuna"
SQL_ENGINE_KWARGS = {"connect_args": {
    "connect_timeout": 10,
    "ssl_ca": certifi.where(),
    "ssl_verify_cert": True,
    "ssl_verify_identity": True
}}


def get_mysql_url():
    if "OPTUNA_URL" in os.environ:
        if "OPTUNA_USER" not in os.environ or "OPTUNA_PASSWORD" not in os.environ:
            raise ValueError("Need user and password for OPTUNA")
        host = os.environ["OPTUNA_URL"]
        user = os.environ["OPTUNA_USER"]
        password = os.environ["OPTUNA_PASSWORD"]
    else:
        uri = os.environ["MLFLOW_TRACKING_URI"]
        user = os.environ["MLFLOW_TRACKING_USERNAME"]
        password = os.environ["MLFLOW_TRACKING_PASSWORD"]

        if not uri.endswith(":8080"):
            raise NotImplementedError("Only 8080 port for MLflow is supported.")
        host = uri[:-5].split("://")[1]
    url = f"mysql+pymysql://{user}:{quote_plus(password)}@{host}"
    return url


def get_optuna_storage():
    storage = optuna.storages.RDBStorage(
        url=get_mysql_url() + f"/{OPTUNA_DB}",
        engine_kwargs=SQL_ENGINE_KWARGS,
    )
    return storage


class SweepDB:
    def __init__(self, engine=None):
        if engine is None:
            engine = create_engine(
                get_mysql_url() + f"/{OPTUNA_DB}",
                **SQL_ENGINE_KWARGS
            )
        self.engine = engine
        with self.engine.begin() as conn:
            query = sa.text("""
            SELECT *
                    FROM INFORMATION_SCHEMA.TABLES
                    WHERE TABLE_NAME = 'Sweeps';
            """)
            result = conn.execute(query).mappings().all()
            if not result:
                # Create Sweeps table.
                query = sa.text("""
                CREATE TABLE Sweeps (
                    sweep_id TEXT(1024),
                    config TEXT(65535)
                );
                """)
                result = conn.execute(query)

    def get_sweeps_list(self):
        with self.engine.begin() as conn:
            query = sa.text("""
            SELECT sweep_id
                    FROM Sweeps;
            """)
            result = conn.execute(query).mappings().all()
        return result

    def add_sweep(self, name, config):
        with self.engine.begin() as conn:
            query = sa.text(f"""
            INSERT INTO Sweeps (sweep_id, config)
                    VALUES ('{name}', '{json.dumps(config)}');
            """)
            result = conn.execute(query)
            if result.rowcount != 1:
                raise RuntimeError(f"Sweep creation failed: {result.rowcount} records affected")
        return result

    def del_sweep(self, name):
        with self.engine.begin() as conn:
            query = sa.text(f"""
            DELETE FROM Sweeps
                    WHERE sweep_id='{name}';
            """)
            conn.execute(query)

    def get_sweep_config(self, name):
        with self.engine.begin() as conn:
            query = sa.text(f"""
            SELECT config
                    FROM Sweeps
                    WHERE sweep_id='{name}';
            """)
            result = conn.execute(query).mappings().all()
            if len(result) == 0:
                raise KeyError("Sweep not found")
            if len(result) > 1:
                raise RuntimeError("Multiple Sweeps")
        return json.loads(result[0]["config"])


class CategoricalSampler:
    def __init__(self, name, values):
        self.name = name
        self.values = values

    def __call__(self, trial):
        return trial.suggest_categorical(self.name, self.values)


class IntegerSampler:
    def __init__(self, name, min, max, log=False):
        self.name = name
        self.min = min
        self.max = max
        self.log = log

    def __call__(self, trial):
        return trial.suggest_int(self.name, low=self.min, high=self.max, log=self.log)


class FloatSampler:
    def __init__(self, name, min, max, log=False):
        self.name = name
        self.min = min
        self.max = max
        self.log = log

    def __call__(self, trial):
        if self.log:
            return trial.suggest_loguniform(self.name, low=self.min, high=self.max)
        else:
            return trial.suggest_uniform(self.name, low=self.min, high=self.max)


class ParameterSampler:
    def __init__(self, parameters):
        self.parameters = {}
        for name, spec in parameters.items():
            assert isinstance(spec, dict)
            spec = dict(spec)
            if "values" in spec:
                self.parameters[name] = CategoricalSampler(name, **spec)
            elif ("min" in spec) and ("max" in spec):
                distribution = spec.pop("distribution", None)
                if distribution == "log_uniform_values":
                    spec["log"] = True
                elif distribution is not None:
                    raise ValueError(f"Unknown distribution: {distribution}")
                if isinstance(spec["min"], int) and isinstance(spec["max"], int):
                    self.parameters[name] = IntegerSampler(name, **spec)
                else:
                    self.parameters[name] = FloatSampler(name, **spec)
            else:
                raise NotImplementedError(f"Unexpected specification: {spec}")

    def sample(self, trial):
        """Get parameters."""
        return {name: sampler(trial)
                for name, sampler in self.parameters.items()}
