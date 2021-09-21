import sys
import shutil
import mlflow
from typing import Callable
from mlflow import log_artifacts
from tempfile import mkdtemp
from pathlib import Path
from pytorch_lightning.metrics.functional import accuracy


def set_mlflow(f: callable) -> callable:
    """
    Decorator to put before a main function to log the experiment

    Args:
        f: main function
    """
    def wrapper(*args, **kwargs):
        # Initiate mlflow instance
        with mlflow.start_run():
            # Create result path
            result_path = Path(mkdtemp())

            f(result_path, *args, **kwargs)

            # Log the output
            log_artifacts(result_path)
            shutil.rmtree(result_path.resolve(), ignore_errors=True)
    return wrapper

