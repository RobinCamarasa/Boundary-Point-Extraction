import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from diameter_learning.settings import MLRUN_PATH


def get_training_param(run_, param):
    """Get a parameter of the considered run

    :param run_: Run of the experiment
    :param run_: Considered run
    :return: The experiment name
    """
    with (run_ / 'params' / 'run_id').open('r') as handle:
        run__ = handle.readline()
    training_exp = list(MLRUN_PATH.glob(f'*/{run__}/meta.yaml'))[0].parent
    with (training_exp / 'params' / param).open('r') as handle:
        return handle.readline()


def get_experiment_name(experiment_folder):
    """Get the name of the experiment from an experiment folder

    :param experiment_folder: Folder of the experiment
    :return: The experiment name
    """
    with (experiment_folder / 'meta.yaml').open('r') as handle:
        return [line[6:] for line in handle.readlines() if 'name:' in line][0]


parser = argparse.ArgumentParser(
	description='Analyse experiment'
	)
parser.add_argument('--experiment_id', type=int)
params = parser.parse_args()
experiment_folder = Path(MLRUN_PATH / f'{params.experiment_id}')
mlflow.set_tag('experiment', get_experiment_name(experiment_folder))
runs = list(experiment_folder.glob('[!m]*'))
assert len(runs) == 12

artifact_path: Path = Path(
    mlflow.get_artifact_uri().split('file://')[-1]
    )

# Get the metrics name
metrics = [
    metric_results.parent.stem
    for metric_results in runs[0].glob('**/result.csv')
    ]
dataframes = pd.DataFrame()

# Loop over the metrics
for metric in metrics:
    results = {f'seed_{i}': pd.DataFrame() for i in range(3)}

    # Loop over the run
    for i, run_ in enumerate(runs):
        seed = get_training_param(run_, 'seed')
        result = pd.read_csv(
            run_ / 'artifacts' / metric / 'result.csv'
            ).set_index('slice_id')
        result = result.rename(columns={'values': f'values_{run_.stem}'})
        results[f'seed_{seed}'] = results[f'seed_{seed}'].join(
            result, how='outer'
            )
    for key in results:
        dataframes[f'{metric.lower()}_{key}'] =  results[key][list(results[key])].sum(axis=1)

dataframes.to_csv(artifact_path / 'metrics.csv')
