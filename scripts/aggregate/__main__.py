import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from diameter_learning.settings import MLRUN_PATH


parser = argparse.ArgumentParser(
	description='Analyse experiment'
	)
parser.add_argument('--experiment_id', type=int)
params = parser.parse_args()
experiment_folder = Path(MLRUN_PATH / f'{params.experiment_id}')
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
dataframes = {metric: None  for metric in metrics}

# Loop over the metrics
for metric in metrics:
    results = pd.DataFrame()

    # Loop over the run
    for i, tmp in enumerate(runs):
        result = pd.read_csv(
            tmp / 'artifacts' / metric / 'result.csv'
            ).set_index('slice_id')
        result = result.rename(columns={'values': f'values_{tmp.stem}'})
        results = results.join(
            result, how='outer',
            )
    values = []
    for i, row in results.iterrows():
        value = row.sum() / row.count()
        if value != np.inf:
            values.append(value)
        else:
            values.append(np.nan)
    results['values'] = values
    dataframes[metric] = results[['values']]

# Filter empty segmentations
dataframes['HaussdorffCallback'] = dataframes['HaussdorffCallback'][
        dataframes['HaussdorffCallback']['values'].notnull()
        ]
mlflow.log_metric(
        'missing',
        100 * (
                1 - dataframes['HaussdorffCallback'].shape[0] /\
                dataframes['DiceCallback'].shape[0]
            )
        )

# Loop over dataframes dictionnary
for key, dataframe in dataframes.items():
    dataframe['slice_id'] = dataframe.index
    dataframe['patient_id'] = dataframe['slice_id'].apply(
        lambda x: '_'.join(x.split('_')[:2])
        )
    if not 'Dice' in key and not 'Haussdorff' in key:
        # Filter dataframe
        dataframe = dataframe[
            dataframe.index.isin(dataframes['HaussdorffCallback'].index)
            ]
    dataframe[['values']].to_csv(
        artifact_path / f'result_{key}.csv'
        )
    aggregation = dataframe.groupby('patient_id').mean()
    aggregation[['values']].to_csv(
        artifact_path / f'aggregated_{key}.csv'
        )
    mlflow.log_metric(f'{key}_mean', aggregation['values'].mean())
    mlflow.log_metric(f'{key}_std', aggregation['values'].std())
