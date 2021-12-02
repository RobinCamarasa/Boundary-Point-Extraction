from itertools import product
from pathlib import Path
import pandas as pd
from diameter_learning.settings import MLRUN_PATH


id_ = 5
experiment_folder = Path(MLRUN_PATH / f'{id_}')
runs = list(experiment_folder.glob('[!m]*'))
assert len(runs) == 12

# Get the metrics name
metrics = [
    metric_results.parent.stem
    for metric_results in runs[0].glob('**/result.csv')
    ]

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
        values.append(row.sum() / row.count())
    results['values'] = values
    results['slice_id'] = results.index
    results['patient_id'] = results['slice_id'].apply(
        lambda x: '_'.join(x.split('_')[:2])
        )
    aggregation = results.groupby('patient_id').agg(['mean', 'std'])
