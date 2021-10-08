import pandas as pd
from tqdm import tqdm
from pathlib import Path
from diameter_learning.apps import CarotidChallengeDataset
from diameter_learning.transforms import (
    LoadCarotidChallengeSegmentation, LoadCarotidChallengeAnnotations
    )
import mlflow
import matplotlib.pyplot as plt
from monai.transforms import (Compose, LoadImaged, SpatialPadd)
from pandas_profiling import ProfileReport
from diameter_learning.settings import DATA_PRE_PATH


for folds, subset in zip(
    [
        [0], [1], [2], [3], [4], [0, 1, 2, 3], 
        [0, 1, 2, 3, 4]
        ],
    [
        'training_fold_0', 'training_fold_1',
        'training_fold_2', 'training_fold_3',
        'test', 'training_set', 'all_dataset'
        ],
    ):
    print(f"\n\nSet: {subset}")
    carotid_challenge_dataset = CarotidChallengeDataset(
        root_dir=DATA_PRE_PATH,
        annotations=('internal_right', 'internal_left'),
        transforms=Compose(
            [
                LoadImaged("image"),
                LoadCarotidChallengeAnnotations("gt"),
                LoadCarotidChallengeSegmentation(),
                SpatialPadd(["image", "gt_lumen_processed_contour"], (160, 768))
                ]
            ),
        seed=0,
        folds=folds,
        num_fold=5,
        cache_rate=0
        )
    data: dict = {
        'dimension_x': [],
        'dimension_y': [],
        'volume': [],
        'diameter': [],
        'spacing_x': [],
        'spacing_y': []
        }
    output_path: Path = Path(
        mlflow.get_artifact_uri().split('://')[-1]
        ) / subset
    output_path.mkdir()
    for i, carotid_challenge_slice in tqdm(enumerate(carotid_challenge_dataset)):
        if i < 5:
            plt.clf()
            fig, ax = plt.subplots(nrows=2, ncols=1)
            ax = ax.ravel()
            ax[0].axis('off')
            ax[0].imshow(carotid_challenge_slice['image'][0], cmap='gray')
            ax[1].axis('off')
            ax[1].imshow(carotid_challenge_slice['image'][0], cmap='gray')
            ax[1].imshow(
                carotid_challenge_slice['gt_lumen_processed_contour'][0],
                cmap='Reds', alpha=0.3
                )
            plt.savefig(output_path / f'sample_{i}.png', dpi=300)

        # Obtain mesurements of interest
        data['dimension_x'].append(
            carotid_challenge_slice['image_meta_dict']['spatial_shape'][1]
            )        
        data['dimension_y'].append(
            carotid_challenge_slice['image_meta_dict']['spatial_shape'][2]
            )        
        data['spacing_x'].append(carotid_challenge_slice[
                'image_meta_dict'
                ]['spacing'][1]
            )
        data['spacing_y'].append(
            carotid_challenge_slice[
                'image_meta_dict'
                ]['spacing'][2]
            )
        data['volume'].append(
            carotid_challenge_slice[
                'gt_lumen_processed_contour'
                ].sum() * data['spacing_x'][-1] * data['spacing_y'][-1]
            )
        data['diameter'].append(
            carotid_challenge_slice['gt_lumen_processed_diameter'] *
            data['spacing_x'][-1]
            )
    # Generate report
    profile = ProfileReport(
        pd.DataFrame(data),
        title=f"Profiling Report: {subset}"
        )
    profile.to_file(output_path / f'analysis.html')
