from pathlib import Path
import argparse
import pytorch_lightning as pl
import mlflow
import torch
from torch.utils.data import DataLoader
from diameter_learning.handlers import (
    RelativeDiameterError, DiceCallback, ImageVisualizer,
    SegmentationVisualizer, LandmarksVisualizer, GroundTruthVisualizer,
    HaussdorffCallback, AbsoluteDiameterError
    )
from monai.transforms import KeepLargestConnectedComponent
from diameter_learning.settings import MLRUN_PATH
from baselines.geodesic.plmodules import CarotidArteryChallengeGeodesicNet
from diameter_learning.transforms import SegmentationToDiameter


# Parse user input
parser = argparse.ArgumentParser(
    description='Analyse experiment'
    )
parser.add_argument('--run_id', type=str)
parser.add_argument('--metric', type=str)
params = parser.parse_args()

artifact_path: Path = Path(
    mlflow.get_artifact_uri().split('file://')[-1]
    )

# Load model
experiment_path = list(MLRUN_PATH.glob(f'**/{params.run_id}'))[0]
checkpoint_path = list(
    experiment_path.glob(f'**/*{params.metric}*.ckpt')
    )[0]
model = CarotidArteryChallengeGeodesicNet.load_from_checkpoint(
    str(checkpoint_path)
    )
model.hparams.training_cache_rate = 0

def get_pred_seg(batch, module):
    segmentation = torch.nn.Softmax(dim=1)(module(
        batch
        ))[0, [1]]
    segmentation = torch.unsqueeze(
        KeepLargestConnectedComponent(1)(segmentation > 0.5), 0
        )
    return segmentation


# Define callbacks useful methods
segmentation_to_diameter = SegmentationToDiameter(.5)
get_input = lambda batch: batch['image']
get_gt_seg = lambda batch: batch['gt_lumen_processed_contour']
get_gt_diam = lambda batch: batch['gt_lumen_processed_diameter']
get_gt_landmarks = lambda batch: batch['gt_lumen_processed_landmarks']

get_pred_diam = lambda batch, module: segmentation_to_diameter(
    get_pred_seg(batch, module)
)
slice_id_key = 'slice_id'
spacing_key = 'image_meta_dict_spacing'

trainer = pl.Trainer(
        progress_bar_refresh_rate=1,
        default_root_dir=artifact_path,
        gpus=1,
        callbacks=[
            RelativeDiameterError(
                result_path=artifact_path,
                get_gt=get_gt_diam,
                get_pred=get_pred_diam,
                slice_id_key=slice_id_key
                ),
            DiceCallback(
                result_path=artifact_path,
                get_gt=get_gt_seg,
                get_pred=get_pred_seg,
                slice_id_key=slice_id_key
                ),
            ImageVisualizer(
                result_path=artifact_path,
                get_pred=None,
                get_gt=None,
                get_input=get_input,
                slice_id_key=slice_id_key,
                number_of_images=None
                ),
            GroundTruthVisualizer(
                result_path=artifact_path,
                get_pred=get_pred_seg,
                get_gt=get_gt_seg,
                get_input=get_input,
                slice_id_key=slice_id_key,
                number_of_images=None
                ),
            SegmentationVisualizer(
                result_path=artifact_path,
                get_pred=get_pred_seg,
                get_gt=get_gt_seg,
                get_input=get_input,
                slice_id_key=slice_id_key,
                number_of_images=None
                ),
            HaussdorffCallback(
                result_path=artifact_path,
                get_gt=get_gt_seg,
                get_pred=get_pred_seg,
                slice_id_key=slice_id_key,
                spacing_key=spacing_key
                ),
            AbsoluteDiameterError(
                result_path=artifact_path,
                get_gt=get_gt_diam,
                get_pred=get_pred_diam,
                slice_id_key=slice_id_key,
                spacing_key=spacing_key
                )
            ]
        )

# Test model
trainer.test(model)
