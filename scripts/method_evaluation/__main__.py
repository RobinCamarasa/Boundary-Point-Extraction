from pathlib import Path
import argparse
import pytorch_lightning as pl
import mlflow
import torch
from torch.utils.data import DataLoader
from monai.transforms import KeepLargestConnectedComponent
from diameter_learning.handlers import (
    RelativeDiameterError, DiceCallback, ImageVisualizer,
    SegmentationVisualizer, LandmarksVisualizer, GroundTruthVisualizer,
    HaussdorffCallback, AbsoluteDiameterError
    )
from diameter_learning.settings import MLRUN_PATH
from diameter_learning.plmodules import CarotidArteryChallengeDiameterModule


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
model = CarotidArteryChallengeDiameterModule.load_from_checkpoint(
    str(checkpoint_path)
    )
model.hparams.training_cache_rate=0

# Change post processing
def post_process(x):
    """Compute the segmentation, the center of mass, the radiuses
    and the diameter

    :return: The segmentation, the center of mass, the radiuses
        and the diameter
    """
    segmentation = model.sigmoid(model.model(x['image']))[0]
    segmentation = torch.unsqueeze(
        KeepLargestConnectedComponent(1)(segmentation > 0.5), -1
        ).unsqueeze(0)
    center_of_mass = model.center_of_mass_extractor(
        segmentation
        )
    radiuses = model.gaussian_radius_extractor(
        segmentation, center_of_mass
        )
    diameter = model.vanilla_diameter_extractor(
        torch.mean(radiuses, 0)
        )
    return segmentation, center_of_mass, radiuses, diameter

# Define callbacks useful methods
get_input=lambda batch: batch['image']

get_gt_seg = lambda batch: batch['gt_lumen_processed_contour']
get_gt_diam = lambda batch: batch['gt_lumen_processed_diameter']
get_gt_landmarks = lambda batch: batch['gt_lumen_processed_landmarks']
get_pred_seg = lambda batch, module: post_process(
    batch
    )[0][:, :, :, :, 0]
get_pred_diam = lambda batch, module: post_process(
    batch
    )[3][:, :, 0]
get_pred_landmarks = lambda batch, module: post_process(batch)[1:3]
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
            LandmarksVisualizer(
                result_path=artifact_path,
                get_pred=get_pred_landmarks,
                get_gt=get_gt_landmarks,
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
