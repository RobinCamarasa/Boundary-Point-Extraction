from pathlib import Path
import argparse
import pytorch_lightning as pl
import mlflow
from torch.utils.data import DataLoader
from diameter_learning.handlers import (
    RelativeDiameterError, DiceCallback, ImageVisualizer,
    SegmentationVisualizer, LandmarksVisualizer, GroundTruthVisualizer,
    HaussdorffCallback, AbsoluteDiameterError
    )
from diameter_learning.settings import MLRUN_PATH
from baselines.geodesic.plmodules import CarotidArteryChallengeGeodesicNet


# Parse user input
parser = argparse.ArgumentParser(
	description='Analyse experiment'
	)
parser.add_argument('--run_id', type=str)
params = parser.parse_args()

artifact_path: Path = Path(
    mlflow.get_artifact_uri().split('file://')[-1]
    )

# Load model
experiment_path = list(MLRUN_PATH.glob(f'**/{params.run_id}'))[0]
checkpoint_path = list(experiment_path.glob('**/epoch=*.ckpt'))[0]
model = CarotidArteryChallengeGeodesicNet.load_from_checkpoint(
    str(checkpoint_path)
    )
model.hparams.training_cache_rate=0

# Define callbacks useful methods
get_input=lambda batch: batch['image']

get_gt_seg = lambda batch: batch['gt_lumen_processed_contour']
get_gt_diam = lambda batch: batch['gt_lumen_processed_diameter']
get_gt_landmarks = lambda batch: batch['gt_lumen_processed_landmarks']
get_pred_seg = lambda batch, module: module(
    batch
    )[:, [1], :, :]
# TODO: Implement
get_pred_landmarks = lambda batch, module: module(batch)[1:3]
slice_id_key = 'slice_id'
spacing_key = 'image_meta_dict_spacing'

trainer = pl.Trainer(
        progress_bar_refresh_rate=1,
        default_root_dir=artifact_path,
        gpus=1,
        callbacks=[
            # RelativeDiameterError(
            #     result_path=artifact_path,
            #     get_gt=get_gt_diam,
            #     get_pred=get_pred_diam,
            #     slice_id_key=slice_id_key
            #     ),
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
            # AbsoluteDiameterError(
            #     result_path=artifact_path,
            #     get_gt=get_gt_diam,
            #     get_pred=get_pred_diam,
            #     slice_id_key=slice_id_key,
            #     spacing_key=spacing_key
            #     )
            ]
        )

# Test model
trainer.test(model)
