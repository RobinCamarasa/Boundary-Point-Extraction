from pathlib import Path
import argparse
import pytorch_lightning as pl
import mlflow
from torch.utils.data import DataLoader
from diameter_learning.handlers import (
    RelativeDiameterError, DiceCallback, ImageVisualizer,
    SegmentationVisualizer, LandmarksVisualizer,
    HaussdorffCallback
    )
from diameter_learning.settings import MLRUN_PATH
from diameter_learning.plmodules import (
    CarotidArteryChallengeDiameterModule,
    CarotidArteryChallengeDiameterResNet
    )
from monai.transforms import (
    KeepLargestConnectedComponent
    )


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
import ipdb; ipdb.set_trace() ###!!!BREAKPOINT!!!
experiment_path = list(MLRUN_PATH.glob(f'**/{params.run_id}'))[0]
checkpoint_path = list(experiment_path.glob('**/epoch=*.ckpt'))[0]
model = CarotidArteryChallengeDiameterModule.load_from_checkpoint(
    str(checkpoint_path),
    )

# Define trainer
trainer = pl.Trainer(
        progress_bar_refresh_rate=1,
        default_root_dir=artifact_path,
        gpus=1,
        callbacks=[
            RelativeDiameterError(
                artifact_path,
                gt_key='gt_lumen_processed_diameter',
                slice_id='slice_id',
                forward_to_pred=lambda batch, module: module(
                    batch
                    )[3][:, :, 0]
                ),
            DiceCallback(
                artifact_path,
                gt_key='gt_lumen_processed_contour',
                slice_id='slice_id',
                forward_to_pred=lambda batch, module: 1. * KeepLargestConnectedComponent(applied_labels=[1])(
                        module(
                            batch
                        )[0][:, :, :, :, 0] > .5
                    )            
                ),
            ImageVisualizer(
                artifact_path,
                number_of_images=0
                ),
            LandmarksVisualizer(
                artifact_path,
                number_of_images=0
                ),
            SegmentationVisualizer(
                artifact_path,
                forward_to_pred=lambda batch, module: 1. * KeepLargestConnectedComponent(applied_labels=[1])(
                        module(
                            batch
                        )[0][:, :, :, :, 0] > .5
                    ),
                number_of_images=None
                ),
            HaussdorffCallback(
                artifact_path,
                gt_key='gt_lumen_processed_contour',
                slice_id='slice_id',
                forward_to_pred=lambda batch, module: 1. * KeepLargestConnectedComponent(applied_labels=[1])(
                        module(
                            batch
                        )[0][0, :, :, :, 0] > .5,
                    )
                )
            ]
        )

# Test model
trainer.test(model)
