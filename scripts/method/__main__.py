from pathlib import Path
import argparse
import pytorch_lightning as pl
import mlflow
from torch.utils.data import DataLoader
from diameter_learning.plmodules import (
    CarotidArteryChallengeDiameterModule
    )


parser = argparse.ArgumentParser(
	description='Process hyperparameters'
	)
parser = pl.Trainer.add_argparse_args(parser)
parser = CarotidArteryChallengeDiameterModule.add_model_specific_args(
    parser
    )
hparams = parser.parse_args()

# Create module
artifact_path = mlflow.get_artifact_uri().split('file://')[-1]
model: CarotidArteryChallengeDiameterModule \
    = CarotidArteryChallengeDiameterModule(
        hparams
        )

artifact_path: Path = Path(
    mlflow.get_artifact_uri().split('file://')[-1]
    )

mlflow.pytorch.autolog()
trainer = pl.Trainer.from_argparse_args(
        hparams, progress_bar_refresh_rate=1,
        default_root_dir=artifact_path,
        gpus=1,
        )
trainer.fit(model)
trainer.test(model)
