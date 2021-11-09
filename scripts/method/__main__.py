import argparse
from pathlib import Path

import random
import numpy as np
import torch
import mlflow
import pytorch_lightning as pl

from diameter_learning.plmodules import (
    CarotidArteryChallengeDiameterModule
    )
from pytorch_lightning.callbacks import ModelCheckpoint


# Set parser
parser = argparse.ArgumentParser(
	description='Process hyperparameters'
	)

child_parser = parser.add_argument_group("Experiment parameters")
child_parser.add_argument('--experiment_seed', type=int)
parser = pl.Trainer.add_argparse_args(parser)
parser = CarotidArteryChallengeDiameterModule.add_model_specific_args(
    parser
    )
hparams = parser.parse_args()

# Log mlflow
mlflow.set_tag('experiment', 'method')

# Set experiment seed
random.seed(hparams.experiment_seed)
np.random.seed(seed=hparams.experiment_seed)
torch.manual_seed(seed=hparams.experiment_seed)
torch.cuda.manual_seed(hparams.experiment_seed)
torch.cuda.manual_seed_all(hparams.experiment_seed)

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
        callbacks=[ModelCheckpoint(monitor="validation_dice")]
        )
trainer.fit(model)
trainer.test(model)
