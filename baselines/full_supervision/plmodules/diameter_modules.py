"""File containing geodesic trainers"""
from typing import Any, Mapping, Tuple, List

import monai
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from monai.transforms import (
    LoadImaged, SpatialPadd, AsChannelFirstd, AddChanneld,
    ToTensord
    )
from monai.networks.nets import BasicUNet, UNet
from monai.losses import DiceLoss
from monai.utils import LossReduction

from diameter_learning.plmodules import CarotidArteryChallengeModule
from diameter_learning.settings import DATA_PRE_PATH
from diameter_learning.apps import CarotidChallengeDataset
from diameter_learning.nets.layers import (
    CenterOfMass2DExtractor, VanillaDiameterExtractor,
    MomentGaussianRadiusExtractor
    )


class CarotidArteryFullSupervisionNet(
    CarotidArteryChallengeModule, pl.LightningModule
    ):
    def __init__(self, hparams, *args, **kwargs):
        # Process the arguments
        super(
            CarotidArteryFullSupervisionNet, self
            ).__init__(hparams, *args, **kwargs)
        self.save_hyperparameters(hparams)
        self._process_args()
        super()._set_dataset()
        super()._set_transform_toolchain()

        # Define torch layers and modules
        self.model: torch.nn.Module = BasicUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2
            )
        self.loss = torch.nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax(dim=1).float()

    def forward(self, x) -> Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
            ]:
        """Compute the segmentation, the center of mass, the radiuses
        and the diameter

        :return: The segmentation, the center of mass, the radiuses
            and the diameter
        """
        # The unsqueeze is there to make the segmentation 3D
        # with a third dimension of size 1
        return self.softmax(self.model(x['image']))

    def compute_losses(
        self, batch, batch_idx
        ) -> torch.Tensor:
        """Compute the different losses

        :param batch: Batch evaluated
        :param batch_idx: Id of the batch evaluated
        """
        pred=self(batch)
        gt=batch['gt_lumen_processed_contour'][:, 0].long()
        return self.loss(pred, gt)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Define the training step

        :param batch: Batch of the current training step
        :param batch_idx: Id of the current training batch
        :return: The value of the loss
        """
        loss = self.compute_losses(batch, batch_idx)
        self.log('training_loss', loss.item(), on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx) -> DataLoader:
        """Define the validation step

        :param batch: Batch of the current validation step
        :param batch_idx: Id of the current validation batch
        :return: The value of the loss
        """
        loss = self.compute_losses(batch, batch_idx)
        dice = 1 - DiceLoss(reduction=LossReduction.MEAN)(
            self(batch)[:, [0]],
            batch['gt_lumen_processed_contour']
            )
        self.log('validation_dice', dice, on_epoch=True)
        self.log('validation_loss', loss, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx) -> DataLoader:
        """Define the test step

        :param batch: Batch of the current test step
        :param batch_idx: Id of the current test batch
        :return: The value of the loss
        """
        loss = self.compute_losses(batch, batch_idx)
        self.log('test_loss', loss, on_epoch=True)
        return loss


    @classmethod
    def add_model_specific_args(cls, parent_parser):
        """Add the argument in the argparser that are model specific

        :param cls: Class
        """
        parent_parser = super(
            CarotidArteryFullSupervisionNet, cls
            ).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("VoxelWiseNet")
        parser.add_argument('--tolerance', type=float)
        return parent_parser

    def _process_args(self):
        try:
            super()._process_args()
        except:
            pass
