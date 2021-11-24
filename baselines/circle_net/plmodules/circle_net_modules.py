"""File containing CircleNet trainers"""
from typing import Tuple
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
from baselines.circle_net.nets import BasicCircleNet
from baselines.circle_net.transforms import (
    TransformToCircleNetMaps, CircleNetToSegmentation
    )
from monai.losses import FocalLoss


class CarotidArteryChallengeCircleNet(
    CarotidArteryChallengeModule, pl.LightningModule
    ):
    def __init__(self, hparams, *args, **kwargs):
        # Process the arguments
        super(
            CarotidArteryChallengeCircleNet, self
            ).__init__(hparams, *args, **kwargs)
        self.save_hyperparameters(hparams)
        self._process_args()
        super()._set_dataset()
        super()._set_transform_toolchain()

        self.postprocess_transforms = [
            TransformToCircleNetMaps(),
            ToTensord(
                [
                    "image", "gt_lumen_processed_diameter",
                    "gt_lumen_processed_landmarks",
                    "radius", "heatmap", "radius_mask",
                    "gt_lumen_processed_contour",
                    ]
                )
            ]
        self.to_segmentation_transform = CircleNetToSegmentation()

        # Define torch layers and modules
        self.model: torch.nn.Module = BasicCircleNet(
            spatial_dims=2,
            in_channels=1,
            out_channels_heatmap=1,
            out_channels_radius=1
            )
        self.sigmoid = torch.nn.Sigmoid().float()
        self.relu = torch.nn.ReLU().float()
        self.heatmap_loss = torch.nn.MSELoss().float()
        self.radius_loss = torch.nn.L1Loss()

    def forward(self, x) -> Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
            ]:
        """Compute the segmentation, the center of mass, the radiuses
        and the diameter

        :return: The segmentation, the center of mass, the radiuses
            and the diameter
        """
        return self.model(x['image'])

    def compute_losses(
        self, batch, batch_idx
        ) -> torch.Tensor:
        """Compute the different losses

        :param batch: Batch evaluated
        :param batch_idx: Id of the batch evaluated
        """
        prediction = self(batch)

        # Compute heatmap loss
        heatmap_prediction = prediction['heatmap']
        ground_truth_heatmap = batch['heatmap']
        heatmap_loss_value = self.heatmap_loss(
            self.sigmoid(heatmap_prediction.float()),
            ground_truth_heatmap.float()
            )

        # Compute radius loss
        radius_prediction = (
            self.relu(prediction['radius']) * batch['radius_mask']
            ).sum(axis=(-1, -2))
        ground_truth_radius = batch['radius'].sum(axis=(-1, -2))
        radius_loss_value = self.radius_loss(
            radius_prediction, ground_truth_radius
            )

        # Total loss value
        total_loss_value = self.hparams.loss_radius_weighting *\
            radius_loss_value + self.hparams.loss_heatmap_weighting *\
            heatmap_loss_value
        return total_loss_value, heatmap_loss_value, radius_loss_value

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Define the training step

        :param batch: Batch of the current training step
        :param batch_idx: Id of the current training batch
        :return: The value of the loss
        """
        total_loss, heatmap_loss, radius_loss = self.compute_losses(
            batch, batch_idx
            )
        self.log('training_heatmap_loss', heatmap_loss.item(), on_epoch=True)
        self.log('training_radius_loss', radius_loss.item(), on_epoch=True)
        self.log('training_loss', total_loss.item(), on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx) -> DataLoader:
        """Define the validation step

        :param batch: Batch of the current validation step
        :param batch_idx: Id of the current validation batch
        :return: The value of the loss
        """
        total_loss, heatmap_loss, radius_loss = self.compute_losses(
            batch, batch_idx
            )
        dice = 1 - DiceLoss(reduction=LossReduction.MEAN)(
            self.to_segmentation_transform(self(batch)),
            batch['gt_lumen_processed_contour']
            )
        self.log('validation_dice', dice.item(), on_epoch=True)
        self.log('validation_heatmap_loss', heatmap_loss.item(), on_epoch=True)
        self.log('validation_radius_loss', radius_loss.item(), on_epoch=True)
        self.log('validation_loss', total_loss.item(), on_epoch=True)
        return total_loss

    def test_step(self, batch, batch_idx) -> DataLoader:
        """Define the test step

        :param batch: Batch of the current test step
        :param batch_idx: Id of the current test batch
        :return: The value of the loss
        """
        total_loss, heatmap_loss, radius_loss = self.compute_losses(
            batch, batch_idx
            )
        self.log('test_heatmap_loss', heatmap_loss.item(), on_epoch=True)
        self.log('test_radius_loss', radius_loss.item(), on_epoch=True)
        self.log('test_loss', total_loss.item(), on_epoch=True)
        return total_loss


    @classmethod
    def add_model_specific_args(cls, parent_parser):
        """Add the argument in the argparser that are model specific

        :param cls: Class
        """
        parent_parser = super(
            CarotidArteryChallengeCircleNet, cls
            ).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("CircleNetModule")
        parser.add_argument('--loss_heatmap_weighting', type=float)
        parser.add_argument('--loss_radius_weighting', type=float)
        parser.add_argument('--heatmap_sigma', type=str)
        return parent_parser

    def _process_args(self):
        try:
            super()._process_args()
        except:
            pass
