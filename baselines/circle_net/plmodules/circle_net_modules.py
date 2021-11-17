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
from models.losses import FocalLoss, FocalLoss_mask
from diameter_learning.plmodules import CarotidArteryChallengeModule
from baselines.circle_net.nets import BasicCircleNet
from trains.circledet import CircleLoss


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

        # Define torch layers and modules
        self.model: torch.nn.Module = BasicCircleNet(
            spatial_dims=2,
            in_channels=1,
            out_channels_heatmap=1,
            out_channels_radius=1,
            out_channels_offset=2,
            )
        self.sigmoid = torch.nn.Sigmoid().float()
        self.loss = CircleLoss(self.hparams)

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
        # output need the following keys:
        # ['wh'], ['reg'], 
        # batch need the following keys:
        # ['reg_mask'], ['ind'], ['wh']
        pass

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
            CarotidArteryChallengeCircleNet, cls
            ).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("CircleNetModule")

        parser.add_argument('--mse_loss', type=str)
        parser.add_argument('--cat_spec_wh', type=str, default='test')
        parser.add_argument('--dense_wh', type=str, default='test')
        parser.add_argument('--center_thresh', type=float, default=0.1)

        parser.add_argument('--eval_oracle_hm', type=str, default=True)
        parser.add_argument('--eval_oracle_offset', type=str, default=True)
        parser.add_argument('--eval_oracle_wh', type=str, default=True)

        parser.add_argument('--hm_weight', type=float, default=1)
        parser.add_argument('--off_weight', type=float, default=1)
        parser.add_argument('--wh_weight', type=float, default=0.1)

        parser.add_argument('--norm_wh', type=str)
        parser.add_argument('--num_stacks', type=int, default=1)
        parser.add_argument('--reg_loss', type=str, default='l1')
        parser.add_argument('--reg_offset', type=str, default='test')
        parser.add_argument('--device', type=str, default='cuda')
        return parent_parser

    def _process_args(self):
        try:
            super()._process_args()
            self.hparams.mse_loss = eval(self.hparams.mse_loss)
            self.hparams.dense_wh = eval(self.hparams.dense_wh)
            self.hparams.norm_wh = eval(self.hparams.norm_wh)
            self.hparams.cat_spec_wh = eval(self.hparams.cat_spec_wh)
        except:
            pass
