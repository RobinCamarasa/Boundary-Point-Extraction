"""File containing diameter trainers"""
from typing import Any, Mapping, Tuple, List
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import monai
from monai.transforms import (
    LoadImaged, SpatialPadd, AsChannelFirstd, AddChanneld,
    ToTensord
    )
from monai.networks.nets import BasicUNet
from diameter_learning.settings import DATA_PRE_PATH
from diameter_learning.apps import CarotidChallengeDataset
from diameter_learning.nets.layers import (
    CenterOfMass2DExtractor, VanillaDiameterExtractor,
    MomentGaussianRadiusExtractor
    )
from diameter_learning.transforms import (
    LoadCarotidChallengeSegmentation, LoadCarotidChallengeAnnotations,
    CropImageCarotidChallenge, PopKeysd
    )


class CarotidArteryChallengeModule():
    """Abstract class that allows training on the Carotid Artery
    Challenge Dataset.

    :param hparams: Hyperparameters of the method
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def _set_dataset(self):
        # Define datasets parameters
        self.dataset_parameters: Mapping[str, Any] = {
           "root_dir": DATA_PRE_PATH,
           "num_fold": self.hparams.num_fold,
           "seed": self.hparams.seed,
           "annotations": ['internal_left', 'internal_right']
        }

    def _set_transform_toolchain(self):
        # Define transform toolchain
        self.preprocess_transforms: List[
            monai.transforms.MapTransform
            ] = [
            LoadImaged("image"),
            PopKeysd("image_meta_dict"),
            AsChannelFirstd("image"),
            LoadCarotidChallengeAnnotations("gt"),
            AddChanneld(
                [
                    "gt_lumen_processed_landmarks",
                    "gt_lumen_processed_diameter"
                    ],
                ),
            AddChanneld(["gt_lumen_processed_diameter"]),
            LoadCarotidChallengeSegmentation(),
            SpatialPadd(
                ["image", "gt_lumen_processed_contour"],
                (
                    self.hparams.image_dimension_x,
                    self.hparams.image_dimension_y
                    )
                ),
            CropImageCarotidChallenge(
                ["image", "gt_lumen_processed_contour"]
                )
            ]
        self.data_augmentation_transforms = [
            ]

        self.postprocess_transforms = [
            ToTensord(
                [
                    "image", "gt_lumen_processed_diameter",
                    "gt_lumen_processed_landmarks",
                    "gt_lumen_processed_contour",
                    ]
                )
            ]

    def train_dataloader(self) -> DataLoader:
        """Obtain the training DataLoader

        :return: Training DataLoader
        """
        training_dataset: CarotidChallengeDataset = CarotidChallengeDataset(
                transforms=self.preprocess_transforms +
                self.data_augmentation_transforms +
                self.postprocess_transforms,
                folds=[
                    i
                    for i in range(self.hparams.num_fold)
                    if i not in self.hparams.test_folds +
                        self.hparams.validation_folds
                    ],
                cache_rate=self.hparams.training_cache_rate,
                **self.dataset_parameters
            )

        # return training_dataset
        return DataLoader(
            training_dataset, batch_size=self.hparams.batch_size,
            shuffle=True
            )

    def val_dataloader(self) -> DataLoader:
        """Obtain the validation DataLoader

        :return: Validation DataLoader
        """
        validation_dataset: CarotidChallengeDataset = CarotidChallengeDataset(
                transforms=self.preprocess_transforms +
                self.postprocess_transforms,
                folds=self.hparams.validation_folds,
                cache_rate=self.hparams.training_cache_rate,
                **self.dataset_parameters
            )

        # return validation_dataset
        return DataLoader(
            validation_dataset, batch_size=1,
            shuffle=False
            )

    def test_dataloader(self) -> DataLoader:
        """Obtain the validation DataLoader

        :return: Validation DataLoader
        """
        test_dataset: CarotidChallengeDataset = CarotidChallengeDataset(
                transforms=self.preprocess_transforms +
                self.postprocess_transforms,
                folds=self.hparams.test_folds,
                cache_rate=self.hparams.training_cache_rate,
                **self.dataset_parameters
            )

        # return test_dataset
        return DataLoader(
            test_dataset, batch_size=1,
            shuffle=False
            )

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        """Add the argument in the argparser that are model specific

        :param cls: Class
        """
        parser = parent_parser.add_argument_group("CarotidArteryChallengeModule")
        parser.add_argument('--num_fold', type=int)
        parser.add_argument('--seed', type=int)
        parser.add_argument('--lr', type=float)
        parser.add_argument('--image_dimension_x', type=int)
        parser.add_argument('--image_dimension_y', type=int)
        parser.add_argument('--training_cache_rate', type=float)
        parser.add_argument('--batch_size', type=float)
        parser.add_argument('--test_folds', type=str)
        parser.add_argument('--validation_folds', type=str)
        return parent_parser

    def _process_args(self):
        self.hparams.test_folds = eval(self.hparams.test_folds)
        self.hparams.validation_folds = eval(self.hparams.validation_folds)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr
            )


class CarotidArteryChallengeDiameterModule(
    CarotidArteryChallengeModule, pl.LightningModule
    ):
    """Class that extends `pytorch_lightning.LightningModule` and
    `diameter_learning.plmodules.CarotidArteryChallengeModule`.
    It allow the training of the diameter.

    :param hparams: Hyperparameters of the method
    """
    def __init__(self, hparams, *args, **kwargs):
        # Process the arguments
        super(
            CarotidArteryChallengeDiameterModule, self
            ).__init__(hparams, *args, **kwargs)
        self.save_hyperparameters(hparams)
        self._process_args()
        super()._set_dataset()
        super()._set_transform_toolchain()

        # Define torch layers and modules
        self.model: torch.nn.Module = BasicUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            )
        self.loss = torch.nn.MSELoss()
        self.sigmoid = torch.nn.Sigmoid()
        self.center_of_mass_extractor = CenterOfMass2DExtractor()
        self.gaussian_radius_extractor = MomentGaussianRadiusExtractor(
                moments=self.hparams.model_moments,
                nb_radiuses=self.hparams.model_nb_radiuses,
                sigma=self.hparams.model_sigma
            )
        self.vanilla_diameter_extractor = VanillaDiameterExtractor(
                nb_radiuses=self.hparams.model_nb_radiuses
            )


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
        segmentation = torch.unsqueeze(
            self.sigmoid(self.model(x['image'])),
            -1
            )
        center_of_mass = self.center_of_mass_extractor(
                segmentation
                )
        radiuses = self.gaussian_radius_extractor(
            segmentation, center_of_mass
            )
        diameter = self.vanilla_diameter_extractor(
            torch.mean(radiuses, 0)
            )
        return segmentation, center_of_mass, radiuses, diameter


    def compute_losses(
        self, batch, batch_idx
        ) -> Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
            ]:
        """Compute the different losses

        :param batch: Batch evaluated
        :param batch_idx: Id of the batch evaluated
        """
        # Evaluate the batch
        _, center_of_mass, radiuses, diameter = self(batch)

        # Compute the diameter MSE
        diameter_loss = self.loss(
            diameter, batch['gt_lumen_processed_diameter']
            )

        # Compute the center shift MSE
        gt_landmarks_center = batch['gt_lumen_processed_landmarks'].mean(-2)
        center_shift_loss = 1/2 * (
            self.loss(
                center_of_mass.real, gt_landmarks_center[:, :, 1]
                ) +
            self.loss(
                center_of_mass.imag, gt_landmarks_center[:, :, 0]
                )
            )

        # Compute the consistency loss
        consistency_loss = torch.var(radiuses, axis=0).mean()
        total_loss = diameter_loss +\
            self.hparams.loss_center_shift_weighting * center_shift_loss +\
            self.hparams.loss_consistency_weighting * consistency_loss

        return diameter_loss, center_shift_loss, consistency_loss, total_loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Define the training step

        :param batch: Batch of the current training step
        :param batch_idx: Id of the current training batch
        :return: The value of the loss
        """
        losses = self.compute_losses(batch, batch_idx)
        self.log('training_diameter_loss', losses[0].item(), on_epoch=True)
        self.log('training_center_shift_loss', losses[1].item(), on_epoch=True)
        self.log('training_consistency_loss', losses[2].item(), on_epoch=True)
        self.log('training_loss', losses[3].item(), on_epoch=True)
        return losses[3]

    def validation_step(self, batch, batch_idx) -> DataLoader:
        """Define the validation step

        :param batch: Batch of the current validation step
        :param batch_idx: Id of the current validation batch
        :return: The value of the loss
        """
        losses = self.compute_losses(batch, batch_idx)
        self.log('validation_diameter_loss', losses[0].item(), on_epoch=True)
        self.log(
            'validation_center_shift_loss', losses[1].item(), on_epoch=True
            )
        self.log(
            'validation_consistency_loss', losses[2].item(), on_epoch=True
            )
        self.log('validation_loss', losses[3].item(), on_epoch=True)
        return losses[3]

    def test_step(self, batch, batch_idx) -> DataLoader:
        """Define the test step

        :param batch: Batch of the current test step
        :param batch_idx: Id of the current test batch
        :return: The value of the loss
        """
        losses = self.compute_losses(batch, batch_idx)
        self.log('test_diameter_loss', losses[0].item(), on_epoch=True)
        self.log('test_center_shift_loss', losses[1].item(), on_epoch=True)
        self.log('test_consistency_loss', losses[2].item(), on_epoch=True)
        self.log('test_loss', losses[3].item(), on_epoch=True)
        return losses[3]


    @classmethod
    def add_model_specific_args(cls, parent_parser):
        """Add the argument in the argparser that are model specific

        :param cls: Class
        """
        parent_parser = super(CarotidArteryChallengeDiameterModule, cls).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("DiameterLearningModule")
        parser.add_argument('--model_moments', type=str)
        parser.add_argument('--model_nb_radiuses', type=int)
        parser.add_argument('--model_sigma', type=float)
        parser.add_argument('--loss_center_shift_weighting', type=float)
        parser.add_argument('--loss_consistency_weighting', type=float)
        return parent_parser

    def _process_args(self):
        super()._process_args()
        self.hparams.model_moments = eval(self.hparams.model_moments)


class CarotidArteryChallengeDiameterResNet(
    CarotidArteryChallengeModule, pl.LightningModule
    ):
    def __init__(self, hparams, *args, **kwargs):
        super(
            CarotidArteryChallengeDiameterResNet, self
            ).__init__(hparams, *args, **kwargs)
        self.save_hyperparameters(hparams)
        self._process_args()
        super()._set_dataset()
        super()._set_transform_toolchain()

        self.model = torch.hub.load(
            'pytorch/vision:v0.10.0', 'resnet50', pretrained=False
            )
        # The following lines are hard-coded because they correspond
        # to pytorch default values
        self.model.conv1 = torch.nn.Conv2d(
                1, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False
            )
        self.model.fc = torch.nn.Linear(
            in_features=2048, out_features=1, bias=True
            )
        self.loss = torch.nn.MSELoss()
        self.sigmoid = torch.nn.ReLU()

    def forward(self, x) -> torch.Tensor:
        """Compute the diameter

        :return: The diameter
        """
        return self.sigmoid(self.model(x['image']).unsqueeze(-1))

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Define the training step

        :param batch: Batch of the current training step
        :param batch_idx: Id of the current training batch
        :return: The value of the loss
        """
        loss = self.loss(
            self(batch), batch['gt_lumen_processed_diameter']
            )
        self.log('training_loss', loss.item(), on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx) -> DataLoader:
        """Define the validation step

        :param batch: Batch of the current validation step
        :param batch_idx: Id of the current validation batch
        :return: The value of the loss
        """
        loss = self.loss(
            self(batch), batch['gt_lumen_processed_diameter']
            )
        self.log('validation_loss', loss.item(), on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx) -> DataLoader:
        """Define the test step

        :param batch: Batch of the current test step
        :param batch_idx: Id of the current test batch
        :return: The value of the loss
        """
        loss = self.loss(
            self(batch), batch['gt_lumen_processed_diameter']
            )
        self.log('test_loss', loss.item(), on_epoch=True)
        return loss
