"""Implement dice related callback
"""
from pathlib import Path
from pytorch_lightning.callbacks import Callback
import torch
import numpy as np
import matplotlib.pyplot as plt
from diameter_learning.transforms import ControlPointPostprocess


class SegmentationVisualizer(Callback):
    """Implement segmentation visualizer

    Args:
        result_path: path where the results are stored
        image_key: key of the image
        segmentation_key: key of the segmentation
        forward_to_pred: Function to transform a batch and a module
            to a landmarks prediction (tensor of dimension 3)
        number_of_images: Number of saved images
    """
    def __init__(
            self, result_path: Path,
            image_key: str = 'image',
            id_key: str = 'slice_id',
            segmentation_key: str = 'gt_lumen_processed_contour',
            forward_to_pred: callable = lambda batch, pl_module: pl_module(
                batch
            ),
            number_of_images: int = 5
    ):
        super().__init__()
        self.result_path = result_path / self.__class__.__name__
        self.result_path.mkdir(exist_ok=True)
        self.image_key = image_key
        self.segmentation_key = segmentation_key
        self.forward_to_pred = forward_to_pred
        self.number_of_images = number_of_images
        self.id_key = id_key

    def __call__(
        self, image: np.array, segmentation: np.array,
        cmap: str
    ) -> None:
        """Create the plot

        :param image: Numpy array that contains the image shape (nx, ny)
        :param segmentation: Numpy array that contains the 
            segmentation (nx, ny)
        :param cmap: colormap
        """
        plt.clf()
        plt.axis('off')
        plt.imshow(np.transpose(image), cmap='gray')
        segmentation[np.where(segmentation < .2)] = np.nan
        plt.imshow(
            np.transpose(segmentation), cmap=cmap,
            vmin=0, vmax=1, alpha=.8
            )

    def on_test_batch_start(
            self,
            trainer,
            pl_module,
            batch,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        """Code launched at test batch start to obtain
        the dice

        :param trainer: pytorch_lightning under study
        :param pl_module: pytorch_lightning module under study
        :param batch: Batch under study
        :param batch_idx: Id of the batch under study
        :param dataloader_idx: Id of the dataloader understudy
        """
        prediction = self.forward_to_pred(
            batch, pl_module
            )[0, 0].detach().cpu().numpy()
        if self.number_of_images is None or \
                batch_idx < self.number_of_images:
            self(
                batch[self.image_key][0, 0].detach().cpu().numpy(),
                batch[self.segmentation_key][0, 0].detach().cpu().numpy(),
                cmap='viridis'
                )
            plt.savefig(
                self.result_path / f'gt_{batch[self.id_key][0]}',
                bbox_inches='tight', dpi=300,
                )
            self(
                batch[self.image_key][0, 0].detach().cpu(), prediction,
                cmap='viridis'
                )
            plt.savefig(
                self.result_path / f'pred_{batch[self.id_key][0]}',
                bbox_inches='tight', dpi=300
            )


class ImageVisualizer(Callback):
    """Implement image visualizer

    Args:
        result_path: path where the results are stored
        image_key: key of the image
        number_of_images: Number of saved images
    """
    def __init__(
            self, result_path: Path,
            image_key: str = 'image',
            id_key: str = 'slice_id',
            number_of_images: int = 5
    ):
        super().__init__()
        self.result_path = result_path / self.__class__.__name__
        self.result_path.mkdir(exist_ok=True)
        self.image_key = image_key
        self.id_key = id_key
        self.number_of_images = number_of_images

    def __call__(
        self, image: np.array
    ) -> None:
        """Create the plot

        :param image: Numpy array that contains the image shape (nx, ny)
        :param segmentation: Numpy array that contains the 
            segmentation (nx, ny)
        :param cmap: colormap
        """
        plt.clf()
        plt.axis('off')
        plt.imshow(np.transpose(image), cmap='gray')

    def on_test_batch_start(
            self,
            trainer,
            pl_module,
            batch,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        """Code launched at test batch start to obtain
        the dice

        :param trainer: pytorch_lightning under study
        :param pl_module: pytorch_lightning module under study
        :param batch: Batch under study
        :param batch_idx: Id of the batch under study
        :param dataloader_idx: Id of the dataloader understudy
        """
        if self.number_of_images is None or \
                batch_idx < self.number_of_images:
            self(
                batch[self.image_key][0, 0].detach().cpu().numpy(),
                )
            plt.savefig(
                self.result_path / f'image_{batch[self.id_key][0]}',
                bbox_inches='tight', dpi=300,
                )


class LandmarksVisualizer(Callback):
    """Implement landmarks visualizer

    Args:
        result_path: path where the results are stored
        image_key: key of the image
        landmarks_key: key of the landmarks
        forward_to_pred: Function to transform a batch and a module
            to a landmarks prediction (tensor of dimension 3)
        number_of_images: Number of saved images
    """
    def __init__(
            self, result_path: Path,
            image_key: str = 'image',
            id_key: str = 'slice_id',
            landmarks_key: str = 'gt_lumen_processed_landmarks',
            forward_to_pred: callable = lambda batch, pl_module: pl_module(
                batch
            ),
            number_of_images: int = 5
    ):
        super().__init__()
        self.result_path = result_path / self.__class__.__name__
        self.result_path.mkdir(exist_ok=True)
        self.image_key = image_key
        self.id_key = id_key
        self.landmarks_key = landmarks_key
        self.forward_to_pred = forward_to_pred
        self.number_of_images = number_of_images
        self.post_process = ControlPointPostprocess()

    def __call__(
        self, image: np.array, landmarks: np.array,
        control_points: np.array
    ) -> None:
        """Create the plot

        :param image: Numpy array that contains the image shape (nx, ny)
        :param landmarks: Numpy array that contains the 
            landmarks
        :param control_points: Numpy array that contains the 
            points on the border
        :param cmap: colormap
        """
        plt.clf()
        plt.axis('off')
        plt.imshow(np.transpose(image), cmap='gray')
        landmarks[np.where(landmarks < .2)] = np.nan
        plt.scatter(
            landmarks[:, 0], landmarks[:, 1],
            s=1
            )
        plt.scatter(
            control_points[:, 1], control_points[:, 0],
            s=1
            )

    def on_test_batch_start(
            self,
            trainer,
            pl_module,
            batch,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        """Code launched at test batch start to obtain
        the dice

        :param trainer: pytorch_lightning under study
        :param pl_module: pytorch_lightning module under study
        :param batch: Batch under study
        :param batch_idx: Id of the batch under study
        :param dataloader_idx: Id of the dataloader understudy
        """
        if self.number_of_images is None or \
                batch_idx < self.number_of_images:
            _, center_of_mass, radiuses, _ = pl_module(batch)
            control_points =  self.post_process(
                center_of_mass.cpu().detach().numpy(),
                torch.mean(radiuses, dim=0).cpu().detach().numpy(),
                batch['image'].shape + (1,)
                )[2]
            self(
                batch[self.image_key][0, 0].detach().cpu().numpy(),
                batch[self.landmarks_key][0, 0].detach().cpu().numpy(),
                control_points[0, 0, :, :, 0]
                )
            plt.savefig(
                self.result_path / f'pred_{batch[self.id_key][0]}',
                bbox_inches='tight', dpi=300,
                )
