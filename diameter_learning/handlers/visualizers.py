"""Implement dice related callback
"""
from pathlib import Path
from pytorch_lightning.callbacks import Callback
import torch
import numpy as np
import matplotlib.pyplot as plt
from diameter_learning.transforms import ControlPointPostprocess


class VisualizerCallback(Callback):
    def __init__(
            self, result_path: Path,
            get_pred: callable,
            get_input: callable,
            get_gt: callable,
            slice_id_key: str = 'slice_id',
            number_of_images: int = 5
    ):
        super().__init__()
        self.result_path = result_path / self.__class__.__name__
        self.result_path.mkdir()
        self.get_gt = get_gt
        self.get_pred = get_pred
        self.get_input = get_input
        self.slice_id_key = slice_id_key
        self.number_of_images = number_of_images

    def process_batch(
        self, trainer, pl_module,
        batch, batch_idx: int, dataloader_idx: int,
    ):
        pass

    def on_test_batch_start(
            self,
            trainer,
            pl_module,
            batch,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        if self.number_of_images is None or \
                batch_idx < self.number_of_images:
            self.process_batch(
                trainer, pl_module, batch,
                batch_idx, dataloader_idx
                )
            plt.savefig(
                self.result_path / batch[self.slice_id_key][0],
                bbox_inches='tight', dpi=300,
                )


class SegmentationVisualizer(VisualizerCallback):
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
            get_pred: callable,
            get_gt: callable,
            get_input: callable,
            slice_id_key: str = 'slice_id',
            number_of_images: int = 5
    ):
        super().__init__(
            result_path=result_path,
            get_pred=get_pred,
            get_gt=get_gt,
            get_input=get_input,
            slice_id_key=slice_id_key,
            number_of_images=number_of_images
        )

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
        plt.imshow(
            np.transpose(segmentation), cmap=cmap,
            vmin=0, vmax=1, alpha=segmentation
            )

    def process_batch(
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
        self(
            self.get_input(batch)[0, 0].detach().cpu(),
            self.get_pred(batch, pl_module)[0, 0].detach().cpu(),
            cmap='viridis'
            )


class GroundTruthVisualizer(SegmentationVisualizer):
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
            get_pred: callable,
            get_gt: callable,
            get_input: callable,
            slice_id_key: str = 'slice_id',
            number_of_images: int = 5
    ):
        super().__init__(
            result_path=result_path,
            get_pred=get_pred,
            get_gt=get_gt,
            get_input=get_input,
            slice_id_key=slice_id_key,
            number_of_images=number_of_images
        )

    def process_batch(
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
        self(
            self.get_input(batch)[0, 0].detach().cpu().numpy(),
            self.get_gt(batch)[0, 0].detach().cpu().numpy(),
            cmap='viridis'
            )


class ImageVisualizer(VisualizerCallback):
    """Implement image visualizer

    Args:
        result_path: path where the results are stored
        image_key: key of the image
        number_of_images: Number of saved images
    """
    def __init__(
            self, result_path: Path,
            get_pred: callable,
            get_gt: callable,
            get_input: callable,
            slice_id_key: str = 'slice_id',
            number_of_images: int = 5
    ):
        super().__init__(
            result_path=result_path,
            get_pred=get_pred,
            get_gt=get_gt,
            get_input=get_input,
            slice_id_key=slice_id_key,
            number_of_images=number_of_images
            )

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

    def process_batch(
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
        self(
            self.get_input(batch)[0, 0].detach().cpu().numpy(),
            )


class LandmarksVisualizer(VisualizerCallback):
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
            get_pred: callable,
            get_gt: callable,
            get_input: callable,
            slice_id_key: str = 'slice_id',
            number_of_images: int = 5
    ):
        super().__init__(
            result_path=result_path,
            get_pred=get_pred,
            get_gt=get_gt,
            get_input=get_input,
            slice_id_key=slice_id_key,
            number_of_images=number_of_images
            )
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

    def process_batch(
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
            center_of_mass, radiuses = self.get_pred(batch, pl_module)

            control_points =  self.post_process(
                center_of_mass.cpu().detach().numpy(),
                torch.mean(radiuses, dim=0).cpu().detach().numpy(),
                batch['image'].shape + (1,)
                )[2]
            self(
                self.get_input(batch)[0, 0].detach().cpu().numpy(),
                self.get_gt(batch)[0, 0].detach().cpu().numpy(),
                control_points[0, 0, :, :, 0]
                )
