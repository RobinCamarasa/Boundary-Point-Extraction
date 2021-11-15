from pathlib import Path
from pytorch_lightning.callbacks import Callback
import torch
import numpy as np
import matplotlib.pyplot as plt


class SegmentationSaver(Callback):
    """Class that extends `pytorch_lightning.callbacks.Callback` class
    and save segmentation as numpy arrays

    :param result_path: Path of the result folder
    :param get_pred: Function to transform the batch into segmentation
    :param number_of_images: Number of saved images (if None all images
        are saved)
    """
    def __init__(
            self, result_path: Path,
            get_pred: callable,
            slice_id_key: str = 'slice_id',
            number_of_images: int = 5
    ):
        super().__init__()
        self.result_path = result_path / self.__class__.__name__
        self.result_path.mkdir()
        self.get_pred = get_pred
        self.slice_id_key = slice_id_key
        self.number_of_images = number_of_images

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
            np.save(
                self.result_path / f'{batch[self.slice_id_key][0]}',
                self.get_pred(
                    batch, pl_module
                    ).cpu().detach().numpy()
                )
