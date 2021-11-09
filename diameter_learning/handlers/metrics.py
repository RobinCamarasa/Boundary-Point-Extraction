"""Implement diameter error related callback
"""
import pandas as pd
from pathlib import Path
import torch
from monai.losses import DiceLoss
from monai.utils import LossReduction
from monai.metrics import compute_hausdorff_distance
from pytorch_lightning.callbacks import Callback
import numpy as np
import mlflow


class MetricCallback(Callback):
    """
    Calculate the dice coefficient for each class of a segmentation

    Args:
        result_path: path where the results are stored
        gt_key: key of the ground truth
        forward_to_pred: Function to transform a batch and a module
            to a prediction (tensor of dimension 4)
    """
    def __init__(
            self, result_path: Path,
            slice_id_key: str,
            get_gt: callable,
            get_pred: callable,
    ):
        super().__init__()
        self.result_path = result_path / self.__class__.__name__
        self.result_path.mkdir(exist_ok=True)
        self.slice_id_key = slice_id_key
        self.get_gt = get_gt
        self.get_pred = get_pred
        self.values = []

    def compute_metric(pred, gt, batch):
        pass

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
        # Compute diameter
        pred = self.get_pred(batch, pl_module).detach().cpu()
        gt = self.get_gt(batch).detach().cpu()
        self.values.append(
            {
                "values": self.compute_metric(pred, gt, batch),
                "slice_id": batch[self.slice_id_key][0]
                }
        )

    def on_test_end(
            self, trainer, pl_module
    ) -> None:
        """Code launched at test batch end to obtain
        the dice

        :param trainer: pytorch_lightning under study
        :param pl_module: pytorch_lightning module under study
        """
        # Process values
        dataframe = pd.DataFrame(
                self.values
            ).set_index('slice_id')
        dataframe.to_csv(self.result_path / 'result.csv')

        # Log in mlflow
        mlflow.log_metric(
            f"{self.result_path.stem.lower()}_mean",
            dataframe['values'].mean()
            )
        mlflow.log_metric(
            f"{self.result_path.stem.lower()}_std",
            dataframe['values'].std()
            )


class RelativeDiameterError(MetricCallback):
    def __init__(
        self, result_path: Path, 
        get_gt: callable, 
        get_pred: callable,
        slice_id_key: str,
        ):
        super().__init__(
            result_path=result_path,
            get_gt=get_gt,
            get_pred=get_pred,
            slice_id_key=slice_id_key,
            )

    def compute_metric(self, pred, gt, batch):
        return (torch.abs(gt.sum() - pred.sum()) / gt.sum()).item()


class DiceCallback(MetricCallback):
    def __init__(
        self, result_path: Path, 
        get_gt: callable, 
        get_pred: callable,
        slice_id_key: str,
        ):
        super().__init__(
            result_path=result_path,
            get_gt=get_gt,
            get_pred=get_pred,
            slice_id_key=slice_id_key,
            )

    def compute_metric(self, pred, gt, batch):
        return 1 - DiceLoss(reduction=LossReduction.MEAN)(
            pred, gt
            ).item()


class HaussdorffCallback(MetricCallback):
    def __init__(
            self, result_path: Path,
            get_gt: callable, 
            get_pred: callable,
            slice_id_key: str,
            spacing_key: str,
            threshold: float = .5
        ):
        super().__init__(
            result_path=result_path,
            get_gt=get_gt,
            get_pred=get_pred,
            slice_id_key=slice_id_key,
            )
        self.threshold = threshold
        self.spacing_key = spacing_key

    def compute_metric(self, pred, gt, batch):
        return compute_hausdorff_distance(
            pred > self.threshold, gt
        )[0][0].item() * batch[self.spacing_key].item()


class AbsoluteDiameterError(MetricCallback):
    def __init__(
        self, result_path: Path, 
        get_gt: callable, 
        get_pred: callable,
        slice_id_key: str,
        spacing_key: str,
        ):
        super().__init__(
            result_path=result_path,
            get_gt=get_gt,
            get_pred=get_pred,
            slice_id_key=slice_id_key,
            )
        self.spacing_key = spacing_key

    def compute_metric(self, pred, gt, batch):
        return torch.abs(
            gt.sum() - pred.sum()
        ).item() * batch[self.spacing_key].item()
