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
            self, result_path: Path, gt_key: str,
            slice_id_key: str,
            forward_to_pred: callable,
            metric_method: callable,
    ):
        super().__init__()
        self.result_path = result_path / self.__class__.__name__
        self.result_path.mkdir(exist_ok=True)
        self.gt_key = gt_key
        self.slice_id_key = slice_id_key
        self.forward_to_pred = forward_to_pred
        self.metric_method = metric_method
        self.values = []
        self.slice_ids = []

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
        pred = self.forward_to_pred(batch, pl_module).detach().cpu()
        gt = batch[self.gt_key].detach().cpu()[0]
        self.values.append(float(self.metric_method(pred, gt)))
        self.slice_ids.append(batch[self.slice_id_key][0])

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
            {
                'slice_id': self.slice_ids,
                'values': self.values
                }
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
        self, result_path: Path, gt_key: str, slice_id: str,
        forward_to_pred: callable,
        ):
        super().__init__(
            result_path, gt_key, slice_id, forward_to_pred,
            lambda pred, gt: torch.abs(gt.sum() - pred.sum()) / gt.sum()
            )


class DiceCallback(MetricCallback):
    def __init__(
        self, result_path: Path, gt_key: str, slice_id: str,
        forward_to_pred: callable,
        ):
        super().__init__(
            result_path, gt_key, slice_id, forward_to_pred,
            lambda pred, gt: 1 - DiceLoss(reduction=LossReduction.MEAN)(
                pred, gt.unsqueeze(0)
                ).item()
            )


class HaussdorffCallback(MetricCallback):
    def __init__(
        self, result_path: Path, gt_key: str,
        slice_id: str, forward_to_pred: callable,
        percentile: int = 95, threshold: float = .5
        ):
        super().__init__(
            result_path, gt_key, slice_id, forward_to_pred,
            lambda pred, gt: compute_hausdorff_distance(
                    pred.unsqueeze(0) > threshold, gt.unsqueeze(0),
                    percentile
                ).item()
            )
        self.percentile = percentile
        self.threshold = threshold
