"""Implement dice related callback
"""
import json
from pathlib import Path
from pytorch_lightning.callbacks import Callback
from monai.metrics import compute_hausdorff_distance
import numpy as np
import mlflow


class HaussdorfCallback(Callback):
    """
    Calculate the Haussdorf distance for each class of a segmentation

    Args:
        result_path: path where the results are stored
        classes: dict where the keys are the classes used for the segmentation
        gt_key: key of the ground truth
        forward_to_pred: Function to transform a batch and a module
        percentile: Percentile of the Haussdorf distance
        to a prediction (tensor of dimension 4)
        threshold: For segmentation
    """
    def __init__(
            self, result_path: Path,
            class_names: dict, gt_key: str,
            forward_to_pred: callable, percentile: float = 95,
            threshold: float = .5
    ):
        super().__init__()
        self.result_path = result_path / self.__class__.__name__
        self.result_path.mkdir(exist_ok=True)
        self.class_names = class_names
        self.gt_key = gt_key
        self.haussdorf = {
            class_name: []
            for class_name in class_names
            }
        self.forward_to_pred = forward_to_pred
        self.percentile = percentile
        self.threshold = threshold

    def on_test_batch_start(
            self,
            trainer,
            pl_module,
            batch,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        """Code launched at test batch start to obtain
        the Haussdorf distance

        :param trainer: pytorch_lightning under study
        :param pl_module: pytorch_lightning module under study
        :param batch: Batch under study
        :param batch_idx: Id of the batch under study
        :param dataloader_idx: Id of the dataloader understudy
        """
        # Compute the dice score
        pred = self.forward_to_pred(batch, pl_module).to(
            pl_module.device
        ).detach().cpu()

        gt = batch[self.gt_key].detach().cpu()
        hausdorff_distance = compute_hausdorff_distance(
            pred, gt, self.percentile
            ).numpy()[0] > self.threshold

        # Save in the haussdorf dictionnary
        for i, class_name in enumerate(self.class_names):
            self.haussdorf[class_name].append(float(hausdorff_distance[i]))

    def on_test_end(
            self, trainer, pl_module
    ) -> None:
        """Code launched at test batch end to obtain
        the Haussdorf distance

        :param trainer: pytorch_lightning under study
        :param pl_module: pytorch_lightning module under study
        """
        # Process haussdorf
        aggregated_haussdorf = {
            key: {
                'mean': np.array(value).mean(),
                'std': np.array(value).std(),
                'values': value
                }
            for key, value in self.haussdorf.items()
            }

        # Save as json file
        with (self.result_path/'haussdorf_per_class.json').open('w') as handle:
            json.dump(aggregated_haussdorf, handle, indent=4)

        # Log in mlflow
        for key in self.class_names:
            mlflow.log_metric(
                f"haussdorf_{key}_mean",
                aggregated_haussdorf[key]['mean']
                )
            mlflow.log_metric(
                f"haussdorf_{key}_std",
                aggregated_haussdorf[key]['std']
                )
