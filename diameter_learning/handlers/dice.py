"""Implement dice related callback
"""
import json
from pathlib import Path
from pytorch_lightning.callbacks import Callback
from monai.losses import DiceLoss
from monai.utils import LossReduction
import numpy as np
import mlflow


class DiceCallback(Callback):
    """
    Calculate the dice coefficient for each class of a segmentation

    Args:
        result_path: path where the results are stored
        classes: dict where the keys are the classes used for the segmentation
        gt_key: key of the ground truth
        forward_to_pred: Function to transform a batch and a module
        to a prediction (tensor of dimension 4)
    """
    def __init__(
            self, result_path: Path,
            class_names: dict, gt_key: str,
            forward_to_pred: callable
    ):
        super().__init__()
        self.result_path = result_path / self.__class__.__name__
        self.result_path.mkdir(exist_ok=True)
        self.class_names = class_names
        self.gt_key = gt_key
        self.dices = {
                class_name: []
                for class_name in class_names
                }
        self.loss = DiceLoss(reduction=LossReduction.NONE)
        self.forward_to_pred = forward_to_pred

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
        # Compute the dice score
        pred = self.forward_to_pred(batch, pl_module).to(
            pl_module.device
        ).detach().cpu()

        gt = batch[self.gt_key].detach().cpu()
        loss_values = 1. - self.loss(pred, gt).numpy()[0]

        # Save in the dices dictionnary
        for i, class_name in enumerate(self.class_names):
            self.dices[class_name].append(float(loss_values[i]))

    def on_test_end(
            self, trainer, pl_module
    ) -> None:
        """Code launched at test batch end to obtain
        the dice

        :param trainer: pytorch_lightning under study
        :param pl_module: pytorch_lightning module under study
        """
        # Process dices
        aggregated_dices = {
            key: {
                'mean': np.array(value).mean(),
                'std': np.array(value).std(),
                'values': value
                }
            for key, value in self.dices.items()
            }

        # Save as json file
        with (self.result_path/'dice_per_class.json').open('w') as handle:
            json.dump(aggregated_dices, handle, indent=4)

        # Log in mlflow
        for key in self.class_names:
            mlflow.log_metric(
                f"dice_{key}_mean",
                aggregated_dices[key]['mean']
                )
            mlflow.log_metric(
                f"dice_{key}_std",
                aggregated_dices[key]['std']
                )
