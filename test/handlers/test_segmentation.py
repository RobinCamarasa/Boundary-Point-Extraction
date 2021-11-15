import argparse
import shutil
from pathlib import Path
import mlflow
from mlflow.store.tracking.file_store import FileStore
import pytorch_lightning as pl
from diameter_learning.plmodules import CarotidArteryChallengeDiameterModule
from diameter_learning.handlers import SegmentationSaver
from diameter_learning.settings import TEST_OUTPUT_PATH


def test_segmentation_saver():
    """Test SegmentationSaver callback"""
    shutil.rmtree(TEST_OUTPUT_PATH, ignore_errors=True)
    TEST_OUTPUT_PATH.mkdir(exist_ok=True)

    FileStore(root_directory = str(TEST_OUTPUT_PATH / 'mlruns'))
    mlflow.set_tracking_uri(
        f'{TEST_OUTPUT_PATH.as_posix()}/mlruns/'
    )

    # Set parameters
    parser = argparse.ArgumentParser(
        description='Process hyperparameters'
        )
    parser = CarotidArteryChallengeDiameterModule.add_model_specific_args(
        parser
        )
    parser.set_defaults(
        num_fold=20,
        seed=5,
        lr=(10**-4),
        image_dimension_x=768,
        image_dimension_y=160,
        training_cache_rate=0,
        test_folds='[4]',
        validation_folds='[3]',
        batch_size=5,
        model_sigma=0.15,
        model_nb_radiuses=24,
        model_moments='[0, 1]',
        loss_consistency_weighting=1,
        loss_center_shift_weighting=1,
        loss_diameter_weighting=1
        )
    hparams = parser.parse_args([])
    model:  CarotidArteryChallengeDiameterModule = \
        CarotidArteryChallengeDiameterModule(hparams)
    with mlflow.start_run():
        artifact_path: Path = Path(
            mlflow.get_artifact_uri().split('file://')[-1]
            )

        trainer = pl.Trainer(
            max_epochs=1, logger=False, gpus=1,
            callbacks=[
                SegmentationSaver(
                    result_path=artifact_path,
                    get_pred=lambda batch, module: module(
                        batch
                        )[0][:, :, :, :, 0],
                    )
                ],
            default_root_dir=TEST_OUTPUT_PATH
        )
        mlflow.pytorch.autolog()
        trainer.test(model)
        for i in [
            '0_P723_U_slice_126_internal_right',
            '0_P723_U_slice_130_internal_right',
            '0_P723_U_slice_135_internal_right',
            '0_P723_U_slice_184_internal_left',
            '0_P723_U_slice_249_internal_left'
        ]:
            assert (
                artifact_path / 'SegmentationSaver' / f'{i}.npy'
                ).exists()
