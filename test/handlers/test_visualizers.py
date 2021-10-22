"""Test `diameter_learning.handlers.visualizers`"""
import argparse
import shutil
from pathlib import Path
import mlflow
import pytorch_lightning as pl
from diameter_learning.plmodules import CarotidArteryChallengeDiameterModule
from diameter_learning.handlers import (
    SegmentationVisualizer, ImageVisualizer,
    LandmarksVisualizer
    )
from diameter_learning.settings import TEST_OUTPUT_PATH


def test_segmentation_visualizer():
    """Test SegmentationVisualizer callback"""
    shutil.rmtree(TEST_OUTPUT_PATH, ignore_errors=True)
    TEST_OUTPUT_PATH.mkdir(exist_ok=True)

    mlflow.set_tracking_uri(
        f'file://{TEST_OUTPUT_PATH.as_posix()}/mlruns/'
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
                SegmentationVisualizer(
                    artifact_path,
                    forward_to_pred=lambda batch, module: module(
                        batch
                        )[0][:, :, :, :, 0]
                    )
                ],
            default_root_dir=TEST_OUTPUT_PATH
        )
        mlflow.pytorch.autolog()
        trainer.test(model)
        for i in range(5):
            assert (
                artifact_path / 'SegmentationVisualizer' /
                f'test_pred_{i}.png'
                ).exists()
            assert (
                artifact_path / 'SegmentationVisualizer' /
                f'test_gt_{i}.png'
                ).exists()
    mlflow.end_run()


def test_image_visualizer():
    """Test ImageVisualizer callback"""
    shutil.rmtree(TEST_OUTPUT_PATH, ignore_errors=True)
    TEST_OUTPUT_PATH.mkdir(exist_ok=True)

    mlflow.set_tracking_uri(
        f'file://{TEST_OUTPUT_PATH.as_posix()}/mlruns/'
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
                ImageVisualizer(artifact_path)
                ],
            default_root_dir=TEST_OUTPUT_PATH
        )
        mlflow.pytorch.autolog()
        trainer.test(model)
        for i in range(5):
            assert (
                artifact_path / 'ImageVisualizer' /
                f'test_image_{i}.png'
                ).exists()
    mlflow.end_run()


def test_landmarks_visualizer():
    """Test LandmarksVisualizer callback"""
    shutil.rmtree(TEST_OUTPUT_PATH, ignore_errors=True)
    TEST_OUTPUT_PATH.mkdir(exist_ok=True)

    mlflow.set_tracking_uri(
        f'file://{TEST_OUTPUT_PATH.as_posix()}/mlruns/'
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
        landmarks_dimension_x=768,
        landmarks_dimension_y=160,
        training_cache_rate=0,
        test_folds='[4]',
        validation_folds='[3]',
        batch_size=5,
        model_sigma=0.15,
        model_nb_radiuses=24,
        model_moments='[0, 1]',
        loss_consistency_weighting=1,
        loss_center_shift_weighting=1,
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
                LandmarksVisualizer(artifact_path)
                ],
            default_root_dir=TEST_OUTPUT_PATH
        )
        mlflow.pytorch.autolog()
        trainer.test(model)
        for i in range(5):
            assert (
                artifact_path / 'LandmarksVisualizer' /
                f'test_landmarks_{i}.png'
                ).exists()
    mlflow.end_run()
