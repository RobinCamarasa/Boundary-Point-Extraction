"""Test `diameter_learning.plmodules.diameter_modules`
"""
import argparse
from torch.utils.data import DataLoader
from baselines.geodesic.plmodules import CarotidArteryChallengeGeodesicNet


def test_geodesic_module_forward():
    """Test forward method"""
    parser = argparse.ArgumentParser(
        description='Process hyperparameters'
        )
    parser = CarotidArteryChallengeGeodesicNet.add_model_specific_args(
        parser
        )
    parser.set_defaults(
        num_fold=5,
        seed=5,
        lr=(10**-4),
        image_dimension_x=768,
        image_dimension_y=160,
        training_cache_rate=0,
        test_folds='[4]',
        validation_folds='[3]',
        batch_size=5
        )
    hparams = parser.parse_args([])

    # Create module
    module: CarotidArteryChallengeGeodesicNet \
        = CarotidArteryChallengeGeodesicNet(
            hparams
            )
    example_batch: DataLoader = next(iter(module.train_dataloader()))
    segmentation = module(
        example_batch
        )
    assert segmentation.shape == (5, 2, 384, 160)


def test_geodesic_module_training_step():
    """Test training_step method"""
    parser = argparse.ArgumentParser(
        description='Process hyperparameters'
        )
    parser = CarotidArteryChallengeGeodesicNet.add_model_specific_args(
        parser
        )
    parser.set_defaults(
        num_fold=5,
        seed=5,
        lr=(10**-4),
        image_dimension_x=768,
        image_dimension_y=160,
        training_cache_rate=0,
        test_folds='[4]',
        validation_folds='[3]',
        batch_size=5
        )
    hparams = parser.parse_args([])

    # Create module
    module: CarotidArteryChallengeGeodesicNet \
        = CarotidArteryChallengeGeodesicNet(
            hparams
        )
    module.log = lambda x, y, on_epoch: None

    example_batch = next(iter(module.train_dataloader()))
    loss = module.training_step(example_batch, 0)
    assert loss.shape == tuple()


def test_geodesic_module_validation_step():
    """Test validation_step method"""
    parser = argparse.ArgumentParser(
        description='Process hyperparameters'
        )
    parser = CarotidArteryChallengeGeodesicNet.add_model_specific_args(
        parser
        )
    parser.set_defaults(
        num_fold=5,
        seed=5,
        lr=(10**-4),
        image_dimension_x=768,
        image_dimension_y=160,
        training_cache_rate=0,
        test_folds='[4]',
        validation_folds='[3]',
        batch_size=5
        )
    hparams = parser.parse_args([])

    # Create module
    module: CarotidArteryChallengeGeodesicNet \
        = CarotidArteryChallengeGeodesicNet(
            hparams
        )
    module.log = lambda x, y, on_epoch: None

    example_batch = next(iter(module.val_dataloader()))
    loss = module.validation_step(example_batch, 0)
    assert loss.shape == tuple()


def test_geodesic_module_test_step():
    """Test test_step method"""
    parser = argparse.ArgumentParser(
        description='Process hyperparameters'
        )
    parser = CarotidArteryChallengeGeodesicNet.add_model_specific_args(
        parser
        )
    parser.set_defaults(
        num_fold=5,
        seed=5,
        lr=(10**-4),
        image_dimension_x=768,
        image_dimension_y=160,
        training_cache_rate=0,
        test_folds='[4]',
        validation_folds='[3]',
        batch_size=5
        )
    hparams = parser.parse_args([])

    # Create module
    module: CarotidArteryChallengeGeodesicNet \
        = CarotidArteryChallengeGeodesicNet(
            hparams
        )
    module.log = lambda x, y, on_epoch: None

    example_batch = next(iter(module.test_dataloader()))
    loss = module.test_step(example_batch, 0)
    assert loss.shape == tuple()


def test_geodesic_module_compute_losses():
    """Test compute_losses"""
    parser = argparse.ArgumentParser(
            description='Process hyperparameters'
        )
    parser = CarotidArteryChallengeGeodesicNet.add_model_specific_args(
        parser
        )
    parser.set_defaults(
        num_fold=5,
        seed=5,
        lr=(10**-4),
        image_dimension_x=768,
        image_dimension_y=160,
        training_cache_rate=0,
        test_folds='[4]',
        validation_folds='[3]',
        batch_size=5
        )
    hparams = parser.parse_args([])

    # Create module
    module: CarotidArteryChallengeGeodesicNet \
        = CarotidArteryChallengeGeodesicNet(
            hparams
        )
    example_batch = next(iter(module.train_dataloader()))
    loss = module.compute_losses(example_batch, 0)
    assert loss.shape == tuple()
