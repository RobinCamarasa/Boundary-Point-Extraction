"""Test `diameter_learning.plmodules.diameter_modules`
"""
import argparse
from torch.utils.data import DataLoader
from diameter_learning.plmodules import CarotidArteryChallengeDiameterModule


def test_forward():
    """Test forward method"""
    parser = argparse.ArgumentParser(
        description='Process hyperparameters'
        )
    parser = CarotidArteryChallengeDiameterModule.add_model_specific_args(
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
        batch_size=5,
        model_sigma=0.15,
        model_nb_radiuses=24,
        model_moments='[0, 1]',
        loss_consistency_weighting=1,
        loss_center_shift_weighting=1,
        )
    hparams = parser.parse_args([])

    # Create module
    module: CarotidArteryChallengeDiameterModule \
        = CarotidArteryChallengeDiameterModule(
            hparams
            )
    example_batch: DataLoader = next(iter(module.train_dataloader()))
    segmentation, center_of_mass, radiuses, diameter = module(
        example_batch
        )
    assert segmentation.shape == (5, 1, 384, 160, 1)
    assert center_of_mass.shape == (5, 1, 1)
    assert radiuses.shape == (2, 5, 1, 24, 1)
    assert diameter.shape == (5, 1, 1)


def test_train_dataloader():
    """Test train dataloader method"""
    parser = argparse.ArgumentParser(
        description='Process hyperparameters'
        )
    parser = CarotidArteryChallengeDiameterModule.add_model_specific_args(
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
            batch_size=5,
            model_sigma=0.15,
            model_nb_radiuses=24,
            model_moments='[0, 1]',
            loss_consistency_weighting=1,
            loss_center_shift_weighting=1,
            )
    hparams = parser.parse_args([])

    # Create module
    module: CarotidArteryChallengeDiameterModule \
        = CarotidArteryChallengeDiameterModule(
            hparams
        )
    training_dataloader: DataLoader = iter(module.train_dataloader())

    # Check the 5 first batches
    for _ in range(5):
        element = next(training_dataloader)
        assert element['gt_lumen_processed_diameter'].shape == (5, 1, 1)
        assert element['gt_lumen_processed_landmarks'].shape == (5, 1, 2, 2)
        assert element['image'].shape == (5, 1, 384, 160)
        assert element['gt_lumen_processed_contour'].shape == (5, 1, 384, 160)


def test_val_dataloader():
    """Test val dataloader method"""
    parser = argparse.ArgumentParser(
            description='Process hyperparameters'
            )
    parser = CarotidArteryChallengeDiameterModule.add_model_specific_args(
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
        batch_size=5,
        model_sigma=0.15,
        model_nb_radiuses=24,
        model_moments='[0, 1]',
        loss_consistency_weighting=1,
        loss_center_shift_weighting=1,
        )
    hparams = parser.parse_args([])

    # Create module
    module: CarotidArteryChallengeDiameterModule \
        = CarotidArteryChallengeDiameterModule(
            hparams
        )
    validation_dataloader: DataLoader = iter(module.val_dataloader())

    # Check the 5 first batches
    for _ in range(5):
        element = next(validation_dataloader)
        assert element['gt_lumen_processed_diameter'].shape == (1, 1, 1)
        assert element['gt_lumen_processed_landmarks'].shape == (1, 1, 2, 2)
        assert element['image'].shape == (1, 1, 384, 160)
        assert element['gt_lumen_processed_contour'].shape == (1, 1, 384, 160)


def test_test_dataloader():
    """Test test_dataloader method"""
    parser = argparse.ArgumentParser(
        description='Process hyperparameters'
        )
    parser = CarotidArteryChallengeDiameterModule.add_model_specific_args(
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
        batch_size=5,
        model_sigma=0.15,
        model_nb_radiuses=24,
        model_moments='[0, 1]',
        loss_consistency_weighting=1,
        loss_center_shift_weighting=1,
        )
    hparams = parser.parse_args([])

    # Create module
    module: CarotidArteryChallengeDiameterModule \
        = CarotidArteryChallengeDiameterModule(
            hparams
        )
    validation_dataloader: DataLoader = iter(module.test_dataloader())

    # Check the 5 first batches
    for _ in range(5):
        element = next(validation_dataloader)
        assert element['gt_lumen_processed_diameter'].shape == (1, 1, 1)
        assert element['gt_lumen_processed_landmarks'].shape == (1, 1, 2, 2)
        assert element['image'].shape == (1, 1, 384, 160)
        assert element['gt_lumen_processed_contour'].shape == (1, 1, 384, 160)


def test_training_step():
    """Test training_step method"""
    parser = argparse.ArgumentParser(
        description='Process hyperparameters'
        )
    parser = CarotidArteryChallengeDiameterModule.add_model_specific_args(
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
        batch_size=5,
        model_sigma=0.15,
        model_nb_radiuses=24,
        model_moments='[0, 1]',
        loss_consistency_weighting=1,
        loss_center_shift_weighting=1,
        )
    hparams = parser.parse_args([])

    # Create module
    module: CarotidArteryChallengeDiameterModule \
        = CarotidArteryChallengeDiameterModule(
            hparams
        )
    module.log = lambda x, y, on_epoch: None

    example_batch = next(iter(module.train_dataloader()))
    loss = module.training_step(example_batch, 0)
    assert loss.shape == tuple()


def test_validation_step():
    """Test validation_step method"""
    parser = argparse.ArgumentParser(
        description='Process hyperparameters'
        )
    parser = CarotidArteryChallengeDiameterModule.add_model_specific_args(
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
        batch_size=5,
        model_sigma=0.15,
        model_nb_radiuses=24,
        model_moments='[0, 1]',
        loss_consistency_weighting=1,
        loss_center_shift_weighting=1,
        )
    hparams = parser.parse_args([])

    # Create module
    module: CarotidArteryChallengeDiameterModule \
        = CarotidArteryChallengeDiameterModule(
            hparams
        )
    module.log = lambda x, y, on_epoch: None

    example_batch = next(iter(module.val_dataloader()))
    loss = module.validation_step(example_batch, 0)
    assert loss.shape == tuple()


def test_test_step():
    """Test test_step method"""
    parser = argparse.ArgumentParser(
        description='Process hyperparameters'
        )
    parser = CarotidArteryChallengeDiameterModule.add_model_specific_args(
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
        batch_size=5,
        model_sigma=0.15,
        model_nb_radiuses=24,
        model_moments='[0, 1]',
        loss_consistency_weighting=1,
        loss_center_shift_weighting=1,
        )
    hparams = parser.parse_args([])

    # Create module
    module: CarotidArteryChallengeDiameterModule \
        = CarotidArteryChallengeDiameterModule(
            hparams
        )
    module.log = lambda x, y, on_epoch: None

    example_batch = next(iter(module.test_dataloader()))
    loss = module.test_step(example_batch, 0)
    assert loss.shape == tuple()


def test_compute_losses():
    """Test compute_losses"""
    parser = argparse.ArgumentParser(
            description='Process hyperparameters'
        )
    parser = CarotidArteryChallengeDiameterModule.add_model_specific_args(
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
        batch_size=5,
        model_sigma=0.15,
        model_nb_radiuses=24,
        model_moments='[0, 1]',
        loss_consistency_weighting=1,
        loss_center_shift_weighting=1,
        )
    hparams = parser.parse_args([])

    # Create module
    module: CarotidArteryChallengeDiameterModule \
        = CarotidArteryChallengeDiameterModule(
            hparams
        )
    example_batch = next(iter(module.train_dataloader()))
    diameter_loss, center_shift_loss, consistency_loss, total_loss \
        = module.compute_losses(example_batch, 0)
    assert diameter_loss.shape == tuple()
    assert center_shift_loss.shape == tuple()
    assert consistency_loss.shape == tuple()
    assert total_loss.shape == tuple()
