import argparse
from torch.utils.data import DataLoader
from baselines.circle_net.plmodules import CarotidArteryChallengeCircleNet


def test_circle_net_module_forward():
    """Test forward method"""
    parser = argparse.ArgumentParser(
        description='Process hyperparameters'
        )
    parser = CarotidArteryChallengeCircleNet.add_model_specific_args(
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
        loss_heatmap_weighting=1,
        loss_radius_weighting=.1,
        heatmap_sigma=20
        )
    hparams = parser.parse_args([])

    # Create module
    module: CarotidArteryChallengeCircleNet\
        = CarotidArteryChallengeCircleNet(
            hparams
            )
    example_batch: DataLoader = next(iter(module.train_dataloader()))
    segmentation = module(
        example_batch
        )
    assert set(segmentation.keys()) == {
        'heatmap', 'radius'
        }
    assert segmentation['heatmap'].shape == (5, 1, 384, 160)
    assert segmentation['radius'].shape == (5, 1, 384, 160)


def test_circle_net_module_compute_losses():
    """Test compute_losses method"""
    parser = argparse.ArgumentParser(
        description='Process hyperparameters'
        )
    parser = CarotidArteryChallengeCircleNet.add_model_specific_args(
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
        loss_heatmap_weighting=1,
        loss_radius_weighting=.1,
        heatmap_sigma=20
        )
    hparams = parser.parse_args([])

    # Create module
    module: CarotidArteryChallengeCircleNet\
        = CarotidArteryChallengeCircleNet(
            hparams
            )
    example_batch: DataLoader = next(iter(module.train_dataloader()))
    total_loss, heatmap_loss, radius_loss = module.compute_losses(
        example_batch, 0
        )
    assert total_loss.shape == ()
    assert heatmap_loss.shape == ()
    assert radius_loss.shape == ()


def test_circle_net_module_compute_losses():
    """Test compute_losses method"""
    parser = argparse.ArgumentParser(
        description='Process hyperparameters'
        )
    parser = CarotidArteryChallengeCircleNet.add_model_specific_args(
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
        loss_heatmap_weighting=1,
        loss_radius_weighting=.1,
        heatmap_sigma=20
        )
    hparams = parser.parse_args([])

    # Create module
    module: CarotidArteryChallengeCircleNet\
        = CarotidArteryChallengeCircleNet(
            hparams
            )
    example_batch: DataLoader = next(iter(module.train_dataloader()))
    total_loss, heatmap_loss, radius_loss = module.compute_losses(
        example_batch, 0
        )
    assert total_loss.shape == ()
    assert heatmap_loss.shape == ()
    assert radius_loss.shape == ()


def test_circle_net_module_training_step():
    """Test training_step method"""
    parser = argparse.ArgumentParser(
        description='Process hyperparameters'
        )
    parser = CarotidArteryChallengeCircleNet.add_model_specific_args(
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
        loss_heatmap_weighting=1,
        loss_radius_weighting=.1,
        heatmap_sigma=20
        )
    hparams = parser.parse_args([])

    # Create module
    module: CarotidArteryChallengeCircleNet\
        = CarotidArteryChallengeCircleNet(
            hparams
            )
    example_batch: DataLoader = next(iter(module.train_dataloader()))
    module.log = lambda name, value, on_epoch: None
    train_loss = module.training_step(
        example_batch, 0
        )
    assert train_loss.shape == ()


def test_circle_net_module_test_step():
    """Test test_step method"""
    parser = argparse.ArgumentParser(
        description='Process hyperparameters'
        )
    parser = CarotidArteryChallengeCircleNet.add_model_specific_args(
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
        loss_heatmap_weighting=1,
        loss_radius_weighting=.1,
        heatmap_sigma=20
        )
    hparams = parser.parse_args([])

    # Create module
    module: CarotidArteryChallengeCircleNet\
        = CarotidArteryChallengeCircleNet(
            hparams
            )
    example_batch: DataLoader = next(iter(module.test_dataloader()))
    module.log = lambda name, value, on_epoch: None
    test_step = module.test_step(
        example_batch, 0
        )
    assert test_step.shape == ()


def test_circle_net_module_validation_step():
    """Test validation_step method"""
    parser = argparse.ArgumentParser(
        description='Process hyperparameters'
        )
    parser = CarotidArteryChallengeCircleNet.add_model_specific_args(
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
        loss_heatmap_weighting=1,
        loss_radius_weighting=.1,
        heatmap_sigma=20
        )
    hparams = parser.parse_args([])

    # Create module
    module: CarotidArteryChallengeCircleNet\
        = CarotidArteryChallengeCircleNet(
            hparams
            )
    example_batch: DataLoader = next(iter(module.val_dataloader()))
    module.log = lambda name, value, on_epoch: None
    validation_step = module.validation_step(
        example_batch, 0
        )
    assert validation_step.shape == ()
