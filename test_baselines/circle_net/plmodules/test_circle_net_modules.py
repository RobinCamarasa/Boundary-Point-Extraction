import argparse
from torch.utils.data import DataLoader
from baselines.circle_net.plmodules import CarotidArteryChallengeCircleNet


def test_geodesic_module_forward():
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
        batch_size=5
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
    assert segmentation.shape == (5, 3, 384, 160)
