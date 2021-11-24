"""Test `diameter_learning.plmodules.diameter_modules`
"""
import shutil
import argparse
from itertools import product
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from baselines.geodesic.plmodules import CarotidArteryChallengeGeodesicNet
from diameter_learning.settings import TEST_OUTPUT_PATH


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
        batch_size=5,
        tolerance=4
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
        batch_size=5,
        tolerance=4
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
        batch_size=5,
        tolerance=4
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
        batch_size=5,
        tolerance=4
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
    shutil.rmtree(TEST_OUTPUT_PATH, ignore_errors=True)
    TEST_OUTPUT_PATH.mkdir()
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
        batch_size=5,
        tolerance=4
        )
    hparams = parser.parse_args([])

    # Create module
    module: CarotidArteryChallengeGeodesicNet \
        = CarotidArteryChallengeGeodesicNet(
            hparams
        ).cuda()
    example_batch = next(iter(module.train_dataloader()))
    example_batch['gt_lumen_processed_landmarks_geodesic'] = example_batch[
        'gt_lumen_processed_landmarks_geodesic'
        ].cuda()
    example_batch['image'] = example_batch['image'].cuda()
    gt = example_batch['gt_lumen_processed_landmarks_geodesic'].long().detach().cpu().numpy()
    plt.clf()
    plt.imshow(gt[0, 0])
    plt.savefig(TEST_OUTPUT_PATH / 'gt_0.png')
    plt.clf()
    plt.imshow(gt[0, 1])
    plt.savefig(TEST_OUTPUT_PATH / 'gt_1.png')
    plt.clf()
    plt.imshow(gt[0, 0] + gt[0, 1] + 12 * (1 - (gt[0, 1] + gt[0, 0])))
    plt.savefig(TEST_OUTPUT_PATH / 'gt_2.png')
    def plot_grad(module, grad_input, grad_output):
        for i, j in product(
            range(grad_output[0].shape[0]), range(grad_output[0].shape[1])
            ):
            plt.clf()
            plt.imshow(grad_output[0][i, j].detach().cpu().numpy())
            plt.colorbar()
            plt.savefig(TEST_OUTPUT_PATH / f'grad_{i}_{j}.png')
    module.model.final_conv.register_backward_hook(plot_grad)
    loss = module.compute_losses(example_batch, 0)
    loss.backward()
    assert loss.shape == tuple()

