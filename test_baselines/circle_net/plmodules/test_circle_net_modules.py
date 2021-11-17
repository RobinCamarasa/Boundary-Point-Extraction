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

        # CircleNet options
        hm_weight=1,
        off_weight=1,
        wh_weight=0.1,
        mse_loss='False',
        cat_spec_wh='False',
        dense_wh='False',
        norm_wh='True',

        reg_loss='l1',
        reg_offset='True',

        center_thresh=0.1,
        eval_oracle_hm=True,
        eval_oracle_offset=True,
        eval_oracle_wh=True,
        num_stacks=1,
        device='cuda',
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
        'heatmap', 'radius', 'offset'
        }
