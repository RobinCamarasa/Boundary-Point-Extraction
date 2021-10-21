"""Test `diameter_learning.transforms.dictionnary` classes"""
import shutil
import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import (
    Compose, LoadImaged, SpatialPadd,
    AsChannelFirstd
    )
from diameter_learning.apps import CarotidChallengeDataset
from diameter_learning.transforms import (
    LoadCarotidChallengeSegmentation, LoadCarotidChallengeAnnotations,
    CropImageCarotidChallenge, PopKeysd
    )
from diameter_learning.settings import DATA_PRE_PATH, TEST_OUTPUT_PATH


def test_load_carotid_challenge_segmentation():
    """Test LoadCarotidChallengeSegmentation class"""
    shutil.rmtree(TEST_OUTPUT_PATH, ignore_errors=True)
    TEST_OUTPUT_PATH.mkdir()
    carotid_challenge_dataset = CarotidChallengeDataset(
        root_dir=DATA_PRE_PATH,
        annotations=('internal_right', 'internal_left'),
        transforms=Compose(
            [
                LoadImaged("image"),
                AsChannelFirstd("image"),
                LoadCarotidChallengeAnnotations("gt"),
                LoadCarotidChallengeSegmentation()
                ]
            ),
        seed=0,
        folds=[0, 1, 2, 3, 4],
        num_fold=5,
        cache_rate=0
        )
    for i in range(5):
        element = carotid_challenge_dataset[i]
        # Check output
        assert 'gt_lumen_processed_contour' in element.keys()
        assert element['gt_lumen_processed_contour'].shape == (1, 640, 160)

        # Visual assessment
        plt.clf()
        _, ax = plt.subplots(nrows=2, ncols=1)
        ax = ax.ravel()
        ax[0].axis('off')
        ax[1].axis('off')
        ax[0].imshow(np.transpose(element['image'][0, :]), cmap='gray')
        ax[1].imshow(np.transpose(element['image'][0, :]), cmap='gray')
        ax[1].imshow(
            np.transpose(element['gt_lumen_processed_contour'][0, :]),
            cmap='Reds', alpha=0.3
            )
        plt.savefig(
            TEST_OUTPUT_PATH / f'contour_{i}.png', dpi=300
            )


def test_load_carotid_challenge_annotations():
    """Test LoadCarotidChallengeAnnotations class"""
    shutil.rmtree(TEST_OUTPUT_PATH, ignore_errors=True)
    TEST_OUTPUT_PATH.mkdir()
    carotid_challenge_dataset = CarotidChallengeDataset(
        root_dir=DATA_PRE_PATH,
        annotations=('internal_right', 'internal_left'),
        transforms=Compose(
            [
                LoadImaged("image"),
                LoadCarotidChallengeAnnotations("gt")
                ]
            ),
        seed=0,
        folds=[0, 1],
        num_fold=5,
        cache_rate=0
        )
    for i in range(5):
        element = carotid_challenge_dataset[i]
        assert set(element.keys()) == {
            'gt_lumen_processed_diameter', 'image_meta_dict', 'gt',
            'gt_lumen_processed_landmarks', 'image', 'annotation_type',
            'gt_lumen_processed_contour'
            }
        assert element['gt_lumen_processed_landmarks'].shape == (2, 2)
        assert element['gt_lumen_processed_diameter'].shape == ()
        assert element['gt_lumen_processed_contour'].shape[1] == 2


def test_crop_image_carotid_challenge():
    """Test CropImageCarotidChallenge class"""
    shutil.rmtree(TEST_OUTPUT_PATH, ignore_errors=True)
    TEST_OUTPUT_PATH.mkdir()
    carotid_challenge_dataset = CarotidChallengeDataset(
        root_dir=DATA_PRE_PATH,
        annotations=('internal_right', 'internal_left'),
        transforms=Compose(
            [
                LoadImaged("image"),
                AsChannelFirstd("image"),
                LoadCarotidChallengeAnnotations("gt"),
                LoadCarotidChallengeSegmentation(),
                SpatialPadd(
                    ["image", "gt_lumen_processed_contour"],
                    (768, 160)
                    ),
                CropImageCarotidChallenge(
                    ["image", "gt_lumen_processed_contour"]
                    )
                ]
            ),
        seed=0,
        folds=[0, 1],
        num_fold=5,
        cache_rate=0
        )
    for i in range(5):
        element = carotid_challenge_dataset[i]
        assert element['gt_lumen_processed_contour'].shape == (1, 384, 160)
        assert element['image'].shape == (1, 384, 160)

        # Visual assessment
        plt.clf()
        _, ax = plt.subplots(nrows=1, ncols=2)
        ax = ax.ravel()
        ax[0].axis('off')
        ax[1].axis('off')
        ax[0].imshow(np.transpose(element['image'][0, :]), cmap='gray')
        ax[1].imshow(np.transpose(element['image'][0, :]), cmap='gray')
        ax[1].imshow(
                np.transpose(element['gt_lumen_processed_contour'][0, :]),
                cmap='Reds', alpha=0.3
                )
        ax[1].scatter(
                element['gt_lumen_processed_landmarks'][:, 0],
                element['gt_lumen_processed_landmarks'][:, 1],
                s=1
            )
        plt.savefig(
            TEST_OUTPUT_PATH / f'contour_{i}.png', dpi=300
            )


def test_pop_keys_d():
    """Test PopKeysd class"""
    carotid_challenge_dataset = CarotidChallengeDataset(
        root_dir=DATA_PRE_PATH,
        annotations=('internal_right', 'internal_left'),
        transforms=Compose(
            [
                LoadImaged("image"),
                PopKeysd("image_meta_dict"),
                ]
            ),
        seed=0,
        folds=[0, 1],
        num_fold=5,
        cache_rate=0
        )
    for i in range(5):
        element = carotid_challenge_dataset[i]
        assert 'image_meta_dict' not in element.keys()
