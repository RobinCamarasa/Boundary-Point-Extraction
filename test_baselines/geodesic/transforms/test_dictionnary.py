import shutil
import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import (
    Compose, LoadImaged, SpatialPadd,
    AsChannelFirstd, AddChanneld
    )
from diameter_learning.apps import CarotidChallengeDataset
from diameter_learning.transforms import (
    LoadCarotidChallengeSegmentation, LoadCarotidChallengeAnnotations,
    CropImageCarotidChallenge, PopKeysd, LoadVoxelSized
    )
from diameter_learning.settings import DATA_PRE_PATH, TEST_OUTPUT_PATH
from baselines.geodesic.transforms import TransformToGeodesicMapd


def test_transform_to_geodesic_mapd():
    """Test TransformToGeodesicMapd class"""
    carotid_challenge_dataset = CarotidChallengeDataset(
        root_dir=DATA_PRE_PATH,
        annotations=('internal_right', 'internal_left'),
        transforms=Compose(
            [
                LoadImaged("image"),
                AsChannelFirstd("image"),
                LoadCarotidChallengeAnnotations("gt"),
                AddChanneld(
                    [
                        "gt_lumen_processed_landmarks",
                        "gt_lumen_processed_diameter"
                        ],
                    ),
                AddChanneld(["gt_lumen_processed_diameter"]),
                LoadCarotidChallengeSegmentation(),
                SpatialPadd(
                    ["image", "gt_lumen_processed_contour"],
                    (768, 160)
                    ),
                CropImageCarotidChallenge(
                    ["image", "gt_lumen_processed_contour"]
                    ),
                TransformToGeodesicMapd(
                    'gt_lumen_processed_landmarks',
                    tolerance=4
                    )
                ]
            ),
        seed=0,
        folds=[0, 1],
        num_fold=5,
        cache_rate=0
        )
    # Assess visualy
    element = carotid_challenge_dataset[0]
    shutil.rmtree(TEST_OUTPUT_PATH, ignore_errors=True)
    TEST_OUTPUT_PATH.mkdir()
    plt.clf()
    fig, ax = plt.subplots(nrows=1, ncols=3)
    ax = ax.ravel()
    ax[0].imshow(element['image'][0], cmap='gray')

    mask = element['gt_lumen_processed_landmarks_geodesic']
    ax[1].imshow(mask[0], cmap='Reds')
    ax[2].imshow(mask[1] + mask[0], cmap='Reds')
    plt.savefig(TEST_OUTPUT_PATH / 'visual_example', dpi=300)
    assert element[
        'gt_lumen_processed_landmarks_geodesic'
        ].shape == (2, 384, 160)
