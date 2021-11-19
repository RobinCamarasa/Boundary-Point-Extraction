import shutil
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
from baselines.circle_net.transforms import TransformToCircleNetMaps
import matplotlib.pyplot as plt


def plot_array(array, name):
    plt.clf()
    plt.imshow(np.transpose(array))
    plt.colorbar()
    plt.title(name)
    plt.axis('off')
    plt.savefig(TEST_OUTPUT_PATH / f'{name}.png', dpi=300)


def test_transform_to_circlenet_maps():
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
                    TransformToCircleNetMaps()
                    ]
                ),
            seed=0,
            folds=[0, 1],
            num_fold=5,
            cache_rate=0
            )
    element = carotid_challenge_dataset[0]

    #  Visual assessment
    plot_array(element['gt_lumen_processed_contour'][0], 'gt')
    plot_array(element['radius'][0], 'radius')
    plot_array(element['radius_mask'][0], 'radius_mask')
    plot_array(element['heatmap'][0], 'heatmap')
    assert {
        'heatmap', 'radius', 'radius_mask'
    }.issubset(set(element.keys()))

    #  Results
    for key in ['heatmap', 'radius', 'radius_mask']:
        assert element[key].shape == (1, 384, 160)
