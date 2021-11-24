import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt
from monai.transforms import (
    Compose, LoadImaged, SpatialPadd,
    AsChannelFirstd, AddChanneld, ToTensord
    )
from skimage.transform import resize
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from diameter_learning.apps import CarotidChallengeDataset
from diameter_learning.transforms import (
    LoadCarotidChallengeSegmentation, LoadCarotidChallengeAnnotations,
    CropImageCarotidChallenge, PopKeysd, LoadVoxelSized
    )
from diameter_learning.settings import DATA_PRE_PATH, TEST_OUTPUT_PATH
from baselines.circle_net.transforms import (
    TransformToCircleNetMaps, CircleNetToSegmentation
    )


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

    dataloader = DataLoader(
        carotid_challenge_dataset, batch_size=2,
        shuffle=False
        )

    # Modification done due to approximations in the batch
    batch = next(iter(dataloader))
    to_segmentation = CircleNetToSegmentation()
    radius = torch.ones(batch['heatmap'].shape)
    radius[0, :] = batch['radius'][0].max() * radius[0, :]
    radius[1, :] = batch['radius'][1].max() * radius[1, :]
    segmentation = to_segmentation(
        output = {
            'heatmap': batch['heatmap'],
            'radius': radius
            }
    )
    assert segmentation.shape == (2, 1, 384, 160)
    for i in range(2):
        plt.imshow(torch.transpose(segmentation[i, 0], 0, -1))
        plt.savefig(TEST_OUTPUT_PATH / f'segmentation_{i}.png')
        plt.imshow(torch.transpose(batch['heatmap'][i, 0], 0, -1))
        plt.savefig(TEST_OUTPUT_PATH / f'heatmap_{i}.png')
        plt.imshow(
            torch.transpose(batch['image'][i, 0], 0, -1),
            cmap='gray'
            )
        plt.savefig(TEST_OUTPUT_PATH / f'image_{i}.png')
