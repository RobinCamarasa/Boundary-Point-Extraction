"""Test class from `diameter_learning.transform.array` class"""
from itertools import product
import shutil
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import ellipse
from scipy.ndimage import gaussian_filter
import torch
from diameter_learning.transforms import (
    ControlPointPostprocess, SegmentationToDiameter 
    )
from diameter_learning.settings import TEST_OUTPUT_PATH


def test_control_point_postprocess():
    """Test ControlPointPostprocess class"""
    shutil.rmtree(TEST_OUTPUT_PATH, ignore_errors=True)
    TEST_OUTPUT_PATH.mkdir()

    control_point_postprocess: ControlPointPostprocess = \
        ControlPointPostprocess()
    # Create center of mass and radiuses
    center_of_mass: np.array = np.array(
        [
            [
                [
                    [10 + 2 * i, 10 + 2 * i + 5]
                    for i in range(3)
                    ]
                for _ in range(4)
                ]
            for _ in range(5)
            ]
        )
    center_of_mass = center_of_mass[:, :, :, 0] +\
        1j * center_of_mass[:, :, :, 1]
    radiuses = np.array(
        [
            [
                [
                    8 + np.random.rand(24)
                    for i in range(3)
                    ]
                for _ in range(4)
                ]
            for _ in range(5)
            ]
        )
    radiuses = np.swapaxes(radiuses, -1, -2)
    segmentation, contours, coordinates = control_point_postprocess(
        center_of_mass, radiuses, (5, 1, 30, 30, 3)
        )
    assert contours.shape == (5, 4, 300, 2, 3)
    assert coordinates.shape == (5, 4, 24, 2, 3)
    assert segmentation.shape == (5, 4, 30, 30, 3)

    # Visual assessment
    for nb, nf, nz in product(
        range(contours.shape[0]),
        range(contours.shape[1]),
        range(contours.shape[-1])
    ):
        plt.clf()
        plt.imshow(segmentation[nb, nf, :, :, nz])
        plt.scatter(
            coordinates[nb, nf, :, 0, nz], coordinates[nb, nf, :, 1, nz]
            )
        plt.plot(contours[nb, nf, :, 0, nz], contours[nb, nf, :, 1, nz])
        plt.savefig(TEST_OUTPUT_PATH / f'nb_{nb}_nf_{nf}_nz_{nz}.png')


def test_segmentation_to_diameter():
    """Test SegmentationToDiameter class"""
    shutil.rmtree(TEST_OUTPUT_PATH, ignore_errors=True)
    TEST_OUTPUT_PATH.mkdir()

    # Generate segmentation
    segmentation = np.zeros((64, 128))
    rows, centers = ellipse(32, 28, 20, 10)
    segmentation[rows, centers] = 1
    segmentation = gaussian_filter(
        segmentation, sigma=(5, 5)
    )
    segmentation = np.expand_dims(
        np.expand_dims(segmentation, 0), 0
        )
    segmentation_torch = torch.from_numpy(segmentation).cpu()

    # Launch segmentation in the transform
    segmentation_to_diameter = SegmentationToDiameter(
        threshold=.5
        )
    diameters = segmentation_to_diameter(segmentation_torch)
    import ipdb; ipdb.set_trace() ###!!!BREAKPOINT!!!
    assert diameters.shape == (1, 1, 1)
    assert (diameters[0, 0, 0] - 34.2)**2 < 0.01

    # Visual assessment
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax = ax.ravel()
    ax[0].imshow(segmentation[0, 0])
    ax[1].imshow(segmentation[0, 0] > .5)
    ax[1].scatter(
        [28, 28],
        [32 - diameters[0, 0, 0] / 2, 32 + diameters[0, 0, 0] / 2]
        )
    plt.savefig(TEST_OUTPUT_PATH / f'segmentation.png', dpi=300)
