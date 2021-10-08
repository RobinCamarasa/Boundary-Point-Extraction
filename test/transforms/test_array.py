"""Test class from `diameter_learning.transform.array` class"""
from itertools import product
import shutil
import numpy as np
import matplotlib.pyplot as plt
from diameter_learning.transforms import ControlPointPostprocess
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
