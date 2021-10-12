"""Test `diameter_learning.nets.layers.diameter` code
"""
import shutil
from itertools import product
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from diameter_learning.settings import TEST_OUTPUT_PATH
from diameter_learning.nets.layers import (
        CenterOfMass2DExtractor, GaussianRadiusExtractor,
        VanillaDiameterExtractor, MomentGaussianRadiusExtractor
        )
matplotlib.use('Agg')


def generate_ellipse_segmentation(
        shape: tuple,
        center_of_mass: tuple,
        ellipse_axis: tuple
        ):
    """Generate an ellipse shaped segmentation

    :param shape: Shape the 2D image
    :param center_of_mass: Coordinates of the center of mass
    :param ellipse_axis: ellipse parameters (a, b) in equation
        `\\left(x/a\\right)^2 + \\left(y/b\\right)^2 = 1`
    """
    # Generate array of indices
    arr = torch.arange(
            shape[0] * shape[1]
            ).reshape(
                    shape[0], shape[1]
                    )

    # Transform array of indices into a segmentation
    segmentation = (
            (
                (arr // shape[1] - center_of_mass[0]) / ellipse_axis[0]

                ) ** 2 +
            (
                (arr % shape[1] - center_of_mass[1]) / ellipse_axis[1]

                ) ** 2
    ) < 1
    return segmentation


def generate_batch_of_segmentation(
    shape=(64, 64, 16), batch_size=3, number_of_features=2
):
    """Generate batch of cylindric segmentation

    :param shape: 3D shape of the image
    :param batch_size: Batch size
    :param number_of_features: number_of_features
    """
    seg_batch = torch.zeros(batch_size, number_of_features, *shape)
    for nb, nf, nz in product(
            range(batch_size), range(number_of_features), range(shape[-1])
            ):
        seg_batch[nb, nf, :, :, nz] = generate_ellipse_segmentation(
                shape[:2],
                (20 + shape[2] + 3 * nb, 20 + nz),
                (5 * (nf + 1), 6 * (nf + 1))
                )
    return seg_batch


def test_center_of_mass_2d_extractor_forward():
    """Test CenterOfMass2DExtractor forward"""
    # Clean test output folder
    shutil.rmtree(TEST_OUTPUT_PATH, ignore_errors=True)
    TEST_OUTPUT_PATH.mkdir(exist_ok=True)

    # Generate variables
    seg_batch = generate_batch_of_segmentation().cpu()
    center_of_mass_extractor = CenterOfMass2DExtractor()

    # Test differentiability
    torch.manual_seed(0)
    center_of_mass = center_of_mass_extractor.forward(
            torch.nn.Sigmoid()(
                torch.nn.Conv3d(2, 2, kernel_size=1, padding=0, dilation=1)(
                    seg_batch.cpu()
                    )
                )
            )
    loss = torch.nn.MSELoss()(
            center_of_mass.real,
            torch.rand(center_of_mass.shape)
            )
    loss.backward()
    assert (loss - 962) ** 2 <= 1

    # Test result
    center_of_mass = center_of_mass_extractor.forward(
                seg_batch
            )
    assert center_of_mass.shape == (3, 2, 16)

    # Assess visual results
    for nb, nf in product(
            range(seg_batch.shape[0]), range(seg_batch.shape[1])
            ):
        plt.clf()
        plt.imshow(seg_batch[nb, nf, :, :, 0])
        plt.savefig(
            TEST_OUTPUT_PATH / f'batch_{nb}_feature_{nf}_without_dots.png'
            )
        plt.plot(
                center_of_mass[nb, nf, 0].real,
                center_of_mass[nb, nf, 0].imag,
                'ro'
                )
        plt.savefig(TEST_OUTPUT_PATH / f'batch_{nb}_feature_{nf}.png')


def test_gaussian_radius_extractor_get_radiuses():
    """test GaussianRadiusExtractor get_radiuses"""
    shutil.rmtree(TEST_OUTPUT_PATH, ignore_errors=True)
    TEST_OUTPUT_PATH.mkdir(exist_ok=True)

    # Generate radiuses
    seg_batch = generate_batch_of_segmentation().cpu()
    gaussian_radius_extractor = GaussianRadiusExtractor(
            nb_radiuses=24
            )
    center_of_mass_extractor = CenterOfMass2DExtractor()

    # Index array
    index_array = torch.arange(
            seg_batch.shape[2] * seg_batch.shape[3]
            ).reshape(
                    seg_batch.shape[2], seg_batch.shape[3]
                    )
    x_indices = index_array % seg_batch.shape[3]
    y_indices = index_array // seg_batch.shape[3]
    for angle in [-math.pi*11/12, math.pi/3, math.pi*11/12]:
        filters = gaussian_radius_extractor.get_filter(
            center_of_mass_extractor(seg_batch),
            torch.complex(x_indices.float(), y_indices.float()),
            angle
            )
        assert filters.shape == (3, 2, 64, 64, 16)

        # Assess visual results
        for nb, nf in product(
                range(seg_batch.shape[0]), range(seg_batch.shape[1])
                ):
            plt.clf()
            plt.imshow(filters[nb, nf, :, :, 0])
            plt.savefig(
                TEST_OUTPUT_PATH /
                f'batch_{nb}_feature_{nf}_angle_{angle.__round__(2)}.png'
                )


def test_gaussian_radius_extractor_forward():
    """Test GaussianRadiusExtractor forward"""
    shutil.rmtree(TEST_OUTPUT_PATH, ignore_errors=True)
    TEST_OUTPUT_PATH.mkdir(exist_ok=True)

    # Intialize modules and test segmentation
    seg_batch = generate_batch_of_segmentation().cpu()
    gaussian_radius_extractor = GaussianRadiusExtractor(
            nb_radiuses=24, sigma=0.2
            )
    center_of_mass_extractor = CenterOfMass2DExtractor()

    # Test differentiability
    torch.manual_seed(0)
    input_ = torch.nn.Sigmoid()(
            torch.nn.Conv3d(
                2, 2, kernel_size=1, padding=0, dilation=1)(
                seg_batch.cpu()
            )
        )
    center_of_mass = center_of_mass_extractor(input_)
    radiuses = gaussian_radius_extractor(input_, center_of_mass)
    loss = torch.nn.MSELoss()(
        radiuses,
        torch.rand(radiuses.shape)
        )
    loss.backward()
    assert (loss - 610) ** 2 <= 1

    # Generate radiuses
    center_of_mass = center_of_mass_extractor(seg_batch)
    radiuses = gaussian_radius_extractor.forward(
            seg_batch, center_of_mass
            )
    assert radiuses.shape == (3, 2, 24, 16)

    # Test gaussian
    x_tensor = torch.linspace(-3, 3, 1001)
    y_tensor = gaussian_radius_extractor.gaussian(x_tensor, 0)
    gaussian_area = 6 * y_tensor.sum() / 1001
    plt.clf()
    plt.plot(x_tensor, y_tensor)
    plt.title(
        f'''
        Area {
            6 * gaussian_radius_extractor.gaussian(x_tensor, 0).sum() / 1001
        }
        '''
        )
    plt.savefig(TEST_OUTPUT_PATH / 'gaussian.png')
    assert (gaussian_area - 1) ** 2 < 0.00001

    # Assess visual results
    for nb, nf in product(
            range(seg_batch.shape[0]), range(seg_batch.shape[1])
            ):
        plt.clf()
        plt.imshow(seg_batch[nb, nf, :, :, 0])
        plt.plot(
            center_of_mass[nb, nf, 0].real,
            center_of_mass[nb, nf, 0].imag,
            'ro'
            )
        plt.scatter(
                [
                    center_of_mass[nb, nf, 0].real + radiuses[nb, nf, j, 0] *
                    np.cos(angle)
                    for j, angle in enumerate(gaussian_radius_extractor.angles)
                    ],
                [
                    center_of_mass[nb, nf, 0].imag + radiuses[nb, nf, j, 0] *
                    np.sin(angle)
                    for j, angle in enumerate(gaussian_radius_extractor.angles)
                    ]
                )
        plt.savefig(TEST_OUTPUT_PATH / f'batch_{nb}_feature_{nf}.png')


def test_moment_gaussian_radius_extractor_get_centered_plan():
    """Test MomentGaussianRadiusExtractor get_centered_plan"""
    shutil.rmtree(TEST_OUTPUT_PATH, ignore_errors=True)
    TEST_OUTPUT_PATH.mkdir(exist_ok=True)

    # Initialize modules and test segmentation
    for moment in [[0], [1], [2]]:
        seg_batch = generate_batch_of_segmentation().cpu()
        gaussian_radius_extractor = MomentGaussianRadiusExtractor(
                moments=moment, nb_radiuses=24, sigma=0.2
                )
        center_of_mass_extractor = CenterOfMass2DExtractor()

        # Index array
        index_array = torch.arange(
                seg_batch.shape[2] * seg_batch.shape[3]
                ).reshape(
                        seg_batch.shape[2], seg_batch.shape[3] 
                        )
        x_indices = index_array % seg_batch.shape[3]
        y_indices = index_array // seg_batch.shape[3]
        centered_plan = gaussian_radius_extractor.get_centered_plan(
                center_of_mass_extractor(seg_batch),
                torch.complex(x_indices.float(), y_indices.float())
                )
        assert centered_plan.shape == (3, 2, 64, 64, 16)

        # Assess visual results
        for nb, nf in product(
                range(seg_batch.shape[0]), range(seg_batch.shape[1])
                ):
            plt.clf()
            plt.imshow(centered_plan[nb, nf, :, :, 0].abs())
            plt.savefig(
                TEST_OUTPUT_PATH / f'centered_plan_batch_{nb}_feature_{nf}.png'
                )


def test_moment_gaussian_radius_extractor_forward():
    """Test MomentGaussianRadiusExtractor forward"""
    shutil.rmtree(TEST_OUTPUT_PATH, ignore_errors=True)
    TEST_OUTPUT_PATH.mkdir(exist_ok=True)

    # Intialize modules and test segmentation
    seg_batch = generate_batch_of_segmentation().cpu()
    gaussian_radius_extractor = MomentGaussianRadiusExtractor(
            moments=[0], nb_radiuses=24, sigma=0.2
            )
    center_of_mass_extractor = CenterOfMass2DExtractor()

    # Test differentiability
    torch.manual_seed(0)
    input_ = torch.nn.Sigmoid()(
            torch.nn.Conv3d(
                2, 2, kernel_size=1, padding=0, dilation=1)(
                seg_batch.cpu()
            )
        )
    center_of_mass = center_of_mass_extractor(input_)
    radiuses = gaussian_radius_extractor(input_, center_of_mass)
    loss = torch.nn.MSELoss()(
            radiuses,
            torch.rand(radiuses.shape)
            )
    loss.backward()
    assert (loss - 610) ** 2 <= 1

    # Generate radiuses
    center_of_mass = center_of_mass_extractor(seg_batch)
    radiuses = gaussian_radius_extractor.forward(
            seg_batch, center_of_mass
            )
    assert radiuses.shape == (1, 3, 2, 24, 16)

    # Test gaussian
    x_tensor = torch.linspace(-3, 3, 1001)
    y_tensor = gaussian_radius_extractor.gaussian(x_tensor, 0)
    gaussian_area = 6 * y_tensor.sum() / 1001
    plt.plot(x_tensor, y_tensor)
    plt.title(
        f'''
        Area {6 * gaussian_radius_extractor.gaussian(x_tensor, 0).sum() / 1001}
        '''
        )
    plt.savefig(TEST_OUTPUT_PATH / 'gaussian.png')
    assert (gaussian_area - 1) ** 2 < 0.00001

    # Assess visual results
    for nb, nf in product(
            range(seg_batch.shape[0]), range(seg_batch.shape[1])
            ):
        plt.clf()
        plt.imshow(seg_batch[nb, nf, :, :, 0])
        plt.plot(
            center_of_mass[nb, nf, 0].real,
            center_of_mass[nb, nf, 0].imag,
            'ro'
            )
        plt.scatter(
                [
                    center_of_mass[nb, nf, 0].real +
                    radiuses[0, nb, nf, j, 0] * np.cos(angle)
                    for j, angle in enumerate(gaussian_radius_extractor.angles)
                    ],
                [
                    center_of_mass[nb, nf, 0].imag +
                    radiuses[0, nb, nf, j, 0] * np.sin(angle)
                    for j, angle in enumerate(gaussian_radius_extractor.angles)
                    ]
                )
        plt.savefig(TEST_OUTPUT_PATH / f'batch_{nb}_feature_{nf}.png')


def test_vanilla_diameter_extrator_forward():
    """Test VanillaDiameterExtrator forward"""
    # Test value
    vanilla_diameter_extractor = VanillaDiameterExtractor(8)
    x_tensor = torch.tensor(
        [
            3, 2 * math.sqrt(2),
            4, math.sqrt(2), 5,
            3 * math.sqrt(2), 2, 3 * math.sqrt(2)
            ]
        ).reshape(1, 1, 8, 1)
    diameters = vanilla_diameter_extractor.forward(x_tensor)
    assert diameters.shape == (1, 1, 1)
    assert (diameters.sum() - math.sqrt(73)) ** 2 < 0.0001

    # Test backpropagation
    torch.manual_seed(0)
    x_tensor = torch.nn.Sigmoid()(
        torch.nn.Conv2d(1, 1, 1, padding=0)(x_tensor)
        )
    diameters = vanilla_diameter_extractor(x_tensor)
    loss = torch.nn.MSELoss()(diameters, torch.rand(1, 1, 1))
    loss.backward()
