"""Contains the usefull transforms of arrays
"""
from itertools import product
from typing import Tuple
from itertools import product
from skimage.draw import polygon
from scipy import interpolate
import numpy as np
from monai.transforms import Transform
import torch


class ControlPointPostprocess(Transform):
    """Post-process the control points output by the method to obtain a
    segmentation, the contour and the coordinates

    :param number_of_points: number of points used to create the contour
    """
    def __init__(
        self, number_of_points: int = 300
    ):
        super().__init__()
        self.number_of_points = number_of_points

    def __call__(
        self, center_of_mass: np.array,
        radiuses: np.array,
        batch_shape: Tuple[int, int, int, int, int]
    ) -> Tuple[np.array, np.array, np.array]:
        """Method called to post-process the control points in this
        documentation: nb means batch size, nf means number of output features,
        nx means dimension x, ny means dimension y, nz means dimension z, na
        means number of angles, ni, number of input features

        :param center_of_mass: numpy array of complex of shape (nb, nf, nz)
        :param radiuses: numpy array of complex of shape (nb, nf, na, nz)
        :param batch_shape: Tuple corresponding to (nb, ni, nx, ny, nz)
        :return: The segmentation (array of shape (nb, nf, nf, nx, ny, nz)),
        the contours (array of shape (nb, nf, number_of_points, 2, ny, nz)) and
        the coordinates (array of shape (nb, nf, na, 2, ny, nz))
        """
        redundant_radiuses: np.array = np.concatenate(
            (radiuses, radiuses[:, :, [0]]), axis=-2
            )
        segmentations: np.array = np.zeros(
            (batch_shape[0], radiuses.shape[1]) + batch_shape[2:]
            )
        coordinates: np.array = np.zeros(
            (
                batch_shape[0], radiuses.shape[1], radiuses.shape[-2],
                2, batch_shape[-1]
                )
            )
        contours: np.array = np.zeros(
            (
                batch_shape[0], radiuses.shape[1], self.number_of_points,
                2, batch_shape[-1]
                )
            )
        angles: np.array = np.linspace(
            0, 2 * np.pi, redundant_radiuses.shape[2]
            ) - np.pi
        for nb, nf, nz in product(
            range(batch_shape[0]),
            range(radiuses.shape[1]),
            range(batch_shape[-1])
        ):
            x: np.array = center_of_mass[nb, nf, nz].real +\
                redundant_radiuses[nb, nf, :, nz] * np.cos(angles)
            y: np.array = center_of_mass[nb, nf, nz].imag +\
                redundant_radiuses[nb, nf, :, nz] * np.sin(angles)
            coordinates[nb, nf, :, 0, nz] = x[:radiuses.shape[-2]]
            coordinates[nb, nf, :, 1, nz] = y[:radiuses.shape[-2]]
            # Evaluate the spline the coordinates
            tck, _ = interpolate.splprep([x, y], s=0)
            x_contour, y_contour = interpolate.splev(
                np.linspace(0, 1, self.number_of_points), tck
                )
            # The order seems inversed because r means row and therefore
            # correspond to x coordinates (same remark for y)
            rows, columns = polygon(r=y_contour, c=x_contour)
            segmentations[nb, nf, rows, columns, nz] = 1
            contours[nb, nf, :, 0, nz] = x_contour
            contours[nb, nf, :, 1, nz] = y_contour
        return segmentations, contours, coordinates


class SegmentationToDiameter(Transform):
    """Transform a segmentation into diameter

    :param threshold: Threshold of the segmentation
    """
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def __call__(
        self, segmentation: np.array
    ) -> Tuple[np.array, np.array, np.array]:
        """Method called to post-process the segmentation into a diameter
        in this documentation: nb means batch size,
        nf means number of output features,
        nx means dimension x, ny means dimension y, nz means dimension z
        means number of angles, ni, number of input features

        :param segmentation: numpy of float of shape (nb, nf, nx, ny)
        :param spacing: Physical spacing of a voxel
        :return: Diameters of shape array of shape (nb, nf, 1)
        """
        seg_array = segmentation.cpu().detach().numpy()
        nb, nf, nx, ny = seg_array.shape
        thresholded_seg_array = seg_array > self.threshold
        diameters = np.zeros((nb, nf, 1))
        for batch, feature in product(range(nb), range(nf)):
            x_indices, y_indices = np.where(
                thresholded_seg_array[batch, feature] == 1
                )
            x_matrix = np.matmul(
                np.ones((x_indices.shape[0], 1)),
                np.expand_dims(x_indices, 0)
                )
            y_matrix = np.matmul(
                np.ones((y_indices.shape[0], 1)),
                np.expand_dims(y_indices, 0)
                )
            distance = np.sqrt(
                (x_matrix - np.transpose(x_matrix))**2 +\
                (y_matrix - np.transpose(y_matrix))**2
            )
            
            point_1, point_2 = np.unravel_index(
                np.argmax(distance), distance.shape
                )
            diameters[batch, feature] = np.linalg.norm(
                    np.array(
                        [
                            x_indices[point_1] - x_indices[point_2],
                            y_indices[point_1] - y_indices[point_2]
                            ]
                        )
                    )
        return torch.from_numpy(diameters).to(
            segmentation.device
            )
