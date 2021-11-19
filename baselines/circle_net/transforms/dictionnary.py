"""Implement transforms for training"""
import json
from typing import List, Union, Mapping, Sequence
import numpy as np
from skimage.draw import polygon, line, disk
from monai.transforms import MapTransform


class TransformToCircleNetMaps(MapTransform):
    """Process the landmarks as a geodesic map
    on if it is a left or a right annotation

    :param keys: Keys to transform
    :param image_key: Key of the image
    :param suffix: Suffix of the generated key
    :param tolerance: Extra voxels in the disk radius
    """
    def __init__(
            self,
            keys: Union[str, Sequence[str]] = ['gt_lumen_processed_contour'],
            landmarks_key: Union[
                str, Sequence[str]
                ] = 'gt_lumen_processed_landmarks',
            diameter_key: Union[
                str, Sequence[str]
                ] = 'gt_lumen_processed_diameter',
            heatmap_key: str = 'heatmap',
            radius_key: str = 'radius',
            radius_mask_key: str = 'radius_mask',
            sigma: str = 20,
    ):
        super().__init__(keys)
        self.landmarks_key = landmarks_key
        self.heatmap_key = heatmap_key
        self.diameter_key = diameter_key
        self.radius_key = radius_key
        self.radius_mask_key = radius_mask_key
        self.sigma = sigma

    def __call__(self, data: dict) -> dict:
        """Method called to crop the correct part of the image

        :param data: Data before LoadCarotidChallengeSegmentation processing
        :return: Updated dictionnary
        """
        # Get the dimension of both the image space and the down-sampled space
        dimensions = data[self.keys[0]].shape

        # space
        centroid = data[self.landmarks_key].mean(axis=-2)

        # Compute the heatmap
        x_coordinates = np.arange(
            dimensions[1] * dimensions[2]
            ).reshape(dimensions) // dimensions[2] - centroid[0, 0]
        y_coordinates = np.arange(
            dimensions[1] * dimensions[2]
            ).reshape(dimensions) % dimensions[2] - centroid[0, 1]
        data[self.heatmap_key] = np.exp(
            - (x_coordinates ** 2 + y_coordinates **2)/(2 * self.sigma)
            )

        # Compute the radius map
        data[self.radius_key] = np.zeros(dimensions)
        data[self.radius_key][
            0, 
            int(centroid[0, 0]),
            int(centroid[0, 1]),
            ] = data[self.diameter_key][0][0]
        data[self.radius_mask_key] = 1.0 * (data[self.radius_key] > 0)
        return data


