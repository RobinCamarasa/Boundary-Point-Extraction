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
            offset_key: str = 'offset',
            offset_mask_key: str = 'offset_mask',
            radius_key: str = 'radius',
            radius_mask_key: str = 'radius_mask',
            down_scaling_factor: int = 8,
            sigma: str = 5,
    ):
        super().__init__(keys)
        self.landmarks_key = landmarks_key
        self.heatmap_key = heatmap_key
        self.diameter_key = diameter_key
        self.offset_key = offset_key
        self.offset_mask_key = offset_mask_key
        self.radius_key = radius_key
        self.radius_mask_key = radius_mask_key
        self.down_scaling_factor = down_scaling_factor
        self.sigma = sigma

    def __call__(self, data: dict) -> dict:
        """Method called to crop the correct part of the image

        :param data: Data before LoadCarotidChallengeSegmentation processing
        :return: Updated dictionnary
        """
        # Get the dimension of both the image space and the down-sampled space
        dimensions = data[self.keys[0]].shape
        down_sampled_dimensions = (
                dimensions[0],
                int(dimensions[1] / self.down_scaling_factor),
                int(dimensions[2] / self.down_scaling_factor)
                )

        # Compute the centroid of both the image space and the down-sampled 
        # space
        centroid = data[self.landmarks_key].mean(axis=-2)
        down_sampled_centroid = data[
            self.landmarks_key
            ].mean(axis=-2) // self.down_scaling_factor

        # Compute the offset in the down-sampled space
        down_sampled_offset = centroid / self.down_scaling_factor -\
            down_sampled_centroid

        # Compute the heatmap
        down_sampled_x = np.arange(
            down_sampled_dimensions[1] * down_sampled_dimensions[2]
            ).reshape(down_sampled_dimensions) // down_sampled_dimensions[2]\
        - centroid[0, 0] / self.down_scaling_factor
        down_sampled_y = np.arange(
            down_sampled_dimensions[1] * down_sampled_dimensions[2]
            ).reshape(down_sampled_dimensions) % down_sampled_dimensions[2]\
        - centroid[0, 1] / self.down_scaling_factor
        data[self.heatmap_key] = np.exp(
            - (down_sampled_x ** 2 + down_sampled_y **2)/(2 * self.sigma)
            )

        # Compute the radius map
        data[self.radius_key] = np.zeros(down_sampled_dimensions)
        data[self.radius_key][
            0, 
            int(down_sampled_centroid[0, 0]),
            int(down_sampled_centroid[0, 1]),
            ] = data[self.diameter_key][0][0] / self.down_scaling_factor
        data[self.radius_mask_key] = 1.0 * (data[self.radius_key] > 0)

        # Compute the offset map
        data[self.offset_key] = np.zeros(
            (2,) + down_sampled_dimensions[1:]
            )
        data[self.offset_key][
            :, 
            int(down_sampled_centroid[0, 0]),
            int(down_sampled_centroid[0, 1]),
            ] = down_sampled_offset[0]
        data[self.offset_mask_key] = 1.0 * (data[self.offset_key] > 0)
        return data


