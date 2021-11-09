"""Implement transforms for geodesic training"""
import json
from typing import List, Union, Mapping, Sequence
import numpy as np
from skimage.draw import polygon, line, disk
from monai.transforms import MapTransform


class TransformToGeodesicMapd(MapTransform):
    """Process the landmarks as a geodesic map
    on if it is a left or a right annotation

    :param keys: Keys to transform
    :param image_key: Key of the image
    :param suffix: Suffix of the generated key
    """
    def __init__(
            self, keys: Union[str, Sequence[str]] = ['gt_lumen_processed_landmarks'],
            image_key: str = 'image',
            suffix: str = '_geodesic',
    ):
        super().__init__(keys)
        self.image_key = image_key
        self.suffix = suffix

    def __call__(self, data: dict) -> dict:
        """Method called to crop the correct part of the image

        :param data: Data before LoadCarotidChallengeSegmentation processing
        :return: Updated dictionnary
        """
        dimension = (2,) + data[self.image_key].shape[1:]
        for key in self.keys:
            post_process_key =  f'{key}{self.suffix}'
            data[post_process_key] = np.zeros(dimension)

            # Create geometrical objects
            rows_line, columns_line = line(
                *tuple(
                    np.array(
                        np.around(data[key][0].flatten()),
                        dtype=int).tolist()
                    )
                )
            rows_circle, columns_circle = disk(
                tuple(
                    np.array(
                        data[key][0].mean(0), dtype=int
                        ).tolist()
                    ),
                    int(np.linalg.norm(data[key][0, 0] - data[key][0, 1]))/2
                )
            # Draw the line in the foreground
            data[post_process_key][1, rows_line, columns_line] = 1.

            # Draw the background
            data[post_process_key][0] = 1.
            data[post_process_key][0, rows_line, columns_line] = 1.
            data[post_process_key][0, rows_circle, columns_circle] = 0.
        return data
