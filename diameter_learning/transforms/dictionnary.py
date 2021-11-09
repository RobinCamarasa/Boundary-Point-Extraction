"""Transforms to pre-process the dataset
"""
import json
from typing import List, Union, Mapping, Sequence
import numpy as np
from skimage.draw import polygon, line, disk
from monai.transforms import MapTransform


class LoadCarotidChallengeSegmentation(MapTransform):
    """Load the contour of a given annotation file

    :param contour_key: Key of the contour class
    :param image_key: Key of that correspond to the loaded image
    """
    def __init__(
            self, keys: str = 'gt_lumen_processed_contour',
            image_key: str = 'image'
    ):
        super().__init__(keys)
        self.image_key = image_key

    def __call__(self, data: dict) -> dict:
        """Method called to load the contour

        :param data: Data before LoadCarotidChallengeSegmentation processing
        :return: Updated dictionnary
        """
        # Load contour
        contour: np.array = np.array(data[self.keys[0]])
        # Sort contour per angle
        complex_contour: np.array = contour[:, 0] + 1j * contour[:, 1]
        center_of_mass: np.complex128 = complex_contour.mean()
        complex_contour = complex_contour - center_of_mass
        contour = contour[np.argsort(np.angle(complex_contour)), :]
        # Fill contour
        filled_contour: np.array = np.zeros(data[self.image_key].shape)
        rows, columns = polygon(
                contour[:, 0],
                contour[:, 1]
            )
        filled_contour[0, rows, columns] = 1
        data[self.keys[0]] = filled_contour
        return data


class LoadCarotidChallengeAnnotations(MapTransform):
    """Load annotations contained in the annotation file

    :param keys: Key of that contain the path to the json file
    :param annotations: Dictionnary containing the loaded keys for a
        given class
    """
    def __init__(
            self,
            keys: str = 'gt',
            annotations: Mapping[str, List[str]] = {
                'lumen': [
                    'processed_landmarks', 'processed_diameter',
                    'processed_contour'
                    ]
                }
    ):
        super().__init__(keys)
        self.annotations = annotations

    def __call__(self, data: dict) -> dict:
        """Method called to load the contour

        :param data: Data before LoadCarotidChallengeContour processing
        :return: Updated dictionnary
        """
        with open(data[self.keys[0]], 'r') as file_descriptor:
            annotations: dict = json.load(file_descriptor)
        for class_, list_of_annotations in self.annotations.items():
            for annotation in list_of_annotations:
                data[f'{self.keys[0]}_{class_}_{annotation}'] = np.array(
                    annotations[class_][annotation]
                    )
        return data


class LoadVoxelSized(MapTransform):
    """Load the voxel dimension (assuming isotropy)

    :param keys: Key of that contain image metadata
    """
    def __init__(
            self,
            keys: str = 'image_meta_dict',
            suffix: str = '_spacing',
    ):
        super().__init__(keys)
        self.suffix = suffix

    def __call__(self, data: dict) -> dict:
        """Method to obtain voxel dimension

        :param data: Data after LoadImaged
        :return: Updated dictionnary
        """
        data[self.keys[0] + self.suffix] = data[self.keys[0]]['spacing'][0]
        return data


class CropImageCarotidChallenge(MapTransform):
    """Crop the left or the right part of the image depending
    on if it is a left or a right annotation

    :param keys: Keys to crop
    :param contour_key: Key of the contour class
    :param landmark_keys: List of keys of the landmarks to adjust
    :param annotation: Key of that correspond to the annotation type
    :param meta_key: Key of the meta data
    """
    def __init__(
            self, keys: Union[str, Sequence[str]],
            landmark_keys: str = ['gt_lumen_processed_landmarks'],
            annotation_key: str = 'annotation_type',
            transform_key: str = 'image_transforms'
    ):
        super().__init__(keys)
        self.annotation_key = annotation_key
        self.transform_key = transform_key
        self.landmark_keys = landmark_keys

    def __call__(self, data: dict) -> dict:
        """Method called to crop the correct part of the image

        :param data: Data before LoadCarotidChallengeSegmentation processing
        :return: Updated dictionnary
        """
        initial_dimension = list(
            data[self.transform_key][0]['orig_size']
            )
        dimension = data[self.keys[0]].shape
        for key in self.keys:
            if 'right' in data[self.annotation_key]:
                data[key] = data[key][:, :dimension[1] // 2]
            else:
                data[key] = data[key][:, dimension[1] // 2:]
        for key in self.landmark_keys:
            # Apply the shift due to padding
            data[key][:, :, 0] += (dimension[1] - initial_dimension[0]) // 2
            data[key][:, :, 1] += (dimension[2] - initial_dimension[1]) // 2
            # Apply the shift due to cropping
            if 'left' in data[self.annotation_key]:
                data[key][:, :, 0] -= dimension[1] // 2
        return data


class PopKeysd(MapTransform):
    """Pop keys in the transform toolchain

    :param keys: Keys to pop
    """
    def __init__(
            self, keys: Union[str, Sequence[str]]
    ):
        super().__init__(keys)

    def __call__(self, data: dict) -> dict:
        """Method called to pop keys

        :param data: Data to pop keys to
        :return: Updated dictionnary
        """
        for key in self.keys:
            data.pop(key)
        return data
