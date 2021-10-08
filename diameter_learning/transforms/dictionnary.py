"""Transforms to pre-process the dataset
"""
import json
from typing import List, Union, Mapping, Sequence
import numpy as np
from skimage.draw import polygon
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
                contour[:, 1],
                contour[:, 0]
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


class CropImageCarotidChallenge(MapTransform):
    """Crop the left or the right part of the image depending
    on if it is a left or a right annotation

    :param contour_key: Key of the contour class
    :param annotation: Key of that correspond to the annotation type
    """
    def __init__(
            self, keys: Union[str, Sequence[str]],
            annotation_key: str = 'annotation_type'
    ):
        super().__init__(keys)
        self.annotation_key = annotation_key

    def __call__(self, data: dict) -> dict:
        """Method called to crop the correct part of the image

        :param data: Data before LoadCarotidChallengeSegmentation processing
        :return: Updated dictionnary
        """
        for key in self.keys:
            if 'right' in data[self.annotation_key]:
                data[key] = data[key][:, :, :data[key].shape[2] // 2]
            else:
                data[key] = data[key][:, :, data[key].shape[2] // 2:]
        return data
