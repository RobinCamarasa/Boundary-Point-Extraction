"""Code to load and process Patient for challenge
"""
import json
import xml.etree.ElementTree as ET
from itertools import product
from typing import List, Mapping, Tuple, Callable
from pathlib import Path
import numpy as np
from diameter_learning.types import SliceAnnotation


class Patient():
    """Class to load the dataset of the carotid artery
    segmentation challenge

    :param patient_folder: Path to the patient folder
    """
    alpha: float = (512/720)
    y_shift: int = 310
    cascade_x_transform: Callable = lambda x: Patient.alpha * x
    cascade_x_inversed_transform: Callable = lambda x: (1/Patient.alpha) * x
    cascade_y_transform: Callable = lambda y: Patient.alpha *\
        (y + Patient.y_shift)
    cascade_y_inversed_transform: Callable = lambda y: (1/Patient.alpha) *\
        y - Patient.y_shift

    def __init__(self, patient_folder: str) -> None:
        self.patient_folder: Path = patient_folder
        self.patient_id: int = int(patient_folder.stem.split('_')[1][1:])
        self.slices: Mapping[int, Path] = self.get_slices()
        self.annotation_files: Mapping[str, Path] = self.get_annotation_files()
        self.annotated_slices: Mapping[
                str, List[int]
                ] = self.get_annotated_slices()

    def get_slices(self) -> Mapping[int, Path]:
        """Get the slices present in the patient folder

        :return: Dictionary that maps the slice id to the path to the
        slice image
        """
        return {
                int(path.stem.split('S101I')[-1]): path
                for path in self.patient_folder.glob('*.dcm')
                }

    def get_annotation_files(self) -> Mapping[str, Path]:
        """Get the annotation files of  a patient

        :return: Dictionary that maps the type of annotation to the
        path to the corresponding annotation file
        """
        return {
                f'{class_}_{side}': self.patient_folder /
                f'CASCADE-{class_short}CA{side_short}' /
                f'EP{self.patient_id}S101_L.QVS'
                for (side, side_short), (class_, class_short) in product(
                    [('left', 'L'), ('right', 'R')],
                    [('external', 'E'), ('internal', 'I')]
                    )
                }

    def get_annotated_slices(self) -> Mapping[str, List[int]]:
        """Get the annotated to slice

        :return: Dictionary that maps the type of annotation to the
        list of annotated slice for that annotation
        """
        annotated_slices: Mapping[str, List[int]] = {}
        for annotation in [
                'external_left', 'external_right',
                'internal_left', 'internal_right'
                ]:
            parsed_xml: ET.Element = ET.parse(
                    self.annotation_files[annotation]
                    ).getroot()
            annotated_slices[annotation] = [
                    i + 1
                    for i, qvas_image in enumerate(
                        parsed_xml.findall('QVAS_Image')
                        )
                    if len(qvas_image.findall('QVAS_Contour')) > 0
                ]
        return annotated_slices

    def get_slice_annotation(
            self, annotation: str, slice_id: int
            ) -> SliceAnnotation:
        """Get the slice annotation for a given annotation type
        and a given slice id

        :param annotation: Annotation type
        :param slice_id: Number of the slice
        :return: Dictionary that maps an annotation type to a
        dictionnary of annotation such as the contours processed
        or raw the final dictionnary has the following structure
        ({'lumen':{'raw': [[x, y] ...], 'processed': [[x, y], ...]}})
        """
        # Check if slice have contours
        if slice_id not in self.annotated_slices[annotation]:
            raise ValueError(f'{slice_id} is not accessible for this patient')

        # Get the contours
        slice_annotation: Mapping[str, List[Tuple[float, float]]] = {}
        xml_contours: List[ET.Element] = ET.parse(
            self.annotation_files[annotation]
            ).getroot().findall(
                'QVAS_Image'
                )[slice_id - 1].findall('QVAS_Contour')

        # Process and store contoursthe contours
        for contour in xml_contours:

            # Get contour type
            contour_type: str = contour.find('ContourType').text.replace(
                ' ', '_'
                ).lower()
            contour_points: List[ET.Element] = contour.find('Contour_Point')\
                .findall('Point')
            slice_annotation[contour_type] = {'raw': [], 'processed': []}

            # Get contour points
            for contour_point in contour_points:
                x_coor: float = float(contour_point.get('x'))
                y_coor: float = float(contour_point.get('y'))
                x_processed: float = Patient.cascade_x_inversed_transform(
                    x_coor
                    )
                y_processed: float = Patient.cascade_y_inversed_transform(
                    y_coor
                    )
                slice_annotation[contour_type]['raw'].append((x_coor, y_coor))
                slice_annotation[contour_type]['processed'].append(
                        (x_processed, y_processed)
                        )
            slice_annotation[contour_type]['raw'] = list(
                    set(slice_annotation[contour_type]['raw'])
                    )
            slice_annotation[contour_type]['processed'] = list(
                    set(slice_annotation[contour_type]['processed'])
                    )
        return slice_annotation

    def process_slice_annotation(
            self, slice_annotation: SliceAnnotation
            ) -> SliceAnnotation:
        """Process the slice annotation to obtain diameter and landmarks
        coordinates

        :param slice_annotation: Slice annotation that follow the format
        ({'lumen':{'raw': [(x, y) ...], 'processed': [(x, y), ...]}})
        :return: Dictionary that maps an annotation type to a
        dictionnary of annotation such as the contours processed
        or raw the final dictionnary has the following structure
        (
            {
                'lumen':
                    {
                        'raw': [(x, y), ...],
                        'raw_diameter': X,
                        'raw_landmarks': [[x, y], [x,y]],
                        'processed': [(x, y), ...],
                        'processed_diameter': X,
                        'processed_landmarks': [[x, y], [x,y]]
                    }
                    }
                    )
        """
        updated_slice_annotation: SliceAnnotation = {}
        # Get the contours
        for process_status, value in slice_annotation.items():
            updated_slice_annotation[process_status] = {}
            for contour_type, tuple_coordinates in value.items():
                coordinates: np.array = np.array(
                    [
                        list(coordinate)
                        for coordinate in
                        tuple_coordinates
                        ]
                    )
                # Create distance matrix
                get_coordinates_in_colon: Callable = lambda x:\
                    np.matmul(x, np.ones((1, x.shape[0])))
                get_distance_matrix: Callable = lambda x, y: np.sqrt(
                        (x - np.transpose(x))**2 +
                        (y - np.transpose(y))**2
                        )
                x_colon_coordinates: np.array = get_coordinates_in_colon(
                        coordinates[:, [0]]
                        )
                y_colon_coordinates: np.array = get_coordinates_in_colon(
                        coordinates[:, [1]]
                        )
                distances: np.array = get_distance_matrix(
                        x_colon_coordinates, y_colon_coordinates
                        )

                # Extract diameter and landmarks
                diameter = distances.max()
                landmarks: List[Tuple[float, float]] = [
                    tuple_coordinates[index]
                    for index in np.where(distances == diameter)[0].tolist()
                    ]
                updated_slice_annotation[
                    process_status
                    ][f'{contour_type}_contour'] = tuple_coordinates
                updated_slice_annotation[
                    process_status
                    ][f'{contour_type}_diameter'] = diameter
                updated_slice_annotation[
                    process_status
                    ][f'{contour_type}_landmarks'] = landmarks
        return updated_slice_annotation

    def to_files(self, path: Path) -> None:
        """Export to files

        :param path: Path where the preprocessing is exported
        """
        for annotation, annotated_slices in self.annotated_slices.items():
            for slice_id in annotated_slices:
                result_path: Path = path / f'slice_{slice_id}'
                # Create image if necessary
                if not result_path.exists():
                    result_path.mkdir(parents=True)
                    (result_path / 'image.dcm').symlink_to(
                        self.slices[slice_id]
                        )

                # Save slice annotation
                slice_annotation: SliceAnnotation = self.get_slice_annotation(
                        annotation, slice_id
                        )
                slice_annotation = self.process_slice_annotation(
                    slice_annotation
                    )
                with (
                        result_path / f'{annotation}_annotation.json'
                        ).open('w') as handle:
                    json.dump(slice_annotation, handle, indent=4)
