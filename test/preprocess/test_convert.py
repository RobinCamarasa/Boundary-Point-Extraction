"""Test Carotid artery challenge preprocess
"""
import shutil
from itertools import product
from typing import Mapping, List, Tuple
from pathlib import Path
import numpy as np
from diameter_learning.settings import DATA_REPO_PATH
from diameter_learning.settings import TEST_OUTPUT_PATH
from diameter_learning.types import SliceAnnotation
from diameter_learning.preprocess import Patient


def test_patient_init():
    """test Patient init method"""
    patient: Patient = Patient(DATA_REPO_PATH / '0_P125_U')
    assert patient.patient_id == 125
    assert str(patient.patient_folder) == str(DATA_REPO_PATH / '0_P125_U')


def test_patient_get_slices():
    """test Patient get_slices method"""
    patient: Patient = Patient(DATA_REPO_PATH / '0_P125_U')
    patient_slices: list = patient.get_slices()
    assert len(patient_slices.keys()) == 720
    assert isinstance(list(patient_slices.keys())[0], int)
    assert isinstance(patient_slices, dict)


def test_patient_get_annotation_files():
    """test Patient get_annotation_files method"""
    patient: Patient = Patient(DATA_REPO_PATH / '0_P125_U')
    patient_annotation_files: list = patient.get_annotation_files()
    assert set(patient_annotation_files.keys()) == {
            'internal_left', 'internal_right',
            'external_left', 'external_right'
            }
    for path in patient_annotation_files.values():
        assert path.exists()


def test_patient_get_annotated_slices():
    """test Patient get_annotated_slices method"""
    patient: Patient = Patient(DATA_REPO_PATH / '0_P125_U')
    annotated_slices: Mapping[str, List[int]] = patient.get_annotated_slices()
    assert len(annotated_slices.keys()) == 4
    for annotation, length in [
            ('external_left', 0), ('external_right', 8),
            ('internal_left', 53), ('internal_right', 61)
            ]:
        assert len(annotated_slices[annotation]) == length


def test_patient_get_slice_annotation():
    """test Patient get_slice_annotation method"""
    patient: Patient = Patient(DATA_REPO_PATH / '0_P125_U')
    slice_annotation: Mapping[
        str, List[Tuple[float, float]]
        ] = patient.get_slice_annotation(
            annotation='external_right',
            slice_id=301
            )

    # Check that the contour have the right range of coordinates
    for contour_type, (
            processing_status, (min_x, max_x), (min_y, max_y)
            ) in product(
            ['lumen', 'outer_wall'],
            [('raw', (0, 512), (0, 512)), ('processed', (0, 720), (0, 100))]
            ):
        coordinates: np.array = np.array(
                [
                    list(coordinate)
                    for coordinate in
                    slice_annotation[contour_type][processing_status]
                    ]
                )
        assert coordinates[:, 0].min() >= min_x
        assert coordinates[:, 0].max() <= max_x
        assert coordinates[:, 1].min() >= min_y
        assert coordinates[:, 1].max() <= max_y


def test_patient_process_slice_annotation():
    """test Patient process_slice_annotation method"""
    patient: Patient = Patient(DATA_REPO_PATH / '0_P125_U')
    slice_annotation: SliceAnnotation = patient.get_slice_annotation(
            annotation='external_right',
            slice_id=301
            )
    # Process slice annotation
    slice_annotation = patient.process_slice_annotation(
            slice_annotation
            )
    assert set(slice_annotation['lumen'].keys()) == {
            'raw_contour', 'raw_diameter', 'raw_landmarks',
            'processed_contour', 'processed_diameter', 'processed_landmarks'
            }
    assert isinstance(slice_annotation['lumen']['raw_contour'], list)
    assert isinstance(slice_annotation['lumen']['raw_diameter'], float)
    assert isinstance(slice_annotation['lumen']['raw_landmarks'], list)


def test_patient_to_files():
    """test Patient to_files method"""
    if TEST_OUTPUT_PATH.exists():
        shutil.rmtree(TEST_OUTPUT_PATH)
    output_path: Path = (TEST_OUTPUT_PATH / '0_P125_U')
    output_path.mkdir(parents=True)
    patient: Patient = Patient(DATA_REPO_PATH / '0_P125_U')
    patient.to_files(output_path)
