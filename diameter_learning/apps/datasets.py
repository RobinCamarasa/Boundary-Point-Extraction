"""Code to load the data as a monai dataset
"""
import random
from pathlib import Path
from typing import Callable, Sequence, Union, List, Mapping
from monai.data import CacheDataset
from monai.transforms import LoadImaged


class CarotidChallengeDataset(CacheDataset):
    """Class that extends `monai.data.CacheDataset`

    :param root_dir: Path to the directory containing the preprocessed data
    :param annotation: Type of annotation used either 'external_left',
    :param transforms: Transforms to execute operations on input data
    'external_right', 'internal_left', 'internal_right'
    :param cache_rate: Proportion of the dataset in cache
    :param num_fold: Number of folds
    :param folds: Folds used on this dataset
    :param seed: Random seed
    """
    def __init__(
            self,
            root_dir: Path,
            annotations: Union[List[str], str],
            transforms: Union[Sequence[Callable], Callable],
            cache_rate: float,
            folds: Union[List[int], int],
            num_fold: int,
            seed: int
            ):
        self.root_dir: Path = root_dir
        self.num_fold: int = num_fold
        if isinstance(folds, int):
            self.folds: List[int] = [folds]
        else:
            self.folds: List[int] = folds
        self.seed: int = seed
        if isinstance(annotations, str):
            self.annotations = [annotations]
        else:
            self.annotations = annotations

        if transforms == ():
            self.transforms: Union[Sequence[Callable], Callable] = LoadImaged(
                "image"
                )
        data = self._generate_dataset()
        super().__init__(data, transforms, cache_rate=cache_rate)

    def _generate_dataset(self) -> List[Mapping[str, str]]:
        # Initialise random
        random.seed(self.seed)

        # Get the patient folders of the considered folds
        all_patient_folders: List[Path] = list(self.root_dir.glob('*'))
        random.shuffle(all_patient_folders)
        patient_folders: List[Path] = []
        num_patient_per_fold: int = len(all_patient_folders) // self.num_fold
        for fold in self.folds:
            if fold != self.num_fold - 1:
                patient_folders += all_patient_folders[
                    fold * num_patient_per_fold: (fold+1) *
                    num_patient_per_fold
                    ]
            else:
                patient_folders += all_patient_folders[
                    fold * num_patient_per_fold:
                    ]

        # Get annotation files of interest
        annotation_files: List[Path] = []
        for patient_folder in patient_folders:
            annotation_files += [
                annotation_file
                for annotation_file in
                list(patient_folder.glob('**/*.json'))
                if annotation_file.stem.split('_annotation')[0]
                in list(self.annotations)
                ]
        return [
            {
                'slice_id': '{}_{}'.format(
                    annotation_file.parents[1].stem,
                    annotation_file.parents[0].stem
                    ),
                'gt': str(annotation_file),
                'image': str(annotation_file.parent / 'image.dcm'),
                'annotation_type': annotation_file.stem.split('_annotation')[0]
                }
            for annotation_file in annotation_files
            ]
