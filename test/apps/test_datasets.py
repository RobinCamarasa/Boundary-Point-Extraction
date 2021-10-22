"""Code to test the datasets
"""
from pathlib import Path
from monai.transforms import LoadImaged
from diameter_learning.apps import CarotidChallengeDataset
from diameter_learning.settings import DATA_PRE_PATH


def test_carotid_challenge_dataset_init():
    """Test CarotidChallengeDataset init method"""
    # Check for the annotation
    for annotations in [
                ('internal_right', 'internal_left'),
                ('internal_right')
    ]:
        carotid_challenge_dataset = CarotidChallengeDataset(
            root_dir=DATA_PRE_PATH,
            annotations=annotations,
            transforms=LoadImaged("image"),
            seed=0,
            folds=[0, 1],
            num_fold=5,
            cache_rate=0
            )
        first_element: dict = carotid_challenge_dataset[-1]
        assert set(first_element.keys()) == {
            'gt', 'image_meta_dict', 'image', 'annotation_type',
            'slice_id'
            }
        assert Path(
            first_element['gt']
            ).stem.split('_annotation')[0] in annotations
        assert first_element['annotation_type'] in annotations
