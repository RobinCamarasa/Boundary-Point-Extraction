import tqdm
import shutil
from diameter_learning.preprocess import Patient
from diameter_learning.settings import (
        DATA_PATH, DATA_REPO_PATH, DATA_ZIP_PATH,
        DATA_PRE_PATH
        )
from pathlib import Path


def preprocess_dataset():
    """Preprocess the dataset
    """
    # Create preprocessed path
    if DATA_PRE_PATH.exists():
        shutil.rmtree(DATA_PRE_PATH)
    DATA_PRE_PATH.mkdir()

    # Preprocess patients
    for path in tqdm.tqdm(DATA_REPO_PATH.glob('0_*')):
        patient: Patient = Patient(path)
        patient.to_files(DATA_PRE_PATH / path.stem)


preprocess_dataset()
