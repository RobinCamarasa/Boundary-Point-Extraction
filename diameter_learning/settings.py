"""Global variables of the project
"""
from pathlib import Path


ROOT_PATH: Path = Path(__file__).parents[1]
DATA_PATH: Path = ROOT_PATH / 'data' / 'care_ii_challenge'
DATA_ZIP_PATH: Path = DATA_PATH / 'care_ii_challenge.zip'
DATA_REPO_PATH: Path = DATA_PATH / 'careIIChallenge'
DATA_PRE_PATH: Path = DATA_PATH / 'preprocessed'
TEST_OUTPUT_PATH: Path = ROOT_PATH / 'test_output'
MLRUN_PATH: Path = ROOT_PATH / 'mlruns'
