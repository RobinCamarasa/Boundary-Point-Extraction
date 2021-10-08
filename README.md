# Diameter learning

Train on network on the regression of the lumen diameter to obtain the full lumen segmentation.

## Requirements

- Python 3.7
- Conda (optional)

## Installation

- For conda users:

``` bash
$ conda env create --name [env-name] --file=conda.yaml
```

- For others:

``` bash
$ pip install -r requirements.txt
```

## Structure

- `CONTRIBUTING.md`: File that set up of the continous integration
- `MLproject`: File that set up MLexperiments
- `LICENSE`: File that contains the legal license
- `diameter_learning/`: Contains the code of your project the structure is similar to [MONAI][monai-url] project structure to make it easier to contribute
- `scripts/`: Directory that contains the entry points of the program
- `test/`: Directory that contains the tests of this repository

## Usage

### Obtain data

You can download the data as a zip archive by joining the [Carotid Artery Vessel Wall Segmentation Challenge](https://vessel-wall-segmentation.grand-challenge.org/Index/). Once downloaded, you can place get the data in the right folder by using:

- if you have the zip archive with the data:
```shell
$ make data_zip ZIP_PATH="[your absolute path to the zip archive]"
```

- if you have already inflated the zip archive with the data:
```shell
$ make data_repo REPO_PATH="[your absolute path to inflated folder]"
```

### Preprocess data

- Once you obtained the data you can preprocess them with the command
```shell
$ make preprocess
```

### Run tests

- Run the tests in your environment
```shell
$ make test
```

### Launch experiments

- Launch an mlflow experiment with conda
```shell
$ mlflow run ./ -e [entry-point]
```

- Launch an mlflow experiment without conda
```shell
$ mlflow run ./ -e [entry-point] --no-conda
```

## To contribute

Please follow the recommendation of the [CONTRIBUTING.md][contributing]

## Authors

The following authors contributed :
- [Robin Camarasa][author-gitlab]

[monai-url]: https://github.com/Project-MONAI
[gitlab-ci]: https://gitlab.com/gitlab-org/gitlab/-/blob/master/lib/gitlab/ci/templates/Python.gitlab-ci.yml
[contributing]: https://gitlab.com/robin-camarasa-phd/diameter-learning/diameter-learning/-/blob/master/CONTRIBUTING.md
[author-gitlab]: https://gitlab.com/https://gitlab.com/RobinCamarasa
