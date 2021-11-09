.PHONY: data_unzip, data_repo, tests, preprocess, stats, delete_experiements

PP = PYTHONPATH="$(shell pwd)"

data_zip:
	rm -rf "data/care_ii_challenge"
	mkdir -p "data/care_ii_challenge/"
	ln -s "$${ZIP_PATH}" "data/care_ii_challenge/care_ii_challenge.zip"
	unzip -d "data/care_ii_challenge/" "data/care_ii_challenge/care_ii_challenge.zip"


data_repo:
	rm -rf "data/care_ii_challenge/"
	mkdir -p "data/care_ii_challenge/"
	ln -s "$${REPO_PATH}" "data/care_ii_challenge/careIIChallenge"

tests: data/care_ii_challenge/careIIChallenge
	$(PP) python -m pytest test/

preprocess: data/care_ii_challenge/careIIChallenge
	$(PP) python scripts/preprocess

stats: data/care_ii_challenge/careIIChallenge
	mlflow run --entry-point stats --experiment-name stats ./

delete_experiments: mlruns
	mlflow gc

show_experiments: mlruns
	mlflow ui

method: data/care_ii_challenge/preprocessed
	mlflow run --experiment-name method_training -e method_training ./ --no-conda
