.PHONY: data_unzip, data_repo, tests, preprocess, stats, delete_experiements method method_evaluation, geodesic, geodesic_evaluation, full_supervision, full_supervision_evaluation

PP = PYTHONPATH="$(shell pwd)"
TRAIN_OPT = -P "test_folds=[1]" -P "validation_folds=[2]" -P "seed=5"
TEST_OPT = -P "run_id=324efe89aed94f7497a26e43d68329ab"
EXP = circle_net
EXP_NAME = circle_net

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

training: data/care_ii_challenge/preprocessed
	mlflow run --experiment-name $(EXP)_training -e $(EXP_NAME)_training ./ --no-conda $(TRAIN_OPT)

evaluation: data/care_ii_challenge/preprocessed
	mlflow run --experiment-name $(EXP_NAME)_evaluation -e $(EXP)_evaluation ./ --no-conda $(TEST_OPT)
