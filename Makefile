.PHONY: data_unzip, data_repo, test, preprocess

PP = PYTHONPATH="$(shell pwd)"

data_zip:
	@rm -rf "data/care_ii_challenge"
	@mkdir -p "data/care_ii_challenge/"
	@ln -s "$${ZIP_PATH}" "data/care_ii_challenge/care_ii_challenge.zip"
	@unzip -d "data/care_ii_challenge/" "data/care_ii_challenge/care_ii_challenge.zip"


data_repo:
	@rm -rf "data/care_ii_challenge/"
	@mkdir -p "data/care_ii_challenge/"
	@ln -s "$${REPO_PATH}" "data/care_ii_challenge/careIIChallenge"

test:
	@python -m pytest -s test

preprocess: data/care_ii_challenge/careIIChallenge
	@$(PP) python scripts/preprocess
