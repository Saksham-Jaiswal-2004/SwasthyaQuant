.PHONY: help setup setup-quantum data audit baselines kernel control vqc compare \
        params budget schema models test lint clean all
SHELL := /bin/bash
PY := python
CFG := configs/base.yaml

help:             ## list targets
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/' | expand -t18

setup:            ## core deps only -- a broken quantum install must not block baselines
	$(PY) -m pip install -r requirements-dev.txt && $(PY) -m pip install -e .

setup-quantum:    ## add pennylane / qiskit / torch / shap / streamlit
	$(PY) -m pip install -r requirements-quantum.txt

data:             ## fetch the dataset (Kaggle CLI, or prints manual instructions)
	@bash scripts/fetch_data.sh
	$(PY) -m qheart.cli data --config $(CFG)

audit:            ## RUN THIS FIRST -- every warning is a number you would otherwise get wrong
	$(PY) -m qheart.cli audit --config $(CFG)

baselines:        ## Week 1: the five classical baselines, 25 folds each
	$(PY) -m qheart.cli baselines --config $(CFG) --models logreg svm_rbf rf xgboost mlp

control:          ## Week 2: Control-C, both brackets, sized from the circuit at runtime
	$(PY) -m qheart.cli control --config $(CFG)

kernel:           ## Week 2: quantum kernel -> SVC (check `make budget` first)
	$(PY) -m qheart.cli kernel --config $(CFG)

vqc:              ## Week 3: variational classifier, 96 angles
	$(PY) -m qheart.cli vqc --config $(CFG)

compare:          ## aggregate the ledger into the main table; refuses an incomplete one
	$(PY) -m qheart.cli compare --config $(CFG)

params:           ## parameter accounting + the matched Control-C sizing
	$(PY) -m qheart.cli params

budget:           ## the O(N^2) kernel cost estimate, before committing to a run
	$(PY) -m qheart.cli budget

schema:           ## print the data contract
	$(PY) -m qheart.cli schema

models:           ## print the model registry
	$(PY) -m qheart.cli models

all: audit baselines control compare   ## the classical-only path, end to end

test:             ## quantum / data / sklearn tests auto-skip when deps are absent
	pytest

lint:
	ruff check src tests app

clean:
	rm -rf .pytest_cache .ruff_cache results/kernel_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
