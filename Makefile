# Run each pipeline step. Override the config with: make features CONFIG=configs/other.yaml
PY := .venv/bin/python
CONFIG ?= configs/default.yaml
FREQ ?= all
export PYTHONPATH := src

.PHONY: universe data features phase1 labels baselines phase2 iforest test

universe:   ## Refresh configs/tickers/sp500.csv and us_all.csv
	$(PY) scripts/build_universe.py --config $(CONFIG) --which both

data:       ## Download and cache bars for the configured universe
	$(PY) scripts/download_data.py --config $(CONFIG) --freq $(FREQ)

features:   ## Clean, build features, split, and write results/data/
	$(PY) scripts/build_features.py --config $(CONFIG) --freq $(FREQ)

phase1: data features

test:
	$(PY) -m pytest -q

labels:     ## Phase 2: tag real events, inject synthetic anomalies, write results/labels/
	$(PY) scripts/make_labels.py --config $(CONFIG)

baselines:  ## Phase 2: sanity-check the evaluator with random and z-score scorers
	$(PY) scripts/evaluate_baselines.py --config $(CONFIG)

phase2: labels baselines

iforest:    ## Phase 3: tune Isolation Forest on val, report on test
	$(PY) scripts/train_iforest.py --config $(CONFIG)
