# Run each pipeline step. Override the config with: make features CONFIG=configs/other.yaml
PY := .venv/bin/python
CONFIG ?= configs/default.yaml
FREQ ?= all
export PYTHONPATH := src

.PHONY: universe data features phase1 test

universe:   ## Refresh configs/tickers/sp500.csv and us_all.csv
	$(PY) scripts/build_universe.py --config $(CONFIG) --which both

data:       ## Download and cache bars for the configured universe
	$(PY) scripts/download_data.py --config $(CONFIG) --freq $(FREQ)

features:   ## Clean, build features, split, and write results/data/
	$(PY) scripts/build_features.py --config $(CONFIG) --freq $(FREQ)

phase1: data features

test:
	$(PY) -m pytest -q
