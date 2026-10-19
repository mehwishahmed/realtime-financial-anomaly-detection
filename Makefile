# Run each pipeline step. Override the config with: make features CONFIG=configs/other.yaml
# `make all` reruns the main pipeline end to end from the cached ticker lists.
PY := .venv/bin/python
CONFIG ?= configs/default.yaml
FREQ ?= all
export PYTHONPATH := src

.PHONY: all universe data features phase1 labels baselines phase2 iforest deep ensemble stream results-table scale-100 scale-sp500 scale-us-all test

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

deep:       ## Phase 4: train the LSTM autoencoder and Transformer, save curves, evaluate
	$(PY) scripts/train_deep.py --config $(CONFIG) --model all

ensemble:   ## Phase 5: search ensemble weights on val, report on test
	$(PY) scripts/run_ensemble.py --config $(CONFIG)

results-table:  ## Regenerate the README results table from results/metrics/summary.csv
	$(PY) scripts/results_table.py --config $(CONFIG)

stream:     ## Phase 6: tick-by-tick replay (daily: latency + detection; intraday: latency only)
	$(PY) scripts/run_streaming.py --config $(CONFIG) --run all

scale-100:  ## Scaling run: 100 stocks (deployed pipeline, daily)
	$(PY) scripts/run_scaling.py --config configs/scale_100.yaml

scale-sp500:  ## Scaling run: every S&P 500 stock
	$(PY) scripts/run_scaling.py --config configs/scale_sp500.yaml

scale-us-all:  ## Scaling run: every US-listed common stock (long; stops and reports if a step fails or times out)
	$(PY) scripts/run_scaling.py --config configs/scale_us_all.yaml

all: data features labels baselines iforest deep ensemble stream results-table test  ## The whole main pipeline, end to end
