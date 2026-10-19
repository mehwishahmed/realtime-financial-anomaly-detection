> Originally built Dec 2024–Jan 2025. The original code was lost, and this is a 2026 rebuild.

# realtime-financial-anomaly-detection

An end-to-end ML system that watches a large universe of stocks for anomalies in real time. It has a data pipeline, an ensemble of detection models (Isolation Forest, LSTM autoencoder, small Transformer), and a live-replay benchmark that measures detection latency and false-alarm rate.

**Status:** under construction. The environment is set up and the build phases are in [PLAN.md](PLAN.md). Results will appear here only after the code that produces them has actually been run, and every number will be backed by a file in `results/`.

## Setup

Requires Python 3.12 and runs on a laptop CPU. No GPU needed.

```bash
git clone https://github.com/mehwishahmed/realtime-financial-anomaly-detection.git
cd realtime-financial-anomaly-detection
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
```

The Makefile sets `PYTHONPATH=src`, so `make` commands work even if the editable install doesn't. (On macOS, files under an iCloud-synced Documents folder can be marked hidden, and Python 3.12 skips hidden `.pth` files.) To run a script by hand, use `PYTHONPATH=src python scripts/...`.

## Running the pipeline

All settings are in [`configs/default.yaml`](configs/default.yaml). Change `universe.file` and `universe.max_tickers` to scale from ~100 tickers up to the full S&P 500 or all US-listed stocks.

| Command | What it does |
|---|---|
| `make universe` | Refresh the ticker lists in `configs/tickers/` (S&P 500 from Wikipedia, all US-listed from NASDAQ Trader) |
| `make data` | Download daily and 5-minute bars from yfinance into a local parquet cache (batched, with retries) |
| `make features` | Clean the data, build features, assign time-based splits, and write data-quality reports to `results/data/` |
| `make labels` | Tag real events (earnings shocks, large moves), inject synthetic anomalies into val/test, write `results/labels/` |
| `make baselines` | Sanity-check the evaluator with a random scorer and a simple z-score rule, write `results/metrics/` |
| `make test` | Run the test suite |

## Project layout

| Path | What's there |
|---|---|
| `configs/` | YAML settings and ticker universe files |
| `src/anomaly/` | Library code: data, labels, evaluation, models, streaming |
| `scripts/` | One command-line entry point per pipeline step |
| `notebooks/` | Data exploration and results visualization |
| `tests/` | pytest suite |
| `results/` | CSVs and charts behind every reported number |

## License

MIT. See [LICENSE](LICENSE).
