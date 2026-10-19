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

## How results are measured

Every model goes through the same evaluator (`src/anomaly/evaluation/`). The alarm threshold is tuned on the **validation** period, and results are reported on the **test** period. There are two label tracks: *synthetic* anomalies injected at known times (price spikes, volume bursts, level shifts) and *real* events (earnings-day shocks, large single-day moves).

| Metric | Definition |
|---|---|
| Precision / recall / F1 | Per bar: is each flagged (ticker, day) inside a labeled anomaly? |
| PR-AUC | Area under the precision-recall curve. Random guessing scores about the anomaly rate (~1.5%), not 0.5 |
| Event recall | Share of anomaly events with at least one alarm between their start and 3 bars after their end |
| Detection delay | Bars from an event's start to its first alarm (0 = same bar) |
| **False alarms per day (all tickers)** | Total false alarms ÷ trading days, **summed across every ticker**. This is the alert volume someone watching the whole universe sees, so it grows with the number of tickers |
| **False alarms per ticker-day** | Total false alarms ÷ (tickers × trading days) |
| **False alarms per ticker per month** | False alarms per ticker-day × 21 trading days. Doesn't depend on universe size, so use it to compare runs with different ticker counts |

A false alarm is an alarm on a bar that isn't part of any event. Alarms in the 3 bars after an event (late detections) and alarms on the other track's events are never counted as false alarms.

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
