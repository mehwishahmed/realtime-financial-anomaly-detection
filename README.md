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
