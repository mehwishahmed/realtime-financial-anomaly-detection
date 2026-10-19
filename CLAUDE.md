# CLAUDE.md

Standing instructions for every Claude Code session in this repo. The project description and phases are in [PLAN.md](PLAN.md).

## Context
This is a portfolio project first built in Dec 2024–Jan 2025. The original code was lost, so this repo is a 2026 rebuild. It's the owner's first time using Claude Code.

## How to work
- Go step by step and explain what you're doing in plain English.
- Ask before doing anything big: new dependencies, restructuring, long downloads or training runs, deleting things, pushing.
- Build in the phases listed in PLAN.md. Don't skip ahead.
- After each phase: run it, show the output, explain it in plain English, write tests, then commit with a clear message.

## Honesty rules (non-negotiable)
- Commit with real, current dates. Never backdate commits (no `GIT_AUTHOR_DATE` / `GIT_COMMITTER_DATE` / `--date` tricks).
- The very top of the README must say exactly:
  > Originally built Dec 2024–Jan 2025. The original code was lost, and this is a 2026 rebuild.
- Every number in the README (latency, false alarms, accuracy, number of assets, etc.) must come from code we actually ran. Save those results to files under `results/` so they can be reproduced. Never invent, estimate, or round up numbers to make them look better.

## Technical constraints
- Python 3.11, runs on a laptop CPU (Mac). No GPU required.
- Data from free sources (yfinance). Respect its limits: intraday data only goes back a short window, and large downloads get rate-limited. Use batching, retries, and local parquet caching.
- The number of tickers is configurable: start small (~100) and scale up.
- Every setting lives in a YAML config file. Nothing hardcoded.
- Use a virtual environment (`.venv`) and a pinned `requirements.txt`.
- Set random seeds everywhere so results are reproducible.
- Time-based train/validation/test splits only. No random shuffling (avoid lookahead leakage).
- `.gitignore` covers `data/`, caches, model files, and `.venv`.
- Code has docstrings and type hints. All tests must pass before committing a phase.

## Teaching
After each phase, give a short **interview explanation**: what we built, why we made the key choices, and the tradeoffs, so the owner can explain the project confidently in interviews.
