"""Small shared helpers: seeding, logging, timing, and JSON output."""

from __future__ import annotations

import json
import logging
import os
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and (if installed) PyTorch so runs are repeatable."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)


def get_logger(name: str = "anomaly") -> logging.Logger:
    """Return a logger that prints timestamped lines to the terminal."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    """Measure wall-clock time: ``with timer() as t: ...`` then read ``t["seconds"]``."""
    result: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start


def write_json(data: dict[str, Any], path: Path) -> None:
    """Write a dict as pretty JSON, creating parent folders as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")
