"""Load the YAML config into typed dataclasses and check its values."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml


class ConfigError(ValueError):
    """Raised when the config file is missing a value or has an invalid one."""


@dataclass(frozen=True)
class Paths:
    data_dir: Path
    results_dir: Path
    models_dir: Path


@dataclass(frozen=True)
class UniverseConfig:
    file: Path
    max_tickers: int | None
    selection: Literal["random", "first"]
    market_ticker: str


@dataclass(frozen=True)
class DownloadConfig:
    batch_size: int
    max_retries: int
    backoff_seconds: float
    pause_between_batches: float
    timeout: int
    refresh: bool


@dataclass(frozen=True)
class SplitConfig:
    """Either date boundaries (train_end/val_end) or day fractions (train_frac/val_frac)."""

    train_end: str | None = None
    val_end: str | None = None
    train_frac: float | None = None
    val_frac: float | None = None


@dataclass(frozen=True)
class FrequencyConfig:
    name: str
    enabled: bool
    interval: str
    min_history_bars: int
    max_ffill_bars: int
    volume_z_mode: Literal["rolling", "time_of_day"]
    splits: SplitConfig
    start: str | None = None
    end: str | None = None
    period: str | None = None


@dataclass(frozen=True)
class CleaningConfig:
    max_missing_frac: float
    max_zero_volume_frac: float
    split_tolerance: float
    min_split_ratio: float


@dataclass(frozen=True)
class FeatureConfig:
    vol_windows: tuple[int, ...]
    volume_z_window: int
    volume_z_days: int
    corr_window: int


@dataclass(frozen=True)
class Config:
    seed: int
    root: Path
    paths: Paths
    universe: UniverseConfig
    download: DownloadConfig
    frequencies: dict[str, FrequencyConfig] = field(default_factory=dict)
    cleaning: CleaningConfig | None = None
    features: FeatureConfig | None = None

    def raw_dir(self, freq: str) -> Path:
        """Directory holding one cached parquet file per ticker for a frequency."""
        return self.paths.data_dir / "raw" / freq

    def processed_dir(self) -> Path:
        return self.paths.data_dir / "processed"


def find_repo_root(start: Path) -> Path:
    """Walk up from ``start`` to the folder containing pyproject.toml."""
    for folder in [start, *start.parents]:
        if (folder / "pyproject.toml").exists():
            return folder
    raise ConfigError(f"Could not find the repo root (pyproject.toml) above {start}")


def _require(section: dict[str, Any], key: str, where: str) -> Any:
    if key not in section:
        raise ConfigError(f"Missing '{key}' in config section '{where}'")
    return section[key]


def _parse_frequency(name: str, raw: dict[str, Any]) -> FrequencyConfig:
    where = f"frequencies.{name}"
    splits = SplitConfig(**_require(raw, "splits", where))
    freq = FrequencyConfig(
        name=name,
        enabled=bool(raw.get("enabled", True)),
        interval=str(_require(raw, "interval", where)),
        min_history_bars=int(_require(raw, "min_history_bars", where)),
        max_ffill_bars=int(_require(raw, "max_ffill_bars", where)),
        volume_z_mode=_require(raw, "volume_z_mode", where),
        splits=splits,
        start=raw.get("start"),
        end=raw.get("end"),
        period=raw.get("period"),
    )
    if freq.volume_z_mode not in ("rolling", "time_of_day"):
        raise ConfigError(f"{where}.volume_z_mode must be 'rolling' or 'time_of_day'")
    if freq.period is None and (freq.start is None or freq.end is None):
        raise ConfigError(f"{where} needs either 'period' or both 'start' and 'end'")
    if freq.start and freq.end and pd.Timestamp(freq.start) >= pd.Timestamp(freq.end):
        raise ConfigError(f"{where}.start must be before end")

    uses_dates = splits.train_end is not None or splits.val_end is not None
    uses_fracs = splits.train_frac is not None or splits.val_frac is not None
    if uses_dates == uses_fracs:
        raise ConfigError(f"{where}.splits needs either train_end/val_end or train_frac/val_frac")
    if uses_dates:
        if splits.train_end is None or splits.val_end is None:
            raise ConfigError(f"{where}.splits needs both train_end and val_end")
        if pd.Timestamp(splits.train_end) >= pd.Timestamp(splits.val_end):
            raise ConfigError(f"{where}.splits.train_end must be before val_end")
    else:
        if splits.train_frac is None or splits.val_frac is None:
            raise ConfigError(f"{where}.splits needs both train_frac and val_frac")
        if not (0 < splits.train_frac and 0 < splits.val_frac and splits.train_frac + splits.val_frac < 1):
            raise ConfigError(f"{where}.splits fractions must be positive and sum to less than 1")
    return freq


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    """Read a YAML config file and return a validated :class:`Config`.

    Relative paths inside the file are resolved against the repo root, so
    scripts behave the same no matter which folder they are run from.
    """
    path = Path(path)
    if not path.is_absolute():
        candidate = Path.cwd() / path
        path = candidate if candidate.exists() else find_repo_root(Path(__file__).resolve()) / path
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    root = find_repo_root(path.resolve().parent)

    p = _require(raw, "paths", "root")
    paths = Paths(**{k: root / _require(p, k, "paths") for k in ("data_dir", "results_dir", "models_dir")})

    u = _require(raw, "universe", "root")
    universe = UniverseConfig(
        file=root / _require(u, "file", "universe"),
        max_tickers=None if u.get("max_tickers") is None else int(u["max_tickers"]),
        selection=u.get("selection", "random"),
        market_ticker=str(_require(u, "market_ticker", "universe")),
    )
    if universe.selection not in ("random", "first"):
        raise ConfigError("universe.selection must be 'random' or 'first'")
    if universe.max_tickers is not None and universe.max_tickers < 1:
        raise ConfigError("universe.max_tickers must be at least 1 (or null for all)")

    d = _require(raw, "download", "root")
    download = DownloadConfig(
        batch_size=int(_require(d, "batch_size", "download")),
        max_retries=int(_require(d, "max_retries", "download")),
        backoff_seconds=float(_require(d, "backoff_seconds", "download")),
        pause_between_batches=float(_require(d, "pause_between_batches", "download")),
        timeout=int(_require(d, "timeout", "download")),
        refresh=bool(d.get("refresh", False)),
    )
    if download.batch_size < 1 or download.max_retries < 0:
        raise ConfigError("download.batch_size must be >= 1 and max_retries >= 0")

    frequencies = {name: _parse_frequency(name, f) for name, f in _require(raw, "frequencies", "root").items()}

    c = _require(raw, "cleaning", "root")
    cleaning = CleaningConfig(
        max_missing_frac=float(_require(c, "max_missing_frac", "cleaning")),
        max_zero_volume_frac=float(_require(c, "max_zero_volume_frac", "cleaning")),
        split_tolerance=float(_require(c, "split_tolerance", "cleaning")),
        min_split_ratio=float(_require(c, "min_split_ratio", "cleaning")),
    )
    if cleaning.min_split_ratio <= 1:
        raise ConfigError("cleaning.min_split_ratio must be greater than 1")

    f = _require(raw, "features", "root")
    features = FeatureConfig(
        vol_windows=tuple(int(w) for w in _require(f, "vol_windows", "features")),
        volume_z_window=int(_require(f, "volume_z_window", "features")),
        volume_z_days=int(_require(f, "volume_z_days", "features")),
        corr_window=int(_require(f, "corr_window", "features")),
    )
    if min((*features.vol_windows, features.volume_z_window, features.volume_z_days, features.corr_window)) < 2:
        raise ConfigError("feature windows must be at least 2")

    return Config(
        seed=int(raw.get("seed", 42)),
        root=root,
        paths=paths,
        universe=universe,
        download=download,
        frequencies=frequencies,
        cleaning=cleaning,
        features=features,
    )
