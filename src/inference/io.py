from __future__ import annotations

import hashlib
from pathlib import Path

import xarray as xr
import yaml

from graphcast import checkpoint
from graphcast import graphcast


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def hash_file(path: str | Path) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_stats(config: dict) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    diffs = xr.load_dataset(config["stats"]["diffs_stddev_by_level_path"])
    mean = xr.load_dataset(config["stats"]["mean_by_level_path"])
    stddev = xr.load_dataset(config["stats"]["stddev_by_level_path"])
    return diffs, mean, stddev


def load_model_checkpoint(config: dict) -> graphcast.CheckPoint:
    with open(config["model"]["checkpoint_path"], "rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
    return ckpt


def load_input_dataset(config: dict) -> xr.Dataset:
    return xr.load_dataset(config["input"]["sample_path"])