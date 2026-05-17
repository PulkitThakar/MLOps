from __future__ import annotations

from pathlib import Path

import xarray as xr

from src.inference.io import hash_file
from src.utils.manifest import utc_now_iso


def save_forecast(path: str | Path, predictions: xr.Dataset) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_netcdf(path)


def build_stage1_manifest(
    config: dict,
    forecast_path: str | Path,
) -> dict:
    forecast_path = Path(forecast_path)

    return {
        "created_at_utc": utc_now_iso(),
        "stage": "stage1",
        "run_name": config["run"]["name"],
        "model_name": config["model"]["name"],
        "model_provider": config["model"]["provider"],
        "checkpoint_path": config["model"]["checkpoint_path"],
        "checkpoint_sha256": hash_file(config["model"]["checkpoint_path"]),
        "input_sample_path": config["input"]["sample_path"],
        "input_sample_exists": Path(config["input"]["sample_path"]).exists(),
        "input_sample_sha256": hash_file(config["input"]["sample_path"]),
        "forecast_path": str(forecast_path),
        "forecast_exists": forecast_path.exists(),
        "stats": {
            "diffs_stddev_by_level_path": config["stats"]["diffs_stddev_by_level_path"],
            "mean_by_level_path": config["stats"]["mean_by_level_path"],
            "stddev_by_level_path": config["stats"]["stddev_by_level_path"],
        },
    }