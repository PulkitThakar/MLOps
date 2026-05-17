from __future__ import annotations

from pathlib import Path

from src.inference.artifacts import build_stage1_manifest, save_forecast
from src.inference.graphcast_runner import generate_predictions
from src.inference.io import (
    load_config,
    load_input_dataset,
    load_model_checkpoint,
    load_stats,
)
from src.utils.manifest import build_run_dir, write_manifest


def run_forecast_pipeline(config_path: str | Path) -> None:
    config = load_config(config_path)

    run_name = config["run"]["name"]
    output_root = config["run"]["output_root"]
    seed = config["run"]["seed"]

    run_dir = build_run_dir(output_root=output_root, run_name=run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    forecast_path = run_dir / config["artifact"]["forecast_filename"]
    manifest_path = run_dir / config["artifact"]["manifest_filename"]

    ckpt = load_model_checkpoint(config)
    diffs_stddev_by_level, mean_by_level, stddev_by_level = load_stats(config)
    dataset = load_input_dataset(config)

    predictions = generate_predictions(
        ckpt=ckpt,
        diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level,
        dataset=dataset,
        seed=seed,
    )

    save_forecast(forecast_path, predictions)

    manifest = build_stage1_manifest(
        config=config,
        forecast_path=forecast_path,
    )
    write_manifest(manifest_path, manifest)

    print(f"Run complete. Outputs saved to: {run_dir}")