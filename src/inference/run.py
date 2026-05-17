from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
from pathlib import Path

import haiku as hk
import jax
import xarray as xr
import yaml

from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout

from src.utils.manifest import build_run_dir, utc_now_iso, write_manifest


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


def load_model_checkpoint(config: dict):
    with open(config["model"]["checkpoint_path"], "rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
    return ckpt


def load_input_dataset(config: dict) -> xr.Dataset:
    return xr.load_dataset(config["input"]["sample_path"])


def construct_wrapped_graphcast(
    model_config: graphcast.ModelConfig,
    task_config: graphcast.TaskConfig,
    diffs_stddev_by_level: xr.Dataset,
    mean_by_level: xr.Dataset,
    stddev_by_level: xr.Dataset,
):
    predictor = graphcast.GraphCast(model_config=model_config, task_config=task_config)
    predictor = casting.Bfloat16Cast(predictor)
    predictor = normalization.InputsAndResiduals(
        predictor=predictor,
        diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level,
    )
    predictor = autoregressive.Predictor(predictor)
    return predictor


def run_forward(
    model_config: graphcast.ModelConfig,
    task_config: graphcast.TaskConfig,
    diffs_stddev_by_level: xr.Dataset,
    mean_by_level: xr.Dataset,
    stddev_by_level: xr.Dataset,
    inputs: xr.Dataset,
    targets_template: xr.Dataset,
    forcings: xr.Dataset,
):
    predictor = construct_wrapped_graphcast(
        model_config=model_config,
        task_config=task_config,
        diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level,
    )
    return predictor(
        inputs=inputs,
        targets_template=targets_template,
        forcings=forcings,
    )


def prepare_inputs_targets_forcings(
    dataset: xr.Dataset,
    task_config: graphcast.TaskConfig,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        dataset=dataset,
        target_lead_times=slice("6h", "6h"),
        **dataclasses.asdict(task_config),
    )
    return inputs, targets, forcings


def save_forecast(path: str | Path, predictions: xr.Dataset) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_netcdf(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
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

    inputs, targets_template, forcings = prepare_inputs_targets_forcings(
        dataset=dataset,
        task_config=ckpt.task_config,
    )

    @hk.transform_with_state
    def run_model(inputs, targets_template, forcings):
        return run_forward(
            model_config=ckpt.model_config,
            task_config=ckpt.task_config,
            diffs_stddev_by_level=diffs_stddev_by_level,
            mean_by_level=mean_by_level,
            stddev_by_level=stddev_by_level,
            inputs=inputs,
            targets_template=targets_template,
            forcings=forcings,
        )

    params = ckpt.params
    state = {}

    def run_forward_apply(rng, inputs, targets_template, forcings):
        predictions, _ = run_model.apply(
            params,
            state,
            rng,
            inputs,
            targets_template,
            forcings,
        )
        return predictions

    run_forward_jitted = jax.jit(run_forward_apply)

    predictions = rollout.chunked_prediction(
        run_forward_jitted,
        rng=jax.random.PRNGKey(seed),
        inputs=inputs,
        targets_template=targets_template,
        forcings=forcings,
    )

    save_forecast(forecast_path, predictions)

    manifest = {
        "created_at_utc": utc_now_iso(),
        "stage": "stage1",
        "run_name": run_name,
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
    write_manifest(manifest_path, manifest)

    print(f"Run complete. Outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()