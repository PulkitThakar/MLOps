from __future__ import annotations

import dataclasses

import haiku as hk
import jax
import xarray as xr

from graphcast import autoregressive
from graphcast import casting
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout


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


def generate_predictions(
    ckpt: graphcast.CheckPoint,
    diffs_stddev_by_level: xr.Dataset,
    mean_by_level: xr.Dataset,
    stddev_by_level: xr.Dataset,
    dataset: xr.Dataset,
    seed: int,
) -> xr.Dataset:
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
    return predictions