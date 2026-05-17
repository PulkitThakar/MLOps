from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from src.evaluate.summary import maybe_select_level, reduce_extra_dims, summarize_da


DEFAULT_VARS = [
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]


def score_variable(
    forecast: xr.Dataset,
    truth: xr.Dataset,
    var_name: str,
    level: int | None = None,
) -> dict:
    if var_name not in forecast.data_vars:
        return {"variable": var_name, "status": "missing_in_forecast"}
    if var_name not in truth.data_vars:
        return {"variable": var_name, "status": "missing_in_truth"}

    da_fcst = reduce_extra_dims(forecast[var_name])
    da_truth = reduce_extra_dims(truth[var_name])

    forecast_summary = summarize_da(da_fcst)

    da_fcst, da_truth = maybe_select_level(da_fcst, da_truth, level)
    da_fcst, da_truth = xr.align(da_fcst, da_truth, join="inner")

    if da_fcst.size == 0 or da_truth.size == 0:
        return {
            "variable": var_name,
            "status": "no_overlap_after_align",
            "forecast_summary": forecast_summary,
        }

    diff = da_fcst - da_truth

    return {
        "variable": var_name,
        "status": "ok",
        "forecast_summary": forecast_summary,
        "truth_summary": summarize_da(da_truth),
        "comparison_shape": list(da_fcst.shape),
        "comparison_dims": list(da_fcst.dims),
        "mae": float(np.abs(diff).mean().item()),
        "rmse": float(np.sqrt((diff ** 2).mean().item())),
        "bias": float(diff.mean().item()),
    }


def summarize_only(
    forecast: xr.Dataset,
    variables: list[str],
) -> list[dict]:
    results = []
    for var_name in variables:
        if var_name not in forecast.data_vars:
            results.append(
                {
                    "variable": var_name,
                    "status": "missing_in_forecast",
                }
            )
            continue

        da_fcst = reduce_extra_dims(forecast[var_name])
        results.append(
            {
                "variable": var_name,
                "status": "skipped_no_truth",
                "forecast_summary": summarize_da(da_fcst),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast", required=True)
    parser.add_argument("--truth", default=None)
    parser.add_argument("--vars", nargs="*", default=DEFAULT_VARS)
    parser.add_argument("--level", type=int, default=None)
    parser.add_argument("--output", default="outputs/metrics/metrics.json")
    args = parser.parse_args()

    forecast = xr.load_dataset(args.forecast)

    report = {
        "forecast_path": args.forecast,
        "truth_path": args.truth,
        "variables_requested": args.vars,
        "level": args.level,
    }

    if args.truth is None:
        report["mode"] = "summary_only"
        report["results"] = summarize_only(forecast, args.vars)
    else:
        truth = xr.load_dataset(args.truth)
        report["mode"] = "forecast_vs_truth"
        report["results"] = [
            score_variable(
                forecast=forecast,
                truth=truth,
                var_name=var_name,
                level=args.level,
            )
            for var_name in args.vars
        ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Metrics written to: {output_path.resolve()}")


if __name__ == "__main__":
    main()