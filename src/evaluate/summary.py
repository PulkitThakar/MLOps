from __future__ import annotations

import xarray as xr


def summarize_da(da: xr.DataArray) -> dict:
    return {
        "dims": list(da.dims),
        "shape": list(da.shape),
        "min": float(da.min().item()),
        "max": float(da.max().item()),
        "mean": float(da.mean().item()),
        "std": float(da.std().item()),
    }


def reduce_extra_dims(da: xr.DataArray) -> xr.DataArray:
    indexers = {}
    for dim in da.dims:
        if dim in {"time", "lat", "lon", "level"}:
            continue
        indexers[dim] = 0
    if indexers:
        da = da.isel(**indexers)
    return da


def maybe_select_level(
    da_fcst: xr.DataArray,
    da_truth: xr.DataArray,
    level: int | None,
) -> tuple[xr.DataArray, xr.DataArray]:
    if "level" not in da_fcst.dims and "level" not in da_truth.dims:
        return da_fcst, da_truth

    if level is None:
        if "level" in da_fcst.dims:
            da_fcst = da_fcst.isel(level=0)
        if "level" in da_truth.dims:
            da_truth = da_truth.isel(level=0)
        return da_fcst, da_truth

    if "level" in da_fcst.dims:
        da_fcst = da_fcst.sel(level=level)
    if "level" in da_truth.dims:
        da_truth = da_truth.sel(level=level)
    return da_fcst, da_truth