from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr


def inspect_nc(path: str | Path) -> None:
    ds = xr.load_dataset(path)
    print("File:", path)
    print("Dimensions:")
    for k, v in ds.sizes.items():
        print(f"  {k}: {v}")
    print("Variables:")
    for name, da in ds.data_vars.items():
        shape = "x".join(str(s) for s in da.shape)
        print(f"  {name}: shape={shape}, dtype={da.dtype}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    inspect_nc(args.path)


if __name__ == "__main__":
    main()