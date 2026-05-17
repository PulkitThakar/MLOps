from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr


PLOT_VAR = "2m_temperature"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--output", default="outputs/plots/forecast_2m_temperature.png")
    args = parser.parse_args()

    ds = xr.load_dataset(args.path)
    da = ds[PLOT_VAR].isel(time=0, batch=0)
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    da.plot.contourf(ax=ax, x="lon", y="lat", cmap="coolwarm", levels=21)
    ax.set_title(f"GraphCast forecast: {PLOT_VAR}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()