from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from src.utils.manifest import build_run_dir, utc_now_iso, write_manifest


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def hash_file(path: str | Path) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_stub_forecast(path: str | Path, config: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "stub_forecast_created",
        "message": "Replace this stub with the real GraphCast inference call.",
        "model": config["model"]["name"],
        "input_sample": config["input"]["sample_path"],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    run_name = config["run"]["name"]
    output_root = config["run"]["output_root"]

    run_dir = build_run_dir(output_root=output_root, run_name=run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    forecast_path = run_dir / config["artifact"]["forecast_filename"]
    manifest_path = run_dir / config["artifact"]["manifest_filename"]

    save_stub_forecast(forecast_path, config)

    manifest = {
        "created_at_utc": utc_now_iso(),
        "stage": "stage1",
        "run_name": run_name,
        "model_name": config["model"]["name"],
        "model_provider": config["model"]["provider"],
        "input_sample_path": config["input"]["sample_path"],
        "input_sample_exists": Path(config["input"]["sample_path"]).exists(),
        "input_sample_sha256": hash_file(config["input"]["sample_path"]),
        "forecast_path": str(forecast_path),
        "forecast_exists": forecast_path.exists(),
    }
    write_manifest(manifest_path, manifest)

    print(f"Run complete. Outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()