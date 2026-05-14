from pathlib import Path

from src.inference.run import load_config


def test_load_config():
    config_path = Path("configs/inference.yaml")
    config = load_config(config_path)
    assert config["run"]["name"] == "graphcast_stage1"
    assert config["model"]["name"] == "GraphCast_small"