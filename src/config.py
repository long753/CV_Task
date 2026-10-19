"""读取统一 Ultralytics 训练与评测配置。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    """读取 YAML 并将配置中约定的相对路径解析为绝对路径。"""
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Ultralytics configuration must be a mapping")

    models = config.get("models")
    if not isinstance(models, dict):
        raise ValueError("models must be a mapping")

    config["models"] = {
        name: (config_path.parent / Path(model_path)).resolve()
        for name, model_path in models.items()
    }
    for key in ("data", "project", "trained_models"):
        config[key] = (config_path.parent / Path(config[key])).resolve()
    config["config_path"] = config_path
    return config
