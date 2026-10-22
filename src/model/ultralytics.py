"""加载项目支持的 Ultralytics 检测模型。"""

from __future__ import annotations

from pathlib import Path

from ultralytics import RTDETR, YOLO


_MODEL_NAMES = frozenset({"yolov8s", "yolov11s", "rtdetr_l"})


def _has_rtdetr_decoder(model: object) -> bool:
    """检查已载入网络的检测头，兼容官方历史 RT-DETR 检查点。"""
    layers = getattr(model, "model", None)
    return bool(layers) and "RTDETRDecoder" in layers[-1].__class__.__name__


def load_ultralytics_model(weights: Path, model_name: str | None = None) -> YOLO | RTDETR:
    """按已知架构加载模型，或从检查点内实际结构识别其架构。"""
    weights_path = Path(weights).resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"Ultralytics weights do not exist: {weights_path}")

    if model_name is not None:
        if model_name not in _MODEL_NAMES:
            raise ValueError(f"Unsupported Ultralytics model: {model_name}")
        if model_name == "rtdetr_l":
            return RTDETR(str(weights_path))
        return YOLO(str(weights_path))

    loaded = YOLO(str(weights_path))
    # 固定依赖版本会在检测到 RTDETRDecoder 后将 YOLO 实例转换为 RTDETR。
    # 显式检查检查点内的末端层，而非 best.pt / last.pt 等文件名；旧版官方权重
    # 的主体虽为 DetectionModel，末端同样是 RTDETRDecoder。
    if _has_rtdetr_decoder(loaded.model) and not isinstance(loaded, RTDETR):
        return RTDETR(str(weights_path))
    return loaded
