"""TorchVision detector prediction adapter for the common COCO evaluator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from src.evaluation.prediction import coco_detections_from_xyxy
from src.model.torchvision import configure_determinism

_TORCHVISION_LABELS = {
    "fasterrcnn_resnet50_fpn": {1: 1},
    "retinanet_resnet50_fpn": {0: 1},
}


def device_from_config(config: dict[str, Any]) -> torch.device:
    """Resolve the configured CPU or CUDA device without changing its policy."""
    configured = config["training"].get("device", 0)
    if isinstance(configured, int):
        return torch.device(f"cuda:{configured}")
    return torch.device(str(configured))


def _rgb_tensor(image_path: Path) -> tuple[torch.Tensor, tuple[int, int]]:
    """Read one original image as an unscaled RGB float tensor."""
    with Image.open(image_path) as source:
        rgb = source.convert("RGB")
        image_size = rgb.size
        tensor = pil_to_tensor(rgb).to(dtype=torch.float32).div_(255.0)
    return tensor, image_size


def _prediction_to_coco(
    output: dict[str, torch.Tensor],
    *,
    model_name: str,
    image_id: int,
    score_floor: float,
    max_detections: int,
    image_size: tuple[int, int],
) -> tuple[list[dict[str, Any]], int]:
    """Map TorchVision's architecture-specific labels to COCO category 1."""
    try:
        label_to_category = _TORCHVISION_LABELS[model_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported TorchVision detector: {model_name}") from exc
    return coco_detections_from_xyxy(
        output["boxes"].detach().cpu().tolist(),
        output["scores"].detach().cpu().tolist(),
        output["labels"].detach().cpu().tolist(),
        image_id=image_id,
        label_to_category=label_to_category,
        score_floor=score_floor,
        max_detections=max_detections,
        image_size=image_size,
    )


def predict_torchvision_split(
    model: torch.nn.Module,
    model_name: str,
    config: dict[str, Any],
    images: list[dict[str, Any]],
    *,
    split: str,
) -> dict[str, Any]:
    """Run original-size images through a TorchVision detector one image at a time.

    The detector's ``GeneralizedRCNNTransform`` performs the fixed-square stretch and
    maps output boxes back to original pixels.  No second inverse resize is applied.
    """
    configure_determinism(False)
    device = device_from_config(config)
    data_root = Path(config["data"]).parent
    score_floor = float(config["evaluation"]["score_floor"])
    max_detections = int(config["evaluation"]["max_detections"])
    model = model.to(device).float().eval()

    predictions: list[dict[str, Any]] = []
    discarded_nonpositive_boxes = 0
    with torch.inference_mode():
        for image in images:
            file_name = Path(str(image["file_name"]))
            image_path = data_root / "images" / file_name
            tensor, image_size = _rgb_tensor(image_path)
            output = model([tensor.to(device)])[0]
            image_predictions, discarded = _prediction_to_coco(
                output,
                model_name=model_name,
                image_id=int(image["id"]),
                score_floor=score_floor,
                max_detections=max_detections,
                image_size=image_size,
            )
            predictions.extend(image_predictions)
            discarded_nonpositive_boxes += discarded

    return {
        "predictions": predictions,
        "discarded_nonpositive_boxes": discarded_nonpositive_boxes,
        "device": str(device),
        "input_protocol": (
            "original-size RGB float tensor in [0, 1]; TorchVision GeneralizedRCNNTransform "
            "applies fixed-square stretch, ImageNet normalization, and restores output boxes"
        ),
    }


def detect_bgr_torchvision(
    model: torch.nn.Module,
    model_name: str,
    bgr: np.ndarray,
    device: torch.device,
    *,
    score_floor: float,
    max_detections: int,
) -> np.ndarray:
    """Time the complete TorchVision path from in-memory BGR to CPU detections.

    The caller places this immutable eval model on ``device`` once before warmup;
    moving it per image would make the timing include avoidable work.
    """
    rgb = np.ascontiguousarray(bgr[..., ::-1])
    height, width = rgb.shape[:2]
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(dtype=torch.float32).div_(255.0)
    with torch.inference_mode():
        output = model([tensor.to(device)])[0]
    # 计时只物化最终 CPU 检测，不往返构造 COCO 字典或重复计算 xywh。
    detections = torch.cat(
        (output["boxes"], output["scores"][:, None], output["labels"][:, None]), dim=1
    ).cpu().numpy()
    detections[:, [0, 2]] = np.clip(detections[:, [0, 2]], 0, width)
    detections[:, [1, 3]] = np.clip(detections[:, [1, 3]], 0, height)
    valid = (
        (detections[:, 2] > detections[:, 0])
        & (detections[:, 3] > detections[:, 1])
        & (detections[:, 4] >= score_floor)
    )
    detections = detections[valid][:max_detections]
    detections[:, 5] = 0
    return detections
