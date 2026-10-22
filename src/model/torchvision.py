"""不触发网络下载的 TorchVision 单类检测模型工厂。"""

from __future__ import annotations
from functools import partial
import os

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models.detection import FasterRCNN, RetinaNet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.ops.misc import FrozenBatchNorm2d

MODEL_NAMES = frozenset({"fasterrcnn_resnet50_fpn", "retinanet_resnet50_fpn"})
_IMAGE_MEAN = [0.485, 0.456, 0.406]
_IMAGE_STD = [0.229, 0.224, 0.225]


def configure_determinism(enabled: bool) -> None:
    """显式选择 CUDA 数值后端，避免训练状态泄漏到重载评测。"""
    if enabled:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = enabled
    torch.use_deterministic_algorithms(enabled, warn_only=True)


def _fixed_transform(imgsz: int) -> GeneralizedRCNNTransform:
    """固定为正方形拉伸，同时保留官方 COCO ImageNet 归一化语义。"""
    return GeneralizedRCNNTransform(
        min_size=imgsz,
        max_size=imgsz,
        image_mean=_IMAGE_MEAN,
        image_std=_IMAGE_STD,
        fixed_size=(imgsz, imgsz),
    )


def _faster_backbone() -> nn.Module:
    """复刻官方 COCO 检测骨干的 FrozenBN、三层可训练骨干和完整 FPN。"""
    return resnet_fpn_backbone(
        backbone_name="resnet50",
        weights=None,
        norm_layer=partial(FrozenBatchNorm2d, eps=0.0),
        trainable_layers=3,
    )


def _retinanet_backbone() -> nn.Module:
    """复刻官方 COCO RetinaNet 的 FrozenBN、C3--C5 FPN 与 P6/P7。"""
    return resnet_fpn_backbone(
        backbone_name="resnet50",
        weights=None,
        norm_layer=partial(FrozenBatchNorm2d, eps=0.0),
        trainable_layers=3,
        returned_layers=[2, 3, 4],
        extra_blocks=LastLevelP6P7(256, 256),
    )


def _load_official_coco_weights(model: nn.Module, pretrained_path: Path) -> None:
    path = Path(pretrained_path)
    if not path.is_file():
        raise FileNotFoundError(f"请先下载官方 COCO 预训练 state_dict：{path}")
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict):
        raise ValueError(f"官方 COCO 权重必须是 state_dict：{path}")
    model.load_state_dict(state_dict, strict=True)


def build_model(
    model_name: str,
    imgsz: int = 640,
    pretrained_path: Path | None = None,
    score_thresh: float = 0.001,
    nms_thresh: float = 0.7,
    max_detections: int = 100,
) -> nn.Module:
    """构造单类模型；给定本地 COCO 权重时先严格加载 91 类头再替换输出头。"""
    if model_name not in MODEL_NAMES:
        raise ValueError(f"model_name must be one of {sorted(MODEL_NAMES)}, got {model_name!r}")
    if imgsz <= 0:
        raise ValueError("imgsz must be positive")
    if max_detections <= 0:
        raise ValueError("max_detections must be positive")

    if model_name == "fasterrcnn_resnet50_fpn":
        model = FasterRCNN(
            _faster_backbone(),
            num_classes=91,
            box_score_thresh=score_thresh,
            box_nms_thresh=nms_thresh,
            box_detections_per_img=max_detections,
        )
        model.transform = _fixed_transform(imgsz)
        if pretrained_path is not None:
            _load_official_coco_weights(model, Path(pretrained_path))
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        return model

    model = RetinaNet(
        _retinanet_backbone(),
        num_classes=91,
        score_thresh=score_thresh,
        nms_thresh=nms_thresh,
        detections_per_img=max_detections,
    )
    model.transform = _fixed_transform(imgsz)
    if pretrained_path is not None:
        _load_official_coco_weights(model, Path(pretrained_path))
    classification_head = model.head.classification_head
    model.head.classification_head = RetinaNetClassificationHead(
        model.backbone.out_channels,
        classification_head.num_anchors,
        num_classes=1,
    )
    # 只重新学习类别输出，保留 COCO 预训练分类塔。
    model.head.classification_head.conv.load_state_dict(classification_head.conv.state_dict())
    return model


def _checkpoint_config(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint configuration needs a {key} mapping")
    return value


def load_model(weights_path: Path, config: dict[str, Any]) -> tuple[nn.Module, str]:
    """按本项目训练检查点重建模型；不根据文件名猜测架构。"""
    checkpoint = torch.load(Path(weights_path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"TorchVision checkpoint must be a mapping: {weights_path}")
    model_name = checkpoint.get("model_name")
    if not isinstance(model_name, str) or model_name not in MODEL_NAMES:
        raise ValueError(f"TorchVision checkpoint has no supported model_name: {weights_path}")
    model_state = checkpoint.get("model")
    checkpoint_config = checkpoint.get("config")
    if not isinstance(model_state, dict):
        raise ValueError(f"TorchVision checkpoint has no model state_dict: {weights_path}")
    if not isinstance(checkpoint_config, dict) or checkpoint_config.get("model_name") != model_name:
        raise ValueError(f"TorchVision checkpoint has inconsistent saved config: {weights_path}")

    training = _checkpoint_config(config, "training")
    evaluation = _checkpoint_config(config, "evaluation")
    model = build_model(
        model_name,
        imgsz=int(training["imgsz"]),
        score_thresh=float(evaluation["score_floor"]),
        nms_thresh=float(evaluation["nms_iou"]),
        max_detections=int(evaluation["max_detections"]),
    )
    model.load_state_dict(model_state, strict=True)
    return model, model_name


__all__ = ["MODEL_NAMES", "build_model", "load_model"]
