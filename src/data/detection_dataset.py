"""TorchVision 检测器使用的冻结 COCO 数据集读取器。"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

_FASTERRCNN = "fasterrcnn_resnet50_fpn"
_RETINANET = "retinanet_resnet50_fpn"
_ALLOWED_SPLITS = {"train", "val", "test"}


class DetectionDataset(Dataset[tuple[Tensor, dict[str, Tensor]]]):
    """从处理后的 COCO 标注读取原始 RGB 图像和 xyxy 框，不在这里缩放图像。"""

    def __init__(
        self,
        data_yaml: Path,
        split: str,
        model_name: str,
        augment: bool = False,
        augmentation: dict[str, Any] | None = None,
    ) -> None:
        if split not in _ALLOWED_SPLITS:
            raise ValueError(f"split must be one of {sorted(_ALLOWED_SPLITS)}, got {split!r}")
        if model_name not in {_FASTERRCNN, _RETINANET}:
            raise ValueError(f"unsupported TorchVision model {model_name!r}")

        self.root = Path(data_yaml).resolve().parent
        self.split = split
        self.model_name = model_name
        self.augment = bool(augment)
        self.augmentation = dict(augmentation or {})
        self._flip_probability = float(self.augmentation.get("horizontal_flip", 0.0))
        self._brightness = float(self.augmentation.get("brightness", 0.0))
        if not 0.0 <= self._flip_probability <= 1.0:
            raise ValueError("augmentation.horizontal_flip must be between 0 and 1")
        if not 0.0 <= self._brightness <= 1.0:
            raise ValueError("augmentation.brightness must be between 0 and 1")

        try:
            data = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(f"cannot read dataset YAML {data_yaml}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("dataset YAML must be a mapping")

        annotation_path = self.root / "annotations" / f"instances_{split}.json"
        try:
            self.annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read COCO annotations {annotation_path}: {exc}") from exc
        if not isinstance(self.annotations, dict):
            raise ValueError(f"COCO annotations must be an object: {annotation_path}")

        raw_images = self.annotations.get("images")
        raw_annotations = self.annotations.get("annotations")
        if not isinstance(raw_images, list) or not isinstance(raw_annotations, list):
            raise ValueError(f"COCO annotations need images and annotations lists: {annotation_path}")
        self.images = sorted(raw_images, key=lambda image: int(image["id"]))
        self._targets = self._index_targets(raw_annotations)
        self._label = 1 if model_name == _FASTERRCNN else 0

    @staticmethod
    def _index_targets(raw_annotations: list[dict[str, Any]]) -> dict[int, list[list[float]]]:
        """将 COCO xywh 保持为按 image_id 索引的原始框，读取时转换为 xyxy。"""
        targets: dict[int, list[list[float]]] = defaultdict(list)
        for annotation in raw_annotations:
            if not isinstance(annotation, dict):
                raise ValueError("COCO annotation must be an object")
            if int(annotation.get("category_id", -1)) != 1:
                raise ValueError("crack dataset must use COCO category_id=1")
            image_id = int(annotation["image_id"])
            bbox = annotation.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"invalid bbox for COCO image {image_id}")
            x, y, width, height = (float(value) for value in bbox)
            if width <= 0.0 or height <= 0.0:
                raise ValueError(f"COCO image {image_id} has a non-positive bbox")
            targets[image_id].append([x, y, x + width, y + height])
        return dict(targets)

    def __len__(self) -> int:
        return len(self.images)

    def _image_path(self, image: dict[str, Any]) -> Path:
        file_name = image.get("file_name")
        if not isinstance(file_name, str):
            raise ValueError("COCO image file_name must be a string")
        relative = Path(file_name)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != self.split:
            raise ValueError(f"COCO image is not in {self.split}/: {file_name!r}")
        return self.root / "images" / relative

    def _augment(self, image: Tensor, boxes: Tensor) -> tuple[Tensor, Tensor]:
        if not self.augment:
            return image, boxes
        if self._flip_probability and torch.rand(()) < self._flip_probability:
            image = F.hflip(image)
            if boxes.numel():
                width = image.shape[-1]
                boxes = boxes.clone()
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        if self._brightness:
            factor = 1.0 + float(torch.empty(()).uniform_(-self._brightness, self._brightness))
            image = F.adjust_brightness(image, factor)
        return image, boxes

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        image_info = self.images[index]
        image_id = int(image_info["id"])
        image_path = self._image_path(image_info)
        try:
            with Image.open(image_path) as source:
                image = F.to_tensor(source.convert("RGB"))
        except OSError as exc:
            raise ValueError(f"cannot read COCO image {image_path}: {exc}") from exc

        raw_boxes = self._targets.get(image_id, [])
        boxes = (
            torch.tensor(raw_boxes, dtype=torch.float32)
            if raw_boxes
            else torch.empty((0, 4), dtype=torch.float32)
        )
        image, boxes = self._augment(image, boxes)
        target = {
            "boxes": boxes,
            "labels": torch.full((boxes.shape[0],), self._label, dtype=torch.int64),
            "image_id": torch.tensor(image_id, dtype=torch.int64),
        }
        return image, target


def collate_fn(
    batch: list[tuple[Tensor, dict[str, Tensor]]],
) -> tuple[tuple[Tensor, ...], tuple[dict[str, Tensor], ...]]:
    """保留变长原始图像和变长检测框，交给 TorchVision transform 成批。"""
    images, targets = zip(*batch, strict=True)
    return images, targets


__all__ = ["DetectionDataset", "collate_fn"]
