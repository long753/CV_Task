"""五种裂缝检测器共用的冻结权重评测流程。"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from pickle import UnpicklingError
from typing import Any

import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ULTRALYTICS_CONFIG_DIR = PROJECT_ROOT / "outputs" / ".ultralytics"
MATPLOTLIB_CONFIG_DIR = PROJECT_ROOT / "outputs" / ".matplotlib"
ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(ULTRALYTICS_CONFIG_DIR)
os.environ["MPLCONFIGDIR"] = str(MATPLOTLIB_CONFIG_DIR)
os.environ["YOLO_AUTOINSTALL"] = "false"

from ultralytics import settings

from src.evaluation.benchmark import benchmark_checkpoint
from src.evaluation.metrics import evaluate_predictions
from src.evaluation.prediction import coco_detections_from_xyxy
from src.evaluation.torchvision import predict_torchvision_split
from src.model.torchvision import MODEL_NAMES as TORCHVISION_MODEL_NAMES
from src.model.torchvision import load_model as load_torchvision_model
from src.model.ultralytics import load_ultralytics_model

_ALLOWED_SPLITS = {"val", "test"}
_ULTRALYTICS_MODEL_NAMES = frozenset({"yolov8s", "yolov11s", "rtdetr_l"})
_SUPPORTED_MODEL_NAMES = TORCHVISION_MODEL_NAMES | _ULTRALYTICS_MODEL_NAMES


def configure_ultralytics() -> None:
    """将 Ultralytics 状态、权重和可选集成限定在项目目录内。"""
    updates: dict[str, Any] = {
        "weights_dir": str(PROJECT_ROOT / "models" / "pretrained"),
        "tensorboard": True,
    }
    for key in (
        "clearml",
        "comet",
        "dvc",
        "hub",
        "mlflow",
        "neptune",
        "raytune",
        "wandb",
        "sync",
    ):
        if key in settings:
            updates[key] = False
    settings.update(updates)


configure_ultralytics()


def load_split_annotations(config: dict[str, Any], split: str) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    """读取并校验指定留出数据划分的 COCO 标注。"""
    if split not in _ALLOWED_SPLITS:
        raise ValueError(f"split must be one of {sorted(_ALLOWED_SPLITS)}, got {split!r}")
    data_root = Path(config["data"]).parent
    gt_path = data_root / "annotations" / f"instances_{split}.json"
    with gt_path.open(encoding="utf-8") as handle:
        annotations = json.load(handle)
    images = annotations.get("images")
    if not isinstance(images, list):
        raise ValueError(f"Invalid COCO {split} images in {gt_path}")

    split_images = sorted(images, key=lambda image: int(image["id"]))
    for image in split_images:
        file_name = Path(image["file_name"])
        if (
            file_name.is_absolute()
            or not file_name.parts
            or file_name.parts[0] != split
            or ".." in file_name.parts
        ):
            raise ValueError(f"COCO image is not in {split}/: {image['file_name']}")
    return gt_path, annotations, split_images


def _coco_detections(
    boxes: Any,
    image_id: int,
    *,
    score_floor: float = 0.0,
    max_detections: int = 100,
    image_size: tuple[int, int] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """导出 YOLO/RT-DETR 正面积框，并将内部类别 0 映射为 COCO 类别 1。"""
    if boxes is None:
        return [], 0
    return coco_detections_from_xyxy(
        boxes.xyxy.detach().cpu().tolist(),
        boxes.conf.detach().cpu().tolist(),
        boxes.cls.detach().cpu().tolist(),
        image_id=image_id,
        label_to_category={0: 1},
        score_floor=score_floor,
        max_detections=max_detections,
        image_size=image_size,
    )


def predict_split(
    model: Any,
    config: dict[str, Any],
    split: str = "val",
    *,
    model_name: str | None = None,
) -> dict[str, Any]:
    """以 batch=1 流式处理 Ultralytics 数据划分并导出原图 COCO 坐标。"""
    gt_path, annotations, images = load_split_annotations(config, split)
    data_root = Path(config["data"]).parent
    image_paths = {(data_root / "images" / image["file_name"]).resolve(): image for image in images}
    if len(image_paths) != len(images):
        raise ValueError(f"COCO {split} file names must be unique")

    evaluation = config["evaluation"]
    arguments: dict[str, Any] = {
        "source": str(data_root / "images" / split),
        "stream": True,
        "batch": 1,
        "half": False,
        "augment": False,
        "imgsz": config["training"]["imgsz"],
        "rect": False,
        "device": config["training"]["device"],
        "conf": evaluation["score_floor"],
        "max_det": evaluation["max_detections"],
        "verbose": False,
        "save": False,
    }
    if model_name != "rtdetr_l":
        arguments["iou"] = evaluation["nms_iou"]

    predictions: list[dict[str, Any]] = []
    seen_image_ids: set[int] = set()
    discarded_nonpositive_boxes = 0
    for result in model.predict(**arguments):
        image = image_paths.get(Path(result.path).resolve())
        if image is None:
            raise RuntimeError(f"Prediction returned an unexpected {split} image path: {result.path}")
        image_id = int(image["id"])
        if image_id in seen_image_ids:
            raise RuntimeError(f"Prediction returned duplicate {split} image {result.path}")
        seen_image_ids.add(image_id)
        image_predictions, discarded = _coco_detections(
            result.boxes,
            image_id,
            score_floor=float(evaluation["score_floor"]),
            max_detections=int(evaluation["max_detections"]),
            image_size=(int(image["width"]), int(image["height"])),
        )
        predictions.extend(image_predictions)
        discarded_nonpositive_boxes += discarded

    expected_image_ids = {int(image["id"]) for image in images}
    if seen_image_ids != expected_image_ids:
        raise RuntimeError(f"Prediction stream did not return every {split} image exactly once")
    return {
        "ground_truth": gt_path,
        "annotations": annotations,
        "images": images,
        "predictions": predictions,
        "discarded_nonpositive_boxes": discarded_nonpositive_boxes,
        "input_protocol": (
            "Ultralytics RT-DETR square stretch, then boxes restored to original-image pixels"
            if model_name == "rtdetr_l"
            else "Ultralytics YOLO letterbox, then boxes restored to original-image pixels"
        ),
    }


def _preview_images(images: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """按来源前缀轮转选择确定性的样本。"""
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for image in images:
        by_source[Path(image["file_name"]).stem.split("_", 1)[0]].append(image)
    for source in by_source:
        by_source[source].sort(key=lambda image: (str(image["file_name"]), int(image["id"])))

    selected: list[dict[str, Any]] = []
    positions = {source: 0 for source in by_source}
    while len(selected) < count:
        added = False
        for source in sorted(by_source):
            position = positions[source]
            if position < len(by_source[source]) and len(selected) < count:
                selected.append(by_source[source][position])
                positions[source] += 1
                added = True
        if not added:
            break
    if len(selected) != count:
        raise ValueError(f"Requested {count} previews, but split contains only {len(selected)} images")
    return selected


def render_prediction_previews(
    annotations: dict[str, Any],
    images: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    config: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    """绘制确定性的 GT 绿色与预测橙色对比图。"""
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    gt_by_image: dict[int, list[list[float]]] = defaultdict(list)
    for annotation in annotations.get("annotations", []):
        if int(annotation["category_id"]) == 1:
            gt_by_image[int(annotation["image_id"])].append(annotation["bbox"])

    confidence = float(config["evaluation"]["confidence"])
    predictions_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        if prediction["score"] >= confidence:
            predictions_by_image[int(prediction["image_id"])].append(prediction)

    data_root = Path(config["data"]).parent
    preview_paths: list[str] = []
    for index, image_data in enumerate(
        _preview_images(images, int(config["evaluation"]["preview_count"])), start=1
    ):
        image_id = int(image_data["id"])
        image_path = data_root / "images" / image_data["file_name"]
        with Image.open(image_path) as source_image:
            preview = source_image.convert("RGB")
        draw = ImageDraw.Draw(preview)
        line_width = max(2, min(preview.size) // 200)
        draw.rectangle((0, 0, preview.width, 22), fill=(0, 0, 0))
        draw.text((4, 4), "GT green | prediction orange", fill=(255, 255, 255))
        for x, y, width, height in gt_by_image[image_id]:
            draw.rectangle((x, y, x + width, y + height), outline=(0, 220, 0), width=line_width)
        for prediction in predictions_by_image[image_id]:
            x, y, width, height = prediction["bbox"]
            draw.rectangle((x, y, x + width, y + height), outline=(255, 140, 0), width=line_width)
            draw.text((x, max(23, y - 12)), f"{prediction['score']:.2f}", fill=(255, 140, 0))

        source = Path(image_data["file_name"]).stem.split("_", 1)[0]
        preview_path = preview_dir / f"{index:02d}_{source}_{Path(image_data['file_name']).name}"
        preview.save(preview_path, quality=95)
        preview_paths.append(str(preview_path.relative_to(output_dir.parent)))
    return preview_paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _class_names(model: Any) -> dict[int, str]:
    """在导出类别标识前，验证 Ultralytics 冻结权重仅包含 crack 类别。"""
    names = model.names
    normalized = {int(index): str(name) for index, name in names.items()} if isinstance(names, dict) else {
        index: str(name) for index, name in enumerate(names)
    }
    if normalized != {0: "crack"}:
        raise RuntimeError(f"Frozen checkpoint must contain exactly class 0=crack, got {normalized}")
    return normalized


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def _preview_paths(output_dir: Path, previews: list[str]) -> list[str]:
    return [str((output_dir.parent / preview).resolve()) for preview in previews]


def _write_summary_csv(
    output_dir: Path,
    weights_path: Path,
    model_name: str,
    split: str,
    metrics: dict[str, Any],
    benchmark: dict[str, Any],
) -> Path:
    """只写入实际测得的评测和性能值，不生成占位单元格。"""
    profile = benchmark["profile"]
    latency = benchmark["latency"]
    summary_path = output_dir / "summary.csv"
    row = {
        "model": str(weights_path),
        "architecture": model_name,
        "split": split,
        "conditions": benchmark["measurement_conditions"]["label"],
        "P": metrics["precision"],
        "R": metrics["recall"],
        "mAP50": metrics["map50"],
        "mAP50-95": metrics["map50_95"],
        "Params_M": profile["parameters_million"],
        "GFLOPs": profile["gflops"],
        "Latency_mean_ms": latency["mean_ms"],
        "Latency_median_ms": latency["median_ms"],
        "Latency_p95_ms": latency["p95_ms"],
        "Latency_std_ms": latency["std_ms"],
        "FPS": latency["sequential_fps"],
    }
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return summary_path


def _benchmark_summary(benchmark: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol": benchmark["protocol"],
        "representative_images": benchmark["representative_images"],
        "measurement_conditions": benchmark["measurement_conditions"],
        "profile": benchmark["profile"],
        "latency": {key: value for key, value in benchmark["latency"].items() if key != "samples_ms"},
    }


def _resolve_weights_path(weights_path: Path) -> Path:
    resolved_path = weights_path.resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Evaluation weights do not exist: {resolved_path}")
    return resolved_path


def _requested_model_name(config: dict[str, Any]) -> str | None:
    model_name = config.get("model_name")
    if model_name is None:
        return None
    if not isinstance(model_name, str) or model_name not in _SUPPORTED_MODEL_NAMES:
        raise ValueError(f"model_name must be one of {sorted(_SUPPORTED_MODEL_NAMES)}, got {model_name!r}")
    return model_name


def _torchvision_checkpoint_model_name(weights_path: Path) -> str | None:
    """Identify only project TorchVision checkpoints by their required metadata."""
    try:
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    except UnpicklingError:
        return None
    if not isinstance(checkpoint, dict):
        return None
    model_name = checkpoint.get("model_name")
    return model_name if isinstance(model_name, str) and model_name in TORCHVISION_MODEL_NAMES else None


def _model_name_from_ultralytics_checkpoint(model: Any) -> str:
    """从真实架构和原生配置识别模型，包含官方文件名 yolo11s。"""
    if model.__class__.__name__ == "RTDETR":
        return "rtdetr_l"
    aliases = {"yolov8s": "yolov8s", "yolov11s": "yolov11s", "yolo11s": "yolov11s"}
    hints = [getattr(model.model, "yaml", {}).get("yaml_file")]
    checkpoint = getattr(model, "ckpt", None)
    if isinstance(checkpoint, dict):
        for key in ("train_args", "args"):
            arguments = checkpoint.get(key)
            if isinstance(arguments, dict):
                hints.append(arguments.get("model"))
    hints.append(getattr(model, "overrides", {}).get("model"))
    for hint in hints:
        if isinstance(hint, str) and Path(hint).stem in aliases:
            return aliases[Path(hint).stem]
    raise ValueError("Cannot determine Ultralytics architecture from checkpoint metadata; pass --model")


def _load_checkpoint_prediction(
    config: dict[str, Any], weights_path: Path, split: str
) -> tuple[str, dict[int, str], dict[str, Any]]:
    """加载五种支持架构之一并生成统一 COCO 格式预测。"""
    requested_name = _requested_model_name(config)
    checkpoint_torchvision_name = _torchvision_checkpoint_model_name(weights_path)
    if checkpoint_torchvision_name is not None:
        if requested_name is not None and requested_name != checkpoint_torchvision_name:
            raise ValueError(
                f"TorchVision checkpoint model_name {checkpoint_torchvision_name!r} does not match "
                f"requested {requested_name!r}"
            )
        model, model_name = load_torchvision_model(weights_path, config)
        gt_path, annotations, images = load_split_annotations(config, split)
        prediction = predict_torchvision_split(model, model_name, config, images, split=split)
        prediction.update({"ground_truth": gt_path, "annotations": annotations, "images": images})
        prediction["internal_label_mapping"] = (
            "Faster R-CNN foreground label 1 -> COCO category_id 1"
            if model_name == "fasterrcnn_resnet50_fpn"
            else "RetinaNet sigmoid foreground label 0 -> COCO category_id 1"
        )
        return model_name, {0: "crack"}, prediction

    if requested_name in TORCHVISION_MODEL_NAMES:
        raise ValueError(f"TorchVision checkpoint metadata is missing or incompatible: {weights_path}")
    model = load_ultralytics_model(weights_path, requested_name)
    model_name = requested_name or _model_name_from_ultralytics_checkpoint(model)
    class_names = _class_names(model)
    return model_name, class_names, predict_split(model, config, split=split, model_name=model_name)


def _evaluate_and_render(
    config: dict[str, Any], prediction: dict[str, Any], output_dir: Path
) -> tuple[dict[str, Any], list[str]]:
    metrics = evaluate_predictions(
        prediction["ground_truth"],
        prediction["predictions"],
        output_dir,
        confidence=float(config["evaluation"]["confidence"]),
        match_iou=float(config["evaluation"]["match_iou"]),
        max_detections=int(config["evaluation"]["max_detections"]),
    )
    previews = render_prediction_previews(
        prediction["annotations"], prediction["images"], prediction["predictions"], config, output_dir
    )
    return metrics, previews

def evaluate_model(
    config: dict[str, Any],
    model: Any,
    model_name: str,
    split: str,
    output_dir: Path,
) -> dict[str, Any]:
    """直接评测内存模型，不写临时 checkpoint 或重载网络。

    训练循环可在每个 epoch 结束时调用本函数取得 ``metrics["map50_95"]``；
    它仍写出与冻结权重评测相同的预测、COCO 指标、PR 曲线和预览图。
    """
    if model_name not in _SUPPORTED_MODEL_NAMES:
        raise ValueError(f"model_name must be one of {sorted(_SUPPORTED_MODEL_NAMES)}, got {model_name!r}")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_path, annotations, images = load_split_annotations(config, split)
    if model_name in TORCHVISION_MODEL_NAMES:
        prediction = predict_torchvision_split(model, model_name, config, images, split=split)
        prediction["internal_label_mapping"] = (
            "Faster R-CNN foreground label 1 -> COCO category_id 1"
            if model_name == "fasterrcnn_resnet50_fpn"
            else "RetinaNet sigmoid foreground label 0 -> COCO category_id 1"
        )
        class_names = {0: "crack"}
    else:
        class_names = _class_names(model)
        prediction = predict_split(model, config, split=split, model_name=model_name)
    prediction.update({"ground_truth": gt_path, "annotations": annotations, "images": images})
    metrics, previews = _evaluate_and_render(config, prediction, output_dir)
    return {
        "architecture": model_name,
        "split": split,
        "classes": class_names,
        "metrics": metrics,
        "discarded_nonpositive_boxes": prediction["discarded_nonpositive_boxes"],
        "previews": _preview_paths(output_dir, previews),
        "artifacts": {
            "output_dir": str(output_dir),
            "predictions_coco": str((output_dir / "predictions_coco.json").resolve()),
            "metrics": str((output_dir / "metrics.json").resolve()),
            "pr_curve": str((output_dir / "pr_curve.png").resolve()),
        },
    }


def _build_evaluation_summary(
    config: dict[str, Any],
    weights_path: Path,
    checkpoint_sha256: str,
    model_name: str,
    split: str,
    output_dir: Path,
    class_names: dict[int, str],
    prediction: dict[str, Any],
    metrics: dict[str, Any],
    previews: list[str],
) -> dict[str, Any]:
    return {
        "weights": {"path": str(weights_path), "sha256": checkpoint_sha256},
        "architecture": model_name,
        "split": split,
        "classes": class_names,
        "configuration": {
            "source_path": str(config.get("config_path", "")),
            "resolved": _json_value(config),
        },
        "prediction_protocol": {
            "coordinates": "original-image pixel xywh in COCO predictions",
            "input": prediction["input_protocol"],
            "score_floor": config["evaluation"]["score_floor"],
            "nms_iou": None if model_name == "rtdetr_l" else config["evaluation"]["nms_iou"],
            "max_detections": config["evaluation"]["max_detections"],
            "postprocess": "native RT-DETR query decoding without added NMS; score-sorted top-K cap" if model_name == "rtdetr_l" else "detector-native postprocess with score-sorted top-K cap",
            "internal_label_mapping": prediction.get("internal_label_mapping", "Ultralytics class 0 -> COCO category_id 1"),
            "empty_box_policy": "Clip to original-image bounds, discard boxes with non-positive width or height, and retain all positive sizes.",
            "discarded_nonpositive_boxes": prediction["discarded_nonpositive_boxes"],
            "fixed_threshold_confidence": config["evaluation"]["confidence"],
            "fixed_threshold_match_iou": config["evaluation"]["match_iou"],
            "test_policy": "The selected split is evaluated with the frozen configuration; no threshold tuning occurs here.",
        },
        "artifacts": {
            "output_dir": str(output_dir),
            "ground_truth": str(Path(prediction["ground_truth"]).resolve()),
            "predictions_coco": str((output_dir / "predictions_coco.json").resolve()),
            "metrics": str((output_dir / "metrics.json").resolve()),
            "pr_curve": str((output_dir / "pr_curve.png").resolve()),
            "previews": _preview_paths(output_dir, previews),
            "previews_relative_to_output_parent": previews,
        },
        "metrics": metrics,
    }


def _append_benchmark_summary(
    summary: dict[str, Any],
    config: dict[str, Any],
    weights_path: Path,
    model_name: str,
    split: str,
    output_dir: Path,
    metrics: dict[str, Any],
) -> None:
    benchmark = benchmark_checkpoint(weights_path, config, output_dir, model_name=model_name)
    summary_csv = _write_summary_csv(output_dir, weights_path, model_name, split, metrics, benchmark)
    summary["artifacts"]["benchmark"] = str((output_dir / "benchmark.json").resolve())
    summary["artifacts"]["summary_csv"] = str(summary_csv.resolve())
    summary["benchmark"] = _benchmark_summary(benchmark)


def _write_evaluation_summary(output_dir: Path, summary: dict[str, Any]) -> Path:
    summary_path = output_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    return summary_path


def evaluate_checkpoint(
    config: dict[str, Any],
    weights_path: Path,
    split: str,
    output_dir: Path,
    include_benchmark: bool = False,
) -> dict[str, Any]:
    """评测指定冻结权重，并将全部统一评测产物写入输出目录。"""
    weights_path = _resolve_weights_path(weights_path)
    checkpoint_sha256 = _sha256(weights_path)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name, class_names, prediction = _load_checkpoint_prediction(config, weights_path, split)
    metrics, previews = _evaluate_and_render(config, prediction, output_dir)
    summary = _build_evaluation_summary(
        config,
        weights_path,
        checkpoint_sha256,
        model_name,
        split,
        output_dir,
        class_names,
        prediction,
        metrics,
        previews,
    )
    if include_benchmark:
        _append_benchmark_summary(summary, config, weights_path, model_name, split, output_dir, metrics)
    _write_evaluation_summary(output_dir, summary)
    return summary
