"""YOLO 的统一验证与测试流程，COCO 指标计算复用模型无关的评测器。

本模块负责 YOLO 推理、坐标导出、指标、可视化和可选速度测量；不负责训练。
其他架构应将自己的预测转换为 COCO 格式，再复用 evaluation.metrics 中的指标函数。
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ULTRALYTICS_CONFIG_DIR = PROJECT_ROOT / "outputs" / ".ultralytics"
MATPLOTLIB_CONFIG_DIR = PROJECT_ROOT / "outputs" / ".matplotlib"
ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(ULTRALYTICS_CONFIG_DIR)
os.environ["MPLCONFIGDIR"] = str(MATPLOTLIB_CONFIG_DIR)
os.environ["YOLO_AUTOINSTALL"] = "false"

from ultralytics import YOLO, settings

from src.evaluation.benchmark import benchmark_yolo
from src.evaluation.metrics import evaluate_predictions

_ALLOWED_SPLITS = {"val", "test"}


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


def _coco_detections(boxes: Any, image_id: int) -> tuple[list[dict[str, Any]], int]:
    """导出面积为正的裁剪框；统计空框而不虚构边界。"""
    predictions = []
    discarded = 0
    for xyxy, score, class_id in zip(
        boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist(), strict=True
    ):
        if int(class_id) != 0:
            raise RuntimeError(f"Unexpected predicted class {class_id} for image {image_id}")
        x1, y1, x2, y2 = xyxy
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0:
            discarded += 1
            continue
        predictions.append({
            "image_id": image_id,
            "category_id": 1,
            "bbox": [x1, y1, width, height],
            "score": float(score),
        })
    return predictions, discarded


def predict_split(model: YOLO, config: dict[str, Any], split: str = "val") -> dict[str, Any]:
    """以 batch=1 流式处理一个数据划分，并将原像素坐标框转换为 COCO 格式。"""
    gt_path, annotations, images = load_split_annotations(config, split)
    data_root = Path(config["data"]).parent
    image_paths = {
        (data_root / "images" / image["file_name"]).resolve(): image
        for image in images
    }
    if len(image_paths) != len(images):
        raise ValueError(f"COCO {split} file names must be unique")

    evaluation = config["evaluation"]
    predictions: list[dict[str, Any]] = []
    seen_image_ids: set[int] = set()
    discarded_nonpositive_boxes = 0
    # 在固定的 Ultralytics 版本中，路径列表会合并为一个内存批次。
    # 目录输入会选择文件加载器，并保持 batch=1 的行为。
    results = model.predict(
        source=str(data_root / "images" / split),
        stream=True,
        batch=1,
        half=False,
        augment=False,
        imgsz=config["training"]["imgsz"],
        rect=False,
        device=config["training"]["device"],
        conf=evaluation["score_floor"],
        iou=evaluation["nms_iou"],
        max_det=evaluation["max_detections"],
        verbose=False,
        save=False,
    )
    for result in results:
        image = image_paths.get(Path(result.path).resolve())
        if image is None:
            raise RuntimeError(f"Prediction returned an unexpected {split} image path: {result.path}")
        image_id = int(image["id"])
        if image_id in seen_image_ids:
            raise RuntimeError(f"Prediction returned duplicate {split} image {result.path}")
        seen_image_ids.add(image_id)

        image_predictions, discarded = _coco_detections(result.boxes, image_id)
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
    }


def _preview_images(images: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """按来源前缀轮转选择确定性的样本。"""
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for image in images:
        source = Path(image["file_name"]).stem.split("_", 1)[0]
        by_source[source].append(image)
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
    for index, image_data in enumerate(_preview_images(images, int(config["evaluation"]["preview_count"])), start=1):
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


def _class_names(model: YOLO) -> dict[int, str]:
    """在导出类别标识前，验证冻结权重仅包含 crack 类别。"""
    names = model.names
    if isinstance(names, dict):
        normalized = {int(index): str(name) for index, name in names.items()}
    else:
        normalized = {index: str(name) for index, name in enumerate(names)}
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
    """解析渲染器返回的路径；这些路径有意相对于 output_dir.parent。"""
    return [str((output_dir.parent / preview).resolve()) for preview in previews]


def _write_summary_csv(
    output_dir: Path,
    weights_path: Path,
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
    """在摘要中引用 benchmark.json 的原始计时样本，避免重复写入。"""
    return {
        "protocol": benchmark["protocol"],
        "representative_images": benchmark["representative_images"],
        "measurement_conditions": benchmark["measurement_conditions"],
        "profile": benchmark["profile"],
        "latency": {
            key: value
            for key, value in benchmark["latency"].items()
            if key != "samples_ms"
        },
    }


def _resolve_weights_path(weights_path: Path) -> Path:
    """解析并检查要评测的冻结权重文件。"""
    resolved_path = weights_path.resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Evaluation weights do not exist: {resolved_path}")
    return resolved_path


def _load_checkpoint_prediction(
    config: dict[str, Any],
    weights_path: Path,
    split: str,
) -> tuple[dict[int, str], dict[str, Any]]:
    """载入指定权重，并对目标数据划分生成预测结果。"""
    model = YOLO(str(weights_path))
    class_names = _class_names(model)
    prediction = predict_split(model, config, split=split)
    return class_names, prediction


def _evaluate_and_render(
    config: dict[str, Any],
    prediction: dict[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    """计算指标并渲染预测预览图。"""
    metrics = evaluate_predictions(
        prediction["ground_truth"],
        prediction["predictions"],
        output_dir,
        confidence=float(config["evaluation"]["confidence"]),
        match_iou=float(config["evaluation"]["match_iou"]),
        max_detections=int(config["evaluation"]["max_detections"]),
    )
    previews = render_prediction_previews(
        prediction["annotations"],
        prediction["images"],
        prediction["predictions"],
        config,
        output_dir,
    )
    return metrics, previews


def _build_evaluation_summary(
    config: dict[str, Any],
    weights_path: Path,
    checkpoint_sha256: str,
    split: str,
    output_dir: Path,
    class_names: dict[int, str],
    prediction: dict[str, Any],
    metrics: dict[str, Any],
    previews: list[str],
) -> dict[str, Any]:
    """组装基础评测摘要，不执行性能测量或写盘。"""
    return {
        "weights": {
            "path": str(weights_path),
            "sha256": checkpoint_sha256,
        },
        "split": split,
        "classes": class_names,
        "configuration": {
            "source_path": str(config["config_path"]),
            "resolved": _json_value(config),
        },
        "prediction_protocol": {
            "coordinates": "original-image pixel xywh in COCO predictions",
            "score_floor": config["evaluation"]["score_floor"],
            "nms_iou": config["evaluation"]["nms_iou"],
            "max_detections": config["evaluation"]["max_detections"],
            "empty_box_policy": "Discard boxes with non-positive width or height after clipping; retain all positive sizes.",
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
    split: str,
    output_dir: Path,
    metrics: dict[str, Any],
) -> None:
    """测量性能，并将性能产物和摘要附加到评测结果。"""
    benchmark = benchmark_yolo(weights_path, config, output_dir)
    summary_csv = _write_summary_csv(output_dir, weights_path, split, metrics, benchmark)
    summary["artifacts"]["benchmark"] = str((output_dir / "benchmark.json").resolve())
    summary["artifacts"]["summary_csv"] = str(summary_csv.resolve())
    summary["benchmark"] = _benchmark_summary(benchmark)


def _write_evaluation_summary(output_dir: Path, summary: dict[str, Any]) -> Path:
    """将完整评测摘要写入固定的 JSON 产物文件。"""
    summary_path = output_dir / "evaluation_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary_path


def evaluate_checkpoint(
    config: dict[str, Any],
    weights_path: Path,
    split: str,
    output_dir: Path,
    include_benchmark: bool = False,
) -> dict[str, Any]:
    """评测指定冻结权重，并将全部评测产物写入输出目录。"""
    weights_path = _resolve_weights_path(weights_path)
    checkpoint_sha256 = _sha256(weights_path)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names, prediction = _load_checkpoint_prediction(config, weights_path, split)
    metrics, previews = _evaluate_and_render(config, prediction, output_dir)
    summary = _build_evaluation_summary(
        config,
        weights_path,
        checkpoint_sha256,
        split,
        output_dir,
        class_names,
        prediction,
        metrics,
        previews,
    )
    if include_benchmark:
        _append_benchmark_summary(summary, config, weights_path, split, output_dir, metrics)
    _write_evaluation_summary(output_dir, summary)
    return summary
