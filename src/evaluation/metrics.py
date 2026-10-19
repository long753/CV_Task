"""COCO detection metrics and fixed-threshold engineering metrics."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

matplotlib.use("Agg")
from matplotlib import pyplot as plt


_COCO_IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def _is_integer(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _require_integer(value: Any, field: str) -> int:
    if not _is_integer(value):
        raise ValueError(f"{field} must be an integer")
    return int(value)


def _require_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _require_box(value: Any, field: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{field} must be a four-number xywh box")
    box = [_require_number(component, f"{field}[{index}]") for index, component in enumerate(value)]
    if box[2] <= 0.0 or box[3] <= 0.0:
        raise ValueError(f"{field} width and height must be positive")
    return box


def _load_ground_truth(path: Path) -> tuple[dict[str, Any], set[int], set[int], dict[tuple[int, int], list[list[float]]], int]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read COCO ground truth {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError("COCO ground truth must be a JSON object")
    images = raw.get("images")
    categories = raw.get("categories")
    annotations = raw.get("annotations")
    if not isinstance(images, list) or not isinstance(categories, list) or not isinstance(annotations, list):
        raise ValueError("COCO ground truth needs images, categories, and annotations lists")

    image_ids: set[int] = set()
    for index, image in enumerate(images):
        if not isinstance(image, dict):
            raise ValueError(f"ground-truth images[{index}] must be an object")
        image_id = _require_integer(image.get("id"), f"ground-truth images[{index}].id")
        if image_id in image_ids:
            raise ValueError(f"ground-truth images has duplicate id {image_id}")
        image_ids.add(image_id)

    category_ids: set[int] = set()
    for index, category in enumerate(categories):
        if not isinstance(category, dict):
            raise ValueError(f"ground-truth categories[{index}] must be an object")
        category_id = _require_integer(category.get("id"), f"ground-truth categories[{index}].id")
        if category_id in category_ids:
            raise ValueError(f"ground-truth categories has duplicate id {category_id}")
        category_ids.add(category_id)

    boxes_by_image_category: dict[tuple[int, int], list[list[float]]] = defaultdict(list)
    for index, annotation in enumerate(annotations):
        prefix = f"ground-truth annotations[{index}]"
        if not isinstance(annotation, dict):
            raise ValueError(f"{prefix} must be an object")
        image_id = _require_integer(annotation.get("image_id"), f"{prefix}.image_id")
        category_id = _require_integer(annotation.get("category_id"), f"{prefix}.category_id")
        if image_id not in image_ids:
            raise ValueError(f"{prefix}.image_id references unknown image {image_id}")
        if category_id not in category_ids:
            raise ValueError(f"{prefix}.category_id references unknown category {category_id}")
        if not _is_integer(annotation.get("iscrowd")) or int(annotation["iscrowd"]) != 0:
            raise ValueError(f"{prefix}.iscrowd must be 0; crowd annotations are unsupported")
        if "ignore" in annotation and (
            not _is_integer(annotation["ignore"]) or int(annotation["ignore"]) != 0
        ):
            raise ValueError(f"{prefix}.ignore must be 0; ignored annotations are unsupported")
        boxes_by_image_category[(image_id, category_id)].append(
            _require_box(annotation.get("bbox"), f"{prefix}.bbox")
        )

    return raw, image_ids, category_ids, dict(boxes_by_image_category), len(annotations)


def _normalize_predictions(
    predictions: list[dict], image_ids: set[int], category_ids: set[int]
) -> list[dict[str, Any]]:
    if not isinstance(predictions, list):
        raise ValueError("predictions must be a list")

    normalized: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions):
        prefix = f"predictions[{index}]"
        if not isinstance(prediction, dict):
            raise ValueError(f"{prefix} must be an object")
        image_id = _require_integer(prediction.get("image_id"), f"{prefix}.image_id")
        category_id = _require_integer(prediction.get("category_id"), f"{prefix}.category_id")
        if image_id not in image_ids:
            raise ValueError(f"{prefix}.image_id references unknown image {image_id}")
        if category_id not in category_ids:
            raise ValueError(f"{prefix}.category_id references unknown category {category_id}")
        score = _require_number(prediction.get("score"), f"{prefix}.score")
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"{prefix}.score must be between 0 and 1")
        normalized.append(
            {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": _require_box(prediction.get("bbox"), f"{prefix}.bbox"),
                "score": score,
            }
        )
    return normalized


def _limit_predictions(predictions: list[dict[str, Any]], max_detections: int) -> list[dict[str, Any]]:
    by_image: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, prediction in enumerate(predictions):
        by_image[prediction["image_id"]].append((index, prediction))

    retained_indices: set[int] = set()
    for image_predictions in by_image.values():
        image_predictions.sort(key=lambda item: (-item[1]["score"], item[0]))
        retained_indices.update(index for index, _ in image_predictions[:max_detections])
    return [prediction for index, prediction in enumerate(predictions) if index in retained_indices]


def _iou_xywh(first: list[float], second: list[float]) -> float:
    first_right = first[0] + first[2]
    first_bottom = first[1] + first[3]
    second_right = second[0] + second[2]
    second_bottom = second[1] + second[3]
    intersection_width = max(0.0, min(first_right, second_right) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first_bottom, second_bottom) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    union = first[2] * first[3] + second[2] * second[3] - intersection
    return intersection / union if union > 0.0 else 0.0


def _fixed_threshold_metrics(
    predictions: list[dict[str, Any]],
    boxes_by_image_category: dict[tuple[int, int], list[list[float]]],
    ground_truth_count: int,
    confidence: float,
    match_iou: float,
) -> tuple[int, int, int, float, float]:
    matched = {
        key: [False] * len(boxes)
        for key, boxes in boxes_by_image_category.items()
    }
    by_image: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, prediction in enumerate(predictions):
        if prediction["score"] >= confidence:
            by_image[prediction["image_id"]].append((index, prediction))

    true_positives = 0
    false_positives = 0
    for image_predictions in by_image.values():
        image_predictions.sort(key=lambda item: (-item[1]["score"], item[0]))
        for _, prediction in image_predictions:
            key = (prediction["image_id"], prediction["category_id"])
            candidates = boxes_by_image_category.get(key, [])
            used = matched.get(key, [])
            best_index: int | None = None
            best_iou = match_iou
            for candidate_index, candidate in enumerate(candidates):
                if used[candidate_index]:
                    continue
                iou = _iou_xywh(prediction["bbox"], candidate)
                if iou >= match_iou and (best_index is None or iou > best_iou):
                    best_index = candidate_index
                    best_iou = iou
            if best_index is None:
                false_positives += 1
            else:
                used[best_index] = True
                true_positives += 1

    false_negatives = ground_truth_count - true_positives
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    return true_positives, false_positives, false_negatives, precision, recall


def _coco_max_detections(max_detections: int) -> list[int]:
    return sorted({1, min(10, max_detections), max_detections})


def _empty_detection_results(coco_ground_truth: COCO) -> COCO:
    coco_detections = COCO()
    coco_detections.dataset = {
        "images": list(coco_ground_truth.dataset.get("images", [])),
        "categories": list(coco_ground_truth.dataset.get("categories", [])),
        "annotations": [],
    }
    coco_detections.createIndex()
    return coco_detections


def _mean_valid(values: np.ndarray) -> float:
    valid = np.asarray(values)[np.asarray(values) >= 0]
    return float(np.mean(valid)) if valid.size else 0.0


def _save_pr_curve(coco_evaluation: COCOeval, output_path: Path) -> None:
    iou_index = int(np.flatnonzero(np.isclose(coco_evaluation.params.iouThrs, 0.5))[0])
    precision = coco_evaluation.eval["precision"][iou_index, :, :, 0, -1]
    recall_by_category = coco_evaluation.eval["recall"][iou_index, :, 0, -1]
    valid = (precision >= 0) & (recall_by_category[np.newaxis, :] >= 0)
    valid_by_recall = np.any(valid, axis=1)
    curve = np.zeros(precision.shape[0], dtype=float)
    for recall_index in np.flatnonzero(valid_by_recall):
        curve[recall_index] = float(np.mean(precision[recall_index, valid[recall_index]]))

    figure, axis = plt.subplots(figsize=(6, 4))
    recall_thresholds = coco_evaluation.params.recThrs
    if np.any(valid_by_recall):
        axis.plot(recall_thresholds[valid_by_recall], curve[valid_by_recall], color="tab:blue", linewidth=2)
    else:
        axis.plot([], [])
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("COCO precision-recall curve (IoU=0.50)")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=144)
    plt.close(figure)


def evaluate_predictions(
    gt_path: Path,
    predictions: list[dict],
    output_dir: Path,
    confidence: float = 0.25,
    match_iou: float = 0.5,
    max_detections: int = 100,
) -> dict[str, Any]:
    """Evaluate COCO-format detections and write stable prediction artifacts.

    ``predictions`` use original-image pixel ``xywh`` boxes.  Score-floor filtering
    belongs to the inference caller; this function only applies its per-image
    ``max_detections`` cap and its independent fixed-confidence engineering metric.
    """
    confidence = _require_number(confidence, "confidence")
    match_iou = _require_number(match_iou, "match_iou")
    max_detections = _require_integer(max_detections, "max_detections")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if not 0.0 < match_iou <= 1.0:
        raise ValueError("match_iou must be greater than 0 and no greater than 1")
    if max_detections < 1:
        raise ValueError("max_detections must be at least 1")

    ground_truth_path = Path(gt_path)
    (
        _,
        image_ids,
        category_ids,
        boxes_by_image_category,
        ground_truth_count,
    ) = _load_ground_truth(ground_truth_path)
    normalized_predictions = _normalize_predictions(predictions, image_ids, category_ids)
    limited_predictions = _limit_predictions(normalized_predictions, max_detections)

    true_positives, false_positives, false_negatives, precision, recall = _fixed_threshold_metrics(
        limited_predictions,
        boxes_by_image_category,
        ground_truth_count,
        confidence,
        match_iou,
    )

    coco_ground_truth = COCO(str(ground_truth_path))
    coco_detections = (
        coco_ground_truth.loadRes([dict(prediction) for prediction in limited_predictions])
        if limited_predictions
        else _empty_detection_results(coco_ground_truth)
    )
    coco_evaluation = COCOeval(coco_ground_truth, coco_detections, "bbox")
    coco_evaluation.params.imgIds = sorted(image_ids)
    coco_evaluation.params.catIds = sorted(category_ids)
    coco_evaluation.params.iouThrs = _COCO_IOU_THRESHOLDS.copy()
    coco_max_detections = _coco_max_detections(max_detections)
    coco_evaluation.params.maxDets = coco_max_detections
    coco_evaluation.evaluate()
    coco_evaluation.accumulate()

    coco_precision = coco_evaluation.eval["precision"]
    map50 = _mean_valid(coco_precision[0, :, :, 0, -1])
    map50_95 = _mean_valid(coco_precision[:, :, :, 0, -1])
    protocol = {
        "metric_range": "0_to_1",
        "bbox_format": "xywh_pixels",
        "ground_truth_path": str(ground_truth_path),
        "ground_truth_annotation_policy": "iscrowd_0_and_ignore_0_only",
        "coco_iou_thresholds": [float(value) for value in _COCO_IOU_THRESHOLDS],
        "engineering_match_iou": match_iou,
        "confidence_threshold": confidence,
        "max_detections_per_image": max_detections,
        "coco_max_detections": coco_max_detections,
    }
    metrics: dict[str, Any] = {
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map50_95": map50_95,
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "protocol": protocol,
    }

    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "predictions_coco.json").write_text(
        json.dumps(limited_predictions, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (artifact_dir / "metrics.json").write_text(
        json.dumps(metrics, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _save_pr_curve(coco_evaluation, artifact_dir / "pr_curve.png")
    return metrics
