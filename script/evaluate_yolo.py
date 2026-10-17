#!/home/long753/miniconda3/envs/cv_task/bin/python
"""Evaluate a frozen YOLO checkpoint on the configured validation or test split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Importing the trainer first applies its local Ultralytics cache/tracker policy.
from script.train_yolo import load_config, predict_split, render_prediction_previews
from src.detection_evaluation import evaluate_predictions
from src.yolo_benchmark import benchmark_yolo
from ultralytics import YOLO


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the frozen-checkpoint evaluation protocol arguments."""
    parser = argparse.ArgumentParser(description="Evaluate frozen YOLO weights with COCO metrics")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "train_yolov8s_formal.yaml",
        help="Formal experiment YAML configuration",
    )
    parser.add_argument("--weights", type=Path, required=True, help="Frozen checkpoint to evaluate")
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
        help="Configured split to evaluate (default: val)",
    )
    parser.add_argument("--output", type=Path, required=True, help="Directory for evaluation artifacts")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Also benchmark the checkpoint on the fixed representative validation sample",
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _class_names(model: YOLO) -> dict[int, str]:
    """Validate the frozen checkpoint's sole class before exporting its identity."""
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
    """Resolve renderer paths, which are intentionally relative to output_dir.parent."""
    return [str((output_dir.parent / preview).resolve()) for preview in previews]


def _write_summary_csv(
    output_dir: Path,
    weights_path: Path,
    split: str,
    metrics: dict[str, Any],
    benchmark: dict[str, Any],
) -> Path:
    """Write only measured evaluation and benchmark values, never placeholder cells."""
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
    """Reference raw timing samples in benchmark.json without duplicating them in the summary."""
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    weights_path = args.weights.resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"Evaluation weights do not exist: {weights_path}")
    checkpoint_sha256 = _sha256(weights_path)
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    class_names = _class_names(model)
    prediction = predict_split(model, config, split=args.split)
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

    summary: dict[str, Any] = {
        "weights": {
            "path": str(weights_path),
            "sha256": checkpoint_sha256,
        },
        "split": args.split,
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

    if args.benchmark:
        benchmark = benchmark_yolo(weights_path, config, output_dir)
        summary_csv = _write_summary_csv(output_dir, weights_path, args.split, metrics, benchmark)
        summary["artifacts"]["benchmark"] = str((output_dir / "benchmark.json").resolve())
        summary["artifacts"]["summary_csv"] = str(summary_csv.resolve())
        summary["benchmark"] = _benchmark_summary(benchmark)

    summary_path = output_dir / "evaluation_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
