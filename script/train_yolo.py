#!/home/long753/miniconda3/envs/cv_task/bin/python
"""Train and validate the configured YOLOv8s crack-detection experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_CONFIG_DIR = PROJECT_ROOT / "outputs" / ".ultralytics"
MATPLOTLIB_CONFIG_DIR = PROJECT_ROOT / "outputs" / ".matplotlib"
ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(ULTRALYTICS_CONFIG_DIR)
os.environ["MPLCONFIGDIR"] = str(MATPLOTLIB_CONFIG_DIR)
os.environ["YOLO_AUTOINSTALL"] = "false"

sys.path.insert(0, str(PROJECT_ROOT))

import ultralytics
from ultralytics import YOLO, settings
from ultralytics.utils.files import increment_path

from src.detection_evaluation import evaluate_predictions


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_yolov8s.yaml"
_ALLOWED_SPLITS = {"val", "test"}
_SELECTION_CRITERION = {
    "name": "Ultralytics validation fitness",
    "metric": "val_mAP50-95",
    "framework_metric_key": "metrics/mAP50-95(B)",
    "description": "Pinned Ultralytics fitness weights select validation mAP50-95.",
}


def configure_ultralytics() -> None:
    """Keep Ultralytics state, weights, and optional integrations local."""
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


def _positive_int(value: str) -> int:
    epochs = int(value)
    if epochs <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return epochs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse training inputs and reject overrides that would break native resume."""
    parser = argparse.ArgumentParser(description="Run a YOLOv8s crack-detection experiment")
    parser.add_argument("--config", type=Path, help="Training YAML configuration")
    parser.add_argument("--epochs", type=_positive_int, help="Positive epoch override")
    parser.add_argument("--name", help="Ultralytics run-name override")
    parser.add_argument("--resume", type=Path, help="Native, unstripped Ultralytics checkpoint")
    args = parser.parse_args(argv)
    if args.resume is not None and any((args.config is not None, args.epochs is not None, args.name is not None)):
        parser.error("--resume uses its saved configuration; do not combine it with --config, --epochs, or --name")
    return args


def _validate_config(config: dict[str, Any]) -> None:
    required = ("run_type", "model", "data", "project", "name", "training", "evaluation")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Training configuration is missing {missing}")
    if config["run_type"] not in {"pilot", "formal"}:
        raise ValueError("run_type must be either 'pilot' or 'formal'")
    if not isinstance(config["training"], dict) or not isinstance(config["evaluation"], dict):
        raise ValueError("training and evaluation must be mappings")
    for key in ("model", "data", "project"):
        if not isinstance(config[key], (str, Path)) or not str(config[key]):
            raise ValueError(f"{key} must be a nonempty path string")
    if not isinstance(config["name"], str) or not config["name"]:
        raise ValueError("name must be a nonempty string")


def load_config(path: Path) -> dict[str, Any]:
    """Load source YAML, resolve root paths, and preserve all experiment fields."""
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Training configuration must be a mapping")
    _validate_config(config)
    for key in ("model", "data", "project"):
        config[key] = (config_path.parent / Path(config[key])).resolve()
    config["config_path"] = config_path
    return config


def _plain_value(value: Any) -> Any:
    """Convert runtime-only scalar types to stable YAML/JSON values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    if isinstance(value, torch.Tensor):
        return _plain_value(value.detach().cpu().item() if value.ndim == 0 else value.detach().cpu().tolist())
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return _plain_value(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json_dump(path: Path, value: dict[str, Any]) -> None:
    """Replace JSON state atomically so interruption leaves the prior state usable."""
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_plain_value(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def create_training_view(config: dict[str, Any]) -> Path:
    """Copy only train/val inputs into a writable, isolated Ultralytics view."""
    source_root = Path(config["data"]).parent
    view_parent = PROJECT_ROOT / "outputs"
    view_parent.mkdir(parents=True, exist_ok=True)
    view_root = Path(tempfile.mkdtemp(prefix="yolov8s-data-", dir=view_parent))

    for split in ("train", "val"):
        for directory in ("images", "labels"):
            source = source_root / directory / split
            if not source.is_dir():
                raise FileNotFoundError(f"Missing prepared {directory}/{split}: {source}")
            shutil.copytree(source, view_root / directory / split, copy_function=shutil.copy2)

    view_data = view_root / "data.yaml"
    with view_data.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {"train": "images/train", "val": "images/val", "names": {0: "crack"}},
            handle,
            sort_keys=False,
        )
    return view_data


def _cuda_runtime(device: Any) -> dict[str, Any]:
    if type(device) is not int or device != 0:
        raise ValueError("This experiment requires training.device to be CUDA device 0")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA device 0 is required; CPU fallback is disabled")
    return {
        "device": "cuda:0",
        "cuda_version": str(torch.version.cuda),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "interpreter": sys.executable,
        "python_version": sys.version.split()[0],
        "amp": {"trainer_amp": None, "scaler_enabled": None},
    }


def _completed_epochs(save_dir: Path) -> int:
    """Ultralytics records completed epochs as one-based values in results.csv."""
    with (save_dir / "results.csv").open(newline="", encoding="utf-8") as handle:
        epochs = [int(float(row["epoch"])) for row in csv.DictReader(handle)]
    if not epochs:
        raise RuntimeError(f"No completed epochs recorded in {save_dir / 'results.csv'}")
    return max(epochs)


def _actual_amp_state(trainer: Any) -> dict[str, bool | None]:
    """Record the framework's final AMP state after its startup checks."""
    trainer_amp = getattr(trainer, "amp", None)
    scaler = getattr(trainer, "scaler", None)
    scaler_enabled = scaler.is_enabled() if hasattr(scaler, "is_enabled") else None
    return {
        "trainer_amp": bool(trainer_amp) if trainer_amp is not None else None,
        "scaler_enabled": bool(scaler_enabled) if scaler_enabled is not None else None,
    }


def _save_resolved_config(config: dict[str, Any], view_data: Path, save_dir: Path) -> Path:
    """Save complete resolved provenance, including the immutable run data view."""
    resolved_config = save_dir / "resolved_config.yaml"
    snapshot = {key: _plain_value(value) for key, value in config.items() if key != "config_path"}
    snapshot["source_config"] = str(config["config_path"])
    snapshot["training_data_view"] = str(view_data)
    with resolved_config.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(snapshot, handle, sort_keys=False)
    return resolved_config


def _load_resolved_config(checkpoint: Path) -> tuple[dict[str, Any], Path, Path]:
    """Load only the saved configuration and data view belonging to a native run."""
    checkpoint = Path(checkpoint).resolve()
    if checkpoint.parent.name != "weights":
        raise ValueError(f"Resume checkpoint must be inside a run weights directory: {checkpoint}")
    run_dir = checkpoint.parent.parent
    resolved_config = run_dir / "resolved_config.yaml"
    source_config = run_dir / "source_config.yaml"
    if not resolved_config.is_file() or not source_config.is_file():
        raise FileNotFoundError(
            "Resume requires the run's source_config.yaml and resolved_config.yaml alongside the checkpoint"
        )
    with resolved_config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Resolved training configuration must be a mapping: {resolved_config}")
    _validate_config(config)
    for key in ("model", "data", "project"):
        config[key] = Path(config[key]).resolve()
    training_view = Path(config.get("training_data_view", "")).resolve()
    if not training_view.is_file():
        raise FileNotFoundError(f"Saved training data view is unavailable for resume: {training_view}")
    config["config_path"] = source_config
    config["training_data_view"] = training_view
    return config, run_dir, resolved_config


def _checkpoint_payload(checkpoint: Path) -> dict[str, Any]:
    """Read native checkpoint metadata before giving it to Ultralytics resume."""
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint}")
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(f"Cannot read native Ultralytics checkpoint: {checkpoint}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Resume checkpoint is not a native Ultralytics checkpoint: {checkpoint}")
    return payload


def _validate_resume_checkpoint(checkpoint: Path, training_view: Path) -> dict[str, Any]:
    """Reject complete or stripped checkpoints before native resume can alter a run."""
    payload = _checkpoint_payload(checkpoint)
    train_args = payload.get("train_args")
    if not isinstance(train_args, dict):
        raise ValueError(f"Resume checkpoint has no native train_args: {checkpoint}")
    try:
        completed_epochs = int(payload["epoch"]) + 1
        requested_epochs = int(train_args["epochs"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Resume checkpoint has invalid epoch metadata: {checkpoint}") from error
    if completed_epochs >= requested_epochs:
        raise ValueError(
            f"Resume checkpoint is already complete ({completed_epochs}/{requested_epochs} epochs): {checkpoint}"
        )
    if payload.get("optimizer") is None or payload.get("scaler") is None:
        raise ValueError(
            f"Resume checkpoint is stripped and cannot restore optimizer/scaler state: {checkpoint}"
        )
    recorded_epochs = _completed_epochs(Path(checkpoint).resolve().parent.parent)
    if completed_epochs != recorded_epochs:
        raise ValueError(
            "Resume would rewind or skip the existing training log; use the latest "
            f"consistent checkpoint ({completed_epochs} checkpoint epochs, {recorded_epochs} logged epochs)"
        )
    checkpoint_data = Path(str(train_args.get("data", ""))).resolve()
    if checkpoint_data != training_view:
        raise ValueError(
            "Resume checkpoint data view does not match the run's resolved_config.yaml: "
            f"{checkpoint_data} != {training_view}"
        )
    return payload


def _empty_best_checkpoint_selection() -> dict[str, Any]:
    """Describe native best.pt replacement before any validation checkpoint exists."""
    return {
        "criterion": dict(_SELECTION_CRITERION),
        "selected_epoch": None,
        "selected_validation_mAP50_95": None,
        "selected_framework_fitness": None,
        "tie_behavior": (
            "Ultralytics overwrites best.pt when epoch fitness equals best_fitness; "
            "the latest tied epoch is selected."
        ),
    }


def _selected_best_checkpoint(
    metrics: dict[str, Any], fitness: Any, epoch: int, previous: dict[str, Any],
) -> dict[str, Any]:
    """Record the exact native best.pt event without reconstructing from rounded CSV metrics."""
    selection = dict(previous)
    selection["selected_epoch"] = epoch
    selection["selected_validation_mAP50_95"] = _plain_value(
        metrics.get(_SELECTION_CRITERION["framework_metric_key"])
    )
    selection["selected_framework_fitness"] = _plain_value(fitness)
    return selection


def _state_snapshot(
    *,
    config: dict[str, Any],
    save_dir: Path,
    training_data: Path,
    source_config: Path,
    resolved_config: Path,
    runtime: dict[str, Any],
    completed_epochs: int,
    requested_epochs: int,
    last_path: Path | None,
    best_path: Path | None,
    validation_metrics: dict[str, Any],
    best_checkpoint_selection: dict[str, Any],
    status: str,
    resumable: bool,
) -> dict[str, Any]:
    """Build the durable runtime record written before and during training."""
    return {
        "schema_version": 1,
        "status": status,
        "resumable": resumable,
        "run_type": config["run_type"],
        "run_dir": str(save_dir),
        "requested_epochs": requested_epochs,
        "completed_epochs": completed_epochs,
        "last_checkpoint": str(last_path) if last_path is not None and last_path.is_file() else None,
        "best_checkpoint": str(best_path) if best_path is not None and best_path.is_file() else None,
        "source_config": str(source_config),
        "resolved_config": str(resolved_config),
        "training_data_view": str(training_data),
        "runtime": runtime,
        "validation_metrics": _plain_value(validation_metrics),
        "validation_metrics_source": (
            "native_final_best_checkpoint_revalidation"
            if status == "completed" else "native_latest_epoch_validation"
        ),
        "best_checkpoint_selection": _plain_value(best_checkpoint_selection),
    }


def _write_training_state(path: Path, **snapshot: Any) -> None:
    _atomic_json_dump(path, _state_snapshot(**snapshot))

def _saved_best_checkpoint_selection(state_path: Path) -> dict[str, Any]:
    """Recover prior exact best-checkpoint evidence before a native resume."""
    if not state_path.is_file():
        raise FileNotFoundError(f"Resume requires persisted training state: {state_path}")
    try:
        with state_path.open(encoding="utf-8") as handle:
            state = json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(f"Cannot preserve malformed training state during resume: {state_path}") from error
    selection = state.get("best_checkpoint_selection") if isinstance(state, dict) else None
    if not isinstance(selection, dict):
        raise ValueError(f"Training state lacks best checkpoint evidence: {state_path}")
    return selection


def _progress_metrics(metrics: dict[str, Any]) -> str:
    metric_names = (
        ("P", "metrics/precision(B)"),
        ("R", "metrics/recall(B)"),
        ("mAP50", "metrics/mAP50(B)"),
        ("mAP50-95", "metrics/mAP50-95(B)"),
    )
    parts = []
    for label, key in metric_names:
        value = _plain_value(metrics.get(key))
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            parts.append(f"{label}={float(value):.4f}")
    return " ".join(parts) if parts else "metrics=unavailable"


def _register_training_callbacks(
    model: YOLO,
    *,
    config: dict[str, Any],
    save_dir: Path,
    training_data: Path,
    source_config: Path,
    resolved_config: Path,
    runtime: dict[str, Any],
    state_path: Path,
    best_checkpoint_selection: dict[str, Any],
) -> dict[str, Any]:
    """Persist each native checkpoint and emit sparse, flushed epoch progress."""
    epoch_started: dict[str, float] = {}
    callback_state = {"best_checkpoint_selection": best_checkpoint_selection}

    def on_train_epoch_start(trainer: Any) -> None:
        epoch_started["at"] = time.monotonic()

    def on_model_save(trainer: Any) -> None:
        runtime["amp"] = _actual_amp_state(trainer)
        metrics = dict(getattr(trainer, "metrics", {}) or {})
        completed_epochs = int(trainer.epoch) + 1
        requested_epochs = int(trainer.epochs)
        fitness = getattr(trainer, "fitness", None)
        if getattr(trainer, "best_fitness", None) == fitness:
            callback_state["best_checkpoint_selection"] = _selected_best_checkpoint(
                metrics,
                fitness,
                completed_epochs,
                callback_state["best_checkpoint_selection"],
            )
        _write_training_state(
            state_path,
            config=config,
            save_dir=save_dir,
            training_data=training_data,
            source_config=source_config,
            resolved_config=resolved_config,
            runtime=runtime,
            completed_epochs=completed_epochs,
            requested_epochs=requested_epochs,
            last_path=Path(trainer.last),
            best_path=Path(trainer.best),
            validation_metrics=metrics,
            best_checkpoint_selection=callback_state["best_checkpoint_selection"],
            status="running",
            resumable=True,
        )
        if completed_epochs == 1 or completed_epochs % 10 == 0 or completed_epochs == requested_epochs:
            elapsed_seconds = time.monotonic() - epoch_started.get("at", time.monotonic())
            checkpoint = Path(trainer.last).resolve()
            print(
                f"[train] epoch={completed_epochs}/{requested_epochs} elapsed={elapsed_seconds:.1f}s "
                f"{_progress_metrics(metrics)} checkpoint={checkpoint}",
                flush=True,
            )

    model.callbacks["on_train_epoch_start"].insert(0, on_train_epoch_start)
    model.callbacks["on_model_save"].insert(0, on_model_save)
    return callback_state


def train_model(config: dict[str, Any], resume: Path | None = None) -> dict[str, Any]:
    """Train or natively resume YOLOv8s and retain runtime evidence during every save."""
    if resume is not None:
        resume = Path(resume).resolve()
        config, save_dir, resolved_config = _load_resolved_config(resume)
        training_data = Path(config["training_data_view"])
        checkpoint = _validate_resume_checkpoint(resume, training_data)
        source_config = save_dir / "source_config.yaml"
        resume_completed = int(checkpoint["epoch"]) + 1
    else:
        save_dir = increment_path(
            Path(config["project"]) / config["name"],
            exist_ok=bool(config["training"].get("exist_ok", False)),
        ).resolve()
        save_dir.mkdir(parents=True, exist_ok=False)
        training_data = create_training_view(config)
        source_config = save_dir / "source_config.yaml"
        shutil.copy2(config["config_path"], source_config)
        resolved_config = _save_resolved_config(config, training_data, save_dir)
        resume_completed = 0

    runtime = _cuda_runtime(config["training"].get("device"))
    state_path = save_dir / "training_state.json"
    best_checkpoint_selection = (
        _saved_best_checkpoint_selection(state_path)
        if resume is not None
        else _empty_best_checkpoint_selection()
    )
    requested_epochs = int(config["training"]["epochs"])
    initial_last = resume if resume is not None else None
    initial_best = save_dir / "weights" / "best.pt"
    _write_training_state(
        state_path,
        config=config,
        save_dir=save_dir,
        training_data=training_data,
        source_config=source_config,
        resolved_config=resolved_config,
        runtime=runtime,
        completed_epochs=resume_completed,
        requested_epochs=requested_epochs,
        last_path=initial_last,
        best_path=initial_best,
        validation_metrics={},
        best_checkpoint_selection=best_checkpoint_selection,
        status="running",
        resumable=resume is not None,
    )

    if resume is None:
        model_path = Path(config["model"])
        if not model_path.is_file():
            raise FileNotFoundError(f"Requested pretrained weights do not exist: {model_path}")
        model = YOLO(str(model_path))
    else:
        model = YOLO(str(resume))

    callback_state = _register_training_callbacks(
        model,
        config=config,
        save_dir=save_dir,
        training_data=training_data,
        source_config=source_config,
        resolved_config=resolved_config,
        runtime=runtime,
        state_path=state_path,
        best_checkpoint_selection=best_checkpoint_selection,
    )
    train_args = dict(config["training"])
    train_args.update(
        {
            "data": str(training_data),
            "project": str(save_dir.parent),
            "name": save_dir.name,
            "exist_ok": True,
        }
    )
    if resume is not None:
        train_args["resume"] = str(resume)
    model.train(**train_args)

    trainer = model.trainer
    runtime["amp"] = _actual_amp_state(trainer)
    actual_save_dir = Path(trainer.save_dir).resolve()
    if actual_save_dir != save_dir:
        raise RuntimeError(f"Native training used an unexpected run directory: {actual_save_dir} != {save_dir}")
    best_path = Path(trainer.best).resolve()
    last_path = Path(trainer.last).resolve()
    if not best_path.is_file() or not last_path.is_file():
        raise RuntimeError(f"Expected best.pt and last.pt under {save_dir / 'weights'}")
    completed_epochs = _completed_epochs(save_dir)
    validation_metrics = dict(getattr(trainer, "metrics", {}) or {})
    _write_training_state(
        state_path,
        config=config,
        save_dir=save_dir,
        training_data=training_data,
        source_config=source_config,
        resolved_config=resolved_config,
        runtime=runtime,
        completed_epochs=completed_epochs,
        requested_epochs=int(trainer.epochs),
        last_path=last_path,
        best_path=best_path,
        validation_metrics=validation_metrics,
        best_checkpoint_selection=callback_state["best_checkpoint_selection"],
        status="completed",
        resumable=False,
    )
    return {
        "run_dir": save_dir,
        "best_path": best_path,
        "last_path": last_path,
        "epochs_completed": completed_epochs,
        "epochs_requested": int(trainer.epochs),
        "training_data": training_data,
        "source_config": source_config,
        "resolved_config": resolved_config,
        "training_state": state_path,
        "runtime": runtime,
        "validation_metrics": _plain_value(validation_metrics),
        "best_checkpoint_selection": callback_state["best_checkpoint_selection"],
    }


def load_split_annotations(config: dict[str, Any], split: str) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    """Load and validate COCO annotations for the chosen held-out split."""
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


def _crack_names(model: YOLO) -> dict[int, str]:
    names = model.names
    if isinstance(names, dict):
        normalized = {int(index): str(name) for index, name in names.items()}
    else:
        normalized = {index: str(name) for index, name in enumerate(names)}
    if normalized != {0: "crack"}:
        raise RuntimeError(f"Reloaded best checkpoint must contain exactly class 0=crack, got {normalized}")
    return normalized


def _coco_detections(boxes: Any, image_id: int) -> tuple[list[dict[str, Any]], int]:
    """Export positive-area clipped boxes; count empty boxes rather than inventing extents."""
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
    """Stream one split directory at batch one and convert original-pixel boxes to COCO."""
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
    # Path lists become a single in-memory batch in this pinned Ultralytics release.
    # Directory input selects the file loader and preserves batch=1 behavior.
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
    """Select a deterministic, round-robin sample across source prefixes."""
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
    """Draw deterministic GT-green/prediction-orange comparison images."""
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


def summarize_experiment(
    config: dict[str, Any],
    training: dict[str, Any],
    validation: dict[str, Any],
    metrics: dict[str, Any],
    previews: list[str],
    class_names: dict[int, str],
) -> dict[str, Any]:
    """Format the generic experiment report separately so main stays focused."""
    return {
        "run_type": config["run_type"],
        "run_dir": str(training["run_dir"]),
        "best_checkpoint": str(training["best_path"]),
        "last_checkpoint": str(training["last_path"]),
        "epochs_completed": training["epochs_completed"],
        "epochs_requested": training["epochs_requested"],
        "dataset_split": "val",
        "configuration": {
            "source_config": str(config["config_path"]),
            "saved_source_config": str(training["source_config"]),
            "resolved_config": str(training["resolved_config"]),
            "prepared_data": str(config["data"]),
            "training_data_view": str(training["training_data"]),
            "training_state": str(training["training_state"]),
        },
        "model": {
            "pretrained_weights": str(config["model"]),
            "best_weights": str(training["best_path"]),
            "class_names": class_names,
            "checkpoint_selection": training["best_checkpoint_selection"],
        },
        "runtime": training["runtime"],
        "training_validation_metrics": training["validation_metrics"],
        "metrics": metrics,
        "validation": {
            "ground_truth": str(validation["ground_truth"]),
            "predictions": "validation/predictions_coco.json",
            "metrics": "validation/metrics.json",
            "previews": previews,
            "discarded_nonpositive_boxes": validation["discarded_nonpositive_boxes"],
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.resume is not None:
        config, _, _ = _load_resolved_config(args.resume)
    else:
        config = load_config(args.config or DEFAULT_CONFIG)
        if args.epochs is not None:
            config["training"]["epochs"] = args.epochs
        if args.name is not None:
            config["name"] = args.name

    training = train_model(config, resume=args.resume)
    best_model = YOLO(str(training["best_path"]))
    class_names = _crack_names(best_model)
    validation = predict_split(best_model, config, split="val")

    validation_dir = Path(training["run_dir"]) / "validation"
    metrics = evaluate_predictions(
        validation["ground_truth"],
        validation["predictions"],
        validation_dir,
        confidence=float(config["evaluation"]["confidence"]),
        match_iou=float(config["evaluation"]["match_iou"]),
        max_detections=int(config["evaluation"]["max_detections"]),
    )
    previews = render_prediction_previews(
        validation["annotations"],
        validation["images"],
        validation["predictions"],
        config,
        validation_dir,
    )

    summary = summarize_experiment(config, training, validation, metrics, previews, class_names)
    summary_path = Path(training["run_dir"]) / "experiment_summary.json"
    _atomic_json_dump(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
