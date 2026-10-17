"""Repeatable end-to-end CUDA benchmark for a frozen YOLO checkpoint."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from thop import profile
from ultralytics import YOLO


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration requires a {key} mapping")
    return value


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _benchmark_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the fixed profiling protocol rather than silently changing it."""
    benchmark = _required_mapping(config, "benchmark")
    imgsz = _positive_integer(benchmark.get("imgsz"), "benchmark.imgsz")
    warmup = _positive_integer(benchmark.get("warmup"), "benchmark.warmup")
    iterations = _positive_integer(benchmark.get("iterations"), "benchmark.iterations")
    image_count = _positive_integer(benchmark.get("image_count"), "benchmark.image_count")
    precision = benchmark.get("precision")

    if imgsz != 640:
        raise ValueError("Benchmark protocol requires benchmark.imgsz=640")
    if image_count != 6:
        raise ValueError("Benchmark protocol requires benchmark.image_count=6")
    if precision != "fp32":
        raise ValueError("Benchmark protocol requires benchmark.precision='fp32'")

    evaluation = _required_mapping(config, "evaluation")
    confidence = _finite_number(evaluation.get("confidence"), "evaluation.confidence")
    nms_iou = _finite_number(evaluation.get("nms_iou"), "evaluation.nms_iou")
    max_detections = _positive_integer(
        evaluation.get("max_detections"), "evaluation.max_detections"
    )
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("evaluation.confidence must be between 0 and 1")
    if not 0.0 < nms_iou <= 1.0:
        raise ValueError("evaluation.nms_iou must be greater than 0 and no greater than 1")

    training = _required_mapping(config, "training")
    device_index = training.get("device")
    if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
        raise ValueError("Benchmark requires training.device to be a CUDA device index")
    if not torch.cuda.is_available() or device_index >= torch.cuda.device_count():
        raise RuntimeError(f"Benchmark requires available CUDA device {device_index}")

    return {
        "imgsz": imgsz,
        "warmup": warmup,
        "iterations": iterations,
        "image_count": image_count,
        "precision": precision,
        "confidence": confidence,
        "nms_iou": nms_iou,
        "max_detections": max_detections,
        "device_index": device_index,
    }


def _representative_images(config: dict[str, Any], image_count: int) -> list[dict[str, Any]]:
    """Choose the renderer's deterministic round-robin validation sample."""
    data_path = Path(config["data"])
    annotations_path = data_path.parent / "annotations" / "instances_val.json"
    with annotations_path.open(encoding="utf-8") as handle:
        annotations = json.load(handle)
    images = annotations.get("images")
    if not isinstance(images, list):
        raise ValueError(f"Invalid COCO validation images in {annotations_path}")

    by_source: dict[str, list[dict[str, Any]]] = {}
    for image in images:
        if not isinstance(image, dict):
            raise ValueError(f"Invalid COCO image entry in {annotations_path}")
        file_name = image.get("file_name")
        image_id = image.get("id")
        if not isinstance(file_name, str) or not file_name:
            raise ValueError(f"Invalid COCO image file_name in {annotations_path}")
        if isinstance(image_id, bool) or not isinstance(image_id, int):
            raise ValueError(f"Invalid COCO image id in {annotations_path}")
        source = Path(file_name).stem.split("_", 1)[0]
        by_source.setdefault(source, []).append(image)

    for source_images in by_source.values():
        source_images.sort(key=lambda image: (str(image["file_name"]), int(image["id"])))

    selected: list[dict[str, Any]] = []
    positions = {source: 0 for source in by_source}
    while len(selected) < image_count:
        added = False
        for source in sorted(by_source):
            position = positions[source]
            if position < len(by_source[source]) and len(selected) < image_count:
                selected.append(by_source[source][position])
                positions[source] += 1
                added = True
        if not added:
            break
    if len(selected) != image_count:
        raise ValueError(
            f"Requested {image_count} representative validation images, found only {len(selected)}"
        )
    return selected


def _load_bgr_images(
    config: dict[str, Any], representative_images: list[dict[str, Any]]
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Load selected originals before timing so disk access cannot affect latency."""
    image_root = Path(config["data"]).parent / "images"
    bgr_images: list[np.ndarray] = []
    recorded_images: list[dict[str, Any]] = []
    for image in representative_images:
        file_name = Path(str(image["file_name"]))
        if file_name.is_absolute() or not file_name.parts or file_name.parts[0] != "val" or ".." in file_name.parts:
            raise ValueError(f"Representative image is not under val/: {image['file_name']}")
        image_path = image_root / file_name
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Cannot read representative validation image: {image_path}")
        bgr_images.append(bgr)
        recorded_images.append(
            {
                "id": int(image["id"]),
                "source": file_name.stem.split("_", 1)[0],
                "file_name": str(file_name),
                "path": str(image_path.resolve()),
                "width": int(bgr.shape[1]),
                "height": int(bgr.shape[0]),
            }
        )
    return bgr_images, recorded_images


def _profile_unfused_model(model: YOLO, imgsz: int) -> tuple[int, float]:
    """Count unfused checkpoint parameters and estimate FLOPs on a disposable CPU copy."""
    unfused_model = model.model
    if unfused_model.is_fused():
        raise RuntimeError("Benchmark requires an unfused PyTorch checkpoint for parameter profiling")
    parameter_count = sum(parameter.numel() for parameter in unfused_model.parameters())
    profile_model = copy.deepcopy(unfused_model).cpu().float().eval()
    with torch.inference_mode():
        macs, _ = profile(
            profile_model,
            inputs=(torch.zeros((1, 3, imgsz, imgsz), dtype=torch.float32),),
            verbose=False,
        )
    gflops = float(macs) * 2.0 / 1_000_000_000.0
    if parameter_count <= 0 or not math.isfinite(gflops) or gflops <= 0:
        raise RuntimeError("Profiler did not produce a valid positive operation count")
    return parameter_count, gflops


def _detect_bgr(model: YOLO, bgr: np.ndarray, settings: dict[str, Any]) -> np.ndarray:
    """Run the complete Ultralytics detector and materialize final CPU detections."""
    results = model.predict(
        source=bgr,
        stream=False,
        batch=1,
        augment=False,
        imgsz=settings["imgsz"],
        rect=False,
        device=settings["device_index"],
        half=False,
        conf=settings["confidence"],
        iou=settings["nms_iou"],
        max_det=settings["max_detections"],
        verbose=False,
        save=False,
        save_txt=False,
        save_conf=False,
        save_crop=False,
        show=False,
        visualize=False,
    )
    if len(results) != 1:
        raise RuntimeError(f"Expected one prediction result for one BGR image, got {len(results)}")
    boxes = results[0].boxes
    if boxes is None:
        raise RuntimeError("Detection benchmark received a result without detection boxes")
    detections = boxes.data.cpu().numpy()
    valid = (detections[:, 2] > detections[:, 0]) & (detections[:, 3] > detections[:, 1])
    return detections[valid]


def _latency_summary(latencies_ms: list[float]) -> dict[str, float]:
    samples = np.asarray(latencies_ms, dtype=np.float64)
    if samples.size == 0:
        raise ValueError("Benchmark collected no latency samples")
    total_seconds = float(samples.sum() / 1_000.0)
    return {
        "mean_ms": float(samples.mean()),
        "median_ms": float(np.median(samples)),
        "p95_ms": float(np.percentile(samples, 95)),
        "std_ms": float(samples.std(ddof=0)),
        "sequential_fps": float(samples.size / total_seconds),
    }

def _nvidia_smi_snapshot(device_index: int) -> dict[str, Any]:
    """Capture one contemporaneous GPU-state query without inventing unavailable values."""
    command = [
        "nvidia-smi",
        f"--id={device_index}",
        "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.free,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, check=False, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": f"nvidia-smi unavailable: {exc}"}
    if completed.returncode != 0:
        return {
            "available": False,
            "reason": f"nvidia-smi exited {completed.returncode}: {completed.stderr.strip()}",
        }
    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        return {"available": False, "reason": f"unexpected nvidia-smi output: {completed.stdout!r}"}
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 7:
        return {"available": False, "reason": f"unexpected nvidia-smi row: {rows[0]!r}"}
    try:
        return {
            "available": True,
            "gpu_index": int(fields[0]),
            "gpu_name": fields[1],
            "temperature_c": int(fields[2]),
            "utilization_percent": int(fields[3]),
            "memory_free_mib": int(fields[4]),
            "memory_total_mib": int(fields[5]),
            "driver_version": fields[6],
        }
    except ValueError:
        return {"available": False, "reason": f"unparseable nvidia-smi row: {rows[0]!r}"}


def benchmark_yolo(weights: Path, config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Benchmark a fresh frozen checkpoint under the formal, end-to-end protocol.

    Timing begins with an already-loaded original BGR image and ends after Ultralytics
    has produced a CPU NumPy detection array.  It therefore includes preprocessing,
    host-to-device transfer, model execution, decoding, NMS, rescaling, and device-to-
    host transfer, but excludes checkpoint loading, image decoding, warmup, and plots.
    """
    weights_path = Path(weights).resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"Benchmark weights do not exist: {weights_path}")
    settings = _benchmark_settings(config)
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    representative_images = _representative_images(config, settings["image_count"])
    bgr_images, recorded_images = _load_bgr_images(config, representative_images)

    # This independent instance cannot fuse or otherwise alter the evaluator's model.
    model = YOLO(str(weights_path))
    parameter_count, gflops = _profile_unfused_model(model, settings["imgsz"])
    cuda_device = torch.device(f"cuda:{settings['device_index']}")

    with torch.inference_mode():
        for index in range(settings["warmup"]):
            _detect_bgr(model, bgr_images[index % len(bgr_images)], settings)
        torch.cuda.synchronize(cuda_device)
        gpu_snapshot = _nvidia_smi_snapshot(settings["device_index"])
        tf32_flags = {
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "handling": "Observed after warmup; this benchmark does not modify TF32 flags.",
        }

        latencies_ms: list[float] = []
        for index in range(settings["iterations"]):
            bgr = bgr_images[index % len(bgr_images)]
            torch.cuda.synchronize(cuda_device)
            started = time.perf_counter()
            _detect_bgr(model, bgr, settings)
            torch.cuda.synchronize(cuda_device)
            latencies_ms.append((time.perf_counter() - started) * 1_000.0)

    result = {
        "weights": str(weights_path),
        "device": str(cuda_device),
        "runtime": {
            "python": sys.version.split()[0],
            "interpreter": sys.executable,
            "torch": torch.__version__,
            "torchvision": importlib.metadata.version("torchvision"),
            "ultralytics": importlib.metadata.version("ultralytics"),
            "cuda_build": torch.version.cuda,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "protocol": {
            "input": "in-memory original BGR image",
            "batch_size": 1,
            "precision": "fp32",
            "precision_note": (
                "Inputs and weights are FP32; observed TF32 controls are recorded separately "
                "and were not changed."
            ),
            "tf32_flags": tf32_flags,
            "imgsz": [settings["imgsz"], settings["imgsz"]],
            "confidence": settings["confidence"],
            "nms_iou": settings["nms_iou"],
            "max_detections": settings["max_detections"],
            "timing_scope": (
                "end-to-end preprocess, host-to-device transfer, network, decode, NMS, "
                "rescale, device-to-host transfer, and empty-box filtering to CPU detections"
            ),
            "excluded_from_timing": "checkpoint loading, image file I/O, warmup, and plots",
            "cuda_synchronization": "before and after every measured image",
            "sample_schedule": "representative validation images in listed round-robin order",
        },
        "representative_images": recorded_images,
        "warmup_iterations": settings["warmup"],
        "timed_iterations": settings["iterations"],
        "measurement_conditions": {
            "label": "local loaded-desktop conditions; not isolated-server throughput",
            "nvidia_smi_snapshot_timing": "after warmup and before timed iterations",
            "nvidia_smi": gpu_snapshot,
        },
        "profile": {
            "parameters_unfused": parameter_count,
            "fused": False,
            "parameters_million": parameter_count / 1_000_000.0,
            "gflops": gflops,
            "input_shape": [1, 3, settings["imgsz"], settings["imgsz"]],
            "counter": "THOP MACs multiplied by 2 and divided by 1e9",
            "limitations": (
                "Approximate operation count: THOP uses module hooks and its operator coverage "
                "and MAC-to-FLOP convention may differ from other profilers."
            ),
            "thop_version": importlib.metadata.version("ultralytics-thop"),
        },
        "latency": {
            **_latency_summary(latencies_ms),
            "samples_ms": latencies_ms,
        },
    }
    (output_path / "benchmark.json").write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result
