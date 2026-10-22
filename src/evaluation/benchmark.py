"""统一的冻结检测器端到端性能测量。"""

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

from src.evaluation.torchvision import detect_bgr_torchvision
from src.model.torchvision import MODEL_NAMES as TORCHVISION_MODEL_NAMES
from src.model.torchvision import load_model as load_torchvision_model
from src.model.torchvision import configure_determinism
from src.model.ultralytics import load_ultralytics_model


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


def _benchmark_device(config: dict[str, Any]) -> torch.device:
    configured = _required_mapping(config, "training").get("device")
    if isinstance(configured, int) and not isinstance(configured, bool):
        device = torch.device(f"cuda:{configured}")
    else:
        device = torch.device(str(configured))
    if device.type != "cuda":
        raise ValueError("Benchmark requires training.device to select a CUDA device")
    if not torch.cuda.is_available() or device.index is None or device.index >= torch.cuda.device_count():
        raise RuntimeError(f"Benchmark requires available CUDA device {device}")
    return device


def _benchmark_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Validate fixed input semantics while allowing short smoke timing runs."""
    benchmark = _required_mapping(config, "benchmark")
    imgsz = _positive_integer(benchmark.get("imgsz"), "benchmark.imgsz")
    warmup = _positive_integer(benchmark.get("warmup"), "benchmark.warmup")
    iterations = _positive_integer(benchmark.get("iterations"), "benchmark.iterations")
    image_count = _positive_integer(benchmark.get("image_count"), "benchmark.image_count")
    if imgsz != 640:
        raise ValueError("Benchmark protocol requires benchmark.imgsz=640")
    if image_count != 6:
        raise ValueError("Benchmark protocol requires benchmark.image_count=6")
    if benchmark.get("precision") != "fp32":
        raise ValueError("Benchmark protocol requires benchmark.precision='fp32'")

    evaluation = _required_mapping(config, "evaluation")
    confidence = _finite_number(evaluation.get("confidence"), "evaluation.confidence")
    score_floor = _finite_number(evaluation.get("score_floor"), "evaluation.score_floor")
    nms_iou = _finite_number(evaluation.get("nms_iou"), "evaluation.nms_iou")
    max_detections = _positive_integer(evaluation.get("max_detections"), "evaluation.max_detections")
    if not 0.0 <= confidence <= 1.0 or not 0.0 <= score_floor <= 1.0:
        raise ValueError("evaluation confidence thresholds must be between 0 and 1")
    if not 0.0 < nms_iou <= 1.0:
        raise ValueError("evaluation.nms_iou must be greater than 0 and no greater than 1")

    return {
        "imgsz": imgsz,
        "warmup": warmup,
        "iterations": iterations,
        "image_count": image_count,
        "confidence": confidence,
        "score_floor": score_floor,
        "nms_iou": nms_iou,
        "max_detections": max_detections,
        "device": _benchmark_device(config),
    }


def _representative_images(config: dict[str, Any], image_count: int) -> list[dict[str, Any]]:
    """Choose the same deterministic round-robin validation sample as previews."""
    annotations_path = Path(config["data"]).parent / "annotations" / "instances_val.json"
    with annotations_path.open(encoding="utf-8") as handle:
        annotations = json.load(handle)
    images = annotations.get("images")
    if not isinstance(images, list):
        raise ValueError(f"Invalid COCO validation images in {annotations_path}")

    by_source: dict[str, list[dict[str, Any]]] = {}
    for image in images:
        if not isinstance(image, dict):
            raise ValueError(f"Invalid COCO image entry in {annotations_path}")
        file_name, image_id = image.get("file_name"), image.get("id")
        if not isinstance(file_name, str) or not file_name or isinstance(image_id, bool) or not isinstance(image_id, int):
            raise ValueError(f"Invalid COCO image entry in {annotations_path}")
        by_source.setdefault(Path(file_name).stem.split("_", 1)[0], []).append(image)
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
        raise ValueError(f"Requested {image_count} representative validation images, found only {len(selected)}")
    return selected


def _load_bgr_images(
    config: dict[str, Any], representative_images: list[dict[str, Any]]
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Decode selected originals before timing so disk access cannot affect latency."""
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


def _profile_yolo_model(model: Any, imgsz: int) -> dict[str, Any]:
    """Keep historical YOLO THOP accounting unchanged for benchmark regression."""
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
    if parameter_count <= 0 or not math.isfinite(gflops) or gflops <= 0.0:
        raise RuntimeError("THOP did not produce a valid positive operation count")
    return {
        "parameters_unfused": parameter_count,
        "fused": False,
        "parameters_million": parameter_count / 1_000_000.0,
        "gflops": gflops,
        "input_shape": [1, 3, imgsz, imgsz],
        "counter": "THOP MACs multiplied by 2 and divided by 1e9",
        "limitations": (
            "Approximate operation count: THOP uses module hooks and its operator coverage "
            "and MAC-to-FLOP convention may differ from other profilers."
        ),
        "thop_version": importlib.metadata.version("ultralytics-thop"),
    }


def _profile_profiler_model(model: torch.nn.Module, imgsz: int, model_name: str) -> dict[str, Any]:
    """Report only FLOPs that torch.profiler exposes for TorchVision and RT-DETR."""
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    profile_model = copy.deepcopy(model).cpu().float().eval()
    input_tensor = torch.zeros((1, 3, imgsz, imgsz), dtype=torch.float32)
    is_torchvision = model_name in TORCHVISION_MODEL_NAMES
    with torch.inference_mode(), torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True
    ) as profiler:
        if is_torchvision:
            profile_model([input_tensor[0]])
        else:
            profile_model(input_tensor)

    operator_flops: dict[str, int] = {}
    zero_flop_operators: list[str] = []
    for event in profiler.key_averages():
        flops = int(event.flops)
        key = str(event.key)
        if flops > 0:
            operator_flops[key] = flops
        elif event.count:
            zero_flop_operators.append(key)
    total_flops = sum(operator_flops.values())
    if parameter_count <= 0 or total_flops <= 0:
        raise RuntimeError(
            "torch.profiler did not report positive FLOPs; refusing to present an unsupported zero estimate"
        )
    return {
        "parameters_unfused": parameter_count,
        "fused": None,
        "parameters_million": parameter_count / 1_000_000.0,
        "gflops": total_flops / 1_000_000_000.0,
        "input_shape": [1, 3, imgsz, imgsz],
        "profile_scope": (
            "TorchVision eval forward including its model-side proposal and postprocess operators"
            if is_torchvision
            else "raw RT-DETR network forward; wrapper-level decode and CPU result materialization are not FLOP-counted"
        ),
        "counter": "torch.profiler CPU activities with with_flops=True; sum of reported operator FLOPs",
        "reported_operator_flops": dict(sorted(operator_flops.items())),
        "zero_flop_operator_events": sorted(set(zero_flop_operators)),
        "limitations": (
            "Illustrative partial count only. Operators reported with zero FLOPs are listed rather "
            "than assumed free; custom proposal, NMS, ROI, attention, and other kernels may be "
            "unreported or use a different FLOP convention."
        ),
    }


def _detect_bgr_ultralytics(model: Any, bgr: np.ndarray, settings: dict[str, Any], model_name: str) -> np.ndarray:
    """Run one Ultralytics detector through its full BGR-to-CPU result path."""
    arguments: dict[str, Any] = {
        "source": bgr,
        "stream": False,
        "batch": 1,
        "augment": False,
        "imgsz": settings["imgsz"],
        "rect": False,
        "device": settings["device"].index,
        "half": False,
        "conf": settings["confidence"],
        "max_det": settings["max_detections"],
        "verbose": False,
        "save": False,
        "save_txt": False,
        "save_conf": False,
        "save_crop": False,
        "show": False,
        "visualize": False,
    }
    if model_name != "rtdetr_l":
        arguments["iou"] = settings["nms_iou"]
    results = model.predict(**arguments)
    if len(results) != 1:
        raise RuntimeError(f"Expected one prediction result for one BGR image, got {len(results)}")
    boxes = results[0].boxes
    if boxes is None:
        return np.empty((0, 6), dtype=np.float32)
    detections = boxes.data.detach().cpu().numpy()
    valid = (detections[:, 2] > detections[:, 0]) & (detections[:, 3] > detections[:, 1])
    detections = detections[valid]
    if detections.size == 0:
        return detections.reshape((0, 6))
    if model_name == "rtdetr_l":
        order = np.argsort(-detections[:, 4], kind="stable")
        return detections[order[: settings["max_detections"]]]
    return detections


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


def _nvidia_smi_snapshot(device: torch.device) -> dict[str, Any]:
    """Capture one contemporaneous GPU-state query without inventing unavailable values."""
    command = [
        "nvidia-smi",
        f"--id={device.index}",
        "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.free,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, check=False, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": f"nvidia-smi unavailable: {exc}"}
    if completed.returncode != 0:
        return {"available": False, "reason": f"nvidia-smi exited {completed.returncode}: {completed.stderr.strip()}"}
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


def _load_benchmark_model(
    weights_path: Path, config: dict[str, Any], model_name: str
) -> tuple[Any, str, bool]:
    if model_name in TORCHVISION_MODEL_NAMES:
        configure_determinism(False)
        timing_config = {**config, "evaluation": {**config["evaluation"], "score_floor": config["evaluation"]["confidence"]}}
        model, checkpoint_name = load_torchvision_model(weights_path, timing_config)
        if checkpoint_name != model_name:
            raise ValueError(
                f"TorchVision checkpoint model_name {checkpoint_name!r} does not match requested {model_name!r}"
            )
        return model, checkpoint_name, True
    selected_name = model_name if model_name in {"yolov8s", "yolov11s", "rtdetr_l"} else None
    return load_ultralytics_model(weights_path, selected_name), model_name, False


def benchmark_checkpoint(
    weights: Path,
    config: dict[str, Any],
    output_dir: Path,
    *,
    model_name: str,
) -> dict[str, Any]:
    """Benchmark one frozen supported detector under a common end-to-end CUDA protocol."""
    weights_path = Path(weights).resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"Benchmark weights do not exist: {weights_path}")
    settings = _benchmark_settings(config)
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    representative_images = _representative_images(config, settings["image_count"])
    bgr_images, recorded_images = _load_bgr_images(config, representative_images)
    model, resolved_model_name, is_torchvision = _load_benchmark_model(weights_path, config, model_name)
    profile_result = (
        _profile_profiler_model(model, settings["imgsz"], resolved_model_name)
        if is_torchvision
        else _profile_yolo_model(model, settings["imgsz"])
        if resolved_model_name != "rtdetr_l"
        else _profile_profiler_model(model.model, settings["imgsz"], resolved_model_name)
    )
    if is_torchvision:
        model = model.to(settings["device"]).float().eval()


    def detect(bgr: np.ndarray) -> np.ndarray:
        if is_torchvision:
            return detect_bgr_torchvision(
                model,
                resolved_model_name,
                bgr,
                settings["device"],
                score_floor=settings["confidence"],
                max_detections=settings["max_detections"],
            )
        return _detect_bgr_ultralytics(model, bgr, settings, resolved_model_name)

    with torch.inference_mode():
        for index in range(settings["warmup"]):
            detect(bgr_images[index % len(bgr_images)])
        torch.cuda.synchronize(settings["device"])
        gpu_snapshot = _nvidia_smi_snapshot(settings["device"])
        tf32_flags = {
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "handling": "Observed after warmup; this benchmark does not modify TF32 flags.",
        }
        latencies_ms: list[float] = []
        for index in range(settings["iterations"]):
            bgr = bgr_images[index % len(bgr_images)]
            torch.cuda.synchronize(settings["device"])
            started = time.perf_counter()
            detect(bgr)
            torch.cuda.synchronize(settings["device"])
            latencies_ms.append((time.perf_counter() - started) * 1_000.0)

    result = {
        "weights": str(weights_path),
        "architecture": resolved_model_name,
        "device": str(settings["device"]),
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
            "precision_note": "Inputs and weights are FP32; observed TF32 controls are recorded separately and were not changed.",
            "tf32_flags": tf32_flags,
            "imgsz": [settings["imgsz"], settings["imgsz"]],
            "score_floor": settings["score_floor"],
            "inference_score_threshold": settings["confidence"],
            "confidence": settings["confidence"],
            "nms_iou": None if resolved_model_name == "rtdetr_l" else settings["nms_iou"],
            "max_detections": settings["max_detections"],
            "rtdetr_postprocess": "native query decoding without an additional NMS; sorted top-K cap applied" if resolved_model_name == "rtdetr_l" else None,
            "timing_scope": "end-to-end preprocess, host-to-device transfer, network, decode, postprocess, rescale, device-to-host transfer, and empty-box filtering to CPU detections",
            "excluded_from_timing": "checkpoint loading, image file I/O, warmup, and plots",
            "cuda_synchronization": "before and after every measured image",
            "sample_schedule": "representative validation images in listed round-robin order",
        },
        "representative_images": recorded_images,
        "warmup_iterations": settings["warmup"],
        "timed_iterations": settings["iterations"],
        "measurement_conditions": {
            "label": "single-process sequential inference; see nvidia-smi snapshot",
            "nvidia_smi_snapshot_timing": "after warmup and before timed iterations",
            "nvidia_smi": gpu_snapshot,
        },
        "profile": profile_result,
        "latency": {**_latency_summary(latencies_ms), "samples_ms": latencies_ms},
    }
    (output_path / "benchmark.json").write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )
    return result
