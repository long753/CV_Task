"""Dataset artifact export, integrity validation, and audit-preview rendering.

The functions in this module deliberately operate on plain record dictionaries so the
preprocessing pipeline does not need a framework-specific dataset object.  Images are
copied byte-for-byte into a single image tree shared by the YOLO and COCO exports.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError


_SPLITS = ("train", "val", "test")
_SOURCES = frozenset(("CFD", "crack500", "AsphaltCrack300"))
_SAFE_SAMPLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SAFE_SUFFIX = re.compile(r"^\.[A-Za-z0-9]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_YOLO_TOLERANCE_PIXELS = 1e-5


class _DatasetError(ValueError):
    """Internal marker used to give public callers a consistent ValueError."""


def _fail(message: str) -> None:
    raise _DatasetError(message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(value: Any, field: str, *, minimum: int | None = None) -> int:
    if not _is_int(value):
        _fail(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        _fail(f"{field} must be at least {minimum}")
    return value


def _require_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{field} must be a finite number")
    return result


def _require_safe_sample_id(value: Any) -> str:
    if not isinstance(value, str) or not _SAFE_SAMPLE_ID.fullmatch(value):
        _fail("sample_id must be a nonempty path-safe identifier")
    if "\x00" in value:
        _fail("sample_id contains a NUL byte")
    return value


def _require_absolute_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        _fail(f"{field} must be a nonempty absolute path string")
    path = Path(value)
    if not path.is_absolute():
        _fail(f"{field} must be absolute")
    return path


def _image_suffix(image_path: Path) -> str:
    suffix = image_path.suffix
    if not _SAFE_SUFFIX.fullmatch(suffix):
        _fail(f"image_path has an unsupported or unsafe suffix: {image_path}")
    return suffix


def _pixel_hash(width: int, height: int, pixels: np.ndarray) -> str:
    """Hash pixels according to the public dataset-record convention."""
    payload = f"{width}x{height}:".encode("ascii") + pixels.tobytes()
    return hashlib.sha256(payload).hexdigest()


def _load_rgb(path: Path) -> tuple[np.ndarray, int, int]:
    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            pixels = np.array(rgb, dtype=np.uint8, copy=True)
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        _fail(f"cannot decode image {path}: {exc}")
    if width <= 0 or height <= 0 or pixels.shape != (height, width, 3):
        _fail(f"invalid decoded RGB image shape for {path}")
    return pixels, width, height


def _decode_binary_mask(path: Path, foreground_values: list[int]) -> np.ndarray:
    """Decode the record's raw mask codes without palette-to-RGB conversion."""
    try:
        with Image.open(path) as image:
            mode = image.mode
            raw = np.array(image, copy=True)
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        _fail(f"cannot decode mask {path}: {exc}")

    if mode in {"1", "L", "P"}:
        if raw.ndim != 2:
            _fail(f"mask {path} is not a single-channel indexed mask")
        codes = raw
    elif mode in {"RGB", "RGBA"}:
        if raw.ndim != 3 or raw.shape[2] != len(mode):
            _fail(f"mask {path} has an invalid {mode} shape")
        if mode == "RGBA" and not np.all(raw[..., 3] == 255):
            _fail(f"mask {path} contains nonopaque alpha")
        if not (np.array_equal(raw[..., 0], raw[..., 1]) and np.array_equal(raw[..., 1], raw[..., 2])):
            _fail(f"mask {path} uses non-grayscale RGB codes")
        codes = raw[..., 0]
    else:
        _fail(f"unsupported mask mode {mode!r} for {path}")

    if codes.ndim != 2 or codes.shape[0] <= 0 or codes.shape[1] <= 0:
        _fail(f"mask {path} has invalid dimensions")
    return np.isin(codes, np.asarray(foreground_values, dtype=codes.dtype))


def _validate_boxes(boxes: Any, width: int, height: int, field: str) -> list[list[int]]:
    if not isinstance(boxes, list):
        _fail(f"{field} must be a list")
    checked: list[list[int]] = []
    for index, box in enumerate(boxes):
        if not isinstance(box, list) or len(box) != 4:
            _fail(f"{field}[{index}] must be a four-integer list")
        x1, y1, x2, y2 = (
            _require_int(value, f"{field}[{index}][{axis}]")
            for axis, value in enumerate(box)
        )
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            _fail(f"{field}[{index}] is outside the {width}x{height} image")
        checked.append([x1, y1, x2, y2])
    return checked


def _validate_record(record: Any, *, verify_source: bool) -> dict[str, Any]:
    """Validate one manifest-compatible record and return export metadata."""
    if not isinstance(record, dict):
        _fail("each record must be a dictionary")

    required = (
        "sample_id",
        "source",
        "image_path",
        "mask_path",
        "width",
        "height",
        "group_id",
        "split",
        "boxes",
        "foreground_values",
        "image_sha256",
        "mask_sha256",
        "is_augmented",
        "exclusion_reason",
    )
    for key in required:
        if key not in record:
            _fail(f"record is missing required field {key!r}")

    sample_id = _require_safe_sample_id(record["sample_id"])
    source = record["source"]
    if source not in _SOURCES:
        _fail(f"record {sample_id} has unsupported source {source!r}")
    image_path = _require_absolute_path(record["image_path"], "image_path")
    mask_path = _require_absolute_path(record["mask_path"], "mask_path")
    suffix = _image_suffix(image_path)
    width = _require_int(record["width"], "width", minimum=1)
    height = _require_int(record["height"], "height", minimum=1)

    group_id = record["group_id"]
    if not isinstance(group_id, str) or not group_id:
        _fail(f"record {sample_id} has an invalid group_id")
    if "original_group_id" in record:
        original_group_id = record["original_group_id"]
        if original_group_id is not None and (not isinstance(original_group_id, str) or not original_group_id):
            _fail(f"record {sample_id} has an invalid original_group_id")

    split = record["split"]
    if split not in (*_SPLITS, None):
        _fail(f"record {sample_id} has an invalid split")
    reason = record["exclusion_reason"]
    if reason is not None and (not isinstance(reason, str) or not reason):
        _fail(f"record {sample_id} has an invalid exclusion_reason")
    if split is None and reason is None:
        _fail(f"excluded record {sample_id} requires an exclusion_reason")
    if split is not None and reason is not None:
        _fail(f"active record {sample_id} must not have an exclusion_reason")

    foreground_values = record["foreground_values"]
    if not isinstance(foreground_values, list):
        _fail(f"record {sample_id} needs a foreground_values list")
    for index, value in enumerate(foreground_values):
        _require_int(value, f"foreground_values[{index}]", minimum=0)
        if value > 255:
            _fail(f"foreground_values[{index}] must be no greater than 255")

    boxes = _validate_boxes(record["boxes"], width, height, f"record {sample_id} boxes")
    for digest_field in ("image_sha256", "mask_sha256"):
        digest = record[digest_field]
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            _fail(f"record {sample_id} has an invalid {digest_field}")
    if not isinstance(record["is_augmented"], bool):
        _fail(f"record {sample_id} is_augmented must be boolean")

    if verify_source:
        image_pixels, actual_width, actual_height = _load_rgb(image_path)
        if (actual_width, actual_height) != (width, height):
            _fail(f"record {sample_id} dimensions do not match decoded source image")
        if _pixel_hash(width, height, image_pixels) != record["image_sha256"]:
            _fail(f"record {sample_id} image_sha256 does not match decoded source image")
        mask = _decode_binary_mask(mask_path, foreground_values)
        if mask.shape != (height, width):
            _fail(f"record {sample_id} mask dimensions do not match image dimensions")
        mask_hash = _pixel_hash(width, height, mask.astype(np.uint8, copy=False))
        if mask_hash != record["mask_sha256"]:
            _fail(f"record {sample_id} mask_sha256 does not match decoded source mask")

    return {
        "record": record,
        "sample_id": sample_id,
        "source": source,
        "image_path": image_path,
        "mask_path": mask_path,
        "suffix": suffix,
        "width": width,
        "height": height,
        "group_id": group_id,
        "split": split,
        "boxes": boxes,
        "foreground_values": foreground_values,
        "image_sha256": record["image_sha256"],
        "mask_sha256": record["mask_sha256"],
        "is_augmented": record["is_augmented"],
    }


def _json_line(record: dict[str, Any]) -> str:
    try:
        return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        _fail(f"record {record.get('sample_id', '<unknown>')!r} is not JSON serializable: {exc}")


def _write_new_text(path: Path, text: str) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    except FileExistsError:
        _fail(f"refusing to overwrite existing artifact {path}")


def _copy_new_file(source: Path, destination: Path) -> None:
    try:
        with source.open("rb") as source_handle, destination.open("xb") as destination_handle:
            shutil.copyfileobj(source_handle, destination_handle, length=1024 * 1024)
    except FileExistsError:
        _fail(f"refusing to overwrite existing artifact {destination}")
    except OSError as exc:
        _fail(f"cannot copy {source} to {destination}: {exc}")


def _format_yolo(value: float) -> str:
    # 17 significant digits round-trip an IEEE-754 float and keep pixel errors
    # well below the validation tolerance even for very large source images.
    return format(value, ".17g")


def _yolo_text(boxes: list[list[int]], width: int, height: int) -> str:
    lines: list[str] = []
    for x1, y1, x2, y2 in boxes:
        box_width = x2 - x1
        box_height = y2 - y1
        values = (
            (x1 + x2) / (2.0 * width),
            (y1 + y2) / (2.0 * height),
            box_width / width,
            box_height / height,
        )
        lines.append("0 " + " ".join(_format_yolo(value) for value in values))
    return "\n".join(lines) + ("\n" if lines else "")


def _coco_document(split: str, infos: list[dict[str, Any]]) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    for image_id, info in enumerate(infos, start=1):
        images.append(
            {
                "id": image_id,
                "file_name": f"{split}/{info['sample_id']}{info['suffix']}",
                "width": info["width"],
                "height": info["height"],
                "license": 1,
            }
        )
        for x1, y1, x2, y2 in info["boxes"]:
            box_width = x2 - x1
            box_height = y2 - y1
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x1, y1, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    return {
        "info": {"description": "Crack detection dataset", "version": "1.0"},
        "licenses": [{"id": 1, "name": "Unknown"}],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "crack"}],
    }


def _prepare_output_directory(output_dir: Path) -> None:
    protected = ("images", "labels", "annotations", "splits", "manifest.jsonl", "data.yaml")
    if output_dir.exists():
        if not output_dir.is_dir() or output_dir.is_symlink():
            _fail(f"output_dir must be a real directory: {output_dir}")
        collisions = [name for name in protected if (output_dir / name).exists() or (output_dir / name).is_symlink()]
        if collisions:
            _fail(f"output directory already contains export artifacts: {', '.join(collisions)}")
    else:
        try:
            output_dir.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            _fail(f"cannot create output directory {output_dir}: {exc}")

    try:
        for root in ("images", "labels"):
            root_path = output_dir / root
            root_path.mkdir(exist_ok=False)
            for split in _SPLITS:
                (root_path / split).mkdir(exist_ok=False)
        (output_dir / "annotations").mkdir(exist_ok=False)
        (output_dir / "splits").mkdir(exist_ok=False)
    except OSError as exc:
        _fail(f"cannot create export directories under {output_dir}: {exc}")

def _assert_active_image_mask_consistency(infos: list[dict[str, Any]]) -> None:
    """Reject active RGB-identical samples whose binary masks disagree.

    Quarantined records have ``split is None`` and intentionally do not take part in
    this check: their full provenance remains in the manifest without contaminating
    exportable split integrity.
    """
    masks_by_image_hash: dict[str, set[str]] = {}
    for info in infos:
        if info["split"] is not None:
            masks_by_image_hash.setdefault(info["image_sha256"], set()).add(info["mask_sha256"])
    conflicting = sorted(digest for digest, masks in masks_by_image_hash.items() if len(masks) > 1)
    if conflicting:
        _fail(f"active records have conflicting masks for decoded-image hashes: {conflicting}")


def export_dataset(records: list[dict[str, Any]], output_dir: Path) -> None:
    """Export source images once, together with matching YOLO and COCO artifacts.

    The output directory may already contain unrelated ``audit/`` material created by
    the caller, but existing export artifacts are always rejected instead of replaced.
    """
    if not isinstance(records, list):
        _fail("records must be a list")
    output_dir = Path(output_dir)

    infos: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            _fail("each record must be a dictionary")
        # Quarantined records remain manifest-only; do not read their raw files.
        info = _validate_record(record, verify_source=record.get("split") is not None)
        sample_id = info["sample_id"]
        if sample_id in sample_ids:
            _fail(f"duplicate sample_id {sample_id!r}")
        sample_ids.add(sample_id)
        _json_line(record)
        infos.append(info)

    infos.sort(key=lambda item: item["sample_id"])
    _assert_active_image_mask_consistency(infos)

    by_split = {
        split: [info for info in infos if info["split"] == split]
        for split in _SPLITS
    }

    _prepare_output_directory(output_dir)
    for split in _SPLITS:
        for info in by_split[split]:
            image_target = output_dir / "images" / split / f"{info['sample_id']}{info['suffix']}"
            label_target = output_dir / "labels" / split / f"{info['sample_id']}.txt"
            _copy_new_file(info["image_path"], image_target)
            _write_new_text(label_target, _yolo_text(info["boxes"], info["width"], info["height"]))

        split_ids = "".join(f"{info['sample_id']}\n" for info in by_split[split])
        _write_new_text(output_dir / "splits" / f"{split}.txt", split_ids)
        document = _coco_document(split, by_split[split])
        _write_new_text(
            output_dir / "annotations" / f"instances_{split}.json",
            json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        )

    manifest = "".join(_json_line(info["record"]) + "\n" for info in infos)
    _write_new_text(output_dir / "manifest.jsonl", manifest)

    # Deliberately omit `path`: Ultralytics treats an omitted path as the directory
    # containing this YAML file, unlike a relative `path: .` in some releases.
    data_yaml = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "crack"},
    }
    _write_new_text(output_dir / "data.yaml", yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True))


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        _fail(f"cannot read manifest {path}: {exc}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line:
            _fail(f"manifest has a blank line at {line_number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            _fail(f"manifest line {line_number} is invalid JSON: {exc}")
        if not isinstance(record, dict):
            _fail(f"manifest line {line_number} is not an object")
        records.append(record)
    return records


def _require_directory(path: Path, name: str) -> None:
    if not path.is_dir() or path.is_symlink():
        _fail(f"{name} must be a real directory: {path}")


def _direct_regular_files(directory: Path, description: str) -> set[str]:
    _require_directory(directory, description)
    names: set[str] = set()
    try:
        entries = list(directory.iterdir())
    except OSError as exc:
        _fail(f"cannot read {description}: {exc}")
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            _fail(f"{description} contains a non-file or symlink: {entry.name}")
        names.add(entry.name)
    return names


def _require_exact_child_directories(directory: Path, expected: set[str], description: str) -> None:
    _require_directory(directory, description)
    try:
        entries = list(directory.iterdir())
    except OSError as exc:
        _fail(f"cannot read {description}: {exc}")
    actual = {entry.name for entry in entries}
    if actual != expected:
        _fail(f"{description} children differ; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    for entry in entries:
        _require_directory(entry, f"{description}/{entry.name}")


def _read_text(path: Path, description: str) -> str:
    if path.is_symlink() or not path.is_file():
        _fail(f"{description} is missing or not a regular file: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        _fail(f"{description} is not UTF-8: {path}: {exc}")
    except OSError as exc:
        _fail(f"cannot read {description} {path}: {exc}")


def _read_json(path: Path, description: str) -> Any:
    content = _read_text(path, description)
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        _fail(f"invalid JSON in {description} {path}: {exc}")


def _assert_expected_files(actual: set[str], expected: set[str], description: str) -> None:
    if actual != expected:
        _fail(f"{description} differ; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")


def _safe_coco_file_name(value: Any, split: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        _fail("COCO image file_name is not a safe relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != 2 or path.parts[0] != split:
        _fail(f"COCO image file_name escapes the {split} image root: {value!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        _fail(f"COCO image file_name is unsafe: {value!r}")
    return value


def _assert_box_multiset(
    expected: list[list[int]], actual: list[tuple[float, float, float, float]], description: str
) -> None:
    if len(expected) != len(actual):
        _fail(f"{description} count differs: expected {len(expected)}, got {len(actual)}")
    # Manifest coordinates are integer pixel boundaries. Recover each boundary
    # within tolerance before multiset comparison: lexicographically sorting
    # raw floats mispairs boxes whose equal boundaries round differently.
    restored = []
    for box in actual:
        rounded = tuple(round(value) for value in box)
        if any(abs(value - integer) > _YOLO_TOLERANCE_PIXELS
               for value, integer in zip(box, rounded, strict=True)):
            _fail(f"{description} differs from integer boundaries beyond {_YOLO_TOLERANCE_PIXELS} pixels")
        restored.append(rounded)
    if Counter(tuple(box) for box in expected) != Counter(restored):
        _fail(f"{description} box multiset differs from manifest")


def _parse_yolo_boxes(path: Path, width: int, height: int) -> list[tuple[float, float, float, float]]:
    content = _read_text(path, "YOLO label")
    if content == "":
        return []
    boxes: list[tuple[float, float, float, float]] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            _fail(f"YOLO label {path} contains a blank line")
        parts = line.split()
        if len(parts) != 5:
            _fail(f"YOLO label {path} line {line_number} must have five fields")
        category = _require_number(_parse_float(parts[0], f"YOLO class at {path}:{line_number}"), "YOLO class")
        if category != 0.0:
            _fail(f"YOLO label {path} line {line_number} must use class 0")
        center_x, center_y, box_width, box_height = (
            _parse_float(value, f"YOLO coordinate at {path}:{line_number}") for value in parts[1:]
        )
        if not (box_width > 0.0 and box_height > 0.0):
            _fail(f"YOLO label {path} line {line_number} has a nonpositive size")
        left = center_x - box_width / 2.0
        top = center_y - box_height / 2.0
        right = center_x + box_width / 2.0
        bottom = center_y + box_height / 2.0
        tolerance = _YOLO_TOLERANCE_PIXELS / min(width, height)
        if left < -tolerance or top < -tolerance or right > 1.0 + tolerance or bottom > 1.0 + tolerance:
            _fail(f"YOLO label {path} line {line_number} is outside normalized image bounds")
        boxes.append((left * width, top * height, right * width, bottom * height))
    return boxes


def _parse_float(value: str, field: str) -> float:
    try:
        parsed = float(value)
    except ValueError:
        _fail(f"{field} is not numeric")
    if not math.isfinite(parsed):
        _fail(f"{field} must be finite")
    return parsed


def _validate_data_yaml(path: Path) -> None:
    content = _read_text(path, "data.yaml")
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        _fail(f"invalid data.yaml: {exc}")
    expected = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "crack"},
    }
    if data != expected:
        _fail("data.yaml must use the movable YAML-root-relative image layout")


def _validate_coco_split(
    path: Path,
    split: str,
    infos: list[dict[str, Any]],
) -> tuple[dict[str, list[tuple[float, float, float, float]]], int]:
    document = _read_json(path, f"COCO {split} annotations")
    if not isinstance(document, dict):
        _fail(f"COCO {split} document must be an object")
    if document.get("categories") != [{"id": 1, "name": "crack"}]:
        _fail(f"COCO {split} categories are invalid")
    images = document.get("images")
    annotations = document.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        _fail(f"COCO {split} images and annotations must be lists")

    expected_by_file = {
        f"{split}/{info['sample_id']}{info['suffix']}": info
        for info in infos
    }
    image_by_id: dict[int, dict[str, Any]] = {}
    image_info_by_id: dict[int, dict[str, Any]] = {}
    seen_file_names: set[str] = set()
    for index, image in enumerate(images):
        if not isinstance(image, dict):
            _fail(f"COCO {split} image {index} is not an object")
        image_id = _require_int(image.get("id"), f"COCO {split} image id", minimum=1)
        if image_id in image_by_id:
            _fail(f"COCO {split} duplicates image id {image_id}")
        file_name = _safe_coco_file_name(image.get("file_name"), split)
        if file_name in seen_file_names:
            _fail(f"COCO {split} duplicates image file_name {file_name!r}")
        seen_file_names.add(file_name)
        info = expected_by_file.get(file_name)
        if info is None:
            _fail(f"COCO {split} references unmanifested image {file_name!r}")
        if _require_int(image.get("width"), "COCO image width", minimum=1) != info["width"]:
            _fail(f"COCO {split} width differs for {file_name}")
        if _require_int(image.get("height"), "COCO image height", minimum=1) != info["height"]:
            _fail(f"COCO {split} height differs for {file_name}")
        image_by_id[image_id] = image
        image_info_by_id[image_id] = info

    _assert_expected_files(seen_file_names, set(expected_by_file), f"COCO {split} image coverage")
    seen_annotation_ids: set[int] = set()
    boxes_by_sample: dict[str, list[tuple[float, float, float, float]]] = {
        info["sample_id"]: [] for info in infos
    }
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            _fail(f"COCO {split} annotation {index} is not an object")
        annotation_id = _require_int(annotation.get("id"), f"COCO {split} annotation id", minimum=1)
        if annotation_id in seen_annotation_ids:
            _fail(f"COCO {split} duplicates annotation id {annotation_id}")
        seen_annotation_ids.add(annotation_id)
        image_id = _require_int(annotation.get("image_id"), "COCO annotation image_id", minimum=1)
        info = image_info_by_id.get(image_id)
        if info is None:
            _fail(f"COCO {split} annotation {annotation_id} references missing image {image_id}")
        if _require_int(annotation.get("category_id"), "COCO annotation category_id", minimum=1) != 1:
            _fail(f"COCO {split} annotation {annotation_id} has an invalid category")
        if _require_int(annotation.get("iscrowd"), "COCO annotation iscrowd", minimum=0) != 0:
            _fail(f"COCO {split} annotation {annotation_id} must have iscrowd=0")
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            _fail(f"COCO {split} annotation {annotation_id} has an invalid bbox")
        x, y, width, height = (
            _require_number(value, f"COCO {split} annotation {annotation_id} bbox") for value in bbox
        )
        if width <= 0.0 or height <= 0.0:
            _fail(f"COCO {split} annotation {annotation_id} has nonpositive dimensions")
        if x < -_YOLO_TOLERANCE_PIXELS or y < -_YOLO_TOLERANCE_PIXELS:
            _fail(f"COCO {split} annotation {annotation_id} starts outside image bounds")
        if x + width > info["width"] + _YOLO_TOLERANCE_PIXELS or y + height > info["height"] + _YOLO_TOLERANCE_PIXELS:
            _fail(f"COCO {split} annotation {annotation_id} exceeds image bounds")
        area = _require_number(annotation.get("area"), f"COCO {split} annotation {annotation_id} area")
        if area < 0.0 or abs(area - width * height) > _YOLO_TOLERANCE_PIXELS:
            _fail(f"COCO {split} annotation {annotation_id} has inconsistent area")
        boxes_by_sample[info["sample_id"]].append((x, y, x + width, y + height))

    for info in infos:
        _assert_box_multiset(info["boxes"], boxes_by_sample[info["sample_id"]], f"COCO {split} {info['sample_id']}")
    return boxes_by_sample, len(annotations)


def validate_dataset(output_dir: Path) -> dict[str, Any]:
    """Reload and semantically validate every exported dataset artifact.

    The validator intentionally uses only the exported images and manifest metadata;
    source image and mask paths may no longer exist after the dataset is moved.
    """
    output_dir = Path(output_dir)
    _require_directory(output_dir, "output_dir")
    manifest_path = output_dir / "manifest.jsonl"
    data_path = output_dir / "data.yaml"
    records = _read_manifest(manifest_path)
    _validate_data_yaml(data_path)

    infos: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    for record in records:
        info = _validate_record(record, verify_source=False)
        if info["sample_id"] in sample_ids:
            _fail(f"manifest duplicates sample_id {info['sample_id']!r}")
        sample_ids.add(info["sample_id"])
        infos.append(info)
    _assert_active_image_mask_consistency(infos)

    by_split = {
        split: sorted((info for info in infos if info["split"] == split), key=lambda item: item["sample_id"])
        for split in _SPLITS
    }

    images_root = output_dir / "images"
    labels_root = output_dir / "labels"
    annotations_root = output_dir / "annotations"
    splits_root = output_dir / "splits"
    _require_exact_child_directories(images_root, set(_SPLITS), "images")
    _require_exact_child_directories(labels_root, set(_SPLITS), "labels")
    _require_directory(annotations_root, "annotations")
    _require_directory(splits_root, "splits")
    _assert_expected_files(
        _direct_regular_files(annotations_root, "annotations"),
        {f"instances_{split}.json" for split in _SPLITS},
        "annotation files",
    )
    _assert_expected_files(
        _direct_regular_files(splits_root, "splits"),
        {f"{split}.txt" for split in _SPLITS},
        "split files",
    )

    source_counts: dict[str, int] = {source: 0 for source in sorted(_SOURCES)}
    group_splits: dict[str, set[str]] = {}
    image_hash_splits: dict[str, set[str]] = {}
    split_counts: dict[str, dict[str, int]] = {}
    total_empty = 0
    total_annotations = 0

    for split in _SPLITS:
        split_infos = by_split[split]
        expected_images = {f"{info['sample_id']}{info['suffix']}" for info in split_infos}
        expected_labels = {f"{info['sample_id']}.txt" for info in split_infos}
        image_dir = images_root / split
        label_dir = labels_root / split
        _assert_expected_files(_direct_regular_files(image_dir, f"images/{split}"), expected_images, f"images/{split}")
        _assert_expected_files(_direct_regular_files(label_dir, f"labels/{split}"), expected_labels, f"labels/{split}")

        split_ids_content = _read_text(splits_root / f"{split}.txt", f"{split} split list")
        split_ids = split_ids_content.splitlines()
        if any(not value for value in split_ids):
            _fail(f"{split} split list contains a blank identifier")
        if len(split_ids) != len(set(split_ids)):
            _fail(f"{split} split list contains duplicate identifiers")
        for sample_id in split_ids:
            _require_safe_sample_id(sample_id)
        _assert_expected_files(set(split_ids), {info["sample_id"] for info in split_infos}, f"{split} split list")

        for info in split_infos:
            image_path = image_dir / f"{info['sample_id']}{info['suffix']}"
            pixels, width, height = _load_rgb(image_path)
            if (width, height) != (info["width"], info["height"]):
                _fail(f"decoded image dimensions differ for {info['sample_id']}")
            decoded_hash = _pixel_hash(width, height, pixels)
            if decoded_hash != info["image_sha256"]:
                _fail(f"decoded image hash differs for {info['sample_id']}")
            image_hash_splits.setdefault(decoded_hash, set()).add(split)
            group_splits.setdefault(info["group_id"], set()).add(split)
            if split in {"val", "test"} and info["is_augmented"]:
                _fail(f"augmented held-out sample {info['sample_id']} is not allowed")
            source_counts[info["source"]] += 1
            if not info["boxes"]:
                total_empty += 1

            yolo_boxes = _parse_yolo_boxes(
                label_dir / f"{info['sample_id']}.txt", info["width"], info["height"]
            )
            _assert_box_multiset(info["boxes"], yolo_boxes, f"YOLO {split} {info['sample_id']}")

        _, annotation_count = _validate_coco_split(
            annotations_root / f"instances_{split}.json", split, split_infos
        )
        expected_annotation_count = sum(len(info["boxes"]) for info in split_infos)
        if annotation_count != expected_annotation_count:
            _fail(f"COCO {split} annotation count differs from manifest")
        total_annotations += annotation_count
        split_counts[split] = {
            "records": len(split_infos),
            "images": len(split_infos),
            "labels": len(split_infos),
            "annotations": annotation_count,
            "empty_records": sum(not info["boxes"] for info in split_infos),
        }

    leaked_groups = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    if leaked_groups:
        _fail(f"group leakage across splits: {leaked_groups}")
    leaked_hashes = sorted(digest for digest, splits in image_hash_splits.items() if len(splits) > 1)
    if leaked_hashes:
        _fail(f"decoded-image hash leakage across splits: {leaked_hashes}")

    active_count = sum(len(items) for items in by_split.values())
    return {
        "records": len(infos),
        "active_records": active_count,
        "excluded_records": len(infos) - active_count,
        "empty_records": total_empty,
        "annotations": total_annotations,
        "splits": split_counts,
        "sources": source_counts,
    }


def _rank_for_preview(seed: int, source: str, sample_id: str, category: str) -> bytes:
    payload = f"{seed}\x00{source}\x00{category}\x00{sample_id}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def _is_elongated(info: dict[str, Any]) -> bool:
    for x1, y1, x2, y2 in info["boxes"]:
        width = x2 - x1
        height = y2 - y1
        if width >= 3 * height or height >= 3 * width:
            return True
    return False


def _touches_edge(info: dict[str, Any]) -> bool:
    for x1, y1, x2, y2 in info["boxes"]:
        if x1 == 0 or y1 == 0 or x2 == info["width"] or y2 == info["height"]:
            return True
    return False


def _choose_preview_infos(infos: list[dict[str, Any]], seed: int, per_source: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    sources = sorted({info["source"] for info in infos})
    for source in sources:
        candidates = [info for info in infos if info["source"] == source]
        chosen: list[dict[str, Any]] = []
        boundary_cases = (
            ("empty", lambda item: not item["boxes"]),
            ("multiple", lambda item: len(item["boxes"]) > 1),
            ("elongated", _is_elongated),
            ("edge", _touches_edge),
        )
        for category, predicate in boundary_cases:
            matches = [item for item in candidates if predicate(item)]
            if matches:
                candidate = min(
                    matches,
                    key=lambda item: _rank_for_preview(seed, source, item["sample_id"], category),
                )
                if candidate not in chosen and len(chosen) < per_source:
                    chosen.append(candidate)
        remaining = sorted(
            (item for item in candidates if item not in chosen),
            key=lambda item: _rank_for_preview(seed, source, item["sample_id"], "sample"),
        )
        chosen.extend(remaining[: per_source - len(chosen)])
        selected.extend(chosen)
    return selected


def _preview_coco_boxes(output_dir: Path, infos: list[dict[str, Any]]) -> dict[str, list[tuple[float, float, float, float]]]:
    wanted = {
        f"{info['split']}/{info['sample_id']}{info['suffix']}": info["sample_id"]
        for info in infos
    }
    result: dict[str, list[tuple[float, float, float, float]]] = {info["sample_id"]: [] for info in infos}
    for split in _SPLITS:
        document = _read_json(output_dir / "annotations" / f"instances_{split}.json", f"COCO {split} annotations")
        if not isinstance(document, dict) or not isinstance(document.get("images"), list) or not isinstance(document.get("annotations"), list):
            _fail(f"COCO {split} annotations are malformed")
        sample_by_image_id: dict[int, str] = {}
        for image in document["images"]:
            if not isinstance(image, dict):
                _fail(f"COCO {split} contains a malformed image")
            file_name = _safe_coco_file_name(image.get("file_name"), split)
            sample_id = wanted.get(file_name)
            if sample_id is not None:
                sample_by_image_id[_require_int(image.get("id"), "COCO image id", minimum=1)] = sample_id
        for annotation in document["annotations"]:
            if not isinstance(annotation, dict):
                _fail(f"COCO {split} contains a malformed annotation")
            sample_id = sample_by_image_id.get(_require_int(annotation.get("image_id"), "COCO annotation image_id", minimum=1))
            if sample_id is None:
                continue
            bbox = annotation.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                _fail(f"COCO {split} contains an invalid bbox")
            x, y, width, height = (_require_number(value, "COCO preview bbox") for value in bbox)
            result[sample_id].append((x, y, x + width, y + height))
    return result


def _draw_box_label(draw: ImageDraw.ImageDraw, box: tuple[float, float, float, float], index: int, color: tuple[int, int, int], font: ImageFont.ImageFont) -> None:
    x1, y1, x2, y2 = box
    draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
    label = str(index + 1)
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    text_width = right - left
    text_height = bottom - top
    label_x = max(0, int(round(x1)))
    label_y = max(0, int(round(y1)) - text_height - 4)
    draw.rectangle((label_x, label_y, label_x + text_width + 4, label_y + text_height + 4), fill=color)
    draw.text((label_x + 2, label_y + 2), label, fill=(0, 0, 0), font=font)


def _make_preview_tile(
    image: Image.Image,
    title: str,
    boxes: list[tuple[float, float, float, float]],
    color: tuple[int, int, int],
) -> Image.Image:
    max_width, max_height = 560, 400
    source = image.convert("RGB")
    scale = min(max_width / source.width, max_height / source.height, 1.0)
    display_size = (max(1, round(source.width * scale)), max(1, round(source.height * scale)))
    if display_size != source.size:
        source = source.resize(display_size, Image.Resampling.LANCZOS)
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(source)
    for index, box in enumerate(boxes):
        scaled = tuple(value * scale for value in box)
        _draw_box_label(draw, scaled, index, color, font)

    header_height = 24
    tile_width = max(source.width, 160)
    tile = Image.new("RGB", (tile_width, source.height + header_height), (245, 245, 245))
    header = ImageDraw.Draw(tile)
    header.text((5, 5), title, fill=(0, 0, 0), font=font)
    tile.paste(source, (0, header_height))
    return tile


def _compose_preview(tiles: list[Image.Image]) -> Image.Image:
    if len(tiles) != 4:
        _fail("preview requires exactly four panels")
    cell_width = max(tile.width for tile in tiles)
    cell_height = max(tile.height for tile in tiles)
    canvas = Image.new("RGB", (cell_width * 2, cell_height * 2), (225, 225, 225))
    for index, tile in enumerate(tiles):
        x = (index % 2) * cell_width
        y = (index // 2) * cell_height
        canvas.paste(tile, (x, y))
    return canvas


def render_previews(records: list[dict[str, Any]], output_dir: Path, seed: int, per_source: int = 6) -> list[str]:
    """Render deterministic audit panels from raw masks and reread exported labels."""
    if not isinstance(records, list):
        _fail("records must be a list")
    if not _is_int(seed):
        _fail("seed must be an integer")
    if not _is_int(per_source) or per_source < 1:
        _fail("per_source must be a positive integer")
    output_dir = Path(output_dir)

    infos: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    for record in records:
        info = _validate_record(record, verify_source=False)
        if info["sample_id"] in sample_ids:
            _fail(f"duplicate sample_id {info['sample_id']!r}")
        sample_ids.add(info["sample_id"])
        if info["split"] is not None:
            infos.append(info)
    selected = _choose_preview_infos(infos, seed, per_source)
    coco_boxes = _preview_coco_boxes(output_dir, selected)

    preview_dir = output_dir / "audit" / "previews"
    try:
        preview_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _fail(f"cannot create preview directory {preview_dir}: {exc}")

    created: list[str] = []
    for info in selected:
        raw_pixels, width, height = _load_rgb(info["image_path"])
        if (width, height) != (info["width"], info["height"]):
            _fail(f"preview source dimensions differ for {info['sample_id']}")
        mask = _decode_binary_mask(info["mask_path"], info["foreground_values"])
        if mask.shape != (height, width):
            _fail(f"preview mask dimensions differ for {info['sample_id']}")

        original = Image.fromarray(raw_pixels)
        overlay_pixels = raw_pixels.copy()
        overlay_pixels[mask] = (
            0.55 * overlay_pixels[mask] + 0.45 * np.array([255, 0, 0])
        ).astype(np.uint8)
        overlay = Image.fromarray(overlay_pixels)
        yolo_boxes = _parse_yolo_boxes(
            output_dir / "labels" / info["split"] / f"{info['sample_id']}.txt", width, height
        )
        manifest_boxes = [tuple(float(value) for value in box) for box in info["boxes"]]
        tiles = [
            _make_preview_tile(original, f"Original: {info['sample_id']}", [], (255, 215, 0)),
            _make_preview_tile(overlay, "Mask overlay + manifest boxes", manifest_boxes, (0, 255, 0)),
            _make_preview_tile(original, "YOLO labels (reread)", yolo_boxes, (255, 160, 0)),
            _make_preview_tile(original, "COCO annotations (reread)", coco_boxes[info["sample_id"]], (0, 180, 255)),
        ]
        preview_path = preview_dir / f"{info['sample_id']}.png"
        try:
            with preview_path.open("xb") as handle:
                _compose_preview(tiles).save(handle, format="PNG")
        except FileExistsError:
            _fail(f"refusing to overwrite existing preview {preview_path}")
        except OSError as exc:
            _fail(f"cannot save preview {preview_path}: {exc}")
        created.append(preview_path.relative_to(output_dir).as_posix())
    return created


__all__ = ["export_dataset", "validate_dataset", "render_previews"]
