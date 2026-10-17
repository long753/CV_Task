"""Independent crack-mask audit, provenance grouping and detection preprocessing.

Raw pixels and files are never modified. All coordinates use half-open xyxy
boundaries; a connected component, not a semantic crack instance, is one target.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

SPLITS = ("train", "val", "test")
SOURCE_PAIRS = {
    "CFD": (("train", "label"), ("val", "val_label")),
    "crack500": (("train", "label"), ("val", "val_label")),
    "AsphaltCrack300": (("train", "label"),),
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _integer(value, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _codes(values, name: str) -> list[int]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a nonempty list of raw pixel values")
    if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 65535 for v in values):
        raise ValueError(f"{name} contains invalid raw pixel values")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate values")
    return values


def _ratios(ratios: dict) -> dict[str, float]:
    if not isinstance(ratios, dict) or set(ratios) != set(SPLITS):
        raise ValueError("split_ratios must contain exactly train, val, test")
    for value in ratios.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("split ratios must be finite positive numbers")
    if not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-9):
        raise ValueError("split ratios must sum to 1")
    return {split: float(ratios[split]) for split in SPLITS}


def load_config(path: Path, output_override: Path | None = None) -> dict:
    """Resolve configuration paths against the YAML file, never the shell cwd."""
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {
        "schema_version", "dataset_root", "output_dir", "seed", "split_ratios",
        "mask", "previews_per_source", "group_overrides", "quarantine_samples",
        "near_duplicate_hamming_distance",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError(f"Configuration keys must be exactly {sorted(required)}")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("Unsupported configuration schema_version")
    _integer(config["seed"], "seed", 0)
    _integer(config["previews_per_source"], "previews_per_source", 1)
    _integer(config["near_duplicate_hamming_distance"], "near_duplicate_hamming_distance", 0)
    if config["near_duplicate_hamming_distance"] > 64:
        raise ValueError("near_duplicate_hamming_distance cannot exceed 64")
    config["split_ratios"] = _ratios(config["split_ratios"])
    mask = config["mask"]
    mask_keys = {
        "background_values", "foreground_values", "min_component_area",
        "closing_kernel_size", "closing_iterations",
        "palette_foreground_rgb", "palette_background_rgb",
    }
    if not isinstance(mask, dict) or set(mask) != mask_keys:
        raise ValueError(f"mask keys must be exactly {sorted(mask_keys)}")
    _codes(mask["foreground_values"], "foreground_values")
    _codes(mask["background_values"], "background_values")
    if set(mask["foreground_values"]) & set(mask["background_values"]):
        raise ValueError("Foreground and background codes overlap")
    for key in ("palette_foreground_rgb", "palette_background_rgb"):
        color = mask[key]
        if not isinstance(color, list) or len(color) != 3 or any(
            type(value) is not int or not 0 <= value <= 255 for value in color
        ):
            raise ValueError(f"{key} must contain three 8-bit RGB components")
    if mask["palette_foreground_rgb"] == mask["palette_background_rgb"]:
        raise ValueError("Palette foreground and background colors overlap")
    _box_parameters(mask["min_component_area"], mask["closing_kernel_size"], mask["closing_iterations"])
    overrides = config["group_overrides"]
    if not isinstance(overrides, dict):
        raise ValueError("group_overrides must be a mapping")
    for name, override in overrides.items():
        if not isinstance(name, str) or "/" not in name:
            raise ValueError("Override keys must be source/stem")
        source, stem = name.split("/", 1)
        infer_provenance(source, stem, {name: override})
    quarantine = config["quarantine_samples"]
    if not isinstance(quarantine, dict) or any(
        not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", key)
        or not isinstance(reason, str) or not reason.strip()
        for key, reason in quarantine.items()
    ):
        raise ValueError("quarantine_samples must map exact sample IDs to nonempty reasons")
    for key in ("dataset_root", "output_dir"):
        value = config[key]
        if not isinstance(value, str) or not value:
            raise ValueError(f"{key} must be a path string")
        config[key] = str((path.parent / value).resolve())
    if output_override is not None:
        config["output_dir"] = str(Path(output_override).resolve())
    raw = Path(config["dataset_root"])
    output = Path(config["output_dir"])
    if not raw.is_dir():
        raise ValueError(f"Dataset root does not exist: {raw}")
    if raw == output or raw in output.parents or output in raw.parents:
        raise ValueError("Output cannot be inside or contain the raw dataset")
    return config


def _mask_array(path: Path) -> tuple[np.ndarray, str]:
    with Image.open(path) as image:
        mode = image.mode
        array = np.asarray(image)
    if array.ndim == 3:
        if array.shape[2] == 4 and np.any(array[..., 3] != 255):
            raise ValueError(f"Nonopaque mask alpha: {path}")
        if array.shape[2] not in (3, 4) or not np.all(array[..., :3] == array[..., :1]):
            raise ValueError(f"Mask must have scalar codes or equal RGB channels: {path}")
        array = array[..., 0]
    if array.ndim != 2 or not all(array.shape) or array.dtype.kind not in "bui":
        raise ValueError(f"Unsupported mask mode/shape {mode}/{array.shape}: {path}")
    return array, mode


def decode_mask(path: Path, foreground_values: list[int], background_values: list[int]) -> np.ndarray:
    """Decode raw L/1/P codes, without palette conversion or EXIF orientation."""
    _codes(foreground_values, "foreground_values")
    _codes(background_values, "background_values")
    if set(foreground_values) & set(background_values):
        raise ValueError("Foreground and background codes overlap")
    array, _ = _mask_array(Path(path))
    unknown = set(int(v) for v in np.unique(array)) - set(foreground_values) - set(background_values)
    if unknown:
        raise ValueError(f"Unknown raw mask values {sorted(unknown)}: {path}")
    return np.isin(array, foreground_values)


def _configured_mask(path: Path, raw: np.ndarray, mode: str, config: dict) -> tuple[np.ndarray, list[int], dict]:
    """Resolve palette indices by explicit colors, not by numeric index order."""
    if mode != "P":
        return decode_mask(path, config["foreground_values"], config["background_values"]), config["foreground_values"], {}
    with Image.open(path) as image:
        palette = image.getpalette("RGB")
        if not palette or "transparency" in image.info:
            raise ValueError(f"Missing or transparent mask palette: {path}")
    foreground, mapping = [], {}
    for value in np.unique(raw):
        index = int(value)
        color = palette[index * 3:index * 3 + 3]
        mapping[str(index)] = color
        if color == config["palette_foreground_rgb"]:
            foreground.append(index)
        elif color != config["palette_background_rgb"]:
            raise ValueError(f"Unknown mask palette color {index}={color}: {path}")
    return np.isin(raw, foreground), foreground, mapping


def _box_parameters(min_component_area: int, closing_kernel_size: int, closing_iterations: int) -> None:
    _integer(min_component_area, "min_component_area", 1)
    _integer(closing_kernel_size, "closing_kernel_size", 1)
    _integer(closing_iterations, "closing_iterations", 0)
    if closing_kernel_size % 2 != 1:
        raise ValueError("closing_kernel_size must be odd")


def mask_to_boxes(binary: np.ndarray, min_component_area: int = 1,
                  closing_kernel_size: int = 3, closing_iterations: int = 0) -> list[list[int]]:
    """One eight-connected foreground component becomes one half-open box."""
    _box_parameters(min_component_area, closing_kernel_size, closing_iterations)
    binary = np.asarray(binary)
    if binary.ndim != 2 or not all(binary.shape) or not np.all((binary == 0) | (binary == 1)):
        raise ValueError("binary must be a nonempty HxW boolean or 0/1 array")
    pixels = binary.astype(np.uint8)
    if closing_iterations:
        kernel = np.ones((closing_kernel_size, closing_kernel_size), dtype=np.uint8)
        # Keep OpenCV's morphology border behavior explicit and reproducible.
        pixels = cv2.morphologyEx(pixels, cv2.MORPH_CLOSE, kernel, iterations=closing_iterations)
    _, _, stats, _ = cv2.connectedComponentsWithStats(pixels, connectivity=8)
    return sorted(
        [[int(x), int(y), int(x + width), int(y + height)]
         for x, y, width, height, area in stats[1:] if area >= min_component_area],
        key=lambda box: (box[1], box[0], box[3], box[2]),
    )


def infer_provenance(source: str, stem: str, overrides: dict) -> tuple[str, bool]:
    """Use verified source-specific names; never silently invent provenance."""
    if source not in SOURCE_PAIRS:
        raise ValueError(f"Unsupported source: {source}")
    key = f"{source}/{stem}"
    if key in overrides:
        override = overrides[key]
        if not isinstance(override, dict) or set(override) != {"group_id", "is_augmented"}:
            raise ValueError(f"Invalid provenance override: {key}")
        group = override["group_id"]
        if not isinstance(group, str) or not group.startswith(source + "/") or not group.removeprefix(source + "/") or type(override["is_augmented"]) is not bool:
            raise ValueError(f"Invalid provenance override: {key}")
        return group, override["is_augmented"]
    if source == "crack500":
        match = re.fullmatch(r"(.+)_(\d+)_(\d+)", stem)
        if match:
            return f"{source}/{match[1]}", False
    elif source == "AsphaltCrack300":
        match = re.fullmatch(r"(\d{3})_(\d+)", stem)
        if match:
            return f"{source}/{match[1]}", False
    else:
        match = re.fullmatch(r"(\d{3})(?:(?:_{1,2}|add|adjust|sub)(\d+))?", stem)
        if match:
            return f"{source}/{match[1]}", match[2] is not None
    raise ValueError(f"Unknown source naming; explicit provenance override required: {key}")


def _pixel_hash(array: np.ndarray) -> str:
    height, width = array.shape[:2]
    return hashlib.sha256(f"{width}x{height}:".encode("ascii") + array.tobytes()).hexdigest()


def _perceptual_hash(rgb: np.ndarray) -> str:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    thumbnail = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    coefficients = cv2.dct(thumbnail.astype(np.float32))[:8, :8].ravel()
    bits = coefficients > np.median(coefficients[1:])
    bits[0] = False
    return f"{sum(int(bit) << index for index, bit in enumerate(bits)):016x}"


def _allocation(count: int, ratios: dict[str, float]) -> dict[str, int]:
    sizes = {split: int(math.floor(count * ratios[split])) for split in SPLITS}
    remaining = count - sum(sizes.values())
    order = sorted(SPLITS, key=lambda s: (-(count * ratios[s] - sizes[s]), SPLITS.index(s)))
    for split in order[:remaining]:
        sizes[split] += 1
    if count >= len(SPLITS):
        for split in SPLITS:
            if sizes[split] == 0:
                donor = max(SPLITS, key=lambda s: sizes[s])
                sizes[donor] -= 1
                sizes[split] += 1
    return sizes


def assign_splits(records: list[dict], seed: int, ratios: dict[str, float]) -> None:
    """Split provenance groups, merging exact-image bridges before assignment.

    Identical images with conflicting masks are errors, not silently selected
    labels. Augmented heldout members and redundant exact copies stay traceable
    in the manifest, but are not exported as evaluation images.
    """
    ratios = _ratios(ratios)
    _integer(seed, "seed", 0)
    ordered = sorted(records, key=lambda r: r["sample_id"])
    if len({r["sample_id"] for r in ordered}) != len(ordered):
        raise ValueError("Duplicate sample_id")
    for record in ordered:
        record["split"] = None
    eligible = [r for r in ordered if not str(r.get("exclusion_reason", "")).startswith("quarantined:")]
    parent = {r["group_id"]: r["group_id"] for r in ordered}

    def find(group):
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    def unite(left, right):
        left, right = sorted((find(left), find(right)))
        parent[right] = left

    by_hash = defaultdict(list)
    for record in eligible:
        record.setdefault("original_group_id", record["group_id"])
        record["split"] = None
        record["exclusion_reason"] = None
        by_hash[record["image_sha256"]].append(record)
    for same_image in by_hash.values():
        if len({r["mask_sha256"] for r in same_image}) != 1:
            raise ValueError("Identical RGB image has conflicting masks: " + ", ".join(r["sample_id"] for r in same_image))
        for record in same_image[1:]:
            unite(same_image[0]["group_id"], record["group_id"])
    for record in eligible:
        record["group_id"] = find(record["group_id"])
    for same_image in by_hash.values():
        preferred = min(same_image, key=lambda r: (r["is_augmented"], r["sample_id"]))
        for record in same_image:
            if record is not preferred:
                record["exclusion_reason"] = f"exact_duplicate_of:{preferred['sample_id']}"
    groups = defaultdict(list)
    for record in eligible:
        groups[record["group_id"]].append(record)
    strata = defaultdict(list)
    for group, members in sorted(groups.items()):
        retained = [r for r in members if r["exclusion_reason"] is None]
        if not retained:
            raise ValueError(f"Provenance group has no retained images: {group}")
        if not any(not r["is_augmented"] for r in retained):
            raise ValueError(f"Provenance group has no unaugmented reference: {group}")
        stratum = "+".join(sorted({r["source"] for r in members}))
        strata[stratum].append(group)
    for stratum, group_ids in sorted(strata.items()):
        rng_seed = int.from_bytes(hashlib.sha256(f"{seed}:{stratum}".encode()).digest(), "big")
        random.Random(rng_seed).shuffle(group_ids)
        sizes = _allocation(len(group_ids), ratios)
        cursor = 0
        for split in SPLITS:
            for group in group_ids[cursor:cursor + sizes[split]]:
                for record in groups[group]:
                    record["group_split"] = split
                    if record["exclusion_reason"] is not None:
                        continue
                    if split != "train" and record["is_augmented"]:
                        record["exclusion_reason"] = "augmentation_in_heldout_group"
                    else:
                        record["split"] = split
            cursor += sizes[split]
    if ordered and any(not any(r["split"] == split for r in ordered) for split in SPLITS):
        raise ValueError("Insufficient independent groups to populate train, val and test")


def _quantiles(values) -> dict:
    if not values:
        return {}
    values = np.asarray(values, dtype=float)
    return {name: float(value) for name, value in zip(
        ("min", "p25", "median", "p75", "p95", "max"), np.quantile(values, [0, .25, .5, .75, .95, 1])
    )}


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class Preprocessor:
    """Audit and publish one new output directory, never overwrite user data."""

    def __init__(self, config: dict):
        self.config = config
        self.root = Path(config["dataset_root"])
        self.output = Path(config["output_dir"])
        self.report = {
            "schema_version": 1,
            "status": "running",
            "runtime": {"python": sys.version.split()[0], "executable": sys.executable,
                        "opencv": cv2.__version__, "numpy": np.__version__,
                        "pillow": Image.__version__, "pyyaml": yaml.__version__},
            "coordinate_convention": "half-open xyxy; raw pixels without EXIF transpose",
            "instance_definition": "one eight-connected component after configured morphology/filtering",
            "split_policy": "seeded source-stratified original groups; merge exact-image bridges; heldout has no offline augmentations",
            "errors": [], "warnings": [], "sources": {},
        }

    def _error(self, path: Path, message: str) -> None:
        self.report["errors"].append({"path": str(path), "message": message})

    def _scan(self) -> list[dict]:
        records = []
        seen_ids = set()
        used_overrides = set()
        seen_quarantine = set()
        mask_config = self.config["mask"]
        for source, pairs in SOURCE_PAIRS.items():
            stats = {"pairs": 0, "input_partitions": {}, "dimensions": Counter(),
                     "mask_modes": Counter(), "mask_codes": Counter(), "palette_mappings": Counter(), "raw_empty_masks": 0,
                     "augmented": 0, "foreground_pixels": [], "boxes_per_image": [],
                     "bbox_areas": [], "bbox_aspect_ratios": []}
            self.report["sources"][source] = stats
            for image_folder, mask_folder in pairs:
                image_dir = self.root / source / image_folder
                mask_dir = self.root / source / mask_folder
                if not image_dir.is_dir() or not mask_dir.is_dir():
                    self._error(image_dir, f"Required image/mask directories missing: {image_dir}, {mask_dir}")
                    continue
                image_map, mask_map = {}, {}
                for folder, mapping in ((image_dir, image_map), (mask_dir, mask_map)):
                    for path in sorted(folder.iterdir()):
                        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                            continue
                        if path.stem in mapping:
                            self._error(path, f"Ambiguous same-stem files: {mapping[path.stem]}, {path}")
                        else:
                            mapping[path.stem] = path
                for stem in sorted(mask_map.keys() - image_map.keys()):
                    self._error(mask_map[stem], "Orphan mask has no matching image")
                for stem in sorted(image_map.keys() - mask_map.keys()):
                    self._error(image_map[stem], "Image has no matching mask")
                for stem in sorted(image_map.keys() & mask_map.keys()):
                    image_path, mask_path = image_map[stem], mask_map[stem]
                    sample_id = f"{source}_{stem}"
                    if not re.fullmatch(r"[A-Za-z0-9_-]+", sample_id):
                        self._error(image_path, f"Unsafe sample filename: {sample_id}")
                        continue
                    if sample_id in seen_ids:
                        self._error(image_path, f"Duplicate sample_id across input partitions: {sample_id}")
                        continue
                    seen_ids.add(sample_id)
                    try:
                        reason = self.config["quarantine_samples"].get(sample_id)
                        if reason:
                            seen_quarantine.add(sample_id)
                        provenance_status = "name_rule"
                        try:
                            group, augmented = infer_provenance(source, stem, self.config["group_overrides"])
                        except ValueError:
                            if not reason:
                                raise
                            group, augmented = f"{source}/UNRESOLVED:{stem}", False
                            provenance_status = "unresolved_quarantined"
                        if f"{source}/{stem}" in self.config["group_overrides"]:
                            used_overrides.add(f"{source}/{stem}")
                        with Image.open(image_path) as image:
                            width, height = image.size
                            orientation = image.getexif().get(274, 1)
                            rgb = np.asarray(image.convert("RGB"))
                        raw_mask, mode = _mask_array(mask_path)
                        binary, foreground_values, palette_mapping = _configured_mask(mask_path, raw_mask, mode, mask_config)
                        if binary.shape != (height, width):
                            raise ValueError(f"Image/mask dimension mismatch: image {(width, height)}, mask {binary.shape[::-1]}")
                        boxes = mask_to_boxes(binary, mask_config["min_component_area"],
                                              mask_config["closing_kernel_size"], mask_config["closing_iterations"])
                        record = {
                            "sample_id": sample_id, "source": source,
                            "image_path": str(image_path.resolve()), "mask_path": str(mask_path.resolve()),
                            "width": width, "height": height, "group_id": group,
                            "original_group_id": group, "input_partition": image_folder,
                            "split": None, "boxes": boxes,
                            "foreground_values": foreground_values,
                            "palette_mapping": palette_mapping,
                            "image_sha256": _pixel_hash(rgb),
                            "perceptual_hash": _perceptual_hash(rgb),
                            "mask_sha256": _pixel_hash(binary.astype(np.uint8)),
                            "is_augmented": augmented,
                            "exclusion_reason": f"quarantined:{reason}" if reason else None,
                            "provenance_status": provenance_status,
                            "foreground_pixels": int(binary.sum()),
                        }
                        records.append(record)
                        stats["pairs"] += 1
                        stats["input_partitions"][image_folder] = stats["input_partitions"].get(image_folder, 0) + 1
                        stats["dimensions"][f"{width}x{height}"] += 1
                        stats["mask_modes"][mode] += 1
                        if palette_mapping:
                            stats["palette_mappings"][json.dumps(palette_mapping, sort_keys=True)] += 1
                        for value in np.unique(raw_mask):
                            stats["mask_codes"][str(int(value))] += 1
                        foreground = record["foreground_pixels"]
                        stats["raw_empty_masks"] += int(foreground == 0)
                        stats["augmented"] += int(augmented)
                        stats["foreground_pixels"].append(foreground)
                        stats["boxes_per_image"].append(len(boxes))
                        for x1, y1, x2, y2 in boxes:
                            stats["bbox_areas"].append((x2 - x1) * (y2 - y1))
                            stats["bbox_aspect_ratios"].append(max((x2-x1)/(y2-y1), (y2-y1)/(x2-x1)))
                        if foreground == width * height:
                            self.report["warnings"].append({"sample_id": sample_id, "message": "Mask is all foreground; inspect encoding/content"})
                        if orientation != 1:
                            self.report["warnings"].append({"sample_id": sample_id, "message": f"EXIF orientation {orientation} ignored for raw-coordinate consistency"})
                    except (ValueError, OSError, cv2.error) as error:
                        self._error(image_path, str(error))
            for key in ("dimensions", "mask_modes", "mask_codes", "palette_mappings"):
                stats[key] = dict(sorted(stats[key].items()))
            counts = stats["boxes_per_image"]
            stats["images_above_100_boxes"] = sum(v > 100 for v in counts)
            stats["images_above_300_boxes"] = sum(v > 300 for v in counts)
            stats["empty_detection_images"] = sum(v == 0 for v in counts)
            for key in ("foreground_pixels", "boxes_per_image", "bbox_areas", "bbox_aspect_ratios"):
                stats[key] = _quantiles(stats[key])
        for key in sorted(set(self.config["group_overrides"]) - used_overrides):
            self._error(self.root, f"Unused provenance override (possible typo): {key}")
        for key in sorted(set(self.config["quarantine_samples"]) - seen_quarantine):
            self._error(self.root, f"Unused quarantine sample ID (possible typo): {key}")
        self.report["total_pairs"] = len(records)
        self._provenance_report(records)
        return records

    def _provenance_report(self, records: list[dict]) -> None:
        groups = defaultdict(list)
        hashes = defaultdict(list)
        for record in records:
            groups[record["original_group_id"]].append(record)
            hashes[record["image_sha256"]].append(record)
        overlap = {group: sorted({r["input_partition"] for r in members})
                   for group, members in groups.items() if len({r["input_partition"] for r in members}) > 1}
        duplicates = [[r["sample_id"] for r in members] for members in hashes.values() if len(members) > 1]
        conflicts = [[r["sample_id"] for r in members] for members in hashes.values()
                     if len({r["mask_sha256"] for r in members}) > 1]
        self.report["provenance"] = {
            "original_groups": len(groups), "input_partition_overlap": overlap,
            "exact_image_duplicate_clusters": duplicates, "conflicting_mask_clusters": conflicts,
            "near_duplicate_check": "cross-split pHash candidates are reported, never treated as confirmed duplicates",
        }
        if overlap:
            self.report["warnings"].append({"message": f"Existing input partitions share {len(overlap)} original groups; rebuild grouped split"})
        unresolved = [members for members in hashes.values()
                      if len({r["mask_sha256"] for r in members if not r["exclusion_reason"]}) > 1]
        if unresolved:
            self._error(self.root, "Active identical RGB images have conflicting masks; explicitly quarantine or resolve labels before export")

    def _split_report(self, records: list[dict]) -> None:
        self.report["splits"] = {}
        for split in SPLITS:
            active = [r for r in records if r["split"] == split]
            self.report["splits"][split] = {
                "images": len(active), "groups": len({r["group_id"] for r in active}),
                "instances": sum(len(r["boxes"]) for r in active),
                "empty_images": sum(not r["boxes"] for r in active),
                "sources": dict(sorted(Counter(r["source"] for r in active).items())),
            }
        self.report["exclusions"] = dict(sorted(Counter(
            r["exclusion_reason"].split(":", 1)[0] for r in records if r["exclusion_reason"]
        ).items()))
        self.report["warnings"].append({"message": "Group ratios are approximate; offline CFD augmentation is retained only in training, so final image ratios differ from group ratios"})

    def _near_duplicate_report(self, records: list[dict]) -> None:
        active = sorted((r for r in records if r["split"]), key=lambda r: r["sample_id"])
        hashes = [int(r["perceptual_hash"], 16) for r in active]
        limit = self.config["near_duplicate_hamming_distance"]
        candidates = []
        for index, left in enumerate(active):
            for other in range(index + 1, len(active)):
                right = active[other]
                if left["split"] == right["split"]:
                    continue
                distance = (hashes[index] ^ hashes[other]).bit_count()
                if distance <= limit:
                    candidates.append({
                        "left": left["sample_id"], "right": right["sample_id"],
                        "splits": [left["split"], right["split"]], "hamming_distance": distance,
                    })
        self.report["near_duplicates"] = {
            "method": "64-bit pHash (DC bit fixed to 0)", "max_hamming_distance": limit,
            "cross_split_candidates": candidates,
            "interpretation": "review candidates only; similarity is not proof of shared provenance",
        }
        if candidates:
            self.report["warnings"].append({"message": f"{len(candidates)} perceptually similar cross-split pairs require provenance review"})

    def run(self, stage: str = "all") -> dict:
        """Publish a complete result or a failed audit, never a silent partial export."""
        if stage not in ("audit", "all"):
            raise ValueError("stage must be audit or all")
        if self.output.exists():
            raise ValueError(f"Output already exists; choose a new --output directory: {self.output}")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{self.output.name}-", dir=self.output.parent))
        try:
            _write_json(staging / "audit" / "config.json", self.config)
            records = self._scan()
            if not records:
                self._error(self.root, "No valid paired samples found")
            if not self.report["errors"]:
                try:
                    assign_splits(records, self.config["seed"], self.config["split_ratios"])
                except ValueError as error:
                    self._error(self.root, str(error))
            if self.report["errors"]:
                self.report["status"] = "failed"
                _write_json(staging / "audit" / "report.json", self.report)
                staging.rename(self.output)
                raise ValueError(f"Audit failed with {len(self.report['errors'])} error(s); see {self.output / 'audit/report.json'}")
            self._split_report(records)
            self._near_duplicate_report(records)
            if stage == "audit":
                with (staging / "manifest.jsonl").open("w", encoding="utf-8") as handle:
                    for record in sorted(records, key=lambda r: r["sample_id"]):
                        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                self.report["status"] = "audit_complete"
            else:
                from src.dataset_io import export_dataset, render_previews, validate_dataset
                export_dataset(records, staging)
                self.report["validation"] = validate_dataset(staging)
                self.report["previews"] = render_previews(records, staging, self.config["seed"], self.config["previews_per_source"])
                self.report["status"] = "complete"
            _write_json(staging / "audit" / "report.json", self.report)
            _write_json(staging / "preprocess_state.json", {"schema_version": 1, "status": self.report["status"]})
            staging.rename(self.output)
            return self.report
        finally:
            if staging.exists():
                shutil.rmtree(staging)
