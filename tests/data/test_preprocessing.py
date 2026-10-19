import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.dataset_io import export_dataset, validate_dataset
from src.data.preprocessor import (
    Preprocessor,
    assign_splits,
    decode_mask,
    infer_provenance,
    mask_to_boxes,
)


def array_hash(array: np.ndarray, width: int, height: int) -> str:
    return hashlib.sha256(
        f"{width}x{height}:".encode("ascii") + array.tobytes()
    ).hexdigest()


def synthetic_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def split_record(
    sample_id: str,
    group_id: str,
    image_hash: str,
    mask_hash: str,
    *,
    source: str = "CFD",
    is_augmented: bool = False,
) -> dict:
    return {
        "sample_id": sample_id,
        "source": source,
        "image_path": f"/tmp/{sample_id}.png",
        "mask_path": f"/tmp/{sample_id}_mask.png",
        "width": 4,
        "height": 3,
        "group_id": group_id,
        "split": None,
        "boxes": [],
        "foreground_values": [1],
        "image_sha256": synthetic_hash(image_hash),
        "mask_sha256": synthetic_hash(mask_hash),
        "is_augmented": is_augmented,
        "exclusion_reason": None,
    }


class MaskDecodingTests(unittest.TestCase):
    def test_decode_mask_handles_1bit_and_palette_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            one_bit = Image.new("1", (3, 2))
            one_bit.putdata([0, 1, 0, 1, 0, 1])
            one_bit_path = root / "one_bit.png"
            one_bit.save(one_bit_path)

            palette = Image.new("P", (3, 2))
            palette.putpalette([17, 99, 201, 0, 0, 0] + [0] * (256 * 3 - 6))
            palette.putdata([0, 1, 0, 1, 1, 0])
            palette_path = root / "palette.png"
            palette.save(palette_path)

            np.testing.assert_array_equal(
                decode_mask(one_bit_path, foreground_values=[1], background_values=[0]),
                np.array([[False, True, False], [True, False, True]]),
            )
            np.testing.assert_array_equal(
                decode_mask(palette_path, foreground_values=[1], background_values=[0]),
                np.array([[False, True, False], [True, True, False]]),
            )

    def test_decode_mask_rejects_unknown_raw_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unknown.png"
            Image.fromarray(np.array([[0, 1], [2, 0]], dtype=np.uint8), mode="L").save(path)
            with self.assertRaises(ValueError):
                decode_mask(path, foreground_values=[1], background_values=[0])

    def test_pipeline_resolves_foreground_from_palette_colors(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            for source, pairs in {
                "CFD": [("train", "label"), ("val", "val_label")],
                "crack500": [("train", "label"), ("val", "val_label")],
                "AsphaltCrack300": [("train", "label")],
            }.items():
                for image_dir, mask_dir in pairs:
                    (raw / source / image_dir).mkdir(parents=True)
                    (raw / source / mask_dir).mkdir(parents=True)
            for index in range(3):
                stem = f"{index + 1:03d}_01"
                Image.new("RGB", (8, 6), (20 + index, 40, 60)).save(
                    raw / "AsphaltCrack300/train" / f"{stem}.png"
                )
                indices = np.ones((6, 8), dtype=np.uint8)
                indices[1:4, 2:6] = 0
                mask = Image.fromarray(indices).convert("P")
                mask.putpalette([255, 255, 255, 0, 0, 0] + [0] * (256 * 3 - 6))
                mask.save(raw / "AsphaltCrack300/label" / f"{stem}.png")
            config = {
                "dataset_root": str(raw), "output_dir": str(root / "out"),
                "seed": 42, "split_ratios": {"train": .7, "val": .1, "test": .2},
                "group_overrides": {}, "quarantine_samples": {},
                "near_duplicate_hamming_distance": 4, "previews_per_source": 1,
                "mask": {
                    "foreground_values": [1], "background_values": [0],
                    "palette_foreground_rgb": [255, 255, 255],
                    "palette_background_rgb": [0, 0, 0],
                    "min_component_area": 1, "closing_kernel_size": 3, "closing_iterations": 0,
                },
            }
            Preprocessor(config).run(stage="audit")
            records = [json.loads(line) for line in (root / "out/manifest.jsonl").read_text().splitlines()]
            for record in records:
                self.assertEqual(record["boxes"], [[2, 1, 6, 4]])
                self.assertEqual(record["foreground_values"], [0])


class BoxExtractionTests(unittest.TestCase):
    def test_mask_to_boxes_uses_half_open_bounds_and_eight_connectivity(self) -> None:
        mask = np.zeros((5, 6), dtype=bool)
        mask[0, 5] = True
        mask[1, 4] = True
        mask[3, 0] = True
        mask[4, 1] = True
        mask[2, 2] = True

        boxes = mask_to_boxes(mask, min_component_area=2, closing_kernel_size=3, closing_iterations=0)
        self.assertCountEqual(boxes, [[4, 0, 6, 2], [0, 3, 2, 5]])

    def test_mask_to_boxes_rejects_malformed_parameters(self) -> None:
        mask = np.zeros((2, 2), dtype=bool)
        for kwargs in (
            {"min_component_area": 0},
            {"closing_kernel_size": 2},
            {"closing_iterations": -1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                mask_to_boxes(mask, **kwargs)


class ProvenanceTests(unittest.TestCase):
    def test_infer_provenance_handles_dataset_names_and_overrides(self) -> None:
        self.assertEqual(
            infer_provenance("crack500", "IMG_2937_1_485", {}),
            ("crack500/IMG_2937", False),
        )
        self.assertEqual(
            infer_provenance("AsphaltCrack300", "100_27", {}),
            ("AsphaltCrack300/100", False),
        )
        self.assertEqual(
            infer_provenance("CFD", "118", {}),
            ("CFD/118", False),
        )
        self.assertEqual(
            infer_provenance("CFD", "118adjust94", {}),
            ("CFD/118", True),
        )
        with self.assertRaises(ValueError):
            infer_provenance("CFD", "unrecognised_cfd_name", {})
        self.assertEqual(
            infer_provenance(
                "CFD",
                "unrecognised_cfd_name",
                {
                    "CFD/unrecognised_cfd_name": {
                        "group_id": "CFD/manual-original",
                        "is_augmented": True,
                    }
                },
            ),
            ("CFD/manual-original", True),
        )


class SplitAssignmentTests(unittest.TestCase):
    def test_assign_splits_is_order_independent_group_safe_and_uses_all_splits(self) -> None:
        records = []
        for group_index in range(15):
            group_id = f"crack500/group-{group_index:02d}"
            for member_index in range(2):
                sample_id = f"sample-{group_index:02d}-{member_index}"
                records.append(
                    split_record(
                        sample_id,
                        group_id,
                        f"image-{sample_id}",
                        f"mask-{sample_id}",
                        source="crack500",
                    )
                )

        reordered = copy.deepcopy(list(reversed(records)))
        assign_splits(records, seed=2718, ratios={"train": 0.6, "val": 0.2, "test": 0.2})
        assign_splits(reordered, seed=2718, ratios={"train": 0.6, "val": 0.2, "test": 0.2})

        splits_by_id = {record["sample_id"]: record["split"] for record in records}
        self.assertEqual(
            splits_by_id,
            {record["sample_id"]: record["split"] for record in reordered},
        )
        self.assertEqual(set(splits_by_id.values()), {"train", "val", "test"})
        for group_index in range(15):
            group_splits = {
                splits_by_id[f"sample-{group_index:02d}-{member_index}"]
                for member_index in range(2)
            }
            self.assertEqual(len(group_splits), 1)

    def test_assign_splits_unifies_groups_connected_by_duplicate_bridges(self) -> None:
        records = [
            split_record("a-duplicate", "CFD/a", "bridge-a", "mask-a"),
            split_record("a-unique", "CFD/a", "a-unique-image", "a-unique-mask"),
            split_record("b-duplicate", "AsphaltCrack300/b", "bridge-a", "mask-a", source="AsphaltCrack300"),
            split_record("b-unique", "AsphaltCrack300/b", "bridge-b", "mask-b", source="AsphaltCrack300"),
            split_record("c-duplicate", "crack500/c", "bridge-b", "mask-b", source="crack500"),
            split_record("c-unique", "crack500/c", "c-unique-image", "c-unique-mask", source="crack500"),
        ]
        for index in range(9):
            records.append(
                split_record(
                    f"extra-{index}",
                    f"CFD/extra-{index}",
                    f"extra-image-{index}",
                    f"extra-mask-{index}",
                )
            )

        assign_splits(records, seed=99, ratios={"train": 0.6, "val": 0.2, "test": 0.2})

        retained = [record for record in records if record["exclusion_reason"] is None]
        bridge_group_splits = {
            record["split"]
            for record in retained
            if record["group_id"] in {"CFD/a", "AsphaltCrack300/b", "crack500/c"}
        }
        self.assertEqual(len(bridge_group_splits), 1)
        self.assertNotIn(None, bridge_group_splits)
        active_splits_by_image = {}
        for record in retained:
            active_splits_by_image.setdefault(record["image_sha256"], set()).add(record["split"])
        self.assertTrue(all(len(splits) == 1 for splits in active_splits_by_image.values()))

    def test_assign_splits_rejects_identical_rgb_with_different_masks(self) -> None:
        records = [
            split_record("first", "CFD/first", "same-rgb", "first-mask"),
            split_record("second", "CFD/second", "same-rgb", "second-mask"),
        ]
        with self.assertRaises(ValueError):
            assign_splits(records, seed=1, ratios={"train": 0.6, "val": 0.2, "test": 0.2})

    def test_assign_splits_excludes_augmented_members_of_heldout_groups(self) -> None:
        records = []
        for index in range(12):
            group_id = f"CFD/original-{index:02d}"
            records.extend(
                [
                    split_record(
                        f"original-{index:02d}",
                        group_id,
                        f"original-image-{index}",
                        f"original-mask-{index}",
                    ),
                    split_record(
                        f"augmented-{index:02d}",
                        group_id,
                        f"augmented-image-{index}",
                        f"augmented-mask-{index}",
                        is_augmented=True,
                    ),
                ]
            )

        assign_splits(records, seed=31415, ratios={"train": 0.5, "val": 0.25, "test": 0.25})

        originals_by_group = {
            record["group_id"]: record
            for record in records
            if not record["is_augmented"]
        }
        heldout_augmented = [
            record
            for record in records
            if record["is_augmented"]
            and originals_by_group[record["group_id"]]["split"] in {"val", "test"}
        ]
        self.assertTrue(heldout_augmented)
        for record in heldout_augmented:
            self.assertIsNone(record["split"])
            self.assertIsNotNone(record["exclusion_reason"])

    def test_quarantined_conflicting_masks_remain_excluded(self) -> None:
        records = [
            split_record("bad-a", "CFD/bad-a", "identical-rgb", "mask-a"),
            split_record("bad-b", "CFD/bad-b", "identical-rgb", "mask-b"),
        ]
        for record in records:
            record["exclusion_reason"] = "quarantined:conflicting_masks"
        for index in range(12):
            records.append(split_record(
                f"good-{index}", f"CFD/group-{index}", f"rgb-{index}", f"mask-{index}"
            ))
        assign_splits(records, seed=42, ratios={"train": 0.7, "val": 0.1, "test": 0.2})
        for record in records[:2]:
            self.assertIsNone(record["split"])
            self.assertEqual(record["exclusion_reason"], "quarantined:conflicting_masks")
        self.assertEqual({r["split"] for r in records[2:]}, {"train", "val", "test"})


class DatasetExportTests(unittest.TestCase):
    def test_export_round_trip_for_boxes_sharing_coordinates(self) -> None:
        # Real crack500 geometry: normalized x=35 restores on opposite sides of
        # 35 for different box widths, so sorting raw floats pairs wrong boxes.
        width, height = 360, 640
        boxes = [[35, 2, 65, 115], [35, 130, 80, 374], [21, 375, 64, 633]]
        image = np.full((height, width, 3), 37, dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        for x1, y1, x2, y2 in boxes:
            mask[y1:y2, x1:x2] = 1
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image_path, mask_path = root / "image.png", root / "mask.png"
            Image.fromarray(image).save(image_path)
            Image.fromarray(mask).save(mask_path)
            record = {
                "sample_id": "shared-x", "source": "crack500",
                "image_path": str(image_path), "mask_path": str(mask_path),
                "width": width, "height": height, "group_id": "crack500/original",
                "split": "train", "boxes": boxes, "foreground_values": [1],
                "image_sha256": array_hash(image, width, height),
                "mask_sha256": array_hash(mask, width, height),
                "is_augmented": False, "exclusion_reason": None,
            }
            output = root / "export"
            export_dataset([record], output)
            self.assertEqual(validate_dataset(output)["annotations"], 3)
            label = output / "labels/train/shared-x.txt"
            lines = label.read_text().splitlines()
            fields = lines[0].split()
            fields[1] = str(float(fields[1]) + 0.25 / width)
            lines[0] = " ".join(fields)
            label.write_text("\n".join(lines) + "\n")
            with self.assertRaises(ValueError):
                validate_dataset(output)

    def test_export_and_validate_detect_semantic_bbox_and_label_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            source_dir.mkdir()
            output_dir = root / "export"

            train_image = np.arange(8 * 6 * 3, dtype=np.uint8).reshape(6, 8, 3)
            train_mask = np.zeros((6, 8), dtype=np.uint8)
            train_mask[1:4, 2:6] = 1
            empty_image = np.full((6, 8, 3), 77, dtype=np.uint8)
            empty_mask = np.zeros((6, 8), dtype=np.uint8)
            test_image = np.full((6, 8, 3), 133, dtype=np.uint8)
            test_mask = np.zeros((6, 8), dtype=np.uint8)

            def make_export_record(sample_id: str, split: str, image: np.ndarray, mask: np.ndarray, boxes: list[list[int]]) -> dict:
                image_path = source_dir / f"{sample_id}.png"
                mask_path = source_dir / f"{sample_id}_mask.png"
                Image.fromarray(image, mode="RGB").save(image_path)
                Image.fromarray(mask, mode="L").save(mask_path)
                height, width = mask.shape
                return {
                    "sample_id": sample_id,
                    "source": "CFD",
                    "image_path": str(image_path.resolve()),
                    "mask_path": str(mask_path.resolve()),
                    "width": width,
                    "height": height,
                    "group_id": f"CFD/{sample_id}",
                    "split": split,
                    "boxes": boxes,
                    "foreground_values": [1],
                    "image_sha256": array_hash(image, width, height),
                    "mask_sha256": array_hash(mask, width, height),
                    "is_augmented": False,
                    "exclusion_reason": None,
                }

            records = [
                make_export_record("train-box", "train", train_image, train_mask, [[2, 1, 6, 4]]),
                make_export_record("val-empty", "val", empty_image, empty_mask, []),
                make_export_record("test-empty", "test", test_image, test_mask, []),
            ]
            export_dataset(records, output_dir)
            report = validate_dataset(output_dir)
            self.assertIsInstance(report, dict)

            train_label = output_dir / "labels" / "train" / "train-box.txt"
            label_values = [float(value) for value in train_label.read_text(encoding="utf-8").split()]
            self.assertEqual(label_values[0], 0.0)
            self.assertEqual(len(label_values), 5)
            self.assertAlmostEqual(label_values[1], 0.5, delta=1e-6)
            self.assertAlmostEqual(label_values[2], 5 / 12, delta=1e-6)
            self.assertAlmostEqual(label_values[3], 0.5, delta=1e-6)
            self.assertAlmostEqual(label_values[4], 0.5, delta=1e-6)
            self.assertEqual(
                (output_dir / "labels" / "val" / "val-empty.txt").read_text(encoding="utf-8"),
                "",
            )

            train_coco = json.loads(
                (output_dir / "annotations" / "instances_train.json").read_text(encoding="utf-8")
            )
            annotation = train_coco["annotations"][0]
            self.assertEqual(annotation["bbox"], [2, 1, 4, 3])
            self.assertEqual(annotation["area"], 12)
            self.assertEqual(annotation["category_id"], 1)
            self.assertEqual(annotation["iscrowd"], 0)
            self.assertEqual(
                train_coco["images"][0]["file_name"],
                "train/train-box.png",
            )

            train_label.write_text("0 0.5 0.4166666667 0.5 0.4\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_dataset(output_dir)

            train_label.write_text("0 0.5 0.4166666667 0.5 0.5\n", encoding="utf-8")
            annotation["bbox"][2] = 3
            (output_dir / "annotations" / "instances_train.json").write_text(
                json.dumps(train_coco), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                validate_dataset(output_dir)


if __name__ == "__main__":
    unittest.main()
