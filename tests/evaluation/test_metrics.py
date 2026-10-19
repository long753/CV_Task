import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.metrics import evaluate_predictions


def write_ground_truth(
    root: Path,
    annotations: list[tuple[int, int, list[float]]],
    *,
    image_ids: tuple[int, ...] = (1,),
    category_ids: tuple[int, ...] = (1,),
) -> Path:
    ground_truth = {
        "info": {"description": "synthetic detection evaluation data"},
        "licenses": [],
        "images": [
            {"id": image_id, "file_name": f"val/{image_id}.jpg", "width": 100, "height": 100}
            for image_id in image_ids
        ],
        "categories": [
            {"id": category_id, "name": f"category-{category_id}", "supercategory": "test"}
            for category_id in category_ids
        ],
        "annotations": [
            {
                "id": index,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": box,
                "area": box[2] * box[3],
                "iscrowd": 0,
            }
            for index, (image_id, category_id, box) in enumerate(annotations, start=1)
        ],
    }
    path = root / "instances_val.json"
    path.write_text(json.dumps(ground_truth), encoding="utf-8")
    return path


class DetectionEvaluationTests(unittest.TestCase):
    def test_perfect_predictions_have_full_precision_recall_and_ap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            ground_truth_path = write_ground_truth(
                root,
                [(1, 1, [0, 0, 10, 10]), (2, 1, [20, 20, 15, 15])],
                image_ids=(1, 2),
            )
            output_dir = root / "results"

            metrics = evaluate_predictions(
                ground_truth_path,
                [
                    {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.95},
                    {"image_id": 2, "category_id": 1, "bbox": [20, 20, 15, 15], "score": 0.90},
                ],
                output_dir,
            )

            self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (2, 0, 0))
            self.assertEqual((metrics["precision"], metrics["recall"]), (1.0, 1.0))
            self.assertAlmostEqual(metrics["map50"], 1.0)
            self.assertAlmostEqual(metrics["map50_95"], 1.0)

    def test_score_order_duplicate_matching_and_confidence_do_not_change_ap_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            ground_truth_path = write_ground_truth(root, [(1, 1, [0, 0, 10, 10])])
            predictions = [
                {"image_id": 1, "category_id": 1, "bbox": [50, 50, 10, 10], "score": 0.95},
                {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.10},
                {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.05},
            ]

            thresholded = evaluate_predictions(
                ground_truth_path,
                predictions,
                root / "thresholded",
                confidence=0.25,
            )
            all_scores = evaluate_predictions(
                ground_truth_path,
                predictions,
                root / "all-scores",
                confidence=0.0,
            )

            self.assertEqual((thresholded["tp"], thresholded["fp"], thresholded["fn"]), (0, 1, 1))
            self.assertEqual((thresholded["precision"], thresholded["recall"]), (0.0, 0.0))
            self.assertAlmostEqual(thresholded["map50"], 0.5)
            self.assertAlmostEqual(thresholded["map50_95"], 0.5)
            self.assertEqual((all_scores["tp"], all_scores["fp"], all_scores["fn"]), (1, 2, 0))
            self.assertAlmostEqual(all_scores["precision"], 1.0 / 3.0)
            self.assertEqual(all_scores["recall"], 1.0)
            self.assertAlmostEqual(all_scores["map50"], thresholded["map50"])


    def test_max_detections_caps_coco_and_fixed_threshold_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            ground_truth_path = write_ground_truth(root, [(1, 1, [0, 0, 10, 10])])
            output_dir = root / "results"

            metrics = evaluate_predictions(
                ground_truth_path,
                [
                    {"image_id": 1, "category_id": 1, "bbox": [50, 50, 10, 10], "score": 0.95},
                    {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.90},
                ],
                output_dir,
                confidence=0.0,
                max_detections=1,
            )

            self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (0, 1, 1))
            self.assertEqual((metrics["map50"], metrics["map50_95"]), (0.0, 0.0))
            self.assertEqual(
                json.loads((output_dir / "predictions_coco.json").read_text(encoding="utf-8")),
                [{"image_id": 1, "category_id": 1, "bbox": [50, 50, 10, 10], "score": 0.95}],
            )

    def test_matching_never_crosses_images_or_categories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            ground_truth_path = write_ground_truth(
                root,
                [(1, 1, [0, 0, 10, 10]), (2, 2, [0, 0, 10, 10])],
                image_ids=(1, 2),
                category_ids=(1, 2),
            )

            metrics = evaluate_predictions(
                ground_truth_path,
                [
                    {"image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.95},
                    {"image_id": 2, "category_id": 2, "bbox": [0, 0, 10, 10], "score": 0.90},
                ],
                root / "results",
            )

            self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 1, 1))
            self.assertEqual((metrics["precision"], metrics["recall"]), (0.5, 0.5))

    def test_empty_predictions_are_valid_zero_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            ground_truth_path = write_ground_truth(root, [(1, 1, [0, 0, 10, 10])])
            output_dir = root / "results"

            metrics = evaluate_predictions(ground_truth_path, [], output_dir)

            self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (0, 0, 1))
            self.assertEqual(
                (metrics["precision"], metrics["recall"], metrics["map50"], metrics["map50_95"]),
                (0.0, 0.0, 0.0, 0.0),
            )

    def test_malformed_references_boxes_scores_and_protocol_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            ground_truth_path = write_ground_truth(root, [(1, 1, [0, 0, 10, 10])])
            invalid_prediction_lists = (
                [{"image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9}],
                [{"image_id": 1, "category_id": 2, "bbox": [0, 0, 10, 10], "score": 0.9}],
                [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 0, 10], "score": 0.9}],
                [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": float("nan")}],
            )

            for predictions in invalid_prediction_lists:
                with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                    evaluate_predictions(ground_truth_path, predictions, root / "invalid")
            with self.assertRaises(ValueError):
                evaluate_predictions(ground_truth_path, [], root / "invalid-confidence", confidence=1.1)
            with self.assertRaises(ValueError):
                evaluate_predictions(ground_truth_path, [], root / "invalid-max-detections", max_detections=0)

            crowded = json.loads(ground_truth_path.read_text(encoding="utf-8"))
            crowded["annotations"][0]["iscrowd"] = 1
            crowded_path = root / "crowded.json"
            crowded_path.write_text(json.dumps(crowded), encoding="utf-8")
            with self.assertRaises(ValueError):
                evaluate_predictions(crowded_path, [], root / "crowded-results")
