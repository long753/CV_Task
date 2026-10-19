"""Real CPU inference regression; no model weights or network access required."""

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image
import torch

from src.evaluation.yolo import _coco_detections, predict_split
from ultralytics import YOLO
from ultralytics.engine.results import Boxes


class YoloInferenceTests(unittest.TestCase):
    def test_empty_clipped_boxes_do_not_break_coco_export_or_remove_tiny_targets(self) -> None:
        boxes = Boxes(torch.tensor([
            [10.0, 2.0, 10.0, 7.0, 0.9, 0.0],
            [1.0, 2.0, 4.0, 7.0, 0.8, 0.0],
            [2.0, 3.0, 2.01, 3.01, 0.7, 0.0],
        ]), orig_shape=(10, 10))
        predictions, discarded = _coco_detections(boxes, image_id=17)
        self.assertEqual(discarded, 1)
        self.assertEqual([row["image_id"] for row in predictions], [17, 17])
        self.assertEqual(predictions[0]["bbox"], [1.0, 2.0, 3.0, 5.0])
        self.assertAlmostEqual(predictions[1]["bbox"][2], 0.01, places=5)
        self.assertAlmostEqual(predictions[1]["bbox"][3], 0.01, places=5)

    def test_multiple_validation_files_remain_single_image_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "images/val").mkdir(parents=True)
            (root / "annotations").mkdir()
            images = []
            for index in range(3):
                file_name = f"val/image_{index}.jpg"
                Image.new("RGB", (64, 64), (30 * index, 20, 40)).save(root / "images" / file_name)
                images.append({"id": index + 1, "file_name": file_name, "width": 64, "height": 64})
            (root / "annotations/instances_val.json").write_text(json.dumps({
                "images": images, "categories": [{"id": 1, "name": "crack"}], "annotations": [],
            }))
            data = root / "data.yaml"
            data.write_text("val: images/val\nnames: {0: crack}\n")
            config = {
                "data": data,
                "training": {"imgsz": 64, "device": "cpu"},
                "evaluation": {"score_floor": 1.0, "nms_iou": 0.7, "max_detections": 100},
            }
            # Random initialization is only a batching fixture, never a substitute
            # for the COCO-pretrained model used in the real GPU pilot.
            model = YOLO("yolov8s.yaml")
            batch_sizes = []
            model.add_callback("on_predict_batch_start", lambda predictor: batch_sizes.append(len(predictor.batch[0])))
            predict_split(model, config)
            self.assertEqual(batch_sizes, [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
