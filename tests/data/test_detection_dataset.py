"""检测标签、空目标和几何变换的消费者行为回归。"""
import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from src.data.detection_dataset import DetectionDataset
from src.models.torchvision import build_model


class DetectionDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'images/train').mkdir(parents=True)
        (self.root / 'annotations').mkdir()
        image = Image.new('RGB', (8, 4))
        image.putpixel((0, 0), (255, 0, 0))
        image.save(self.root / 'images/train/one.png')
        image.save(self.root / 'images/train/empty.png')
        (self.root / 'data.yaml').write_text('train: images/train\nnames: {0: crack}\n')
        (self.root / 'annotations/instances_train.json').write_text(json.dumps({
            'images': [
                {'id': 7, 'file_name': 'train/one.png', 'width': 8, 'height': 4},
                {'id': 8, 'file_name': 'train/empty.png', 'width': 8, 'height': 4},
            ],
            'categories': [{'id': 1, 'name': 'crack'}],
            'annotations': [{'id': 1, 'image_id': 7, 'category_id': 1, 'bbox': [2, 1, 3, 2], 'area': 6, 'iscrowd': 0}],
        }))

    def dataset(self, model, **kwargs):
        return DetectionDataset(self.root / 'data.yaml', 'train', model, **kwargs)

    def test_model_specific_labels_keep_original_image_and_xyxy(self):
        _, faster = self.dataset('fasterrcnn_resnet50_fpn')[0]
        image, retina = self.dataset('retinanet_resnet50_fpn')[0]
        self.assertEqual(faster['labels'].tolist(), [1])
        self.assertEqual(retina['labels'].tolist(), [0])
        torch.testing.assert_close(retina['boxes'], torch.tensor([[2., 1., 5., 3.]]))
        self.assertEqual(tuple(image.shape), (3, 4, 8))
        self.assertEqual(retina['image_id'].item(), 7)

    def test_empty_image_keeps_empty_two_dimensional_boxes(self):
        _, target = self.dataset('retinanet_resnet50_fpn')[1]
        self.assertEqual(target['boxes'].shape, (0, 4))
        self.assertEqual(target['boxes'].dtype, torch.float32)
        self.assertEqual(target['labels'].shape, (0,))
        self.assertEqual(target['labels'].dtype, torch.int64)
        self.assertEqual(target['image_id'].item(), 8)

    def test_horizontal_flip_changes_image_and_box_together(self):
        image, target = self.dataset('fasterrcnn_resnet50_fpn', augment=True, augmentation={'horizontal_flip': 1.0})[0]
        torch.testing.assert_close(target['boxes'], torch.tensor([[3., 1., 6., 3.]]))
        self.assertEqual(image[0, 0, 7].item(), 1.0)
        self.assertEqual(image[0, 0, 0].item(), 0.0)

    def test_fixed_square_transform_restores_original_box_coordinates(self):
        image, target = self.dataset('fasterrcnn_resnet50_fpn')[0]
        model = build_model('fasterrcnn_resnet50_fpn', imgsz=64).eval()
        transformed, targets = model.transform([image], [target])
        self.assertEqual(transformed.image_sizes, [(64, 64)])
        torch.testing.assert_close(targets[0]['boxes'], torch.tensor([[16., 16., 40., 48.]]))
        restored = model.transform.postprocess(
            [{'boxes': targets[0]['boxes']}], transformed.image_sizes, [tuple(image.shape[-2:])],
        )
        torch.testing.assert_close(restored[0]['boxes'], target['boxes'])


if __name__ == '__main__':
    unittest.main()
