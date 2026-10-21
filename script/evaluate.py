#!/usr/bin/env python3
"""使用冻结的五种裂缝检测权重执行统一 COCO 评测。

Ultralytics 训练示例：
uv run --no-sync python script/evaluate.py \
  --config configs/ultralytics.yaml \
  --model yolov8s \
  --weights models/trained/yolov8s_formal/best.pt \
  --split test \
  --output outputs/evaluation/yolov8s --benchmark

TorchVision 训练示例：
uv run --no-sync python script/evaluate.py \
  --config configs/torchvision.yaml \
  --model fasterrcnn_resnet50_fpn \
  --weights models/trained/fasterrcnn_resnet50_fpn_formal/best.pt \
  --split test \
  --output outputs/evaluation/fasterrcnn_resnet50_fpn --benchmark

参数说明：
--config：评测参数
--model：评测模型
    Ultralytics：yolov8s、yolov11s、rtdetr_l
    TorchVision：fasterrcnn_resnet50_fpn、retinanet_resnet50_fpn
--weights：必填，要评测的冻结权重文件路径。
--split：要评测的已配置数据划分:test, val
--output：输出目录
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.evaluation.detection import evaluate_checkpoint


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析冻结权重评测流程的命令行参数。"""
    parser = argparse.ArgumentParser(description="使用 COCO 指标评测冻结的裂缝检测权重")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "ultralytics.yaml",
        help="统一正式实验 YAML 配置文件",
    )
    parser.add_argument("--weights", type=Path, required=True, help="必填，要评测的冻结权重文件")
    parser.add_argument(
        "--model",
        choices=(
            "yolov8s",
            "yolov11s",
            "rtdetr_l",
            "fasterrcnn_resnet50_fpn",
            "retinanet_resnet50_fpn",
        ),
        help="可选，显式选择 checkpoint 架构；省略时根据元数据识别",
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
        help="要评测的已配置数据划分（默认：val）",
    )
    parser.add_argument("--output", type=Path, required=True, help="评测产物输出目录")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="额外在固定的代表性验证样本上测量该权重的性能",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.model is not None:
        config["model_name"] = args.model
    summary = evaluate_checkpoint(
        config, args.weights, args.split, args.output, include_benchmark=args.benchmark
    )
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
