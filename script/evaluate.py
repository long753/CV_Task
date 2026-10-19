#!/usr/bin/env python3
"""使用冻结的 YOLO 权重在指定数据划分上执行 COCO 指标评测。

在 cv_task 项目根目录执行：
python script/evaluate.py \
  --config configs/train_ultralytics.yaml \
  --weights models/trained/yolov8s_formal/best.pt \
  --split val \
  --output outputs/evaluation/yolov8s_val \
  --benchmark

参数说明：
--config：统一正式实验的 YAML 配置文件路径。
--weights：必填，要评测的冻结权重文件路径，明确选择实际模型。
--split：要评测的已配置数据划分，只能为 val 或 test，默认 val。
--output：必填，评测 JSON、曲线、预览图及可选性能结果的输出目录。
--benchmark：提供时，额外在固定的代表性验证样本上测量该权重的性能。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.evaluation.yolo import evaluate_checkpoint


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析冻结权重评测流程的命令行参数。"""
    parser = argparse.ArgumentParser(description="使用 COCO 指标评测冻结的 YOLO 权重")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "train_ultralytics.yaml",
        help="统一正式实验 YAML 配置文件",
    )
    parser.add_argument("--weights", type=Path, required=True, help="必填，要评测的冻结权重文件")
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
    summary = evaluate_checkpoint(
        config, args.weights, args.split, args.output, include_benchmark=args.benchmark
    )
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
