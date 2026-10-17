#!/usr/bin/env python3
"""
Audit, export or independently validate the crack detection dataset.

python script/prepare_dataset.py \
  --config configs/preprocess.yaml \
  --stage validate \
  --output datasets/processed/crack_detection

参数说明
`config`: 配置文件路径
`stage`: 分为 audit, all, validate 三类。其中，
        audit 用于读取原始数据并生成审计报告，
        all 用于执行完整的数据预处理流程，
        validate 用于独立验证已导出的数据集。
`output`: 输出目录路径
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.Preprocessor import Preprocessor, load_config
from src.dataset_io import validate_dataset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """读取命令行参数，并在 parser 的作用域内检查参数组合。"""
    parser = argparse.ArgumentParser(description="Prepare dataset for crack detection")
    parser.add_argument("--config", 
                        type=Path, 
                        default=PROJECT_ROOT / "configs/preprocess.yaml")
    parser.add_argument("--stage", 
                        choices=("audit", "all", "validate"), 
                        default="all")
    parser.add_argument("--output",
                        type=Path,
                        help="New output directory, or existing export for --stage validate")
    args = parser.parse_args(argv)
    if args.stage == "validate" and args.output is None:
        parser.error("--stage validate requires --output; raw data/configuration are not needed")
    return args


def summarize_report(report: dict, output_dir: Path) -> dict:
    """只整理终端展示内容，不参与数据处理或数据校验。"""
    summary = {
        key: report[key]
        for key in ("status", "runtime", "total_pairs", "splits", "exclusions")
    }
    summary["output_dir"] = str(output_dir)
    summary["audit_report"] = str(output_dir / "audit/report.json")
    summary["previews"] = len(report.get("previews", []))
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage == "validate":
        result = validate_dataset(args.output.resolve())
    else:
        config = load_config(args.config, output_override=args.output)
        report = Preprocessor(config).run(stage=args.stage)
        result = summarize_report(report, Path(config["output_dir"]))

    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
