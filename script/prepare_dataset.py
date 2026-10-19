#!/usr/bin/env python3
"""审计、导出或独立验证裂缝检测数据集。

在 cv_task 项目根目录执行：
python script/prepare_dataset.py \
  --config configs/preprocess.yaml \
  --stage validate \
  --output datasets/processed/crack_detection

参数说明：
--config：预处理配置文件路径，默认 configs/preprocess.yaml。
--stage：执行阶段；audit 读取原始数据并生成审计报告，all 执行完整预处理流程，
         validate 独立验证已导出的数据集。
--output：新数据集的输出目录；当 --stage 为 validate 时，指定已有导出数据集目录。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_io import validate_dataset
from src.data.preprocessor import Preprocessor, load_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """读取命令行参数，并在解析器作用域内检查参数组合。"""
    parser = argparse.ArgumentParser(description="准备裂缝检测数据集")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/preprocess.yaml",
        help="预处理配置文件路径",
    )
    parser.add_argument(
        "--stage",
        choices=("audit", "all", "validate"),
        default="all",
        help="执行阶段：audit、all 或 validate（默认：all）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="新输出目录；--stage validate 时为已有导出数据集目录",
    )
    args = parser.parse_args(argv)
    if args.stage == "validate" and args.output is None:
        parser.error("--stage validate 必须提供 --output；此阶段不需要原始数据或配置")
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
