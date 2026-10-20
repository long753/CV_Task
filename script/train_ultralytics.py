#!/usr/bin/env python3
"""基于 Ultralytics 架构训练裂缝检测模型。

完整命令（YOLOv8s）：
uv run --no-sync python script/train_ultralytics.py \
  --config configs/train_ultralytics.yaml \
  --epochs 150 \
  --batch 8 \
  --workers 0 \
  --model yolov8s
训练 YOLOv11s 或 RT-DETR-L 时，将 --model 分别改为 yolov11s 或 rtdetr_l。

恢复尚未完成的训练（自动按运行快照恢复 YOLO 或 RT-DETR 类）：
uv run --no-sync python script/train_ultralytics.py \
  --resume outputs/training/rtdetr_l_formal/weights/last.pt

参数说明：
--config：训练超参数 YAML，省略时使用 configs/train_ultralytics.yaml。
--epochs：覆盖新训练的总轮数，必须大于零。
--batch：覆盖新训练的每设备批次大小，必须大于零。
--workers：覆盖新训练的数据加载工作进程数，必须为非负整数。
--model：新训练模型：yolov8s、yolov11s、rtdetr_l。
--resume：恢复未完成的原生检查点；必须单独使用，不与其他训练参数组合。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
# 先配置项目内的缓存目录和 TensorBoard，再使用 Ultralytics。
from src.evaluation.detection import evaluate_checkpoint
from src.models.ultralytics import load_ultralytics_model
from ultralytics.engine.model import Model
from ultralytics.utils.callbacks import tensorboard as tensorboard_callbacks
from ultralytics.utils.files import increment_path

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_ultralytics.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析模型选择、训练超参数覆盖和原生续训参数。"""
    parser = argparse.ArgumentParser(description="使用 Ultralytics 训练裂缝检测模型")
    parser.add_argument("--config", type=Path, help="共用训练 YAML，省略时使用默认配置")
    parser.add_argument("--epochs", type=int, help="训练轮数：默认 150 轮")
    parser.add_argument("--batch", type=int, help="覆盖新训练的每设备批次大小")
    parser.add_argument("--workers", type=int, help="覆盖新训练的数据加载工作进程数")
    parser.add_argument(
        "--model",
        choices=("yolov8s", "yolov11s", "rtdetr_l"),
        help="训练模型：yolov8s  yolov11s  rtdetr_l",
    )
    parser.add_argument("--resume", type=Path, help="未完成运行的 weights/last.pt")
    args = parser.parse_args(argv)
    if args.epochs is not None and args.epochs <= 0:
        parser.error("--epochs 必须大于零")
    if args.batch is not None and args.batch <= 0:
        parser.error("--batch 必须大于零")
    if args.workers is not None and args.workers < 0:
        parser.error("--workers 必须为非负整数")
    if args.resume is not None and any(
        value is not None for value in (args.config, args.epochs, args.batch, args.workers, args.model)
    ):
        parser.error("--resume 使用原运行的配置，不能与其他训练参数同时使用")
    return args


def prepare_training_data(data_yaml: Path, run_dir: Path) -> Path:
    """复制训练集和验证集，避免框架缓存写入冻结数据；不复制测试集。"""
    source = data_yaml.parent
    target = run_dir / "data_view"
    for split in ("train", "val"):
        for directory in ("images", "labels"):
            shutil.copytree(source / directory / split, target / directory / split)
    dataset = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    training_data = target / "data.yaml"
    training_data.write_text(
        yaml.safe_dump(
            {"path": str(target), "train": "images/train", "val": "images/val", "names": dataset["names"]},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return training_data


def train(args: argparse.Namespace) -> tuple[Model, dict]:
    """选取预训练模型并训练；续训直接交给 Ultralytics 恢复优化器等状态。"""
    if args.resume is not None:
        checkpoint = args.resume.resolve()
        run_dir = checkpoint.parent.parent
        config = load_config(run_dir / "config.yaml")
        model = load_ultralytics_model(checkpoint, config["model_name"])
    else:
        config = load_config(args.config or DEFAULT_CONFIG)
        config["model_name"] = args.model or "yolov8s"
        config["name"] = f"{config['model_name']}_formal"
        training = dict(config["training"])
        training.update(config.get("model_overrides", {}).get(config["model_name"], {}))
        for name, value in (
            ("epochs", args.epochs),
            ("batch", args.batch),
            ("workers", args.workers),
        ):
            if value is not None:
                training[name] = value
        config["training"] = training
        weights = config["models"][config["model_name"]]
        if not weights.is_file():
            raise FileNotFoundError(f"请先下载预训练权重：{weights}")
        model = load_ultralytics_model(weights, config["model_name"])
        run_dir = increment_path(config["project"] / config["name"]).resolve()

    # 只保留必要的防覆盖检查，不改写已有正式模型。
    archive_dir = config["trained_models"] / run_dir.name
    if archive_dir.exists():
        raise FileExistsError(f"模型归档目录已存在，请使用新的运行目录：{archive_dir}")

    if args.resume is not None:
        model.train(resume=True)
    else:
        run_dir.mkdir(parents=True)
        training_data = prepare_training_data(config["data"], run_dir)
        # 保存本次实际配置，包含命令行覆盖值；中断后可直接读取。
        snapshot = json.loads(json.dumps(config, default=str))
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        training_options = dict(config["training"])
        training_options.update(
            data=str(training_data), project=str(run_dir.parent), name=run_dir.name, exist_ok=True,
        )
        model.train(**training_options)
    return model, config


def save_results(model: Model, config: dict) -> dict:
    """关闭事件文件、归档原生训练产物，再对 best.pt 执行统一验证。"""
    run_dir = Path(model.trainer.save_dir).resolve()
    writer = tensorboard_callbacks.WRITER
    if writer is not None:
        writer.flush()
        writer.close()

    archive_dir = config["trained_models"] / run_dir.name
    archive_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_dir / "weights"), str(archive_dir))
    for path in list(run_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            destination = run_dir / "plots"
        elif path.is_file() and path.name.startswith("events.out.tfevents."):
            destination = run_dir / "tensorboard"
        else:
            continue
        destination.mkdir(exist_ok=True)
        shutil.move(str(path), str(destination / path.name))

    best_path = archive_dir / "best.pt"
    validation = evaluate_checkpoint(config, best_path, "val", run_dir / "validation")
    summary = {
        "model": config["model_name"],
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(archive_dir / "last.pt"),
        "epochs": int(model.trainer.epochs),
        "metrics": validation["metrics"],
        "validation": str(run_dir / "validation" / "evaluation_summary.json"),
    }
    (run_dir / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model, config = train(args)
    summary = save_results(model, config)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
