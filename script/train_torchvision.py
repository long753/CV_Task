#!/usr/bin/env python3
"""使用 TorchVision Faster R-CNN 或 RetinaNet 训练裂缝检测模型。

完整命令（Faster R-CNN）：
uv run --no-sync python script/train_torchvision.py \
  --config configs/train_torchvision.yaml \
  --epochs 150 \
  --batch 8 \
  --workers 0 \
  --model fasterrcnn_resnet50_fpn
训练 RetinaNet 时将 --model 改为 retinanet_resnet50_fpn。

恢复未完成训练：
uv run --no-sync python script/train_torchvision.py \
  --resume outputs/training/fasterrcnn_resnet50_fpn_formal/weights/last.pt

参数说明：
--config：训练超参数 YAML，省略时使用 configs/train_torchvision.yaml。
--epochs：覆盖新训练的总轮数，必须大于零。
--batch：覆盖新训练的每设备批次大小，必须大于零。
--workers：覆盖新训练的数据加载工作进程数，必须为非负整数。
--model：新训练的模型：fasterrcnn_resnet50_fpn 或 retinanet_resnet50_fpn。
--resume：恢复本脚本保存的 weights/last.pt；必须单独使用，并读取同一运行目录的 config.yaml。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import yaml
from torch import Tensor
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

matplotlib.use("Agg")
from matplotlib import pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.detection_dataset import DetectionDataset, collate_fn
from src.evaluation.detection import evaluate_checkpoint, evaluate_model
from src.models.torchvision import MODEL_NAMES, build_model, configure_determinism, load_model

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_torchvision.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析模型选择、训练超参数覆盖和本项目检查点续训参数。"""
    parser = argparse.ArgumentParser(description="使用 TorchVision 训练裂缝检测模型")
    parser.add_argument("--config", type=Path, help="训练 YAML，省略时使用默认配置")
    parser.add_argument("--epochs", type=int, help="覆盖新训练的总轮数")
    parser.add_argument("--batch", type=int, help="覆盖新训练的每设备批次大小")
    parser.add_argument("--workers", type=int, help="覆盖新训练的数据加载工作进程数")
    parser.add_argument("--model", choices=sorted(MODEL_NAMES), help="新训练模型")
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
        parser.error("--resume 使用原运行配置，不能与其他训练参数同时使用")
    return args


def _json_config(config: dict[str, Any]) -> dict[str, Any]:
    """将路径等 YAML 可读配置值转为检查点安全的 JSON 数据。"""
    return json.loads(json.dumps(config, default=str))


def _save_config(config: dict[str, Any], run_dir: Path) -> None:
    snapshot = _json_config(config)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def _next_run_dir(config: dict[str, Any]) -> Path:
    base = Path(config["project"]) / str(config["name"])
    trained_models = Path(config["trained_models"])
    candidate = base
    suffix = 2
    while candidate.exists() or (trained_models / candidate.name).exists():
        candidate = base.with_name(f"{base.name}_{suffix}")
        suffix += 1
    return candidate.resolve()


def _resolve_device(value: Any) -> torch.device:
    if isinstance(value, int):
        return torch.device(f"cuda:{value}")
    if isinstance(value, str) and value.isdecimal():
        return torch.device(f"cuda:{value}")
    return torch.device(str(value))


def _seed_everything(seed: int, deterministic: bool) -> None:
    configure_determinism(deterministic)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_scheduler(optimizer: Optimizer, training: dict[str, Any]) -> LambdaLR:
    total_epochs = int(training["epochs"])
    warmup_epochs = int(training["warmup_epochs"])
    final_ratio = float(training["lrf"])

    def learning_rate_factor(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return final_ratio + (1.0 - final_ratio) * (1.0 + math.cos(math.pi * progress)) / 2.0

    return LambdaLR(optimizer, learning_rate_factor)


def _make_optimizer(model: torch.nn.Module, training: dict[str, Any]) -> Optimizer:
    if training.get("optimizer") != "SGD":
        raise ValueError("TorchVision configuration currently requires optimizer: SGD")
    return SGD(
        model.parameters(),
        lr=float(training["lr0"]),
        momentum=float(training["momentum"]),
        weight_decay=float(training["weight_decay"]),
    )


def _checkpoint(
    model_name: str,
    model: torch.nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_map: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_map": best_map,
        "config": _json_config(config),
    }


def _load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"TorchVision checkpoint must be a mapping: {path}")
    return checkpoint


def _make_loader(
    dataset: DetectionDataset,
    training: dict[str, Any],
    device: torch.device,
    *,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(training["seed"]) + (0 if shuffle else 1))
    return DataLoader(
        dataset,
        batch_size=int(training["batch"]),
        shuffle=shuffle,
        num_workers=int(training["workers"]),
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
        generator=generator,
    )


def _move_targets(targets: tuple[dict[str, Tensor], ...], device: torch.device) -> list[dict[str, Tensor]]:
    return [{key: value.to(device, non_blocking=True) for key, value in target.items()} for target in targets]


def _train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    accumulate: int,
) -> tuple[float, dict[str, float]]:
    """训练一个 epoch；最后不满 accumulate 的窗口按真实批次数归一化。"""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    per_loss: dict[str, float] = {}
    examples = 0
    batches = len(loader)
    for batch_index, (images, targets) in enumerate(loader):
        image_list = [image.to(device, non_blocking=True) for image in images]
        target_list = _move_targets(targets, device)
        window_start = (batch_index // accumulate) * accumulate
        window_batches = min(accumulate, batches - window_start)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            losses = model(image_list, target_list)
            loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise FloatingPointError("训练 Loss 出现 NaN/Inf，停止更新参数")
        scaler.scale(loss / window_batches).backward()

        batch_examples = len(image_list)
        examples += batch_examples
        total_loss += float(loss.detach()) * batch_examples
        for name, value in losses.items():
            per_loss[name] = per_loss.get(name, 0.0) + float(value.detach()) * batch_examples

        if (batch_index + 1) % accumulate == 0 or batch_index + 1 == batches:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    if not examples:
        raise ValueError("training split is empty")
    return total_loss / examples, {name: value / examples for name, value in per_loss.items()}


def _prepare_metrics(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def _append_metrics(path: Path, row: dict[str, float]) -> None:
    needs_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def _plot_curves(history: list[dict[str, float]], output_path: Path) -> None:
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(epochs, [row["train_loss"] for row in history], color="tab:blue")
    axes[0].set(xlabel="epoch", ylabel="training loss", title="Training loss")
    axes[1].plot(epochs, [row["map50_95"] for row in history], color="tab:green")
    axes[1].set(xlabel="epoch", ylabel="mAP50-95", title="Validation mAP50-95", ylim=(0, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _restore_or_create(args: argparse.Namespace) -> tuple[dict[str, Any], Path, torch.nn.Module, dict[str, Any] | None]:
    if args.resume is not None:
        checkpoint_path = args.resume.resolve()
        run_dir = checkpoint_path.parent.parent
        config = load_config(run_dir / "config.yaml")
        _seed_everything(int(config["training"]["seed"]), bool(config["training"]["deterministic"]))
        checkpoint = _load_checkpoint(checkpoint_path)
        model, model_name = load_model(checkpoint_path, config)
        if config.get("model_name") != model_name:
            raise ValueError("checkpoint model_name does not match its saved configuration")
        return config, run_dir, model, checkpoint

    config = load_config(args.config or DEFAULT_CONFIG)
    model_name = args.model or "fasterrcnn_resnet50_fpn"
    config["model_name"] = model_name
    config["name"] = f"{model_name}_formal"
    config["training"] = {
        **config["training"], **config.get("model_overrides", {}).get(model_name, {})
    }
    for name, value in (
        ("epochs", args.epochs),
        ("batch", args.batch),
        ("workers", args.workers),
    ):
        if value is not None:
            config["training"][name] = value
    _seed_everything(int(config["training"]["seed"]), bool(config["training"]["deterministic"]))
    pretrained_path = Path(config["models"][model_name])
    if not pretrained_path.is_file():
        raise FileNotFoundError(f"请先下载官方 COCO 预训练权重：{pretrained_path}")
    evaluation = config["evaluation"]
    model = build_model(
        model_name,
        imgsz=int(config["training"]["imgsz"]),
        pretrained_path=pretrained_path,
        score_thresh=float(evaluation["score_floor"]),
        nms_thresh=float(evaluation["nms_iou"]),
        max_detections=int(evaluation["max_detections"]),
    )
    return config, _next_run_dir(config), model, None


def train(args: argparse.Namespace) -> tuple[dict[str, Any], Path, Path]:
    """执行完整原生训练循环，逐轮保存 last/best 并以统一评测选择最佳模型。"""
    config, run_dir, model, resume_state = _restore_or_create(args)
    training = config["training"]
    model_name = str(config["model_name"])
    archive_dir = Path(config["trained_models"]) / run_dir.name
    if resume_state is None:
        if archive_dir.exists():
            raise FileExistsError(f"模型归档目录已存在，请使用新的运行目录：{archive_dir}")
        run_dir.mkdir(parents=True)
        _save_config(config, run_dir)

    device = _resolve_device(training["device"])
    model.to(device)
    optimizer = _make_optimizer(model, training)
    scheduler = _make_scheduler(optimizer, training)
    amp_enabled = bool(training["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    best_map = float("-inf")
    start_epoch = 0
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        scaler.load_state_dict(resume_state["scaler"])
        start_epoch = int(resume_state["epoch"]) + 1
        best_map = float(resume_state["best_map"])

    train_dataset = DetectionDataset(
        config["data"],
        "train",
        model_name,
        augment=True,
        augmentation=config.get("augmentation"),
    )
    train_loader = _make_loader(train_dataset, training, device, shuffle=True)
    if not len(train_loader):
        raise ValueError("training split is empty")
    accumulate = max(1, round(int(training["nbs"]) / int(training["batch"])))

    metrics_path = run_dir / "results.csv"
    history = _prepare_metrics(metrics_path)
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(exist_ok=True)
    last_path = weights_dir / "last.pt"
    best_path = weights_dir / "best.pt"

    try:
        for epoch in range(start_epoch, int(training["epochs"])):
            configure_determinism(bool(training["deterministic"]))
            current_lr = float(optimizer.param_groups[0]["lr"])
            train_loss, component_losses = _train_epoch(
                model, train_loader, optimizer, scaler, device, amp_enabled, accumulate
            )

            # 直接验证当前模型；不复制模型、不把验证集切回训练模式计算 Loss。
            scheduler.step()
            validation = evaluate_model(
                config, model, model_name, "val", run_dir / "validation_epoch"
            )
            metrics = validation["metrics"]
            map50_95 = float(metrics["map50_95"])

            if map50_95 > best_map:
                best_map = map50_95
                torch.save(
                    _checkpoint(model_name, model, optimizer, scheduler, scaler, epoch, best_map, config), best_path
                )

            row = {
                "epoch": float(epoch + 1),
                "lr": current_lr,
                "train_loss": train_loss,
                "map50_95": map50_95,
                "map50": float(metrics["map50"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                **{f"train/{name}": value for name, value in component_losses.items()},
            }
            _append_metrics(metrics_path, row)
            history.append(row)
            writer.add_scalar("train/loss", train_loss, epoch + 1)
            writer.add_scalar("validation/map50_95", map50_95, epoch + 1)
            writer.add_scalar("training/lr", current_lr, epoch + 1)
            for name, value in component_losses.items():
                writer.add_scalar(f"train/{name}", value, epoch + 1)
            _plot_curves(history, run_dir / "plots" / "training_curves.png")
            writer.flush()
            torch.save(
                _checkpoint(model_name, model, optimizer, scheduler, scaler, epoch, best_map, config), last_path
            )
            print(
                json.dumps(
                    {"epoch": epoch + 1, "train_loss": train_loss, "map50_95": map50_95},
                    ensure_ascii=False,
                    allow_nan=False,
                ),
                flush=True,
            )
    finally:
        writer.flush()
        writer.close()

    if not best_path.is_file():
        raise RuntimeError("training did not produce a best checkpoint")
    if archive_dir.exists():
        raise FileExistsError(f"模型归档目录已存在，拒绝覆盖：{archive_dir}")
    archive_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(weights_dir), str(archive_dir))
    return config, run_dir, archive_dir / "best.pt"


def save_results(config: dict[str, Any], run_dir: Path, best_path: Path) -> dict[str, Any]:
    """归档完成后对 best.pt 执行完整统一验证，并记录可复现的实验摘要。"""
    validation = evaluate_checkpoint(config, best_path, "val", run_dir / "validation", include_benchmark=False)
    archive_dir = best_path.parent
    summary = {
        "model": config["model_name"],
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(archive_dir / "last.pt"),
        "epochs": int(config["training"]["epochs"]),
        "metrics": validation["metrics"],
        "validation": str(run_dir / "validation"),
    }
    (run_dir / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config, run_dir, best_path = train(args)
    summary = save_results(config, run_dir, best_path)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
