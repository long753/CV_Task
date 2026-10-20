#!/usr/bin/env bash
# 服务器使用：conda activate cv_task，然后在项目目录执行：
# bash script/train_three_models.sh
# 默认连续训练三个模型各 150 轮；可用环境变量覆盖批次、workers、配置及日志目录。
# 例如：FASTER_BATCH=16 RETINA_BATCH=16 RTDETR_BATCH=8 WORKERS=8 bash script/train_three_models.sh
# 任一模型失败即停止并保留断点；不重试、不自动降 batch、不启动后台服务。
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$PROJECT_ROOT"

# 服务器配额为 16 核，避免主进程线程和 DataLoader workers 过度争抢 CPU。
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

PYTHON_BIN="${PYTHON_BIN:-python}"
EPOCHS="${EPOCHS:-150}"
FASTER_BATCH="${FASTER_BATCH:-8}"
RETINA_BATCH="${RETINA_BATCH:-8}"
RTDETR_BATCH="${RTDETR_BATCH:-16}"
WORKERS="${WORKERS:-4}"
TORCHVISION_CONFIG="${TORCHVISION_CONFIG:-configs/train_torchvision.yaml}"
ULTRALYTICS_CONFIG="${ULTRALYTICS_CONFIG:-configs/train_ultralytics.yaml}"
LOG_DIR="${LOG_DIR:-outputs/training/logs}"

mkdir -p -- "$LOG_DIR"
RUN_LOG_DIR="$LOG_DIR/$(date +%Y%m%d-%H%M%S-%N)"
mkdir -- "$RUN_LOG_DIR"

QUEUE_LOG="$RUN_LOG_DIR/queue.log"
FASTER_LOG="$RUN_LOG_DIR/fasterrcnn_resnet50_fpn.log"
RETINA_LOG="$RUN_LOG_DIR/retinanet_resnet50_fpn.log"
RTDETR_LOG="$RUN_LOG_DIR/rtdetr_l.log"

printf '[%s] 开始训练 Faster R-CNN。\n' "$(date -Is)" | tee -a "$FASTER_LOG" "$QUEUE_LOG"
"$PYTHON_BIN" -u -s -B script/train_torchvision.py \
    --config "$TORCHVISION_CONFIG" \
    --epochs "$EPOCHS" \
    --model fasterrcnn_resnet50_fpn \
    --batch "$FASTER_BATCH" \
    --workers "$WORKERS" 2>&1 | tee -a "$FASTER_LOG" "$QUEUE_LOG"
printf '[%s] Faster R-CNN 已完成。\n' "$(date -Is)" | tee -a "$FASTER_LOG" "$QUEUE_LOG"

printf '[%s] 开始训练 RetinaNet。\n' "$(date -Is)" | tee -a "$RETINA_LOG" "$QUEUE_LOG"
"$PYTHON_BIN" -u -s -B script/train_torchvision.py \
    --config "$TORCHVISION_CONFIG" \
    --epochs "$EPOCHS" \
    --model retinanet_resnet50_fpn \
    --batch "$RETINA_BATCH" \
    --workers "$WORKERS" 2>&1 | tee -a "$RETINA_LOG" "$QUEUE_LOG"
printf '[%s] RetinaNet 已完成。\n' "$(date -Is)" | tee -a "$RETINA_LOG" "$QUEUE_LOG"

printf '[%s] 开始训练 RT-DETR-L。\n' "$(date -Is)" | tee -a "$RTDETR_LOG" "$QUEUE_LOG"
"$PYTHON_BIN" -u -s -B script/train_ultralytics.py \
    --config "$ULTRALYTICS_CONFIG" \
    --epochs "$EPOCHS" \
    --model rtdetr_l \
    --batch "$RTDETR_BATCH" \
    --workers "$WORKERS" 2>&1 | tee -a "$RTDETR_LOG" "$QUEUE_LOG"
printf '[%s] RT-DETR-L 已完成。\n' "$(date -Is)" | tee -a "$RTDETR_LOG" "$QUEUE_LOG"

printf '[%s] 三个模型均已完成。\n' "$(date -Is)" | tee -a "$QUEUE_LOG"
