#!/usr/bin/env bash

# 遇到错误立即停止
set -euo pipefail

# 自动切换到项目根目录
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$PROJECT_ROOT"

# 1. 测评 yolov8s 模型
uv run --no-sync python script/evaluate.py \
  --config configs/ultralytics.yaml \
  --model yolov8s \
  --weights models/trained/yolov8s_formal/best.pt \
  --split test \
  --output outputs/evaluation/yolov8s --benchmark

# 2. 测评 yolov11s 模型
uv run --no-sync python script/evaluate.py \
  --config configs/ultralytics.yaml \
  --model yolov11s \
  --weights models/trained/yolov11s_formal/best.pt \
  --split test \
  --output outputs/evaluation/yolov11s --benchmark

# 3. 测评 rtdetr_l 模型
uv run --no-sync python script/evaluate.py \
  --config configs/ultralytics.yaml \
  --model rtdetr_l \
  --weights models/trained/rtdetr_l_formal/best.pt \
  --split test \
  --output outputs/evaluation/rtdetr_l --benchmark

# 4. 测评 retinanet_resnet50_fpn 模型
uv run --no-sync python script/evaluate.py \
  --config configs/torchvision.yaml \
  --model retinanet_resnet50_fpn \
  --weights models/trained/retinanet_resnet50_fpn_formal/best.pt \
  --split test \
  --output outputs/evaluation/retinanet_resnet50_fpn --benchmark   

# 5. 测评 fasterrcnn_resnet50_fpn 模型
uv run --no-sync python script/evaluate.py \
  --config configs/torchvision.yaml \
  --model fasterrcnn_resnet50_fpn \
  --weights models/trained/fasterrcnn_resnet50_fpn_formal/best.pt \
  --split test \
  --output outputs/evaluation/fasterrcnn_resnet50_fpn --benchmark

echo "全部五个模型评估完成！"