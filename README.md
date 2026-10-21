# 目标检测多框架对比训练与评测

基于建筑裂缝数据集，对比 YOLOv8s、YOLOv11s、RT-DETR-L、Faster R-CNN 和 RetinaNet。

## 代码架构

```text
configs/                              # 数据预处理、训练和评测配置
script/                               # 数据处理、训练及评测入口
src/                                  # 数据集、模型和评测实现
datasets/CrackDataset/                 # 原始数据
datasets/processed/crack_detection/    # 处理后的图片、YOLO 标签和 COCO 标注
models/pretrained/                    # COCO 预训练权重
models/trained/                       # 训练完成的模型权重
outputs/training/                     # 训练日志、配置快照和曲线
outputs/evaluation/                   # 评测指标、预测结果和可视化
```

---
## 具体实现：

### 1. 创建环境

使用 Python 3.12，GPU 训练需要兼容 CUDA 12.8 的 NVIDIA 驱动。以下两种方式任选一种；后续命令均在项目根目录、激活所选环境后执行。

1. 基于 uv 的环境管理方案（需已安装 uv）：

```bash
uv sync --locked --managed-python
source .venv/bin/activate
```

2. 基于 conda 的环境管理方案：

```bash
conda create -n cv_task python=3.12 -y
conda activate cv_task
python -m pip install -r requirements.txt
```

需要下载时，可先在已配置代理的终端执行 `proxy_on`。

### 2. 数据预处理

数据集：[CrackDataset](https://github.com/guoguolord/CrackDataset)。原始数据放入 `datasets/CrackDataset/`，路径与处理参数见 `configs/preprocess.yaml`。

预处理将裂缝掩膜转换为检测框，检查重复与冲突样本，按来源分组划分训练、验证和测试集，同时导出 YOLO 与 COCO 格式。

```bash
# 完整预处理
python script/prepare_dataset.py --config configs/preprocess.yaml --stage all

# 校验已有处理结果；已有数据时可只执行此命令
python script/prepare_dataset.py --stage validate --output datasets/processed/crack_detection
```

### 3. 模型训练

训练前，将对应的 COCO 预训练权重放入 `models/pretrained/`，文件名见配置中的 `models` 字段。

**Ultralytics：YOLOv8s、YOLOv11s、RT-DETR-L**

```bash
python script/train_ultralytics.py \
  --config configs/ultralytics.yaml \
  --epochs 150 \
  --model yolov8s
```

将 `--model` 改为 `yolov11s` 或 `rtdetr_l` 即可切换模型。

**TorchVision：Faster R-CNN、RetinaNet（均使用 ResNet-50-FPN）**

```bash
python script/train_torchvision.py \
  --config configs/torchvision.yaml \
  --epochs 150 \
  --model fasterrcnn_resnet50_fpn
```

训练 RetinaNet 时，将 `--model` 改为 `retinanet_resnet50_fpn`。两种训练脚本均支持 `--batch`、`--workers` 覆盖配置；按显存和实际吞吐调整。

**断点续训：**

```bash
python script/train_ultralytics.py --resume outputs/training/yolov8s_formal/weights/last.pt
```

TorchVision 模型使用 `train_torchvision.py --resume <last.pt路径>`。`--resume` 单独使用，不与其他训练参数组合。

训练日志与曲线位于 `outputs/training/<运行名>/`；训练完成的权重归档到 `models/trained/<运行名>/`，其中 `best.pt` 用于最终评测。

**查看 TensorBoard：**

```bash
tensorboard --logdir outputs/training --host 127.0.0.1 --port 6006
```

本机浏览器访问 `http://127.0.0.1:6006`，选择 SCALARS 查看 loss；远程服务器需通过 SSH 端口转发访问。

### 4. 模型评估

加载指定权重，在验证集或测试集上评测，不更新模型参数。以下以 YOLOv8s 为例：

```bash
python script/evaluate.py \
  --config configs/ultralytics.yaml \
  --model yolov8s \
  --weights models/trained/yolov8s_formal/best.pt \
  --split test \
  --output outputs/evaluation/yolov8s \
  --benchmark
```

其他模型替换 `--model`、`--weights` 和输出目录；Faster R-CNN、RetinaNet 使用 `configs/torchvision.yaml`。`--split` 支持 `val`、`test`；`--benchmark` 额外测量参数量、GFLOPs、单张时延和 FPS，使用固定验证样本进行性能测试。

uv 环境下，可执行 `bash script/evaluate_all.sh` 连续评测五个模型，需先准备好全部权重。

| 输出文件 | 用途 |
| --- | --- |
| `metrics.json` | Precision、Recall、mAP@0.5、mAP@0.5:0.95，以及 TP、FP、FN |
| `pr_curve.png` | IoU＝0.5 时的 Precision–Recall 曲线 |
| `previews/` | 真实框为绿色，预测框为橙色，数字为预测置信度 |
| `predictions_coco.json` | 全部评测图片的预测框，格式为 COCO 的 `[x, y, 宽, 高]` |
| `evaluation_summary.json` | 权重、配置、评测协议及结果的完整记录 |
| `benchmark.json` | 性能测量详情，仅启用 `--benchmark` 时生成 |
| `summary.csv` | 检测与性能指标汇总，便于制作对比表，仅启用 `--benchmark` 时生成 |

指标基于所选划分的全部图片计算，预览只展示部分样本。横向对比前统一输入尺寸、评测阈值和测速硬件，不在测试集上调参。
