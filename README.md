# 裂缝目标检测多架构对比

仓库已保留 YOLOv8s 的 150 轮正式训练、冻结权重和测试集评测，以及 YOLO11s 的 150 轮正式训练和验证结果。新增 Faster R-CNN、RetinaNet、RT-DETR-L 训练与统一评测入口不代表三个新模型已完成正式实验，也不构成五架构优劣结论。既有产物不改写，代码不导入或调用 `tutorial/` 的函数，原始数据不修改、不删除。

## 环境与目录

项目使用 uv 管理 Python 和依赖，默认虚拟环境位于项目根目录的 `.venv/`。所有预处理、训练、评测和检查均在项目根目录执行：

```text
uv run --no-sync python ...
```

无需激活虚拟环境或 Conda 环境，也不直接使用系统 Python。日常命令使用 `uv run --no-sync python` 复用已准备好的项目环境，不安装、升级依赖或改写锁文件。

- `pyproject.toml`：声明 Python 兼容范围、直接依赖和包下载源，覆盖预处理、Ultralytics 训练、COCO 评测和 TensorBoard。
- `uv.lock`：记录完整依赖解析结果，由 uv 管理，不手动编辑。
- `.python-version`：记录本项目默认请求的 Python 版本。
- `.venv/`：本机实际运行环境，不提交到版本控制；前三个文件应提交，以便在其他机器重建环境。

历史环境快照中的主要版本为 PyTorch `2.8.0+cu128`、torchvision `0.23.0+cu128`、Ultralytics `8.3.221`、pycocotools `2.0.10`、TensorBoard `2.21.0`；历史正式运行的环境审计记录保留在其 `audit/` 中。

### 新建 uv 环境（本地或其他服务器）

统一依赖面向 Linux x86_64、Python 3.12 和 NVIDIA GPU，沿用当前 PyTorch CUDA 12.8 基线。服务器需先安装 uv；已有项目无需重新执行 `uv init` 或 `uv lock`。
先用 `nvidia-smi` 检查服务器 GPU 与驱动；驱动需要兼容所选 CUDA wheel。
PyTorch wheel 自带 CUDA 运行库，不要求另外安装同版本的系统 CUDA Toolkit。

```bash
nvidia-smi
# 使用 uv 管理的 Python，按已提交的项目配置与锁文件创建或同步 .venv
uv sync --locked --managed-python
uv pip check --python .venv/bin/python
uv run --no-sync python -c "import torch; print(torch.__version__, torch.version.cuda); print('CUDA 可用:', torch.cuda.is_available())"
```

需要代理时，在安装前配置服务器可用的代理；只有服务器也配置了 `proxy_on` 命令时才直接执行它，
不要把本机的 `127.0.0.1` 代理地址直接当作服务器上的代理。
OpenCV 使用 PyPI 的 `opencv-python==5.0.0.93` wheel，模块报告的 `cv2.__version__` 为 `5.0.0`；
不同时安装另一份 `opencv-python-headless`。Ubuntu/Debian 精简镜像若缺少 `libGL.so.1` 或 GLib，
需由服务器管理员安装 `libgl1`、`libglib2.0-0` 系统运行库。
`torch` 和 `torchvision` 通过 `pyproject.toml` 中的 `tool.uv.sources` 显式绑定到 PyTorch CUDA 12.8 官方索引；该索引设置 `explicit = true`，其他包默认从 PyPI 获取。完整传递依赖由 `uv.lock` 锁定，不再维护独立的 `requirements.txt`。历史实验环境快照仍保存在正式实验的 `audit/`，不随本次环境迁移改写，也不把旧环境中的所有包复制到新环境。

### 当前远程服务器：Conda、数据盘与手动训练

上述 uv 用法保留给本地开发和需要新建环境的情形；当前已部署的服务器在 `/root/cv_task` 使用既有的 Conda 环境，不执行 `uv sync`，也不由训练脚本安装依赖：

```bash
cd /root/cv_task
conda activate cv_task
bash script/train_three_models.sh
```

只有在终端中手动执行上述 `bash` 命令时，脚本才会依次启动 Faster R-CNN、RetinaNet、RT-DETR-L；任一训练失败会停止后续模型并保留已有 checkpoint。脚本使用已激活环境中的 `python`（可用 `PYTHON_BIN` 覆盖），默认每个模型训练 150 轮，batch 分别为 8、8、16，`WORKERS=4`。可临时指定 `EPOCHS`、`FASTER_BATCH`、`RETINA_BATCH`、`RTDETR_BATCH`、`WORKERS`，并用 `TORCHVISION_CONFIG`、`ULTRALYTICS_CONFIG` 覆盖 YAML；不会自动启动 tmux 或 TensorBoard。连续训练入口为 `script/train_three_models.sh`，不是旧的根目录占位脚本。

默认值经过当前 RTX 4080 SUPER（32,760 MiB 显存、16 核 CPU 配额）实测校准：从原训练集抽取 128 张图（包括目标数最多的样本）、验证集抽取 24 张图，每次训练 2 轮，对比 batch 8/16 和 workers 4/8。下表吞吐为第 2 轮纯训练段（含数据加载），显存为两轮训练段的最大 PyTorch reserved；不是完整正式训练或推理 FPS。

| 模型 | 默认 batch | workers | 短测训练吞吐（图/秒） | 峰值 reserved（GiB） |
| --- | ---: | ---: | ---: | ---: |
| Faster R-CNN R50-FPN | 8 | 4 | 38.71 | 4.24 |
| RetinaNet R50-FPN | 8 | 4 | 39.46 | 3.49 |
| RT-DETR-L | 16 | 4 | 31.76 | 19.96 |

本次短测中，两个 torchvision 模型的 batch 8 比 16 更快；RT-DETR-L 则是 16 更快。不要仅为占满显存增大 batch。脚本默认值面向这台服务器，本地小显存 GPU 请覆盖 batch；两个 YAML 的本地默认值保持不变。脚本同时将主进程 `OMP_NUM_THREADS`、`MKL_NUM_THREADS` 默认设为 4，可用同名环境变量覆盖。

部署验证全部在服务器既有 `cv_task` Conda 环境中完成：25 项测试通过；真实 Bash 队列完成三个模型的两轮短训练，均产生非零骨干权重更新；重新加载后的统一验证指标与训练结束时一致；三个模型均完成保存点中断及 `--resume` 续训；失败时队列停止、不启动后续模型。另已实际运行三个模型的 `--benchmark`，确认参数量、GFLOPs、时延和 CSV 输出正常；运行条件描述使用中性文本并附实际 GPU 快照，不把服务器结果误标为本地桌面。

TensorBoard 的 SCALARS 页面已实际查看 loss 折线，HTTP 接口中三模型及续训运行均有第 1、2 轮的有限 loss 值。上述临时数据、检查点、日志、验证程序和临时 TensorBoard 服务均已清理，未启动正式训练；环境中的 60 个发行包版本保持不变。这些短测不是正式实验结果。

服务器迁移已完成，并已校验 13,266 个文件。`/root/cv_task/datasets`、`/root/cv_task/models` 和（原先不存在、现已新建的）`/root/cv_task/outputs` 分别软链到 `/root/autodl-tmp/cv_task/` 下的同名目录，因此默认相对路径无需变更，所有新训练输出（包括脚本默认的 `outputs/training/logs/<时间戳>/` 队列和单模型日志）均位于数据盘。`LOG_DIR` 可在手动启动前覆盖；每次启动都会创建新的时间戳日志目录，不覆盖此前日志。

迁移前的服务器代码保存在 `/root/autodl-tmp/cv_task/backups/server_code_20261020_204441.tar.gz`；服务器已有的 `requirements.txt` 未覆盖。数据盘不随系统镜像保存，重要数据和训练权重应另行备份。

如需查看 TensorBoard，在服务器的另一个 tmux 窗口中手动执行；训练脚本不会在后台启动它：

```bash
cd /root/cv_task
conda activate cv_task
tensorboard --logdir outputs/training --host 127.0.0.1 --port 6006
```

在本地终端建立转发，并保持连接：

```bash
ssh -p 18396 -N -L 16006:127.0.0.1:6006 root@connect.cqa1.seetacloud.com
```

随后在本地浏览器打开 `http://127.0.0.1:16006`。服务器 TensorBoard 只监听回环地址，不直接暴露公网端口。


```text
script/                       # 中文调用说明与命令行入口
├── prepare_dataset.py
├── train_ultralytics.py      # YOLOv8s / YOLO11s / RT-DETR-L
├── train_torchvision.py      # Faster R-CNN / RetinaNet 原生训练循环
├── train_three_models.sh     # 服务器手动连续训练三个新增模型
└── evaluate.py               # 五架构冻结权重统一评测
configs/
├── preprocess.yaml
├── train_ultralytics.yaml    # 三个 Ultralytics 模型的共享配置和 RT-DETR 覆盖
└── train_torchvision.yaml    # 两个 TorchVision 模型的原生训练配置
src/
├── config.py                 # 实验配置读取及路径解析
├── data/
│   ├── preprocessor.py       # 掩码审计、框提取、分组与划分
│   ├── dataset_io.py         # YOLO/COCO 导出及独立校验
│   └── detection_dataset.py  # 原尺寸 TorchVision 检测数据集
├── models/
│   ├── ultralytics.py        # YOLO / RT-DETR 工厂
│   └── torchvision.py        # Faster R-CNN / RetinaNet 工厂
└── evaluation/
    ├── metrics.py            # 模型无关的 COCO AP 和固定阈值 P/R
    ├── detection.py          # 五架构预测、指标、可视化的共用流程
    ├── torchvision.py        # TorchVision 原图推理适配
    ├── prediction.py         # 原生 xyxy 到统一 COCO xywh 的小型转换
    └── benchmark.py          # 参数量、计算量与端到端时延
tests/
└── evaluation/
    ├── test_metrics.py
    └── test_detection.py
models/trained/yolov8s_formal/ # 历史 YOLOv8s 正式模型，含 best/last/epoch*.pt
outputs/training/yolov8s_formal/ # 历史训练记录和冻结评测产物
```

活动训练入口分别是 `script/train_ultralytics.py` 和 `script/train_torchvision.py`，没有 `src.training` 或教程代码依赖。训练后统一验证、独立评测都调用 `src.evaluation.detection.evaluate_checkpoint`；TorchVision 每轮验证直接调用 `evaluate_model`，不写临时 checkpoint。

`configs/train_ultralytics.yaml` 的 `models` 包含 `yolov8s`、`yolov11s`、`rtdetr_l`，公共 `training` 后的 `model_overrides.rtdetr_l` 仅覆盖 RT-DETR 不支持或不适宜的选项。`configs/train_torchvision.yaml` 的 `models` 包含 `fasterrcnn_resnet50_fpn`、`retinanet_resnet50_fpn`，其值是本地官方 COCO state_dict。两个配置均以 YAML 文件为基准解析 `models`、`data`、`project`、`trained_models` 路径；命令行文件路径相对于当前工作目录解析。

每个新运行保存在 `outputs/training/<实际运行名>/`，解析后的 `config.yaml`、`results.csv`、TensorBoard、曲线和验证产物留在该目录。Ultralytics 另有其原生图片和事件整理；TorchVision 的 `results.csv` 记录总训练损失、各 `train/*` 分项、P/R、mAP50 与 mAP50-95。两条训练路径完成后均将整个 `weights/` 目录移动到 `models/trained/<实际运行名>/`，摘要中的 best/last checkpoint 都指向归档路径；不保留第二份副本。

## 常用命令

以下命令在项目根目录执行。首次准备环境可按上节运行 `uv sync --locked --managed-python`；日常任务一律使用 `uv run --no-sync python`，不在命令中安装或升级依赖。

```bash
# 对现有冻结数据进行独立校验，不重新生成数据
uv run --no-sync python script/prepare_dataset.py \
  --stage validate --output datasets/processed/crack_detection

# Ultralytics：YOLOv8s、YOLO11s、RT-DETR-L
uv run --no-sync python script/train_ultralytics.py --model yolov8s
uv run --no-sync python script/train_ultralytics.py --model yolov11s
uv run --no-sync python script/train_ultralytics.py --model rtdetr_l

# TorchVision：Faster R-CNN、RetinaNet
uv run --no-sync python script/train_torchvision.py --model fasterrcnn_resnet50_fpn
uv run --no-sync python script/train_torchvision.py --model retinanet_resnet50_fpn

# 新训练可覆盖轮数；恢复只能给对应运行的 last.pt，不能附带其他训练参数
uv run --no-sync python script/train_torchvision.py \
  --model retinanet_resnet50_fpn --epochs 20
uv run --no-sync python script/train_ultralytics.py \
  --resume outputs/training/<未完成实际运行名>/weights/last.pt

# RT-DETR-L 冻结权重评测和端到端基准
uv run --no-sync python script/evaluate.py \
  --config configs/train_ultralytics.yaml --model rtdetr_l \
  --weights models/trained/rtdetr_l_formal/best.pt \
  --split val --output outputs/evaluation/rtdetr_l_val --benchmark

# Faster R-CNN 冻结权重评测
uv run --no-sync python script/evaluate.py \
  --config configs/train_torchvision.yaml --model fasterrcnn_resnet50_fpn \
  --weights models/trained/fasterrcnn_resnet50_fpn_formal/best.pt \
  --split val --output outputs/evaluation/fasterrcnn_val
```

### 五架构训练与评测约定

Ultralytics 入口的 `--model` 选择 `yolov8s`、`yolov11s` 或 `rtdetr_l`，默认 `yolov8s`；新运行先合并公共 `training` 和模型专属覆盖，再以 `--epochs` 作最终覆盖，最后将有效配置保存到运行的 `config.yaml`。YOLO 保留 AMP、AdamW、`nbs=16` 的框架累计 batch 语义；RT-DETR 的覆盖使用 `amp: false`、`deterministic: false`，避开其匈牙利匹配和 `grid_sample` 的已知限制。RT-DETR 预测使用原生 query 解码，不追加 NMS，并统一按 `score_floor` 和 `max_detections` 截断。

TorchVision 入口的 `--model` 选择 `fasterrcnn_resnet50_fpn` 或 `retinanet_resnet50_fpn`，默认 Faster R-CNN。它从本地官方 COCO state_dict 加载后替换单类输出，保留 FrozenBatchNorm2d、COCO 的 `eps=0` 和三层可训练骨干；RetinaNet 还保留原分类塔，只重新学习类别输出。训练使用原生 PyTorch 循环、SGD、余弦调度、CUDA AMP 和 `nbs / batch` 梯度累积；每轮直接验证内存模型，不为验证复制网络或写临时权重，完成指标和日志后保存 last，验证 mAP50-95 改善时保存 best。分项 Loss、总 Loss、学习率和验证指标写入 `results.csv`、TensorBoard，并生成曲线。验证没有 `val_loss`：本项目用推理模式下的统一 COCO 指标选 best，不切回训练模式伪造 validation loss。

TorchVision 也支持同文件内的 `model_overrides`。Faster R-CNN 专属覆盖为 `deterministic: false`：当前 torchvision 开启确定性算法会把 CUDA RoIAlign 切换到编译回退，已在本机 8GB GPU 测试中触发显存不足。该覆盖保留 640×640、batch 2、随机种子 42 和 AMP，使用常规 CUDA RoIAlign，不保证逐位确定性。RetinaNet 保留确定性设置，并配置 CuBLAS 所需工作区；这些差异随每次有效配置保存。

TorchVision 验证与独立重载评测统一使用 FP32、关闭 cuDNN benchmark 和确定性算法选择，避免评测结果受训练遗留的卷积后端状态影响；每个训练 epoch 开始前再恢复所选模型的训练确定性设置。种子初始化只在创建/恢复运行时执行，不在每轮验证后重置数据增强的随机序列。

TorchVision 数据集读取原始尺寸 RGB `[0,1]` 张量和原图 `xyxy` 框；其 `GeneralizedRCNNTransform` 固定拉伸为 640×640、采用 ImageNet 均值/标准差归一化，并将输出框还原到原图。评测适配层不再做第二次反变换。冻结 COCO 的类别为 `category_id=1`；YOLO/RT-DETR 内部 crack 类别为 0，Faster R-CNN 前景标签为 1，RetinaNet sigmoid 前景标签为 0，导出时全部明确映射回 COCO 1。

TorchVision checkpoint 保存 `model_name`、`model` state_dict、optimizer、scheduler、GradScaler、从 0 开始的已完成 epoch、`best_map` 和 JSON 可序列化配置；恢复使用运行目录的 `config.yaml`。评测脚本的 `--model` 是可选的：TorchVision 必须与 checkpoint 内 `model_name` 一致，省略时从该元数据识别；Ultralytics 同样从 checkpoint 架构/元数据识别，无法识别时必须显式传入。`--split` 仅接受 `val` 或 `test`，`--output` 必填；评测从不使用测试集调阈值。

`--benchmark` 使用同一权重、6 张固定验证原图、batch 1、FP32、50 次预热、300 次轮转计时，并在每次测量前后同步 CUDA。计时从内存原始 BGR 到 CPU 最终检测，包含预处理、H2D、网络、解码/后处理、坐标还原、D2H，排除权重加载、图像 I/O、预热和绘图。临时 smoke 只在临时配置中减少 warmup/iterations。五个模型的时延均采用 `confidence=0.25` 工作点，AP 评测另用 `score_floor=0.001`，不混用两种阈值。YOLO 保留历史 THOP 的 `2×MACs` GFLOPs 口径；TorchVision 与 RT-DETR 使用 `torch.profiler(with_flops=True)` 已报告算子的说明性合计，并列出未报告 FLOPs 的算子。proposal、NMS、ROI、attention 或自定义算子可能漏计，不能把这些不同计数器的数值直接作为完整跨框架 FLOPs 排序。

### 新增三个模型的流程验收

已在现有 uv 环境完成真实 GPU 小测试，不是三个新模型的正式精度结果：

- 使用同一跨来源临时子集：24 张训练图、12 张验证图，640×640、batch 2，每个模型训练 2 轮。
- Faster R-CNN、RetinaNet、RT-DETR-L 都有真实骨干参数更新、有限 Loss、检查点和第 1/2 轮 TensorBoard 记录；生成的曲线已查看。
- 三个模型的最佳权重重载后，统一 P/R、AP、TP/FP/FN 与其训练后验证一致，每图检测上限为 100。
- 三个模型分别模拟第 1 轮保存后中断，再通过各自公开的 `--resume` 完成第 2 轮；优化器、调度器、AMP（启用时）和日志延续路径已验证。续训不保证与从不中断的训练逐位一致。
- 三个模型的 `--benchmark` 路径实跑通过；临时配置使用 1 次预热、3 次测量，仅验证统计路径，不作为正式速度对比。
- YOLOv8s、YOLO11s 各自完整 274 张验证图的指标保持一致。最终 25 项回归测试通过，其中 4 项覆盖新数据集的标签、空目标、翻转和坐标还原。
- 11762 个受保护文件（既有正式实验、模型、冻结数据及预训练权重等）校验一致；uv 的 58 个包版本、`pyproject.toml` 和 `uv.lock` 未改变。

临时数据、配置、训练权重、脚本、日志和报告已随 `/tmp` 测试目录删除；保留必要的行为回归测试。
本次没有启动三个新增模型的 150 轮正式实验，没有在正式训练输出目录留下烟测模型。

所需官方预训练权重已通过 `proxy_on` 下载并校验，保存在下列位置；迁移服务器时也需同步这些文件。
TorchVision 两个文件的 SHA-256 已匹配官方 URL 的哈希前缀；RT-DETR 已实际加载检查其 RTDETRDecoder 和 COCO 80 类结构。

| 本地文件 | 官方来源 |
|---|---|
| `models/pretrained/fasterrcnn_resnet50_fpn_coco.pth` | [TorchVision COCO 权重](https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth) |
| `models/pretrained/retinanet_resnet50_fpn_coco.pth` | [TorchVision COCO 权重](https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth) |
| `models/pretrained/rtdetr-l.pt` | [Ultralytics 官方权重](https://github.com/ultralytics/assets/releases/download/v8.3.0/rtdetr-l.pt) |

## 数据与预处理

输入为 `datasets/CrackDataset/`，含 `crack500`、`AsphaltCrack300`、`CFD`。
这些数据包含道路裂缝，不能直接声称已验证建筑墙面的泛化能力。

1. 严格配对同名图像与掩码；缺失、重名、无法解码、尺寸不同、未知编码均报错，不模糊匹配或缩放修补标注。
2. 读取原始像素，不自动执行 EXIF 旋转，保留原分辨率，不生成增强副本或新增切片。
3. 单通道/1-bit 按配置解析；`P` 模式按每张图自己的调色板白色前景、黑色背景解析，并记录映射。
   AsphaltCrack300 的白色实际为索引 0，CFD 附加图为索引 1，不能统一用索引大于零识别前景。
4. 八邻域连通域生成半开区间框。冻结设置为 `min_component_area=1`、`closing_iterations=0`；面积阈值是前景像素数，不是 bbox 面积。
5. 相交裂缝可能合并，断裂裂缝可能产生多个框；这是掩码派生标注，不是人工实例级真值。合法空掩码保留为空目标图像。

### 防泄漏划分

- Crack500 去掉文件名最后两个坐标字段作为原图 ID；AsphaltCrack300 按切片编号前的三位原图编号分组。
- CFD 原图与 `_N`、`__N`、`addN`、`adjustN`、`subN` 派生样本归同组；未知命名须明确覆盖或隔离。
- 原输入目录存在 107 个交叉原图组，因此重新按来源分层、原图分组，用种子 42 划分约 70%/10%/20%，不拆组。
- RGB 完全重复的图像合并来源组，防止重复桥接；相同图像却掩码冲突的样本必须明确隔离。
- 已知 CFD 离线增强仅保留在训练组，验证/测试组增强不导出；pHash 距离不超过 4 仅报告相似候选，不证明同源或不存在全部近重复。
- 用户确认隔离 32 个样本：AsphaltCrack300 `001/002` 的 14 对相同图像、不同掩码记录，以及 CFD `img1`–`img4` 来源不明附加图。

实际处理 3403 对文件，隔离 32 个异常样本，另有 180 个验证/测试组增强样本不导出；原始文件未删除。

| 集合 | 图像 | 原图分组 | 检测框 |
|---|---:|---:|---:|
| train | 2342 | 325 | 5035 |
| val | 274 | 47 | 518 |
| test | 575 | 94 | 1214 |

冻结导出在 `datasets/processed/crack_detection/`：

```text
images/{train,val,test}/             # 原图字节副本
labels/{train,val,test}/             # YOLO TXT，空目标为空文件
annotations/instances_{split}.json   # COCO
splits/{train,val,test}.txt          # sample_id 清单
manifest.jsonl                      # 包括未导出记录
data.yaml                           # 数据配置
preprocess_state.json               # 完成状态
audit/{config.json,report.json,previews/}
```

YOLO 类别是 0，中心与宽高归一化；COCO 类别是 1，bbox 为原图像素 `[x_min,y_min,width,height]`，area 为框面积。
COCO `file_name` 相对于 `images/`，数据 YAML 不固定绝对根路径。Manifest 的原始来源绝对路径只用于追溯。
独立校验不依赖原始来源仍可访问，检查图包含原图、掩码/派生框、重新读取的 YOLO 框及 COCO 框。

完整导出先写同级临时目录，通过独立校验和检查图生成后才发布；失败保留审计报告。

### 预处理输出字段

| 摘要字段 | 含义 |
|---|---|
| `status` | `audit_complete` 为审计完成，`complete` 为完整导出完成 |
| `runtime` | 实际解释器与图像处理依赖版本，不是训练参数 |
| `total_pairs` | 已读取图像—掩码对数，包括未导出样本 |
| `splits` | 各集合的 images、groups、instances、empty_images、sources |
| `exclusions` | 隔离、验证/测试组增强、完全重复等未导出原因 |
| `output_dir`、`audit_report`、`previews` | 输出位置、审计报告、检查图数量；检查图数量不是训练图片数 |

`validate` 的 `records/active_records/excluded_records` 分别为全部/导出/未导出记录数，
`annotations` 为框数，`empty_records` 为无框记录；各集合的 records/images/labels 分别统计记录、图片、标签文件。
一张图或一个标签文件可以包含多个框。

## 正式训练与评测协议

历史 YOLOv8s 正式实验从 `models/pretrained/yolov8s.pt` 的 COCO 预训练开始，训练 150 轮：
640×640、物理 batch 2、AMP、AdamW、初始学习率 0.001、余弦学习率、种子 42。
`nbs=16` 控制框架累计 batch，warmup 时累计步数变化，不能理解为每一步严格等效于 batch 16。
关闭早停，仅轻度亮度变化和水平翻转，不使用 Mosaic。当前共用 YAML 在 `training`、`evaluation`、`benchmark` 中保留同一正式协议；每次新运行还会保存其解析后的 `config.yaml`。

训练不直接使用冻结数据目录：Ultralytics 即使 `cache=False` 也会写缓存，因此新运行先复制 train/val 到运行内 `data_view/`，不复制测试集。
路径列表会被固定版本合成大 batch，故推理使用目录、`stream=True`、`batch=1`、`rect=False`。
从原图 `xyxy` 转 COCO 左上角 `xywh`，不误用 Ultralytics 的中心坐标 `boxes.xywh`。

- 历史 YOLOv8s 的最佳权重按原生验证集 mAP@0.5:0.95 选取，同 fitness 时采用最新保存事件；未从低精度 CSV 反推最佳 epoch。新运行直接采用原生 Ultralytics 的最佳 checkpoint，不维护自定义事件状态。
- AP 使用 COCO bbox、IoU 0.50:0.05:0.95、分数下限 0.001、每图最多 100 框，NMS IoU 0.70。
- P/R 使用置信度 0.25、匹配 IoU 0.50，按分数一对一匹配；重复检出为 FP，未匹配 GT 为 FN。
- 裁剪回原图后只剔除宽或高不大于零的空框，记录数量，保留所有正面积小框。指标层仍拒绝无效框。
- 冻结评测 GT 为非 crowd、非 ignore 的单类裂缝框，不支持的 crowd/ignore 会报错。
- 框架原生训练曲线与统一评测口径不同，不在同一对比表混用。测试不参与模型选择或阈值搜索。

### 已完成结果

正式实验完成 150/150 轮，在第 40 轮后因切换 tmux/TensorBoard 原生续训。
最终选择第 25 轮 `best.pt`，不是末轮 `last.pt`。最佳权重 SHA-256：
`d1790586c7ff495fa5156aeb610b63cf0682fa121fa0fe5c885ba2e86664639a`。

| 集合 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| val | 42.50% | 42.66% | 37.38% | 19.87% | 221 | 299 | 297 |
| test | 41.59% | 38.71% | 32.45% | 17.88% | 470 | 660 | 744 |

测试包含 575 图、1214 GT；低分数下限下导出 32965 个预测，剔除 1 个空框。这不等于 P/R 工作点的检出数。

| 模型 | Params / M | GFLOPs | 平均时延 / ms | 中位数 / ms | P95 / ms | 标准差 / ms | 顺序 FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8s，单类 | 11.136 | 28.642 | 11.84 | 10.53 | 19.23 | 3.85 | 84.49 |

参数量为未融合模型总参数；GFLOPs 是 THOP 在 `[1,3,640,640]` 的 `2×MACs/1e9` 近似计数，不包括 NMS/预处理。
时延在 RTX 4060 Laptop 上用固定六张验证图片轮流预热 50 次、测量 300 次，batch 1、FP32 张量，逐张前后同步 CUDA。
包含预处理、H2D、网络、解码、NMS、还原坐标、D2H 及空框过滤，排除模型加载、磁盘 I/O、预热、绘图。
CUDA matmul TF32 关闭，cuDNN TF32 开启；保留桌面应用，GPU 快照温度 67°C。
这不是 575 张测试图的平均时延，也不能直接与服务器或纯网络耗时比较；五架构比较须在同一硬件重新测量。

### 主要问题

| 测试来源 | 图片 | GT | Precision | Recall | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| AsphaltCrack300 | 155 | 216 | 28.93% | 68.98% | 366 | 67 |
| CFD | 24 | 69 | 38.46% | 21.74% | 24 | 54 |
| crack500 | 396 | 929 | 53.13% | 32.94% | 270 | 623 |

原图 bbox 面积 `<32×32` 像素的小框共 320 个，在全体 GT 的匹配中只命中 5 个、漏检 315 个，工作点召回率 1.56%。
这不是 AP_small，也没有删去大框重新匹配；极小连通域包含在分母中，需要在训练/验证集审查实例粒度。

六张固定预览图位于 `test/previews/`，GT 绿框、预测橙框：
Asphalt `010_03` 出现过大框覆盖多个 GT（TP=0/FP=1/FN=3）；CFD `001` 有长框与多 GT 粒度不一致（1/1/1），
CFD `005` 的低对比度细长裂缝漏检；crack500 `081113_1281_361` 有边缘小框和定位偏差（0/2/2），
相邻 `081113_1281_721` 则命中两框但多检一框（2/1/0）。完整逐图统计在 `test/error_analysis.json`。
这不是场景分层抽样，没有强反光/遮挡标签，不把亮色石粒直接判为强反光或给出相关定量结论。

`plots/results.png` 显示训练 Loss 下降、验证 Loss 后期回升且原生 Recall 下降，存在训练/验证分化。
应审查小框和掩码连通域标注粒度，而不是根据测试集调阈值或直接增加轮数。
已从保存预测复算精度与计数，并核对 300 次时延的均值、P95 和 FPS，未更换模型或重跑全测试集推理。

## 清理与归档说明

- 最终推理模型在 `models/trained/yolov8s_formal/best.pt`。全部 8 个正式权重均保留，迁移前后 SHA-256 一致；`models/pretrained/` 没有改动。
- 正式指标、预测、曲线、日志和 TensorBoard 事件保留；模型迁移不是重新训练。
- 清理了旧 pilot/断点烟测、四份重复数据副本、TensorBoard wheel 下载缓存，以及源码/测试的生成缓存。
  删除副本前核对 5235 个冻结输入文件及每份副本的 5232 个输入文件，确认可由冻结数据重建；删除临时文件逻辑大小约 860 MiB。
- `outputs` 现在主要保留正式实验及项目内运行缓存；新训练不再在根目录散落随机数据副本。
- `audit/cleanup_manifest.json` 记录删除列表、路径迁移、原始权重哈希和保护清单。
- 迁移前的元数据和旧 README 保存在 `audit/before_cleanup.zip`。当前摘要、CSV 和性能记录中的有效模型路径已更新。
- `args.yaml`、`source_config.yaml`、`resolved_config.yaml` 是历史训练记录，原件中的旧数据副本路径不再可用；不要把已完成实验的历史参数直接当作恢复指令。
  当前完成状态明确不支持恢复，原始数据仍在 `datasets/processed/crack_detection/`。历史哈希不为迎合新目录而重算。
- `test/metrics.json`、预测 JSON、PR 图及 TensorBoard 事件的内容哈希未因清理改变。

`test/summary.csv` 是可加入多模型对比表的完整指标行；`benchmark.json` 保留原始计时，`evaluation_summary.json` 保留评测配置和模型哈希。
历史 pilot 仅用于调通流程，清理后不再作为正式结果目录对外引用。

## 历史目录整理的验收记录

以下为历史目录整理时在 `cv_task` 环境完成的检查记录，没有新增或升级依赖；不代表统一入口已完成本轮验证：

- 重新发现并通过 21 项回归测试；冻结数据独立校验通过，3191 张图、6767 个框。
- 在 `/tmp` 临时目录实跑两轮、5% 训练数据的中断与恢复，恢复 183 组优化器状态和 AMP scaler。
  未完成时 checkpoint 留在运行目录；完成后四个烟测 checkpoint、绘图及两轮事件分别归档，
  完成状态指向新模型位置；同名模型目录冲突在训练前拒绝。临时目录已清除。
- 使用迁移后的正式 `best.pt` 实际评测 274 张验证图，P/R、AP、TP/FP/FN 与保存结果一致；
  可选性能分支以六次计时实跑，只验证入口，不替换正式性能数据。
- 全部 8 个正式权重哈希一致；正式预测、指标、PR 和 TensorBoard 文件哈希保留，
  原训练 Loss 的 1–150 步仍完整。未重新训练正式实验或覆盖正式测试结论。

完整验收字段保存在 `audit/cleanup_manifest.json` 的 `verification` 中。

rsync -avP \
  -e "ssh -p 18396" \
  datasets/\
  root@connect.cqa1.seetacloud.com:/root/cv_task/datasets/