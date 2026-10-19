# 裂缝目标检测多架构对比

已完成数据预处理，以及历史 YOLOv8s 的 150 轮正式训练、冻结权重选择和测试集评测。
目前只有这一项正式模型结果，不是五模型架构优劣结论。代码不导入或调用 `tutorial/` 的函数，原始数据不修改、不删除。

## 环境与目录

所有预处理、训练、评测和检查均使用 `cv_task` Conda 环境中的解释器：

```text
/home/long753/miniconda3/envs/cv_task/bin/python -s -B
```

不使用系统 Python。项目仅保留一份 `requirements.txt`，覆盖预处理、训练、评测和 TensorBoard；代码不自动安装或升级包。
历史环境快照中的主要版本为 PyTorch `2.8.0+cu128`、torchvision `0.23.0+cu128`、Ultralytics `8.3.221`、pycocotools `2.0.10`、TensorBoard `2.21.0`；历史正式运行的环境审计记录保留在其 `audit/` 中。

### 服务器环境安装

统一依赖面向 Linux x86_64、Python 3.12 和 NVIDIA GPU，沿用当前 PyTorch CUDA 12.8 基线。
先用 `nvidia-smi` 检查服务器 GPU 与驱动；驱动需要兼容所选 CUDA wheel。
PyTorch wheel 自带 CUDA 运行库，不要求另外安装同版本的系统 CUDA Toolkit。

```bash
conda create -n cv_task python=3.12 -y
conda activate cv_task
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda); print('CUDA 可用:', torch.cuda.is_available())"
```

需要代理时，在安装前配置服务器可用的代理；只有服务器也配置了 `proxy_on` 命令时才直接执行它，
不要把本机的 `127.0.0.1` 代理地址直接当作服务器上的代理。
本机 OpenCV 来自 Conda，版本号为 `5.0.0`；统一文件使用 PyPI 对应的 `opencv-python==5.0.0.93` wheel，
不同时安装另一份 `opencv-python-headless`。Ubuntu/Debian 精简镜像若缺少 `libGL.so.1` 或 GLib，
需由服务器管理员安装 `libgl1`、`libglib2.0-0` 系统运行库。
这里只锁定项目直接依赖，完整历史环境快照仍保存在正式实验的 `audit/`，不把本机所有包搬到服务器。

```text
script/                       # 中文调用说明与命令行入口
├── prepare_dataset.py
├── train_ultralytics.py      # YOLOv8s / YOLO11s 共用训练入口和训练逻辑
└── evaluate.py               # 冻结权重的统一评测入口
configs/
├── preprocess.yaml
└── train_ultralytics.yaml    # 两个 YOLO 模型共用的正式训练配置
src/
├── config.py                 # 实验配置读取及路径解析
├── data/
│   ├── preprocessor.py       # 掩码审计、框提取、分组与划分
│   └── dataset_io.py         # YOLO/COCO 导出及独立校验
└── evaluation/
    ├── metrics.py            # 模型无关的 COCO AP 和固定阈值 P/R
    ├── yolo.py               # YOLO 推理、指标、可视化的共用流程
    └── benchmark.py          # YOLO 参数量、计算量与端到端时延
tests/                        # 保留全部既有行为回归，按职责分类
├── data/test_preprocessing.py
└── evaluation/
    ├── test_metrics.py
    └── test_yolo.py
models/
├── pretrained/               # 训练前的预训练权重
└── trained/yolov8s_formal/   # 历史 YOLOv8s 正式模型，含 best/last/epoch*.pt
outputs/training/yolov8s_formal/
├── validation/               # 历史统一验证结果
├── test/                     # 历史冻结测试结果
├── plots/                    # 历史 Loss、原生 PR、混淆矩阵、batch 预览
├── tensorboard/              # 历史 TensorBoard 事件
├── logs/                     # 历史 train.log、test.log、tensorboard.log
├── audit/                    # 历史冻结协议、依赖/输入快照、目录迁移记录
├── results.csv
├── training_state.json       # 仅历史运行元数据
├── experiment_summary.json
├── args.yaml
├── source_config.yaml        # 仅历史运行配置副本
└── resolved_config.yaml      # 仅历史解析配置副本
```

当前训练逻辑只位于 `script/train_ultralytics.py`；不存在活动的模型专用训练入口或 `src.training` 用法。
训练后的统一验证和独立评测均调用 `src.evaluation.yolo.evaluate_checkpoint`。其他架构需要自己的加载、预处理和推理适配，转换为标准 COCO 预测后，可复用 `src.evaluation.metrics.evaluate_predictions`。

`configs/train_ultralytics.yaml` 顶层的 `models` 映射维护 `yolov8s`、`yolov11s` 到各自预训练权重的相对路径；`data`、`project`、`trained_models` 也在顶层。配置路径相对于 YAML 文件解析，读取后这些路径为绝对路径；命令行文件路径相对于当前工作目录解析。`training`、`evaluation`、`benchmark` 三段保留正式基线的训练、统一评测和性能测量设置。配置不再包含或要求 `run_type`、单一 `model`、`name`。

新运行的目录由 `project/<实际运行名>/` 决定，模型归档根目录由 `trained_models/<实际运行名>/` 决定。运行目录包含如下当前产物：

```text
outputs/training/<实际运行名>/
├── data_view/                # 仅本次训练复制的 train/ 与 val/；不复制 test/
├── validation/               # 训练完成后的统一验证
├── plots/                    # 训练图片、原生图表等图片产物
├── tensorboard/              # 原生 Ultralytics TensorBoard 事件
├── args.yaml                 # 原生 Ultralytics 参数记录
├── results.csv               # 原生 Ultralytics 训练记录
├── config.yaml               # 含 model_name、name 基名和轮数覆盖值的解析配置
└── experiment_summary.json   # 简洁引用模型、运行目录、checkpoint、指标和验证结果
models/trained/<实际运行名>/
└── best.pt、last.pt、epoch*.pt 等整个原生 weights/ 目录内容
```

完成时整个原生 `weights/` 归档到 `trained_models/<实际运行名>/`，图片归入 `plots/`，原生事件归入 `tensorboard/`。`experiment_summary.json` 记录模型、实际运行目录、最佳及最后 checkpoint、训练轮数、指标和验证摘要路径。`config.yaml` 中的 `name` 是运行名基名，实际递增名称见 `args.yaml` 和摘要中的 `run_dir`。新运行不生成自定义 `training_state.json`、`source_config.yaml` 或 `resolved_config.yaml`，也不通过 CSV 回读训练进度或维护自定义最佳 checkpoint 事件状态。

## 常用命令

以下命令在项目根目录执行。

```bash
# 对现有冻结数据进行独立校验，不重新生成数据
/home/long753/miniconda3/envs/cv_task/bin/python -s -B script/prepare_dataset.py \
  --stage validate --output datasets/processed/crack_detection

# 仅审计：目标目录须尚不存在
/home/long753/miniconda3/envs/cv_task/bin/python -s -B script/prepare_dataset.py \
  --stage audit --output datasets/processed/crack_audit

# 完整预处理：输出目录须尚不存在
/home/long753/miniconda3/envs/cv_task/bin/python -s -B script/prepare_dataset.py

# 使用默认配置启动新的 YOLOv8s 正式训练
/home/long753/miniconda3/envs/cv_task/bin/python -s -B script/train_ultralytics.py

# 使用同一配置启动新的 YOLO11s 正式训练
/home/long753/miniconda3/envs/cv_task/bin/python -s -B script/train_ultralytics.py \
  --model yolov11s

# 新训练时以命令行覆盖 YAML 中的轮数
/home/long753/miniconda3/envs/cv_task/bin/python -s -B script/train_ultralytics.py \
  --model yolov8s --epochs 150

# 只恢复一个未完成运行保留的原生 checkpoint；不能附加其他训练参数
/home/long753/miniconda3/envs/cv_task/bin/python -s -B script/train_ultralytics.py \
  --resume outputs/training/<未完成实际运行名>/weights/last.pt

# 使用冻结权重独立评测验证集，另写新目录，不覆盖正式结果
/home/long753/miniconda3/envs/cv_task/bin/python -s -B script/evaluate.py \
  --config configs/train_ultralytics.yaml \
  --weights models/trained/yolov8s_formal/best.pt \
  --split val --output outputs/evaluation/yolov8s_val

# 递归发现全部行为回归
/home/long753/miniconda3/envs/cv_task/bin/python -s -B -m unittest discover -s tests -v

# 查看某个运行的原生训练曲线；不需要安装 TensorFlow
/home/long753/miniconda3/envs/cv_task/bin/python -s -B -m tensorboard.main \
  --logdir outputs/training/<实际运行名> \
  --host 127.0.0.1 --port 6006 --load_fast=false
```

`train_ultralytics.py --config` 默认 `configs/train_ultralytics.yaml`。`--model` 只能是 `yolov8s` 或 `yolov11s`，默认 `yolov8s`；`--epochs` 只在新训练时覆盖 YAML 的 `training.epochs`。新运行名以 `<model>_formal` 为基础，由 Ultralytics 自动递增以避免覆盖已有运行；没有 `--name` 参数。

`--resume <checkpoint>` 必须单独使用，并直接执行原生 `YOLO(checkpoint).train(resume=True)`。它只能恢复仍未完成、其 `weights/` 尚在运行目录中的 checkpoint；完成运行的模型已经归档，不能继续训练。归档拒绝覆盖同名模型目录。

评测入口为 `script/evaluate.py`：`--weights`、`--output` 必填，`--split` 只能是 `val` 或 `test`（默认 `val`），`--benchmark` 可额外测量固定代表性验证样本上的性能。`validate` 不读取原始数据或预处理配置，必须指定 `--output`；预处理拒绝覆盖已有目标目录。

TensorBoard 地址为 <http://127.0.0.1:6006>。训练期间原生事件写在运行目录根层，结束后移入 `tensorboard/`；将 `--logdir` 指向运行目录即可递归发现事件。SCALARS 中可用 `(train|val)/.*loss` 过滤 Loss。历史 `yolov8s_formal` 事件的前 40 轮曾从真实 CSV 补录，后续由原生回调记录；补录事件的 wall time 是导入时间，因此查看这份历史完整曲线应使用 **STEP** 横轴。该历史事件不被重新生成或改写。

## YOLO11s 训练入口与历史流程验证

YOLO11s 使用统一入口，而不是独立脚本：

```bash
/home/long753/miniconda3/envs/cv_task/bin/python -s -B script/train_ultralytics.py \
  --config configs/train_ultralytics.yaml --model yolov11s
```

该模型在共用 YAML 的 `models.yolov11s` 中选择预训练权重。除模型权重和自动生成的运行名外，正式训练、验证及性能测量配置与 YOLOv8s 正式基线一致：150 轮、640×640、batch 2、AMP、AdamW、种子 42，使用同一冻结数据划分，不用测试集选权重。

官方权重名为 **`yolo11s.pt`**，没有 `v`。历史下载前已在终端执行 `proxy_on`，再从
[Ultralytics 官方发布地址](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt)
下载到 `models/pretrained/yolo11s.pt`，未安装或升级依赖。已实际加载并确认是 s 规格、COCO 80 类、包含 C3k2/C2PSA 的 YOLO11 架构；本地文件 SHA-256 为 `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`。

统一入口已完成本轮小样本验证，尚未启动新的正式 150 轮实验。以下仍为切换入口前的历史记录，不能作为模型正式精度或架构优劣结论：当时在 `cv_task` 环境中用 5% 训练数据（117 张）实际训练了 2 轮，并核对：

- AMP 与 GradScaler 正常启用，训练/验证 Loss 有限，骨干卷积参数确实更新。
- 4 个检查点完成历史归档，原生 TensorBoard 有第 1、2 轮记录，曲线和预测图已查看。
- 保存的最佳模型能够重载为单类 crack；274 张验证图的独立评测与训练后验证一致。
- 原 YOLOv8s 的 274 张验证图指标保持一致，既有正式结果和模型未改写。
- 当时的 21 项原有回归测试通过，现有 158 个包的版本未改变。

该历史小测试使用 `/tmp` 中的临时配置、数据副本、日志和权重，验证后已全部删除，没有向项目 `tests/` 新增测试文件，也没有在正式 `outputs/` 或 `models/trained/` 留下小测试模型。

### 本轮统一入口验证

全部检查在 `cv_task` 环境完成，没有新增永久测试文件或安装、升级依赖：

- 同一临时数据集含 24 张训练图、12 张验证图，分别通过 `--model yolov8s` 和 `--model yolov11s` 真实训练 2 轮。
- 两个模型均有真实骨干权重更新、有限 Loss，AMP 自检通过，原生检查点中的 GradScaler 状态有效。
- 每个模型的原生 4 个检查点完成归档，TensorBoard 有第 1、2 轮 Loss 记录，生成的曲线已查看；保存权重重载后的统一验证指标一致。
- 额外模拟第 1 轮保存后中断，再单独执行 `--resume`，成功完成第 2 轮并保留连续日志和 TensorBoard 步数。
- 使用新公共配置重评原正式模型的完整 274 张验证图，指标与既有结果一致；21 项原有回归测试通过。
- 6481 个受保护文件（既有正式模型、实验结果、预训练权重和冻结数据）内容未变，158 个环境包版本未变。

临时配置、数据副本、训练模型、日志、验证脚本和 AMP 辅助权重均已随 `/tmp` 测试目录删除。
上述仅验证统一入口的训练、续训和评测流程，不代表完整训练后的精度比较。
用户保留的 `configs/train_yolov8s_formal copy.yaml` 仅是备份，当前程序和命令不引用它。

Ultralytics 首次执行 AMP 自检时可能下载 `yolo11n.pt`，这是框架的辅助检查模型，不会改变
`--model` 选中的训练模型。如果需要下载，先在同一终端执行 `proxy_on`；本轮测试也按此执行。

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
