# 裂缝目标检测多架构对比

当前实现数据预处理，以及 YOLOv8s 的短训练、权重重载、验证集推理与统一 COCO 评测闭环。
这还不是五模型正式对比实验；当前 YOLOv8s 只训练了两个 epoch，用于验证流程。

代码独立实现，不导入或调用 `tutorial/` 中的任何模块或函数。原始数据不修改、不删除。

## 环境与运行

所有检查、预处理、训练及评测均使用用户确认的 Conda 环境 `cv_task`，不使用系统 Python。
经用户授权新增了 26 个训练相关包，原有 125 个包的版本全部保持不变；
版本审计保存在 `outputs/training_environment.json`。`requirements.txt` 声明预处理依赖，
`requirements-training.txt` 记录训练的直接依赖；脚本不自动安装或升级包。

以下命令在项目根目录执行：

```bash
# 完整预处理：输出目录必须尚不存在
conda run -n cv_task python -s -B script/prepare_dataset.py

# 仅审计、生成分组清单和 manifest；不复制图片、不导出训练标签
conda run -n cv_task python -s -B script/prepare_dataset.py \
  --stage audit --output datasets/processed/crack_audit

# 对已有输出独立验证；不需要原始数据，也不读取预处理配置
conda run -n cv_task python -s -B script/prepare_dataset.py \
  --stage validate --output datasets/processed/crack_detection

# 回归测试；只使用临时合成样本，不依赖原始数据
conda run -n cv_task python -s -B -m unittest discover -s tests -v
```

默认配置为 `configs/preprocess.yaml`。配置中的路径相对于配置文件解析；
命令行 `--output` 相对于当前工作目录解析。若需重新生成，使用新的 `--output`；
程序拒绝覆盖已有目录，包括已有审计目录，不自动删除旧结果。

## 命令行参数与输出字段

命令行参数是输入，终端 JSON 是处理结果统计，两者不是一回事。

| 输入参数 | 含义 |
|---|---|
| `--config` | YAML 配置文件，默认 `configs/preprocess.yaml`；`validate` 不使用它 |
| `--stage audit` | 审计、掩码转框、分组划分并保存 manifest，不导出图片/训练标签/检查图 |
| `--stage all` | 默认模式；审计、转换、划分、导出、校验及生成检查图 |
| `--stage validate` | 独立检查已有导出，不重新生成数据；必须指定 `--output` |
| `--output` | `audit/all` 时覆盖配置中的输出目录；`validate` 时指定待检查目录 |

`audit/all` 的终端摘要：

| 字段 | 含义 |
|---|---|
| `status` | `audit_complete` 表示审计完成；`complete` 表示完整导出完成 |
| `runtime` | 实际 Python 解释器路径、Python 与 OpenCV/NumPy/Pillow/PyYAML 版本，不是训练参数 |
| `total_pairs` | 成功读取的图像—掩码对数量，包括最终未导出的样本 |
| `splits` | `train/val/test` 各集合的统计，见下文 |
| `exclusions` | 未导出原因及数量：`quarantined` 为人工确认隔离，`augmentation_in_heldout_group` 为验证/测试组增强成员；有完全重复样本时还可能出现 `exact_duplicate_of` |
| `output_dir` | 本次输出根目录 |
| `audit_report` | 详细审计报告文件路径，含警告、编码与尺寸统计等 |
| `previews` | 检查图数量，不是训练图片数量；`audit` 模式为零 |

每个 `splits.<集合>` 中：`images` 是导出图片数，`groups` 是原图来源分组数，
`instances` 是检测框总数，`empty_images` 是没有检测框的图片数，
`sources` 是该集合按来源子集统计的图片数。一张图片可有多个框，因此图片数不等于实例数。

`validate` 直接显示完整验证统计：

- `records`：manifest 总记录数；`active_records`：实际导出数；`excluded_records`：未导出数。
- `annotations`：全部检测框数；`empty_records`：导出图片中无框的图片数。
- `splits.<集合>.records/images/labels`：该集合记录、图片、标签文件数；一个 TXT 可包含多个框。
  `annotations` 是框数，`empty_records` 是无框图片数。
- `sources`：全部导出图片按来源子集统计的数量。

### 简化后的入口职责

`parse_args(argv)` 只读取命令行并检查参数组合；`load_config(...)` 已在
`src/Preprocessor.py` 中负责 YAML 读取和配置校验；`summarize_report(...)` 只整理展示字段；
`main(argv)` 只选择处理流程并打印一次结果。

入口不再统一捕获 `ValueError/OSError`：开发阶段直接保留错误堆栈，便于定位实际失败位置。
命令行参数错误仍由 argparse 显示 usage 并返回退出码 `2`。
标注合法性、分组泄漏、防覆盖及临时目录发布等底层检查保留，避免静默生成错误数据。

## 程序结构

- `src/Preprocessor.py`：配置校验、文件配对与掩码审计、框提取、来源分组、去重及集合划分。
- `src/dataset_io.py`：YOLO/COCO 导出、磁盘结果独立验证和检查图绘制。
- `script/prepare_dataset.py`：命令行入口。
- `configs/preprocess.yaml`：预处理参数及经用户确认的异常样本隔离清单。
- `tests/test_preprocessing.py`：编码、边界、分组泄漏、隔离策略和格式往返回归测试。
- `script/train_yolo.py`：简单函数式训练入口、独立可写数据副本、权重重载、验证集推理与展示。
- `configs/train_yolov8s.yaml`：两轮 YOLOv8s pilot 的模型、路径、训练和评测参数。
- `src/detection_evaluation.py`：可供后续不同模型共用的 COCO AP 和固定阈值 P/R 评测。
- `tests/test_detection_evaluation.py`、`tests/test_yolo_inference.py`：指标匹配和逐张推理批量边界回归。

## 数据与标注规则

输入为 `datasets/CrackDataset/`，包含 `crack500`、`AsphaltCrack300` 和 `CFD`。
原仓库含道路裂缝；不能据此直接声称已经验证建筑墙面的领域泛化。

1. 同名图像与掩码严格配对。缺失、重名、无法解码、尺寸不一致或未知编码均报告错误，
   不通过模糊匹配或自动缩放修补标注。
2. 读取原始像素，不执行 EXIF 自动旋转。图像保留原分辨率；不生成增强副本，不新增切片。
3. 单通道/1-bit 掩码按配置的数值解析。`P` 模式掩码按配置的前景白色、背景黑色解析
   **每张图自己的调色板**，并在 manifest 保存实际前景索引与调色板映射。
   实际数据中 AsphaltCrack300 的白色为索引 `0`，CFD 附加图的白色为索引 `1`，
   不能统一按索引大于零提取前景。
4. 每个八邻域连通域生成一个半开区间框 `[x_min, y_min, x_max, y_max]`。
   默认 `min_component_area=1`、`closing_iterations=0`，保留所有已标注连通域；
   面积阈值单位是前景像素数，不是框面积。参数只依据训练数据确定，之后冻结。
5. 相交裂缝可能合成一个连通域，断裂裂缝可能产生多个框；这些是掩码派生标注，
   不是人工实例级真值。合法空掩码保留为空目标图像。

### 分组划分与异常处理

- Crack500 去掉文件名最后两个数字坐标字段作为原图 ID。
- AsphaltCrack300 以切片编号前的三位数字作为原图 ID。
- CFD 将三位数字原图与 `_N`、`__N`、`addN`、`adjustN`、`subN` 派生样本归为同组。
- 未识别命名必须提供明确的 `group_overrides`，或显式隔离；不假设每张图片独立。
- 不沿用输入目录的 train/val 划分：审计发现其中有 107 个原图组交叉。
  按来源分层、以原图组为单位，用固定种子 `42` 划分约 `70%/10%/20%`。
  整组不可拆分，图片数量不保证恰好满足该比例。
- 解码后的 RGB 完全重复时先合并来源组，防止重复桥接导致泄漏。
  相同图像但掩码冲突会失败，除非相关记录已明确隔离。
- 已知 CFD 离线增强仅保留在训练组；验证/测试组的增强成员不导出，但保留在 manifest。
- pHash 只报告跨集合相似候选，不把相似度误当作已证实的同源关系。
  当前阈值为汉明距离不超过 `4`；它不是不存在所有近重复或相邻拍摄场景的证明。

用户已确认隔离 32 个样本：AsphaltCrack300 中 `001/002` 的 14 对相同图像、不同掩码
记录，以及 CFD 的 `img1`–`img4` 来源不明附加图。隔离仅影响导出，原始文件和原因记录均保留。

## 输出

```text
datasets/processed/crack_detection/
├── images/{train,val,test}/             # 原图字节副本，两个格式共用
├── labels/{train,val,test}/             # YOLO TXT；空目标为空文件
├── annotations/instances_{split}.json   # COCO JSON
├── splits/{train,val,test}.txt          # sample_id 清单
├── manifest.jsonl                      # 全量记录，包括所有未导出样本
├── data.yaml                           # YOLO/RT-DETR 数据配置
├── preprocess_state.json               # 完成状态
└── audit/
    ├── config.json                     # 实际生效配置与解析后的路径
    ├── report.json                     # 审计、统计、划分及验证结果
    └── previews/                       # 18 张检查图
```

YOLO 类别为 `0`；COCO 类别为 `1`。COCO 的框为像素 `xywh`，`area` 为框面积，
`file_name` 相对于输出的 `images/`。YOLO 的中心与宽高归一化到 `[0,1]`。
`data.yaml` 不写固定的 `path`，图片路径相对于其所在目录，便于移动整套数据。
Manifest 中的原始来源路径保留绝对路径用于追溯；已有导出数据的独立校验不要求这些来源路径仍存在。

每张检查图包含原图、掩码叠加与派生框、重新读取的 YOLO 框、重新读取的 COCO 框。
预览路径相对于输出目录，避免发布临时目录后路径失效。

程序先写入同级临时目录。审计失败会保留 `audit/report.json` 并返回非零退出码；
完整导出只有通过独立验证与检查图生成后才发布为 `complete`。

## 已验证结果

实际处理 3403 对文件；隔离 32 个异常样本，另有 180 个验证/测试组的增强样本未导出。
最终导出 3191 张图像、6767 个检测框：

| 集合 | 图像 | 原图分组 | 检测框 |
|---|---:|---:|---:|
| train | 2342 | 325 | 5035 |
| val | 274 | 47 | 518 |
| test | 575 | 94 | 1214 |

预处理的 13 项回归测试通过，包括调色板反向编码、共享框边界的浮点往返误差、
相同图像标签冲突、重复桥接、分组泄漏与隔离样本保留。
真实数据完整导出通过独立验证；已人工查看三个来源的检查图。

检测框来自裂缝掩码，不代表裂缝实例标注已人工全面审核；在正式训练前，应继续审阅检查图及目标尺度统计。

## YOLOv8s 短训练与验证

### 运行方式

训练依赖已安装到 `cv_task`：PyTorch `2.8.0+cu128`、torchvision `0.23.0+cu128`、
Ultralytics `8.3.221`、pycocotools `2.0.10`。GPU 卷积和 CUDA NMS 已实际验证。
需要下载权重时，可先在交互式终端执行用户已有的 `proxy_on`；代码不硬编码代理地址。

```bash
# 默认加载 COCO 预训练 yolov8s.pt，在 CUDA:0 训练两轮，再对验证集推理与评测
conda run -n cv_task python -s -B script/train_yolo.py

# 可选覆盖：是一次新的训练，不是从旧权重续训
conda run -n cv_task python -s -B script/train_yolo.py \
  --config configs/train_yolov8s.yaml --epochs 3 --name yolov8s_pilot_3ep
```

入口分为参数解析、配置读取、训练、预测、评测、可视化和摘要整理几个普通函数，
没有另建通用 Trainer 类。配置路径仍相对于 YAML 文件解析。
默认设置为 `640×640`、物理 batch `2`、AMP、AdamW、学习率 `0.001`、种子 `42`。
仅使用轻度亮度变化和水平翻转，不使用 Mosaic。完整实际设置以运行目录的 `args.yaml` 为准。
重名运行由 Ultralytics 创建带编号的新目录，不覆盖旧结果。

### 数据隔离与推理规则

- 训练前只把 train/val 图片和标签按原字节复制到 `outputs/yolov8s-data-*/`。
  这是可写训练副本，不是重新标注、缩放或新增增强数据。
- Ultralytics 即使设置 `cache=False` 也会写标签缓存，因此不直接训练处理好的原目录。
  已验证原 train/val 图片、标签和相关配置共 5235 个文件的 SHA-256 均未改变。
- 不复制、不推理、不评测真实测试集。模型选择、当前指标和示例图均来自验证集。
- 训练结束后重新实例化 `YOLO(best.pt)`，检查其类别为 `0=crack`。
- 当前版本把 Python 图片路径列表视为一个内存 batch；目录输入才按 `batch=1` 流式读取。
  这里使用验证集目录，实际观察到 274 个 batch，每个恰好一张图，并保留 CPU 回归测试防止再次出现该问题。
- 预测使用固定方形 `640×640` 输入、`rect=False`，输出框已映射回原图坐标。
  Ultralytics 的 `boxes.xywh` 是中心坐标，不能直接当 COCO 左上角坐标；
  本实现从 `xyxy` 转换为 COCO 的 `[x_min,y_min,width,height]`。

### 统一评测口径

`src/detection_evaluation.py` 不依赖具体模型，接收标准 COCO 预测结果。

- AP：COCO bbox，IoU 为 `0.50:0.05:0.95`，每图最多 100 个检测结果。
- 预测保留分数下限 `0.001`，不先用 P/R 的阈值截断 AP 输入。
- P/R：置信度不低于 `0.25`、IoU 不低于 `0.50`，按分数排序、一对一匹配；
  重复检出计 FP，未匹配真值计 FN。
- 当前 GT 为非 crowd、非 ignore 的单类裂缝框；不支持的 crowd/ignore 标注会明确报错。
- JSON 指标范围是 `0..1`；下表换算为百分比。

框架内置 `results.csv` 和 `BoxPR_curve.png` 用于观察训练过程，
不要与此统一评测的 `validation/metrics.json` 混在同一比较表中。
不同输入处理和统计口径可能得到不同数值；正式多模型比较应统一使用后者。

### 本次真实运行与结果

运行目录：`outputs/training/yolov8s_pilot/`。
两个 epoch 已真实完成，检查到 backbone 参数更新、权重能够重新加载，统一指标可从保存的预测重复计算。

| 验证集指标 | 两轮 pilot 结果 |
|---|---:|
| Precision | 23.78% |
| Recall | 7.53% |
| mAP@0.5 | 7.86% |
| mAP@0.5:0.95 | 3.07% |

当前仍有明显漏检，不能作为收敛性能或架构优劣结论。
本次初始训练已完成，但后续路径列表推理触发大 batch 显存不足；
修正为目录流式推理后，复用已保存的 `best.pt` 完成评测，没有重新训练。
原执行日志保留为 `training_console.log`，恢复情况记录在 `pilot_summary.json`。
AMP 检查通过的证据来自原训练日志，不虚构已结束进程的 GradScaler 状态。

```text
outputs/training/yolov8s_pilot/
├── args.yaml / source_config.yaml / resolved_pilot_config.yaml
├── results.csv / results.png            # 每轮指标和 Loss 曲线
├── weights/{best,last}.pt               # 真实训练权重
├── training_console.log                # 原训练及首次推理报错日志
├── pilot_summary.json                  # 运行环境、路径与最终统一评测
└── validation/
    ├── predictions_coco.json
    ├── metrics.json / pr_curve.png
    └── previews/                       # 三个来源各两张，GT 绿框、预测橙框
```

完整 20 项回归测试通过；已查看实际 Loss 曲线、统一 PR 曲线及三个来源的预测图。

## YOLOv8s 正式实验

### 固定协议与执行顺序

正式配置为 `configs/train_yolov8s_formal.yaml`，与两轮 pilot 分开保存：
从原始 COCO 预训练 `models/pretrained/yolov8s.pt` 开始，使用完整训练集训练 150 轮，
`640×640`、物理 batch `2`、AMP、AdamW、`lr0=0.001`、余弦学习率、种子 `42`。
`nbs=16` 控制框架的名义累计 batch，warmup 期间累计步数会变化，不能理解为每一步都严格等效于 batch 16。
关闭早停（`patience=0`），不因测试集表现改变训练预算。

```bash
conda run -n cv_task python -s -B script/train_yolo.py \
  --config configs/train_yolov8s_formal.yaml

# 仅在训练中断后使用：恢复原生 optimizer、AMP scaler 和调度 epoch
conda run -n cv_task python -s -B script/train_yolo.py \
  --resume outputs/training/yolov8s_formal/weights/last.pt
```

`--resume` 使用该运行保存的配置和原来的 train/val 数据副本，
不能与 `--config`、`--epochs` 或 `--name` 混用，也不会新建一次从头训练。
只接受与现有日志一致的最新未完成 checkpoint，不支持回退到较早的周期 checkpoint。
训练完成后框架会剥离 `best.pt`/`last.pt` 的 optimizer，不能再当作未完成断点续训。
原生恢复不保证随机数状态及数据顺序与从不中断的训练逐位一致。

权重选择仅依赖固定版本 Ultralytics 的验证集 mAP@0.5:0.95；相同 fitness 时保留最新一次权重。
`training_state.json` 记录实际保存事件对应的最佳 epoch，而不是从四舍五入的 CSV 反推。
训练结束后的原生最佳权重复验、每轮原生指标和统一 COCO 指标是不同记录，不能混为一谈。
正式训练完成并审阅验证结果、冻结权重后，才执行测试：

```bash
conda run -n cv_task python -s -B script/evaluate_yolo.py \
  --config configs/train_yolov8s_formal.yaml \
  --weights outputs/training/yolov8s_formal/weights/best.pt \
  --split test --output outputs/training/yolov8s_formal/test --benchmark
```

评测入口支持 `--split val`，不训练模型，不扫描训练集标签，也不搜索置信度阈值。
所有模型后续都应遵守同一几何规则：框映射并裁剪回原图后，剔除宽或高不大于零的空框，
记录剔除数量；任何正面积小框都保留，不设置最小面积过滤阈值。
COCO 评测器仍严格拒绝无效框，不通过放宽指标校验掩盖导出错误。

### 复杂度与时延口径

- Params 为未融合、单类 checkpoint 的总参数量，单位 M。
- GFLOPs 在 `[1,3,640,640]` 上用 THOP 统计未融合网络，按 `2 × MACs / 1e9` 计算；
  是模块算子覆盖范围内的近似值，不包括 NMS、数据读取和其他图像预处理。
- 时延使用新加载的模型、FP32 输入/权重、batch 1、方形输入、不使用 TTA；
  记录实际 TF32 开关，不把 FP32 张量等同于保证关闭 TF32。
- 预先读入来自验证集三个来源的固定六张图；轮流预热 50 次、测量 300 次，
  每张前后同步 CUDA。计时包括预处理、H2D、网络、解码、NMS、坐标还原、
  D2H 和空框过滤；不包括磁盘读取、模型加载、预热和绘图。
- 同时保存原始计时、均值、中位数、P95、标准差和顺序 FPS。
  计时使用 `confidence=0.25`，AP 预测仍使用 `score_floor=0.001`。
- 这是本地有桌面应用运行的 RTX 4060 Laptop 实测，记录驱动、软件版本、显存和温度快照。
  后续服务器上的五模型横向比较必须在同一服务器重新测量 YOLOv8s，不能直接混用本机速度。

### 产物与已完成的入口验证

新运行保存通用的 `resolved_config.yaml`、`training_state.json` 和 `experiment_summary.json`；
历史 pilot 的原有文件名和结果不改写。每轮日志和 Loss/mAP 曲线保留在运行目录，
统一验证结果在 `validation/`，冻结后的测试结果在 `test/`。
测试评测另输出 `evaluation_summary.json`、`benchmark.json`、`summary.csv`、
COCO 预测、PR 曲线和 GT/预测对照图。

启动前已完成真实 GPU 中断/恢复检查：恢复 183 组 AdamW 参数状态、AMP scaler、
原调度 epoch，并在同一目录完成第二轮；没有用 mock 训练替代。
零面积边界框的真实失败已复现并修正；当前 23 项回归测试通过，包括真实 TensorBoard 事件的历史步数、重复导入与缺失 epoch 检查。
`outputs/formal_input_snapshot.json` 保存原输入与包版本快照，
`outputs/yolov8s_formal_protocol.json` 保存正式配置、实现代码哈希和评测协议。
启动前确认 5235 个原 train/val 文件及全部 151 个已安装包版本未改变。

### tmux 与 TensorBoard 监控

本次正式训练在第 40 轮完整保存后中断，随后在 tmux 中从第 41 轮恢复，
仍沿用原运行目录、数据副本和 150 轮总预算，没有从头重训。
会话名为 `cv_task`，窗口 `cv_task` 显示训练输出，窗口 `tensorboard` 显示网页服务输出：

```bash
tmux attach -t cv_task

# 也可从另一个终端直接选择窗口
tmux select-window -t cv_task:cv_task
tmux select-window -t cv_task:tensorboard
```

默认快捷键 `Ctrl+b` 后按 `0`/`1` 切换窗口，`Ctrl+b` 后按 `d` 离开会话，
不会停止训练。训练输出继续追加到 `outputs/training/yolov8s_formal_console.log`，
TensorBoard 服务输出写入 `outputs/training/yolov8s_formal_tensorboard_console.log`。
不要用 `Ctrl+c` 代替 detach：它会中断当前窗口中的程序。

浏览器打开 <http://127.0.0.1:6006>，在 **SCALARS** 中查看 `train/*_loss`、
`val/*_loss`、`metrics/*` 和 `lr/*`。Loss 标签过滤器可填 `(train|val)/.*loss`。
右上角齿轮设置中勾选 **Reload data** 可开启网页自动刷新，默认刷新周期为 30 秒；
服务端每 5 秒重新加载事件文件，框架写入也可能有缓冲。
本机监听地址仅为 `127.0.0.1`，不向局域网公开服务。

TensorBoard 使用 `2.21.0`：它不再依赖当前 setuptools 已移除的 `pkg_resources`。
只在 `cv_task` 环境新增 TensorBoard 及其依赖，原有 151 个包的版本未改动；
未安装 TensorFlow，查看这些标量曲线不需要它。

`script/backfill_tensorboard.py` 用于首次接入监控时补录已完成的历史：

```bash
# 必须先停止该运行，并在原生 TensorBoard 回调第一次创建事件文件之前执行
conda run -n cv_task python -s -B script/backfill_tensorboard.py \
  --run outputs/training/yolov8s_formal
```

本次第 1–40 轮已补录，**不要再次执行这个命令**；检测到已有事件文件会拒绝重复写入。
补录仅读取 `training_state.json` 中确认完成的 CSV epoch，沿用原生标签及从 1 开始的 Step；
不插值，不重算 Loss，不访问测试集。历史事件的 wall time 是补录时间，
因此查看完整训练曲线应使用 **STEP** 横轴，而不是 WALL/RELATIVE。
第 41 轮以后由原生回调持续记录，已核对续训后的实际标量与对应 CSV 一致。
迁移时的 checkpoint 哈希、依赖变化和代码哈希保存在
`outputs/yolov8s_formal_monitoring_migration.json`，原正式协议快照保留不覆盖。
已实际打开 TensorBoard 并检查 Loss 曲线，截图为
`outputs/training/yolov8s_formal/tensorboard_loss_preview.png`。

### 正式训练完成后的冻结测试结果

已完成 150/150 轮正式训练。按预先约定的原生验证集 mAP@0.5:0.95 选择，
最终使用第 **25** 轮的 `weights/best.pt`，不是第 150 轮的 `last.pt`。
测试前保存 `pretest_freeze.json`，权重 SHA-256 为
`d1790586c7ff495fa5156aeb610b63cf0682fa121fa0fe5c885ba2e86664639a`。
配置、原始 train/val 文件以及既有包版本核对通过；测试后权重哈希未改变。
没有根据测试集选择其他 checkpoint、搜索阈值或重新训练。

测试集包含 **575 张图片、1214 个 GT 框**。以下均为统一评测器的结果，
不是框架训练日志中自动选取置信度工作点的 P/R：

| 数据集 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| val | 42.50% | 42.66% | 37.38% | 19.87% | 221 | 299 | 297 |
| test | 41.59% | 38.71% | 32.45% | 17.88% | 470 | 660 | 744 |

P/R 使用 `confidence=0.25`、匹配 `IoU=0.50`；AP 使用分数下限 `0.001`、
NMS `IoU=0.70`、每图最多 100 个框，固定方形 640 输入。
测试导出共有 32965 个低分数下限预测框，按冻结规则剔除 1 个非正面积框；
这些低分预测不等于 P/R 工作点上的检出数量。

| 模型 | Params / M | GFLOPs | 平均时延 / ms | 中位数 / ms | P95 / ms | 标准差 / ms | 顺序 FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8s，单类 | 11.136 | 28.642 | 11.84 | 10.53 | 19.23 | 3.85 | 84.49 |

时延是 RTX 4060 Laptop 上的单张端到端实测：FP32 张量、batch 1，
固定六张**验证集**图片轮流预热 50 次、测量 300 次，不是对 575 张测试图片计时取均值。
包含预处理、传输、网络和后处理，不包含磁盘 I/O；CUDA matmul TF32 为关闭、
cuDNN TF32 为开启。测量时保留桌面应用，GPU 快照温度 67°C，
结果不能直接与不同服务器或纯网络耗时相比较。
参数量与 GFLOPs 为未融合网络的统计，后者使用 THOP 的 `2 × MACs` 近似口径。

#### 漏检、误检与训练趋势

`test/error_analysis.json` 对保存的预测执行固定工作点的一对一匹配，记录逐图与来源统计：

| 测试来源 | 图片 | GT 框 | Precision | Recall | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| AsphaltCrack300 | 155 | 216 | 28.93% | 68.98% | 366 | 67 |
| CFD | 24 | 69 | 38.46% | 21.74% | 24 | 54 |
| crack500 | 396 | 929 | 53.13% | 32.94% | 270 | 623 |

小框按原图 bbox 面积 `<32×32` 像素定义：共 320 个，在全体 GT 的原有匹配中
只命中 5 个、漏检 315 个，固定工作点召回率为 **1.56%**。
这不是 COCO AP_small，也没有删除大框后重新匹配；极小连通域框也包含在分母中。
该统计说明本基线的小框检出不足，后续应在训练/验证集审查极小连通域和检测框的实例粒度，
不能直接因测试结果而改动本版冻结标注或阈值。

已查看三个来源共六张 GT 绿框/预测橙框对照图，具体例子：

- `01_AsphaltCrack300_AsphaltCrack300_010_03.jpg`：预测出一个覆盖范围过大的框，
  未达到与三个 GT 的匹配 IoU；统计为 TP=0、FP=1、FN=3。
  看到了大致裂缝区域，不代表定位和实例划分正确。
- `02_CFD_CFD_001.jpg`：一个较长预测框覆盖多个 GT 所在区域，一对一匹配后仍有漏检，
  另有小框误检；统计为 TP=1、FP=1、FN=1。
- `05_CFD_CFD_005.jpg`：细长、低对比度裂缝没有达到固定阈值的预测，FN=1。
- `03_crack500_crack500_20160222_081113_1281_361.jpg`：边缘小框及定位偏差问题明显，
  统计为 TP=0、FP=2、FN=2；相邻的 `06_crack500_crack500_20160222_081113_1281_721.jpg`
  则有两个正确匹配，但仍多检一个框，说明同一来源的难度也有差异。

上述图片不是带有场景标签的分层抽样。本版没有强反光、遮挡等场景标注，
不将亮色石粒直接判作强反光，也不据此给出这些场景的定量结论。
裂缝掩码按连通域导出的多个框与模型预测整段裂缝的粒度差异，需要与原掩码一起审查。

`results.png` 显示训练 Loss 持续下降，而验证 Loss 后期回升、原生验证 Recall 下降，
存在训练/验证表现分化；最后一轮并不优于按规则保存的最佳权重。
因此本次使用早期最佳权重是既定验证集选择的结果，并非查看测试集后回退。
这里只完成了单模型、单种子的本地基线，不能据此判断五种架构的优劣。

#### 测试产物与复核

结果目录为 `outputs/training/yolov8s_formal/test/`：

- `summary.csv`：包含精度、参数量、GFLOPs、时延与 FPS 的完整横向比较行。
- `metrics.json`、`predictions_coco.json`：统一指标及原图坐标的预测。
- `benchmark.json`：硬件/软件与计时口径、300 次原始计时。
- `evaluation_summary.json`：冻结权重哈希、评测配置与产物路径。
- `pr_curve.png`、`previews/`：测试集 PR 曲线和六张可视化对照图。
- `error_analysis.json`：逐图、来源、小框统计和复核状态。

原始控制台记录为 `outputs/training/yolov8s_formal_test_console.log`。
已从保存的预测重算 P/R、AP 与 TP/FP/FN，结果完全一致；
已核对 300 次计时的平均值、P95 以及 `FPS=1000/平均毫秒数`，
没有为复核更换模型、阈值或重跑全测试集推理。