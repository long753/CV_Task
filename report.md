# 建筑／路面裂缝目标检测多架构对比实验报告

## 摘要

本实验基于 CrackDataset 中的 AsphaltCrack300、CFD 和 crack500 数据，将裂缝分割掩膜转换为目标检测标注，完成 YOLOv8s、YOLO11s、Faster R-CNN ResNet-50-FPN、RetinaNet ResNet-50-FPN 和 RT-DETR-L 五个模型的迁移学习训练。五模型均完成 150 轮训练，最终使用各自验证阶段选出的 `best.pt`，在同一测试集上进行统一 COCO 检测评测。

实验结果显示：本次模型配置中，RT-DETR-L 的综合 AP 最高，YOLO 系列的单张推理时延最低；两阶段 Faster R-CNN 在固定阈值下召回率较高，但额外预测较多。所有模型对小标注框的检测能力均明显弱于大框。报告同时提供完整训练记录索引、损失／PR／mAP 曲线及困难样本同图对比。

**实验边界：五模型的历史训练硬件及部分训练设置并不完全一致，因此本报告是“已有五模型训练结果＋同平台统一评测”的实证比较，不能宣称已经完成完全控制变量的同硬件训练实验。强反光专项测试也尚未完成。** 这些缺口在第 9 节列明，没有用推测结果补齐。

## 1. 实验目的与模型选择

实验目标是掌握标注转换、不同技术路线的训练方式及统一评测方法，并分析检测精度、模型规模与推理速度之间的权衡。

| 模型 | 技术路线 | 主要机制 | 当前实现的后处理 |
|---|---|---|---|
| YOLOv8s | 单阶段、Anchor-free | 在多尺度特征上直接预测类别与位置 | 原生 NMS，随后限制检测数量 |
| YOLO11s | 单阶段、Anchor-free | 新一代骨干与特征融合结构，直接密集预测 | 原生 NMS，随后限制检测数量 |
| Faster R-CNN R50-FPN | 两阶段、RPN 使用 Anchor | 共享骨干特征，先生成候选区域，再进行 RoI 分类与框回归 | 候选区域及检测阶段的原生筛选／NMS |
| RetinaNet R50-FPN | 单阶段、Anchor-based | FPN 多尺度密集预测，使用 Focal Loss 缓解前景／背景不平衡 | 原生 NMS，随后限制检测数量 |
| RT-DETR-L | Transformer 端到端检测 | 高效混合编码器、目标查询及集合匹配 | 原生查询解码与分数排序，最多保留 100 框，不额外添加 NMS |

YOLO 两模型覆盖不同世代的无锚框检测器；RetinaNet 补充有锚框单阶段路线；Faster R-CNN 和 RT-DETR-L 分别覆盖候选区域和 Transformer 路线。这里的“单阶段卷积网络”指以卷积特征提取和密集检测为主的体系，不表示现代 YOLO 中完全没有注意力模块。

所有模型均从对应的 COCO 预训练权重迁移到单类别 `crack` 任务，而非从零训练。Faster R-CNN 的模型内部前景标签为 1，背景为 0；RetinaNet 的单类前景标签为 0；Ultralytics 的 `crack` 标签为 0。导出评测结果时统一映射为 COCO `category_id=1`。

## 2. 数据集与预处理

### 2.1 数据来源与发布规模

数据来自 [CrackDataset](https://github.com/guoguolord/CrackDataset)。虽然课题使用“建筑裂缝”的表述，当前数据实际以路面、沥青等表面裂缝为主，不能将结论不加验证地推广到所有墙面或工业材料。

预处理审计共登记 3,403 对图像／掩膜，最终导出 3,191 张有效图像及 6,767 个检测框。

| 划分 | 图像数 | 检测实例数 | AsphaltCrack300 图像 | CFD 图像 | crack500 图像 |
|---|---:|---:|---:|---:|---:|
| 训练集 | 2,342 | 5,035 | 540 | 492 | 1,310 |
| 验证集 | 274 | 518 | 72 | 12 | 190 |
| 测试集 | 575 | 1,214 | 155 | 24 | 396 |
| 合计 | 3,191 | 6,767 | 767 | 528 | 1,896 |

数据规模来源于 [预处理审计报告](datasets/processed/crack_detection/audit/report.json) 和导出的 COCO 标注。三个划分均没有空标注图像，因此现有实验不能评估“完全不含裂缝的阴性图像”上的整图误报率。

### 2.2 掩膜转检测框

预处理执行以下操作：

1. **检查图像与掩膜配对、尺寸、像素编码和数据来源。** 对普通二值／1-bit 掩膜识别 0/1；对调色板掩膜根据颜色解释前景，避免直接把调色板索引 1 一律当作裂缝。AsphaltCrack300 的白色前景恰好可能对应索引 0。
2. **将裂缝前景做八连通域分析。** 每个连通域生成一个轴对齐外接矩形，作为一个检测实例。
3. **保留所有面积不小于 1 像素的连通域。** 配置虽然记录 `closing_kernel_size=3`，但 `closing_iterations=0`，所以本次实际上没有执行闭运算，不能将其描述为“闭运算去噪后的数据”。
4. **统一坐标并导出两种格式。** 内部使用半开区间坐标；YOLO 标签为归一化中心点／宽高，COCO 标签为原图像素 `x,y,w,h`。输出保留原图空间关系，640 输入变换在模型训练／推理时执行。

这一定义意味着：**一个物理裂缝如果在掩膜中不连通，可能被拆成多个实例；1 像素碎片也可能成为 GT。** 因此本实验评测的是“连通域外接框检测”，不是像素级裂缝分割，也不等于对独立物理裂缝数量的直接统计。

### 2.3 重复样本、冲突与划分

原始目录中的既有划分存在 107 个原图组跨分区的情况，因此没有直接照搬原始 train/val 目录。实际采用 seed=42、按来源分层的原图组划分，并合并由完全相同图像连接的原图组，避免原图裁块或离线增强版本跨越训练和留出集。

配置中的组划分目标比例为 0.7／0.1／0.2，而不是要求最终图像数量严格等比例；CFD 离线增强只保留在训练所属原图组中，所以最终图像比例与目标组比例不同。

未导出样本包括：

- 180 对属于验证／测试原图组的离线增强样本；
- 32 对隔离样本：28 对相同 RGB 图像但掩膜冲突的 AsphaltCrack300 样本，以及 4 对来源无法核验的 CFD 补充样本。

隔离和排除通过清单记录，未删除原始数据。跨划分近重复检查使用 64-bit pHash、汉明距离不超过 4 的候选规则，本次审计候选列表为空；这不等于证明不存在一切形式的视觉近重复。

冻结配置见 [audit/config.json](datasets/processed/crack_detection/audit/config.json)，逐样本记录见 [manifest.jsonl](datasets/processed/crack_detection/manifest.jsonl)。

## 3. 训练设置与过程记录

### 3.1 实际训练超参数

下表取自**训练发生时的配置快照和原生 args**，不是当前 `configs/` 或最终测试摘要中的 training 字段。尤其 Faster R-CNN 的当前评测配置不能替代其历史训练配置。

| 参数 | YOLOv8s | YOLO11s | Faster R-CNN | RetinaNet | RT-DETR-L |
|---|---|---|---|---|---|
| 完成轮数 | 150 | 150 | 150 | 150 | 150 |
| 输入尺寸 | 640 | 640 | 640×640 | 640×640 | 640×640 |
| 物理 batch | 2 | 2 | 8 | 8 | 16 |
| nbs | 16 | 16 | 16 | 16 | 16 |
| workers | 2 | 2 | 4 | 4 | 4 |
| 优化器 | AdamW | AdamW | SGD | SGD | AdamW |
| 初始学习率 | 0.001 | 0.001 | 0.005 | 0.005 | 0.0001 |
| momentum 配置值 | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 |
| weight decay | 0.0005 | 0.0005 | 0.0005 | 0.0005 | 0.0001 |
| warmup 轮数 | 3 | 3 | 3 | 3 | 0 |
| 学习率调度 | cosine | cosine | 线性 warmup＋cosine | 线性 warmup＋cosine | cosine |
| lrf | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| AMP | 开启 | 开启 | 开启 | 开启 | 关闭 |
| deterministic | true | true | false | true | false |
| seed | 42 | 42 | 42 | 42 | 42 |

`momentum` 是配置字段；在 AdamW 中不能将其解释为 SGD 的动量更新公式。`nbs=16` 表示标称累积批次目标，不等于 DataLoader 的实际 batch；TorchVision 本次每两个 batch 累积一次更新。Ultralytics 的累积策略还受 warmup 影响，不能概括为全程固定相同累积步数。

数据增强方面，五模型均使用 0.5 概率水平翻转及亮度扰动，但具体实现不同：Ultralytics 为 `hsv_v=0.1`，TorchVision 为亮度因子约 `[0.9,1.1]`。Ultralytics 三模型关闭 mosaic、mixup、copy-paste、旋转、平移、缩放、剪切、透视和垂直翻转。原生 args 中出现的分类通用选项不能据此认定检测训练实际使用了 RandAugment 或随机擦除。

### 3.2 输入变换与历史硬件差异

- YOLOv8s、YOLO11s：使用各自 Ultralytics 路径的 letterbox 输入变换，预测框映射回原图坐标。
- RT-DETR-L：使用方形拉伸输入。
- 两个 TorchVision 模型：原图 RGB float 输入，由 `GeneralizedRCNNTransform` 拉伸至固定方形，并使用 ImageNet 均值／标准差归一化。

**相同的“640”不代表相同的几何变换。** 拉伸与 letterbox 可能改变细长裂缝的视觉形态，这是结果解释中的混杂因素。

历史训练环境证据如下：

| 模型 | 能从保存资料证实的训练环境 |
|---|---|
| YOLOv8s、YOLO11s | 本地 RTX 4060 Laptop GPU；训练日志为 Python 3.12.15、PyTorch 2.8.0+cu128、Ultralytics 8.3.221 |
| Faster R-CNN、RetinaNet | 云端连续队列及同一容器主机名；缺各运行独立、完整的 GPU／软件版本审计快照 |
| RT-DETR-L | 云端日志明确记录 RTX 4080 SUPER，Python 3.12.15、PyTorch 2.8.0+cu128、Ultralytics 8.3.221 |

因此，不能比较上述训练时长来直接评价纯架构训练效率，也不能写成“五模型训练软硬件完全相同”。云端日志还保留了 `OMP_NUM_THREADS` 的 libgomp 警告，尽管训练完成，也不能描述为全程无警告。

### 3.3 Loss、mAP 历史与 best 权重

五份 `results.csv` 均具有连续的第 1—150 轮记录。本次另解析 TensorBoard 事件，已核验的损失标签均覆盖第 1—150 步；证据见 [training_curve_evidence.json](outputs/report_assets/training_curve_evidence.json)。YOLOv8s 在第 40 轮后暂停并从第 41 轮续训，其前 40 轮 TensorBoard 历史由 CSV 补录，不能将其说成全部来自原始在线事件。

![五模型完整损失历史](outputs/report_assets/loss_curves.png)

图 1：各模型使用独立坐标轴展示损失。Ultralytics 面板为 CSV 中记录的三个损失分量之和，仅用于展示趋势，不将其当作跨架构统一的优化目标；TorchVision 面板为训练总损失。两个 TorchVision 训练实现没有计算验证 loss，因此图中没有伪造对应曲线。分量原始值仍保留在 CSV／TensorBoard 中。

![五模型验证mAP历史](outputs/report_assets/map_curves.png)

图 2：YOLO 两模型与 RT-DETR 使用框架原生验证 AP，Faster R-CNN 与 RetinaNet 使用项目统一 COCO AP。**这些历史曲线不是同一评测实现下的五条可直接排序曲线。** 虚线仅标识各自历史序列的最大值所在轮，最终横向精度结论以第 5 节统一测试为准。

| 模型 | 历史验证序列最大值所在轮 | 选择依据／证据边界 |
|---|---:|---|
| YOLOv8s | 25 | 原生 mAP50:95；另有 pretest_freeze／training_state 选模记录 |
| YOLO11s | 114 | CSV 原生 mAP50:95 最大值；未由本报告额外读取 stripped checkpoint 的 epoch 作独立证明 |
| Faster R-CNN | 14 | 训练循环按统一验证 mAP50:95 严格增加保存 best |
| RetinaNet | 66 | 同上 |
| RT-DETR-L | 5 | CSV 原生 mAP50:95 最大值；同分覆盖等行为以框架规则为准 |

训练使用的 Ultralytics 8.3.221 的原生 fitness 取 mAP50:95，而不是套用其他版本的加权公式。原生 AP 与 best 权重在项目 COCO 评测中的 AP 可能不同，这不是同一指标下凭空获得的精度提升。

从曲线可以观察到：训练损失持续下降，但多个模型的验证损失回升或 AP 不再改善；RT-DETR-L 和 Faster R-CNN 的验证峰值出现得较早。**这提示存在泛化退化或训练设置不匹配的可能，但还没有通过消融实验证实具体原因。** 完成 150 轮不代表应使用最后一轮权重，最终评测选择 `best.pt`。

## 4. 统一评测协议与本报告补测

### 4.1 检测精度协议

最终精度评测使用同一测试集：575 张图像、1,214 个 GT、单类别 `crack`。各模型预测框均回到原图坐标，并裁剪到图像边界，去除非正宽高框；保留正尺寸小框，而不是额外删除困难小目标。

统一约定：

- 预测导出的低分门槛 `score_floor=0.001`；
- 固定工作点 `confidence=0.5`；
- Precision／Recall 的匹配阈值 `IoU=0.5`；
- 每图最多保留 100 个检测框；
- CNN 模型沿用原生 NMS，配置阈值为 0.7；RT-DETR 不额外做 NMS。

同一图片、同一类别内，预测按置信度降序处理，与尚未匹配的 GT 按 IoU 做一对一匹配。重复预测不重复计为 TP。

\[
\mathrm{Precision}=\frac{TP}{TP+FP},\qquad
\mathrm{Recall}=\frac{TP}{TP+FN},\qquad
\mathrm{IoU}=\frac{|B_p\cap B_g|}{|B_p\cup B_g|}.
\]

mAP@0.5 使用 IoU=0.5；mAP@0.5:0.95 使用 0.50、0.55、……、0.95 共十个 IoU 阈值，按 COCO API 累积和插值。本项目只有一类，所以类别维度平均并非多个不同语义类的平均。固定阈值 P/R 与 AP 不是同一计算过程；AP 使用低分预测的排序信息，不能先用 0.5 删掉所有低分框再声称得到同一 AP。

**本次纠正的历史不一致：** 原 Faster R-CNN 测试文件采用 confidence=0.75，而其余四模型为 0.5。该旧工作点的 P/R 分别为 43.49%／47.36%，不能与其他模型的原行直接合表。本报告使用已保存、低分门槛相同的 COCO 预测，全部重新按 0.5 计算；五模型 AP 均与原结果一致，其余四模型 P/R 也完全一致。原始文件未覆盖。

本次固定 0.5 是协议对齐，不是根据测试结果搜索最优阈值；也不代表五个模型的置信度已经校准到相同概率含义。今后如果需要选择实际业务阈值，应只在验证集上完成并冻结，再评测测试集。

### 4.2 参数量、GFLOPs 与时延协议

五模型在同一本地平台重新测量，避免继续混用旧阈值的测速结果：

- GPU：NVIDIA GeForce RTX 4060 Laptop GPU，驱动报告显存总量 8,188 MiB；
- 软件：Python 3.12.13，PyTorch 2.8.0+cu128，torchvision 0.23.0+cu128，Ultralytics 8.3.221，OpenCV 模块版本 5.0.0，NumPy 2.5.3；
- 项目 `.venv` 环境，无新增／升级依赖；OMP、MKL 与 PyTorch CPU 线程设置为 4；
- 输入尺度 640，batch=1，FP32 权重与输入；CUDA matmul TF32 关闭，cuDNN 卷积允许 TF32，不能将其误写为所有算子均禁止 TF32；
- 使用同一组 6 张固定验证图，跨三种来源轮转；预热 50 次，再计时 300 次，即 6 图反复测量，而不是 300 张独立图；
- 每次测量前后进行 CUDA 同步。

计时包含：内存 BGR 图像预处理、CPU→GPU 拷贝、网络、解码／后处理、坐标还原、GPU→CPU 结果拷贝及空框过滤。**不包含磁盘读图、权重加载、预热、画图、相机采集或显示。** FPS 是该单进程串行检测路径的吞吐，不是 batch 吞吐，也不是完整视频系统 FPS，更不是 575 张测试图的平均时延。

单张时延取 300 次同步测量的算术平均；`FPS=1000/平均时延(ms)`，而不是对每次瞬时 FPS 再求平均。P95 和标准差用于描述本次循环内的波动，不是多次独立实验的置信区间。

原始文件中 YOLO 的 GFLOPs 来自 THOP，而另三模型来自 `torch.profiler`。本报告补充两个 YOLO 的 profiler 计数，主表统一采用 `torch.profiler CPU activities, with_flops=True` 的已支持算子计数。参数量按画像时未融合模型统计；时延走各框架原生推理路径，未强制统一其内部融合／图优化策略。

参数量以模型参数元素总数除以 10⁶，GFLOPs 以计数器报告的浮点操作数除以 10⁹。画像使用单张 `1×3×640×640` 零输入；候选区域检测的计算还可能随实际候选数量变化，因此该 GFLOPs 不是全测试集图像的平均操作量。

**GFLOPs 仍只是部分算子近似计数：** 部分 NMS、RoI、注意力、插值等操作可能没有 FLOPs 报告，不能把零报告当作零计算；CNN 模型内 proposal／postprocess 与原始网络 forward 的画像边界也不完全相同。因此 GFLOPs 不用于推导严格的跨架构效率倍数，实际速度以同步计时为准。

测速时保留了桌面和浏览器等进程，GPU 快照温度约 70—77°C，未做隔离与多次独立重复。JSON 保留每次时延样本、设备和运行环境。本报告结果属于**当前桌面负载条件下的同平台观测**，不宣称理想空载性能或统计显著差异。

## 5. 五模型综合性能横向对比

下表检测指标以百分数表示；参数量单位为百万，时延为单张均值。GFLOPs† 使用第 4.2 节的统一但不完备的计数方式。精确未舍入数值见 [model_comparison.csv](outputs/report_assets/model_comparison.csv)。

| 模型 | Precision / % | Recall / % | mAP@0.5 / % | mAP@0.5:0.95 / % | Params / M | GFLOPs† | Latency / ms | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8s | 60.18 | 27.76 | 32.45 | 17.88 | 11.136 | 28.43 | 9.00 | 111.09 |
| YOLO11s | 63.03 | 36.66 | 34.62 | 19.65 | 9.428 | 21.42 | 9.73 | 102.80 |
| Faster R-CNN R50-FPN | 25.75 | 52.39 | 42.26 | 24.85 | 41.299 | 181.62 | 48.73 | 20.52 |
| RetinaNet R50-FPN | 57.51 | 43.82 | 41.59 | 23.45 | 32.148 | 162.12 | 43.39 | 23.05 |
| RT-DETR-L | 61.11 | 47.12 | 47.37 | 29.81 | 32.808 | 109.22 | 34.41 | 29.06 |

### 5.1 误检、漏检和时延波动

| 模型 | TP | FP | FN | 时延中位数 / ms | P95 / ms | 时延标准差 / ms |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv8s | 337 | 223 | 877 | 8.23 | 12.61 | 1.77 |
| YOLO11s | 445 | 261 | 769 | 9.15 | 13.96 | 2.35 |
| Faster R-CNN | 636 | 1,834 | 578 | 46.69 | 65.95 | 6.44 |
| RetinaNet | 532 | 393 | 682 | 41.25 | 57.96 | 6.45 |
| RT-DETR-L | 572 | 364 | 642 | 33.69 | 42.63 | 4.58 |

每行均满足 `TP+FN=1,214`。Faster R-CNN 在当前工作点保留更多候选，获得较高召回，但也产生较多无法匹配的框。这不代表两阶段方法在所有阈值下必然低 Precision；不同模型的分数分布／校准不同，单个阈值不能代替完整 PR 分析。

![五模型统一测试PR曲线](outputs/report_assets/pr_curves.png)

图 3：同一测试集、IoU=0.5 下的 COCO 插值 PR 曲线。曲线没有限定为只看 confidence≥0.5 的预测；它与固定工作点表格的用途不同。

### 5.2 尺度分组补充实验

为避免只凭几张图片判断小目标能力，本次对已保存预测额外运行标准 COCO 小／中／大框 AP 分析，不重新训练、不按测试表现修改数据或阈值。

| 模型 | AP-small / % | AP-medium / % | AP-large / % |
|---|---:|---:|---:|
| YOLOv8s | 0.48 | 7.14 | 31.76 |
| YOLO11s | 0.99 | 7.26 | 35.14 |
| Faster R-CNN | 1.20 | 11.77 | 43.29 |
| RetinaNet | 0.64 | 10.10 | 42.00 |
| RT-DETR-L | 1.53 | 15.50 | 50.75 |

这里各列均是 IoU=0.50:0.95 的 AP，面积按**原图 GT 外接框**而非裂缝宽度定义；使用 COCO 的 32²、96² 面积划分及 API 原生边界规则。长而细的裂缝也可能拥有大外接框，因此不能把 AP-small 解释为所有“细裂缝”的检测率。

全部模型的小框 AP 都很低，不能宣称小目标问题已解决。最小连通域面积为 1 的标注策略、特征下采样、框定位偏差等均可能影响结果，但本次没有消融实验证明各因素的因果贡献。数值与计数说明见 [supplementary.json](outputs/report_assets/quantitative/supplementary.json)。

## 6. 典型测试样本的同图可视化

以下图片均由**同一测试原图、同一 GT 和五模型现有预测**生成，统一显示 confidence≥0.5 的结果；未重新推理或调参。左上面板只画 GT，其余面板画绿色 GT 与橙色预测；预测旁标出匹配后的 TP／FP 与分数，未匹配 GT 标为 FN。PNG 原图可放大查看。

这些样本按困难属性人工选取，不是随机抽样，也不是各场景的统计子集；**单图胜负不能替代第 5 节全测试集结果。** 来源和逐图计数保存在 [typical_scene_evidence.json](outputs/report_assets/typical_scene_evidence.json)。

### 6.1 小标注目标／碎裂连通域

样本：`test/crack500_20160222_115833_641_1.jpg`，image_id=202。

![小标注目标对比](outputs/report_assets/scene_small_annotated_targets.png)

图 4：主裂缝被掩膜连通域规则拆成不同尺度的外接框，并包含 1×1 像素级标注碎片。五个模型都匹配到了部分较明显区域，但细碎连通域普遍漏检；有些预测框将多个 GT 片段合在一起，因一对一匹配及 IoU 条件只能获得有限匹配。Faster R-CNN 的额外框较多，并未因此解决该图的碎片漏检。

这个案例同时暴露了任务定义的限制：1 像素 GT 不一定对应一个有独立工程意义的物理裂缝。后续若研究连通域过滤或片段合并，应在训练／验证数据上制定规则并对所有模型统一重做标注，不能只删掉测试集中难检的目标来提高成绩。

### 6.2 标注框重叠／复杂交叉结构

样本：`test/crack500_20160326_162529_641_721.jpg`，image_id=336。

![重叠标注框对比](outputs/report_assets/scene_overlapping_gt_boxes.png)

图 5：多个裂缝连通域的轴对齐外接框存在重叠，其中一对 GT 框 IoU 约为 0.289。五模型均未完整匹配全部标注；部分方法同时输出了额外局部框或合并范围不同的框。

此处是**标注外接框重叠**，不是已经证实的物理对象遮挡。该 GT IoU 也低于配置中的 NMS 阈值 0.7，且不做额外 NMS 的 RT-DETR 同样漏检。因此不能仅凭此图断言“就是 NMS 抑制导致漏检”；验证该原因还需要抑制前候选和 NMS 消融。

### 6.3 低对比度细裂缝

样本：`test/CFD_033.jpg`，image_id=163。

![低对比度裂缝对比](outputs/report_assets/scene_low_contrast_cracks.png)

图 6：灰色路面中的细裂缝与周边纹理对比较弱。RetinaNet 和 Faster R-CNN 在当前标准下匹配了部分标注，但 Faster R-CNN 的重叠额外预测较多；YOLO 两模型与 RT-DETR 虽沿裂缝方向给出预测，部分框过宽或合并了多个 GT 片段，未满足本图的一对一 IoU 匹配条件。

因此，图中的 FP 不都等价于“把完全没有裂缝的背景当成裂缝”，还可能是定位尺度、片段合并或重复预测问题。该图也说明总体 AP 最优的模型并不保证每个困难样本都最优。

“低对比度”在此是视觉观察标签，没有建立全测试集的对比度分组和统计指标，不能据此宣称某模型对所有低对比度场景显著更强。

### 6.4 强反光专项：尚未完成

本次实际查看了 12 张高亮候选图，但主要呈现亮色／白色路面或涂层，没有确认可无歧义认定为局部强镜面反光的案例。**亮色表面不等于强反光，不能用它们冒充已完成强反光实验。** 候选筛查也不证明整个数据集绝无反光样本。

需要后续补充具有可靠标注的真实反光样本，并在五个冻结权重、固定阈值下生成同图比较；如需场景定量结论，还应先定义反光分组、样本数及统计方式。当前报告不提供虚构的强反光检出率。

## 7. 三类架构的精度—速度权衡分析

### 7.1 单阶段卷积检测：密集预测与部署效率

YOLO 系列通过共享骨干／颈部特征，在多尺度特征上直接回归位置和类别，避免逐候选区域的二次特征处理，通常更容易获得较低时延。本次 YOLO 两模型形成速度优先的选择，但固定工作点召回和 AP 仍落后于较大模型配置。

YOLO11s 在本实验中参数量与支持算子 GFLOPs 均低于 YOLOv8s，但时延略高。**参数少、近似 FLOPs 低，不保证实际更快。** 算子类型、内存访问、核函数调度、框架原生融合及后处理均会影响时延；本次尚未做算子级时延剖析，不能指认其中某一项为已证实原因。

RetinaNet 同为单阶段，但采用更重的 R50-FPN 和有锚框密集预测，其速度并不接近小规格 YOLO。Focal Loss 的设计目的是减轻大量易分类背景对训练的支配，不能由名称推出它会自动解决细裂缝误检。该结果说明“单阶段”只是流程分类，不是固定的精度或速度等级。

### 7.2 两阶段检测：候选区域与精修成本

Faster R-CNN 先通过 RPN 产生候选区域，再通过 RoI 特征提取完成分类和框回归。候选区域处理有利于对潜在目标进行专门建模，但带来额外计算、数据组织及后处理成本。

本次结果中，它相对两个 YOLO 取得更高 AP 和召回，但在固定阈值下错误匹配较多、时延较高。RPN、RoI 和框回归的存在并不保证所有细碎目标都能检出，也不保证其 AP 一定超过 Transformer。重叠或细长外接框、实例划分方式及置信度分布同样影响最终指标。

### 7.3 Transformer 端到端检测：集合预测与全局交互

RT-DETR 使用目标查询及集合匹配表达检测结果，并利用高效混合编码器处理多尺度特征，从而减少对额外 NMS 的依赖。该结构可以利用更大范围的上下文，但查询交互、解码、访存等计算也具有不同于普通卷积的开销。

本次 RT-DETR-L 在所选配置中兼顾了较高 AP 和低于两个 R50-FPN 模型的时延。其原生不追加 NMS，但可视化仍出现 FP 和 FN，说明“无 NMS”不是“没有重复或误检”的保证。

它的名称含实时检测，不等于在任意硬件、任意精度和完整视频系统中都能达到固定实时帧率。本次桌面负载下的 FP32 单张结果不足以承诺稳定 30 FPS 视频处理。

### 7.4 如何理解本次取舍

以 AP50:95 越高、时延越低为两个目标，在当前观测点中，RT-DETR-L 同时优于两种 R50-FPN 模型；YOLOv8s 和 YOLO11s 则提供不同的速度／精度折中。实际选择可按需求区分：

- 低时延、端侧吞吐优先：优先考察 YOLO 系列，同时评估漏检成本；
- 当前数据上的总体 AP 优先：优先考察 RT-DETR-L；
- 漏检敏感：不能只看最高 Recall，应结合 FP 成本和完整 PR 曲线，在验证集选择工作点；
- 细碎小目标或弱对比度场景：现有模型均有明显不足，应先完善场景评测与标注策略，而不是只更换模型名称。

这些是**特定模型规格、训练设置、输入变换和单次划分下的结论**，不是对三类架构普适优劣的因果证明。未做等参数量／等计算预算比较，也未通过多随机种子验证微小差异的统计显著性。

## 8. 配置、日志、权重及复核材料索引

### 8.1 实际训练配置与原始日志

| 模型 | 历史训练配置 | 逐轮 CSV | 标准输出日志 |
|---|---|---|---|
| YOLOv8s | [source_config](outputs/training/yolov8s_formal/source_config.yaml)、[resolved_config](outputs/training/yolov8s_formal/resolved_config.yaml)、[args](outputs/training/yolov8s_formal/args.yaml) | [results.csv](outputs/training/yolov8s_formal/results.csv) | [train.log](outputs/training/yolov8s_formal/logs/train.log) |
| YOLO11s | [config](outputs/training/yolov11s_formal/config.yaml)、[args](outputs/training/yolov11s_formal/args.yaml) | [results.csv](outputs/training/yolov11s_formal/results.csv) | [train.log](outputs/training/logs/yolov11s_formal.train.log) |
| Faster R-CNN | [config](outputs/training/fasterrcnn_resnet50_fpn_formal/config.yaml) | [results.csv](outputs/training/fasterrcnn_resnet50_fpn_formal/results.csv) | [train.log](outputs/training/logs/20261020-212542-793903153/fasterrcnn_resnet50_fpn.log) |
| RetinaNet | [config](outputs/training/retinanet_resnet50_fpn_formal/config.yaml) | [results.csv](outputs/training/retinanet_resnet50_fpn_formal/results.csv) | [train.log](outputs/training/logs/20261020-212542-793903153/retinanet_resnet50_fpn.log) |
| RT-DETR-L | [config](outputs/training/rtdetr_l_formal/config.yaml)、[args](outputs/training/rtdetr_l_formal/args.yaml) | [results.csv](outputs/training/rtdetr_l_formal/results.csv) | [train.log](outputs/training/logs/20261020-212542-793903153/rtdetr_l.log) |

完整配置文件已存在，报告表格只抽取关键项；不以当前默认配置代替历史参数。快照中的历史绝对路径是运行记录，跨机器重放时需要适配路径。

五模型权重分别位于 `models/trained/<对应模型名>_formal/`，其中 `best.pt` 为最终评测权重，`last.pt` 为最后训练检查点。权重 SHA-256 记录在 [report_provenance.json](outputs/report_assets/report_provenance.json)，本次实际读取并核对了五个 best 文件与原评测摘要的哈希一致性。

TensorBoard 数据均在 `outputs/training/<运行名>/tensorboard/`，查看命令为：

```bash
.venv/bin/tensorboard --logdir outputs/training --host 127.0.0.1 --port 6006
```

YOLO／RT-DETR 的原始训练总图位于各自 `plots/results.png`，两个 TorchVision 模型为 `plots/training_curves.png`。本报告图 1—2 是从完整 CSV 重绘的汇总图，没有将 YOLOv8 早期约 42 轮的界面截图当成完整训练图。

### 8.2 统一测试与本报告新增证据

- 原始测试结果：`outputs/evaluation/{yolov8s,yolov11s,fasterrcnn_resnet50_fpn,retinanet_resnet50_fpn,rtdetr_l}/`，包括原始预测、摘要及 PR 曲线，均保留未覆盖。
- 本报告统一工作点和复测：每模型的 `outputs/report_assets/quantitative/<模型名>/metrics.json`、`benchmark.json`、`common_profile.json`。
- 五模型完整测量合并记录：[report_measurements.json](outputs/report_assets/quantitative/report_measurements.json)。其中 benchmark 的 YOLO 原始画像仍保留 THOP 字段以忠实记录原函数输出；**主表 GFLOPs 则取各模型 `common_profile.json` 的统一 profiler 补测值**，不能将两种计数混用。
- 统一 profiler／尺度 AP 汇总：[supplementary.json](outputs/report_assets/quantitative/supplementary.json)。
- 可直接汇总到表格的软件数据：[model_comparison.csv](outputs/report_assets/model_comparison.csv)。其中 P、R、AP 保存为 0—1 小数，正文转换为百分数。
- 本报告的评测配置快照：`outputs/report_assets/quantitative/<模型名>/evaluation_config.json`。JSON 兼容项目 YAML 读取器；这是评测配置，**不是历史训练配置**。

例如，使用保存的统一配置复测 Faster R-CNN，可在项目环境中执行：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python script/evaluate.py \
  --config outputs/report_assets/quantitative/fasterrcnn_resnet50_fpn/evaluation_config.json \
  --model fasterrcnn_resnet50_fpn \
  --weights models/trained/fasterrcnn_resnet50_fpn_formal/best.pt \
  --split test \
  --output outputs/evaluation_reproduce/fasterrcnn_resnet50_fpn \
  --benchmark
```

该命令复现检测与时延流程；其默认 YOLO FLOPs 仍按原程序计数，复核报告主表统一计数时还应参考保存的 `common_profile.json`。时延受当次系统负载影响，不能保证逐样本毫秒值完全相同。

## 9. 完成情况、缺失实验及补充顺序

| 要求／检查项 | 当前状态 | 已有证据或仍需补充的内容 |
|---|---|---|
| 不少于五模型，覆盖规定路线 | 已完成 | 两代 YOLO、Faster R-CNN、RetinaNet、RT-DETR-L |
| 五模型完整训练 | 已完成 | 全部 150 轮，配置、CSV、损失记录和 best／last 均存在 |
| Precision、Recall、两种 mAP | 已完成并统一复核 | 从相同低分预测按共同阈值重算，修正 Faster 原工作点差异 |
| Params、GFLOPs、Latency／FPS | 已记录并补做同平台复测 | GFLOPs 为支持算子部分计数，不能视作完备理论运算量 |
| Loss、PR、mAP 曲线及横向表格 | 已完成 | 正文插图、完整源日志、可导出 CSV |
| 小目标、重叠框、低对比度同图分析 | 已完成定性案例 | 另补标准尺度 AP；重叠与低对比度尚非完整场景统计组 |
| 强反光检出／误检分析 | **未完成** | 需要可靠的真实反光样本，不用白色路面冒充 |
| 五模型相同软硬件基准下训练 | **未满足严格要求** | 历史训练硬件不同；同平台测速不能代替同硬件重训 |
| 多随机种子／独立重复 | 未完成，属于结论稳健性增强 | 当前仅 seed=42；时延循环不是多次独立实验 |
| 全程统一 COCO 验证选模 | 部分完成 | TorchVision 已做；Ultralytics 历史曲线为原生指标，仅 best 有统一重验证 |
| 纯阴性图像及外部域测试 | 未完成，属于应用泛化增强 | 当前所有导出图像都有 GT，无法评价无裂缝整图误报率 |

建议优先补充：

1. **若任务严格要求同硬件训练，统一服务器和环境重新完成受控训练。** 固定数据划分、输入协议、训练预算及记录策略；优化器等架构必要差异应事先声明。为每个运行独立保存 GPU、CPU、驱动、依赖版本和配置快照。
2. **补强反光专项数据。** 先建立清晰的场景定义和标注，若为外部采集数据，应与原测试集分开报告；冻结权重和工作点后比较，不把新增数据用于选择原测试结果更好的阈值。
3. **如需场景级结论，建立重叠／低对比度样本分组及统计协议。** 目前图片只支持案例分析；可额外保留 NMS 前后候选，验证抑制与漏检的关系。
4. **完善稳健性和消融。** 在训练／验证集上研究连通域最小面积、输入尺寸、细长框策略、预训练与优化器设置；采用多 seed 重复并报告均值和波动。不要直接修改测试标注来提升成绩。
5. **如需严格硬件效率结论，重做隔离、多轮交错顺序的测速。** 同时控制电源、温度、后台负载和图优化策略；若研究实际视频实时性，再加入采集、读图、显示等环节。

缺验证 loss 的 TorchVision 运行仍然有完整训练 loss、验证 AP 和 PR，因此不是“训练未完成”；若课程额外要求每模型验证 loss，需要另行设计验证损失计算方法，不能通过验证时开启训练态来污染模型统计。

## 10. 结论

本项目已经建立从裂缝掩膜到 YOLO／COCO 标注、五架构训练、冻结权重统一评测及可视化分析的完整实验链路。结果支持在当前配置和数据上采用 YOLO 作为低时延方案、RT-DETR-L 作为总体 AP 优先方案；Faster R-CNN 展现更高固定工作点召回，但具有较多额外预测和更高时延。RetinaNet 体现了有锚框单阶段方案与轻量 YOLO 在骨干规模和密集预测开销上的差异。

本实验也揭示了比模型名称更重要的限制：连通域标注可能产生极小碎片，框级 IoU 会惩罚多片段合并和定位偏差，统一尺寸不等于统一输入几何，较低训练 loss 不代表更好的泛化。现有结果不能证明任一架构普遍更优，也不能掩盖强反光测试与同硬件训练尚未完成。后续应优先补齐这些关键条件，再扩大泛化与消融实验。

## 参考资料

1. CrackDataset 数据集仓库：[https://github.com/guoguolord/CrackDataset](https://github.com/guoguolord/CrackDataset)。
2. Ren et al. *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*：[arXiv:1506.01497](https://arxiv.org/abs/1506.01497)。原论文机制用于架构解释，不将其 VGG 或原数据集速度当作本实验结果。
3. Lin et al. *Focal Loss for Dense Object Detection*：[arXiv:1708.02002](https://arxiv.org/abs/1708.02002)。
4. Zhao et al. *DETRs Beat YOLOs on Real-time Object Detection*：[arXiv:2304.08069](https://arxiv.org/abs/2304.08069)。论文模型与本实验具体 Ultralytics L 变体不混同。
5. Ultralytics YOLOv8 官方说明：[YOLOv8 documentation](https://docs.ultralytics.com/models/yolov8/)。该项目没有独立正式 YOLOv8 论文，本报告引用官方实现资料。
6. Ultralytics YOLO11 Anchor-free 说明：[YOLO11 anchor-free detection](https://www.ultralytics.com/blog/benefits-ultralytics-yolo11-being-anchor-free-detector)。理论和实现说明不替代本实验实测。
7. 项目内部证据：`datasets/processed/crack_detection/audit/`、`outputs/training/`、`outputs/evaluation/` 和本次 `outputs/report_assets/`；报告使用这些文件中的实际值，未调用 `tutorial/` 中的函数。
