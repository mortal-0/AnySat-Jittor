# AnySat-Jittor

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![Jittor](https://img.shields.io/badge/Jittor-1.3.11-orange.svg)](https://github.com/Jittor/jittor)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

本项目是基于遥感多模态基础模型 **AnySat: One Earth Observation Model for Many Resolutions, Scales, and Modalities** 的 Jittor 框架迁移与下游任务复现实验。项目围绕遥感多模态大模型的下游适配展开，重点实现了基于 AnySat 冻结特征的 **TimeSen2Crop 作物类型分类线性探测**，并对 **BraDD-S1TS 森林砍伐检测 / 语义分割任务**进行了复现、可视化与结果分析。

---

## 目录

- [项目背景](#项目背景)
- [整体复现路线](#整体复现路线)
- [项目结构](#项目结构)
- [数据集说明](#数据集说明)
- [环境配置与安装](#环境配置与安装)
- [Git LFS 与大文件管理](#git-lfs-与大文件管理)
- [Jittor 环境检查](#jittor-环境检查)
- [TimeSen2Crop 作物类型分类实验](#timesen2crop-作物类型分类实验)
- [BraDD-S1TS 语义分割复现实验](#bradd-s1ts-语义分割复现实验)
- [可视化脚本](#可视化脚本)
- [已知问题与踩坑记录](#已知问题与踩坑记录)
- [参考文献](#参考文献)
- [License](#license)
- [致谢](#致谢)

---

## 项目背景

遥感数据天然具有多源异构特点，包括：

- **不同传感器模态**：例如 Sentinel-1 SAR、Sentinel-2 光学时间序列、高分航空影像等；
- **不同空间分辨率**：例如米级、十米级甚至更粗分辨率；
- **不同时间维度**：例如单时相影像和多时相时间序列；
- **不同下游任务**：例如分类、语义分割、变化检测等。

AnySat 的核心目标是构建一个能够适配多分辨率、多尺度、多模态遥感数据的统一遥感基础模型。本项目在此基础上主要关注两个问题：

1. 如何将 AnySat 预训练 backbone 提取到的特征用于 Jittor 框架下的下游任务；
2. 如何在有限存储和算力条件下，选择合适的外部数据集进行复现与效果评估。

---

## 整体复现路线

由于 AnySat 预训练数据规模较大，完整预训练过程对存储和算力要求较高。因此，本项目没有重新进行 AnySat 端到端预训练，而是采用更适合框架迁移和课程复现的路线：

![整体复现路线](.media/reproduction_workflow.png)

---

## 项目结构

```text
AnySat-Jittor/
├── .media/                         # README 用到的图片资源
├── configs/                        # 实验配置文件
├── data/                           # 数据说明文件，不直接存放原始数据集
│   └── README.md
├── src/
│   ├── data/                       # 数据读取和数据处理代码
│   ├── jittor_lp/                  # Jittor 线性探测与环境检查脚本
│   ├── models/                     # 模型结构、损失函数与评估指标
│   ├── utils/                      # 工具函数
│   ├── visualization/              # 混淆矩阵、Loss 曲线、预测结果可视化脚本
│   ├── eval.py                     # 评估入口
│   ├── train.py                    # 训练入口
│   └── train.sh                    # 训练脚本示例
├── vis_bradd_preds/                # BraDD 预测结果可视化目录
├── demo.ipynb                      # 示例 Notebook
├── hubconf.py
├── requirements.txt                # 依赖清单
├── LICENSE
└── README.md
```

---

## 数据集说明

本仓库不直接分发原始数据集。请根据官方链接自行下载，并按照 `data/README.md` 中的说明放置到本地目录。

### TimeSen2Crop

| 项目 | 内容 |
|---|---|
| 数据集 | TimeSen2Crop |
| 任务类型 | 作物类型分类 |
| 传感器 / 模态 | Sentinel-2 光学时间序列 |
| 数据形式 | 像素级 Sentinel-2 时间序列样本 |
| 类别数 | 16 类作物 |
| 数据规模 | 约 14.9 GB |

官方数据集链接：

```text
https://zenodo.org/records/4715631
```

备用访问方式：

```text
https://huggingface.co/datasets/monster-monash/TimeSen2Crop
```

建议本地路径：

```text
data/TimeSen2Crop/
```

### BraDD-S1TS

| 项目 | 内容 |
|---|---|
| 数据集 | BraDD-S1TS |
| 全称 | Brazilian Amazon Deforestation Dataset with Sentinel-1 Time Series |
| 任务类型 | 森林砍伐检测 / 变化检测 / 语义分割 |
| 传感器 / 模态 | Sentinel-1 SAR 时间序列 |
| 数据形式 | SAR 图像序列与森林砍伐标签 |
| 类别数 | 2 类，背景 / 变化区域 |
| 数据规模 | 约 17.7 GB |

官方数据集链接：

```text
https://zenodo.org/records/8060250
```

官方 GitHub 仓库：

```text
https://github.com/ecovision-uzh/BraDD-S1TS
```

建议本地路径：

```text
data/BraDD-S1TS/
```

---

## 环境配置与安装

推荐使用 Python 3.9。

```bash
conda create -n anysat_jittor python=3.9
conda activate anysat_jittor
pip install -r requirements.txt
pip install jittor
```

检查 Jittor 是否安装成功：

```bash
python -c "import jittor as jt; print(jt.__version__)"
```

本项目涉及 PyTorch、Lightning、Hydra、TorchMetrics、Jittor、Timm、Rasterio、Geopandas 等依赖。具体依赖以 `requirements.txt` 为准。

---

## Git LFS 与大文件管理

本项目使用 Git LFS 管理较大的 `.npz` 特征文件和模型相关二进制文件。当前建议通过 Git LFS 管理以下类型：

```text
*.npz
*.pkl
*.pth
*.ckpt
*.7z.*
```

克隆仓库前，请先安装并启用 Git LFS：

```bash
git lfs install
```

克隆仓库并拉取 LFS 文件：

```bash
git clone git@github.com:mortal-0/AnySat-Jittor.git
cd AnySat-Jittor
git lfs pull
```

如果某些特征文件因超过 GitHub LFS 单文件大小限制而被拆分为 7-Zip 分卷，例如：

```text
train_features_npz.7z.001
train_features_npz.7z.002
```

可使用以下命令恢复原始 `.npz` 文件：

```bash
7z x train_features_npz.7z.001
```

恢复后应得到：

```text
train_features.npz
```

---

## Jittor 环境检查

在迁移过程中发现，部分 Jittor CUDA 环境可能会出现基础算子异常，导致 loss 失真。例如，对于 16 类全零 logits，理论交叉熵应为：

```text
log(16) = 2.7725887
```

如果 Jittor CUDA 下返回 `0`、`1` 或极大异常值，则该环境下的训练 loss 和 accuracy 不可信。

建议在训练前运行最小算子检查脚本：

```bash
python src/jittor_lp/debug_jittor_ops.py
```

正常结果应接近：

```text
exp(0).sum(1) = 16
logsumexp = 2.7725887
log_softmax = -2.7725887
cross_entropy_loss = 2.7725887
```

如果 CUDA 模式异常，可以优先使用 CPU 模式运行 Jittor 下游任务：

```bash
--use_cuda 0
```

也可以清理 Jittor 缓存并重新测试：

```bash
python -m jittor_utils.clean_cache all
python -m jittor.test.test_example
python -m jittor.test.test_core
python -m jittor.test.test_cuda
```

---

## TimeSen2Crop 作物类型分类实验

### 输入文件格式

TimeSen2Crop 的 Jittor 线性探测脚本使用 AnySat backbone 预先导出的特征文件，格式为 `.npz`。

每个 `.npz` 文件应包含：

```text
x: [N, D] float32 特征矩阵
y: [N] int64 类别标签
```

推荐路径：

```text
src/jittor_lp/outputs/ts2c_features/
├── train_features.npz
├── val_features.npz
└── test_features.npz
```

### 模型结构

Jittor 版本下游分类头采用轻量线性探测结构：

```text
LayerNorm(D) + Linear(D, num_classes)
```

该结构不重新训练 AnySat backbone，只训练下游分类头，用于验证 AnySat 预训练特征在作物类型分类任务上的线性可分性。

### 运行命令

```bash
python src/jittor_lp/train_ts2c_head_jittor.py \
  --train_npz src/jittor_lp/outputs/ts2c_features/train_features.npz \
  --val_npz src/jittor_lp/outputs/ts2c_features/val_features.npz \
  --test_npz src/jittor_lp/outputs/ts2c_features/test_features.npz \
  --batch_size 1024 \
  --epochs 200 \
  --lr 2e-4 \
  --weight_decay 0.05 \
  --use_cuda 0 \
  --out_dir src/jittor_lp/outputs/jittor_ts2c_lp
```

其中：

- `--batch_size` 可根据内存或显存调整；
- `--use_cuda 0` 表示使用 CPU；
- 若 Jittor CUDA 算子检查完全正常，可尝试 `--use_cuda 1`。

### 实验设置

| 参数 | 数值 |
|---|---|
| Epoch | 200 |
| Batch size | 1024 |
| Optimizer | AdamW |
| Learning rate | 2e-4 |

### 测试集准确率对比

| 模型 | 测试集 Accuracy |
|---|---:|
| End-to-End PyTorch | 72.02% |
| Jittor Feature LP | 70.76% |
| PyTorch Feature LP | 70.66% |

从结果看，基于冻结 AnySat 特征的 Jittor 线性探测结果与 PyTorch 线性探测结果非常接近，说明 Jittor 迁移后的下游分类头能够较好复现 feature linear probing 的分类效果。端到端 PyTorch 训练的最终测试精度略高，但其训练过程需要更新更多参数，训练成本也更高。

### Loss 与 Accuracy 曲线讨论

![TimeSen2Crop Loss and Accuracy](.media/timesen2crop_loss_acc.png)

实验观察到以下现象：

1. **端到端 PyTorch 训练收敛较慢**  
   端到端训练需要同时更新 backbone 与下游任务头，优化空间更大，因此前期收敛相对较慢。

2. **非端到端 PyTorch / Jittor 线性探测收敛较快**  
   二者只训练轻量任务头，输入已经是 AnySat 提取好的高维特征，因此验证集准确率能够在较短 epoch 内快速上升。

3. **Jittor Feature LP 与 PyTorch Feature LP 的验证准确率接近**  
   说明 Jittor 下游分类头迁移基本有效，差异主要来自框架实现、数值环境和训练细节。

4. **端到端 PyTorch 最终 loss 更低**  
   端到端方式可以调整 backbone 表征，因此损失函数优化空间更大；但线性探测更强调冻结特征的可分性，而不是追求最低训练 loss。

### 混淆矩阵

![Confusion Matrix Visualization](.media/confusion_matrix.png)

---

## BraDD-S1TS 语义分割复现实验

BraDD-S1TS 是 Sentinel-1 SAR 时间序列森林砍伐检测数据集。相比 TimeSen2Crop 的作物分类任务，BraDD 更接近语义分割 / 变化检测任务，对空间结构、前景区域和类别不平衡更加敏感。

### 运行与可视化

BraDD 相关预测图和可视化结果可以参考：

```text
vis_bradd_preds/
src/visualization/
```

绘制 loss 与 IoU 曲线示例：

```bash
python src/visualization/plot_loss_acc_curve.py
```

该脚本可读取类似下面路径的日志文件：

```text
logs/BraDD_AnySat_LinearProbing_SemSeg/csv/version_0/metrics.csv
```

并绘制：

- train loss
- validation loss
- validation IoU

### 训练现象与结果讨论

![BraDD Loss and IoU](.media/bradd_loss_iou.png)

在 BraDD 复现实验中观察到：

1. **训练集 loss 持续下降**  
   说明模型并非完全无法学习，训练过程存在一定程度的收敛。

2. **验证集 loss 上下波动**  
   说明模型泛化不稳定，可能与数据量、标签分布、类别不平衡和任务复杂度有关。

3. **IoU 在前 20 个 epoch 内提升有限**  
   实验中核心性能指标 IoU 仅有小幅提升，说明仅使用轻量下游头可能不足以充分解决复杂语义分割任务。

4. **模型容易关注背景类**  
   在部分可视化结果中，模型对背景区域预测较强，而对森林砍伐前景区域识别不足。这可能与前景类别占比低、loss 函数设计以及特征与标签空间对齐方式有关。

5. **仍能学习到部分语义信息**  
   尽管整体 IoU 提升有限，但少量样本上可以看到模型对部分变化区域有响应，说明 AnySat 特征对 BraDD 任务仍然具有一定迁移价值。

预测可视化示例：

![BraDD Prediction Visualization](.media/bradd_prediction_examples.png)

---

## 可视化脚本

### 绘制 BraDD Loss 与 IoU 曲线

```bash
python src/visualization/plot_loss_acc_curve.py
```

输出示例：

```text
src/visualization/outputs/bradd_loss_iou_curve.png
```

### 绘制 TimeSen2Crop 混淆矩阵

可根据测试集预测结果与真实标签绘制混淆矩阵，用于分析不同作物类别之间的误分类关系。

---

## 已知问题与踩坑记录

### 1. BraDD 指标维度不匹配

在 BraDD 语义分割评估中曾出现如下错误：

```text
RuntimeError: Predictions and targets are expected to have the same shape,
but got torch.Size([4608, 2]) and torch.Size([96, 48, 2]).
```

原因是预测结果和标签在展平、one-hot 编码和维度排列时没有严格对齐。

更合理的做法是先得到预测类别图：

```python
pred_label = pred.argmax(dim=1).long()  # [B, H, W]
label = gt["label"].long()              # [B, H, W]
```

然后再进行 one-hot 和维度重排：

```python
pred_oh = torch.nn.functional.one_hot(
    pred_label, num_classes=num_classes
).permute(0, 3, 1, 2)

label_oh = torch.nn.functional.one_hot(
    label, num_classes=num_classes
).permute(0, 3, 1, 2)
```

从而保证指标计算时预测和标签形状一致。

### 2. Jittor Loss 停滞或异常

如果出现以下现象：

```text
loss 一直为 0.000000
loss 一直为 1.000000
logits 全 0 时 cross entropy 不是 log(num_classes)
```

应优先运行最小算子测试，而不是继续修改训练代码。若 CUDA 模式异常但 CPU 模式正常，建议先用 CPU 跑通 Jittor 下游逻辑。

### 3. 大文件上传限制

部分 `.npz` 特征文件较大，若单文件超过 GitHub LFS 限制，需要拆分为 7-Zip 分卷：

```bash
7z a -v1900m train_features_npz.7z train_features.npz
```

解压时：

```bash
7z x train_features_npz.7z.001
```

### 4. 原始数据集不随仓库分发

原始 TimeSen2Crop 和 BraDD-S1TS 数据集请从官方链接下载。本仓库只保留代码、配置、部分特征文件或压缩分卷，以及实验可视化结果。

---

## 参考文献

- **AnySat: One Earth Observation Model for Many Resolutions, Scales, and Modalities.** CVPR 2025.
- **TimeSen2Crop: A Million Labeled Samples Dataset of Sentinel 2 Image Time Series for Crop-Type Classification.** IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2021.
- **Deforestation Detection in the Amazon with Sentinel-1 SAR Image Time Series.** ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences, 2023.

---

## License

本项目遵循 MIT License。详情见：

```text
LICENSE
```

---

## 致谢

感谢 AnySat 原作者提供的遥感多模态基础模型思路与开源代码基础。感谢 TimeSen2Crop 和 BraDD-S1TS 数据集作者提供公开数据资源。
