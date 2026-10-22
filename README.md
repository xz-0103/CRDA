# 原始 EEG 跨被试 LOSO Few-shot + Duala 关系一致性与被试级特征扰动

## 1. 项目说明

本项目用于原始 EEG 信号的跨被试 Few-shot 分类。

整体网络结构为：

```text
原始 EEG [B, C, T]
        ↓
多尺度时域编码器（Multi-scale Temporal CNN，保持原始时间分辨率）
        ↓
时序 Token 压缩（默认 400 → 100）
        ↓
空间编码器（Spatial CNN）
        ↓
Transformer Encoder（默认处理 100 tokens）
        ↓
时间维平均池化
        ↓
最终表征 z
        ↓
线性分类头
```


1. **多尺度时域编码器（Multi-scale Temporal Encoder）**
2. **被试级特征扰动（Subject-level Feature Perturbation）**
3. **关系一致性损失 `L_rc`（Relational Consistency Loss）**


---

## 2. 模型结构

### 2.1 多尺度时域编码器

输入 EEG 的形状为：

```text
[B, C, T]
```

其中：

- `B`：batch size
- `C`：EEG 通道数
- `T`：时间点数

默认使用三个不同尺度的时域卷积核：

```text
7, 15, 31
```

三个分支分别提取不同时间尺度的 EEG 时域特征，然后进行特征融合。

卷积核可以通过命令行参数修改，**奇数和偶数卷积核现在都支持**：

```bash
--temporal_kernels 7,15,31
--temporal_kernels 8,16,32
--temporal_kernels 10,20,40
```

实现上使用显式的左右非对称 SAME padding，保证无论卷积核是奇数还是偶数，每个分支输出的时间长度都严格保持为 `T`。例如 `k=8` 时左右分别补 `3/4` 个时间点，因此多个尺度仍然可以直接在通道维拼接。

---

### 2.2 时序 Token 压缩（加速版）

为避免 400 个时间点直接进入后续 Spatial CNN 和 Transformer，模型默认在多尺度时域编码器之后加入时间池化：

```text
[B, D, C, 400]
        ↓ AvgPool(stride=4)
[B, D, C, 100]
```

这样原始 EEG 的 400 点仍会完整经过多尺度时域卷积；压缩发生在已经提取好的时序特征上。随后 Spatial CNN 和 Transformer 都只需处理 100 个时间位置。若长度不能整除目标 token 数，代码自动退回 AdaptiveAvgPool。

默认参数：

```bash
--transformer_tokens 100
```

如果希望恢复原始 400-token 路径：

```bash
--transformer_tokens 0
```

位置编码仍保留 `time_segments` 长度，并在进入 Transformer 前线性插值到实际 token 数，因此旧版模型的位置编码参数形状保持兼容。

---

### 2.3 空间编码器

时域特征提取完成后，使用空间 CNN 对 EEG 不同电极之间的空间关系进行建模。

空间卷积核会覆盖全部 EEG 通道，从而学习跨电极空间特征。

---

### 2.4 Transformer

空间编码后的序列输入 Transformer Encoder，用于建模 EEG 的全局时间依赖。默认情况下，`time_segments=400`、`transformer_tokens=100`，因此 Transformer 的序列长度由 400 降为 100；自注意力中与序列长度平方相关的部分理论规模约降到原来的 `1/16`。

Transformer 的以下参数均可以调整：

```bash
--embed_dim
--transformer_layers
--transformer_heads
--ff_dim
--dropout
```

Transformer 输出经过时间维平均池化得到最终 EEG 表征：

```text
z ∈ R^D
```

然后：

```text
z → Linear Classifier → 分类结果
```

---

## 3. Duala 机制的使用方式

本项目只将 Duala 的 `L_rc` 和被试级特征扰动作用于最终 EEG 表征 `z`。

### 3.1 源被试训练

LOSO 每一折中：

```text
除测试被试之外的所有被试
            ↓
      Source Training
            ↓
          50 epoch
            ↓
       得到源模型
```

不额外划分验证集。

源模型训练完成之后，利用全部源被试计算：

```text
类别均值 μ_c
被试/类别特征偏差
源被试参考关系矩阵 S_ref
```

---

### 3.2 关系一致性损失 L_rc

对于每个源被试和类别，根据归一化后的 `z` 计算类别 prototype。

然后计算类别 prototype 之间的余弦相似度，得到类别关系矩阵。

多个源被试的关系矩阵进行聚合，得到：

```text
S_ref
```

新被试 Few-shot 微调时，根据新被试的原始 `z` 构造类别关系矩阵，并与 `S_ref` 对齐。

因此：

```text
L_rc = 新被试类别关系矩阵与 S_ref 之间的差异
```

**注意：`L_rc` 始终作用在原始、未扰动的 `z` 上。**

---

### 3.3 被试级特征扰动

根据源被试的类别级 embedding 统计量计算：

```text
μ_c
```

以及不同源被试在该类别上的特征变化尺度。

新被试 Few-shot 微调时：

```text
原始 z
   ↓
Subject-level Feature Perturbation
   ↓
扰动后的 z_tilde
   ↓
Classifier
```

分类损失使用：

```text
CE(Classifier(z_tilde), y)
```

扰动强度可以通过：

```bash
--perturb_strength
```

调整。

默认：

```bash
--perturb_strength 1.0
```

---

## 4. Few-shot 微调损失

新被试微调阶段：

```text
EEG
 ↓
Encoder
 ↓
原始 z
 ├────────────────→ L_rc
 │
 ↓
被试级特征扰动
 ↓
z_tilde
 ↓
Classifier
 ↓
Cross Entropy
```

最终损失：

```text
L = CE(Classifier(z_tilde), y)
    + λ_rc * L_rc(z, y)
```

其中 `λ_rc` 可以通过：

```bash
--lambda_rc
```

设置。

例如：

```bash
--lambda_rc 0.1
```

---

## 5. 测试阶段

Few-shot 微调完成后，测试阶段**不进行特征扰动**。

测试流程：

```text
EEG
 ↓
Encoder
 ↓
原始 z
 ↓
Classifier
 ↓
预测结果
```

也就是严格按照：

**扰动后的 z 用于微调分类，原始 z 用于最终测试。**

---

## 6. EEG 数据预处理

所有数据最终统一为 200 Hz。

处理顺序：

```text
连续 EEG trial
      ↓
每个 EEG 通道进行 Z-score 标准化
      ↓
必要时重采样到 200 Hz
      ↓
2 秒无重叠窗口切片
      ↓
[C, 400]
```

默认：

```text
target_fs = 200 Hz
time_segments = 400
```

即：

```text
400 / 200 = 2 秒
```

窗口之间没有重叠。

### DREAMER

EEG 原始采样率：

```text
128 Hz
```

训练时重采样：

```text
128 Hz → 200 Hz
```

默认使用 Valence：

```text
< 3  → 类别 0
>= 3 → 类别 1
```

也可以使用：

```bash
--target arousal
--target dominance
```

### DEAP

EEG 原始采样率：

```text
128 Hz
```

只使用前 32 个 EEG 通道。

训练时：

```text
128 Hz → 200 Hz
```

默认 Valence：

```text
< 5  → 类别 0
>= 5 → 类别 1
```

可通过 `--target` 使用：

```text
valence
arousal
dominance
liking
```

### Stroop

EEG：

```text
16 通道
200 Hz
```

三分类：

```text
低压力：1back、stroop1
中压力：2back、stroop2
高压力：3back、stroop3
```

### SEED

```text
62 通道
200 Hz
3 分类
```

直接使用数据集已有的 200 Hz EEG，不额外重采样。

### SEED-IV

```text
62 通道
200 Hz
4 分类
```

三个 session 的数据按照同一个被试进行 LOSO 分组。

### BCI2a

```text
22 通道
200 Hz
4 分类
```

### BCI2b

```text
3 通道
200 Hz
2 分类
```

### EEGMAT

```text
19 通道
200 Hz
2 分类
```

---

## 7. LOSO + Few-shot 验证方式

每一折选择一个被试作为新被试：

```text
被试 1 → Test Subject
其他所有被试 → Source Subjects
```

首先使用所有 Source Subjects：

```text
训练 50 epoch
```

不划分额外验证集。

然后从新被试中，每个类别随机选择：

```text
n 个样本
```

作为 Few-shot support set。

这里的 **n-shot 指每类 n 个 2 秒 EEG window 样本，而不是 n 个 trial。**

支持：

```text
1-shot
5-shot
10-shot
20-shot
```

Few-shot 样本会从最终测试集合中移除，避免数据泄漏。

例如三分类 5-shot：

```text
类别 0：5 个样本
类别 1：5 个样本
类别 2：5 个样本

Few-shot 总样本数 = 15
```

然后在这 15 个样本上进行全模型小学习率微调。

默认：

```text
Source Training:
50 epoch
lr = 1e-3

Few-shot Fine-tuning:
50 epoch
lr = 1e-4
```

微调时**整个网络都参与更新**，不是只训练分类头。

---

## 8. 安装环境

建议创建独立环境：

```bash
conda create -n eeg_duala python=3.10 -y
conda activate eeg_duala
pip install -r requirements.txt
```

---

## 9. 单个实验运行

以下命令全部使用：

```text
nohup ... &
```

因此关闭终端后训练仍可继续运行。

程序本身还会另外生成完整日志，并在日志开头保存实际执行的 Python 命令。

### Stroop 5-shot

```bash
nohup python train.py \
  --dataset stroop \
  --gpu_id 0 \
  --batch_size 64 \
  --num_of_channels 16 \
  --time_segments 400 \
  --transformer_tokens 100 \
  --n_shot 5 \
  --lambda_rc 0.1 \
  --temporal_kernels 7,15,31 \
  > nohup_stroop_5shot.out 2>&1 &
```

### SEED 5-shot

```bash
nohup python train.py \
  --dataset SEED \
  --gpu_id 0 \
  --batch_size 64 \
  --num_of_channels 62 \
  --time_segments 400 \
  --n_shot 5 \
  > nohup_seed_5shot.out 2>&1 &
```

### SEED-IV 5-shot

```bash
nohup python train.py \
  --dataset SEED_IV \
  --gpu_id 0 \
  --batch_size 64 \
  --num_of_channels 62 \
  --time_segments 400 \
  --n_shot 5 \
  > nohup_seediv_5shot.out 2>&1 &
```

### DREAMER 5-shot

```bash
nohup python train.py \
  --dataset DREAMER \
  --target valence \
  --gpu_id 0 \
  --batch_size 64 \
  --num_of_channels 14 \
  --time_segments 400 \
  --n_shot 5 \
  > nohup_dreamer_5shot.out 2>&1 &
```

### DEAP 5-shot

```bash
nohup python train.py \
  --dataset DEAP \
  --target valence \
  --gpu_id 0 \
  --batch_size 64 \
  --num_of_channels 32 \
  --time_segments 400 \
  --n_shot 5 \
  > nohup_deap_5shot.out 2>&1 &
```

### BCI2a 5-shot

```bash
nohup python train.py \
  --dataset BCI2a \
  --gpu_id 0 \
  --batch_size 64 \
  --num_of_channels 22 \
  --time_segments 400 \
  --n_shot 5 \
  > nohup_bci2a_5shot.out 2>&1 &
```

### BCI2b 5-shot

```bash
nohup python train.py \
  --dataset BCI2b \
  --gpu_id 0 \
  --batch_size 64 \
  --num_of_channels 3 \
  --time_segments 400 \
  --n_shot 5 \
  > nohup_bci2b_5shot.out 2>&1 &
```

### EEGMAT 5-shot

```bash
nohup python train.py \
  --dataset EEGMAT \
  --gpu_id 0 \
  --batch_size 64 \
  --num_of_channels 19 \
  --time_segments 400 \
  --n_shot 5 \
  > nohup_eegmat_5shot.out 2>&1 &
```

---

## 10. 主要命令行参数

### GPU

```bash
--gpu_id 0
```

### Batch size

默认：

```bash
--batch_size 64
```

### EEG 通道数

```bash
--num_of_channels 62
```

如果设置为：

```bash
--num_of_channels 0
```

则自动使用对应数据集默认通道数。

### 时间点数

```bash
--time_segments 400
```

200 Hz 下 400 个采样点对应 2 秒。

### n-shot

```bash
--n_shot 1
--n_shot 5
--n_shot 10
--n_shot 20
```

### L_rc 权重

```bash
--lambda_rc 0.1
```

### 特征扰动强度

```bash
--perturb_strength 1.0
```

### 多尺度卷积核

奇数、偶数都可以，例如：

```bash
--temporal_kernels 7,15,31
--temporal_kernels 8,16,32
--temporal_kernels 10,20,40
```

### Source learning rate

```bash
--lr 1e-3
```

### Few-shot learning rate

```bash
--finetune_lr 1e-4
```

---

## 11. 调参示例

例如修改：

- `L_rc` 权重
- 特征扰动强度
- 多尺度卷积核
- Few-shot 学习率

可以运行：

```bash
nohup python train.py \
  --dataset SEED \
  --gpu_id 0 \
  --batch_size 64 \
  --num_of_channels 62 \
  --time_segments 400 \
  --n_shot 10 \
  --lambda_rc 0.05 \
  --perturb_strength 1.0 \
  --temporal_kernels 5,11,21 \
  --finetune_lr 5e-5 \
  > nohup_seed_tune.out 2>&1 &
```

---

## 12. 多组超参数并行运行

项目提供：

```text
run_parallel.py
```

新版 `run_parallel.py` 使用**动态并行任务池**，不是“启动一批 -> 等整批全部结束 -> 再启动下一批”。只要某个 GPU slot 上的实验结束，下一组超参数会立刻补到这个空闲 slot，因此不会因为同一批里某个实验较慢而让其他 GPU 空等。

可以同时组合：

```text
n-shot
lambda_rc
temporal_kernels
```

### 多张 GPU 并行

例如 4 张 GPU，每张 GPU 同时跑 1 个实验：

```bash
nohup python run_parallel.py \
  --dataset SEED \
  --gpus 0,1,2,3 \
  --jobs_per_gpu 1 \
  --n_shots 1,5,10,20 \
  --lambda_rcs 0.05,0.1,0.5 \
  --kernel_sets '8-16-32;10-20-40' \
  --extra '--batch_size 64 --num_of_channels 62 --time_segments 400 --finetune_lr 1e-4' \
  > nohup_parallel_seed.out 2>&1 &
```

最大并发数为：

```text
GPU 数量 × jobs_per_gpu
```

上面的例子会同时运行 4 组配置；任意一组结束后会立即启动下一组。

### 同一张 GPU 同时跑多组超参

如果只有一张 GPU 且显存足够，可以直接：

```bash
nohup python run_parallel.py \
  --dataset SEED \
  --gpus 0 \
  --jobs_per_gpu 2 \
  --n_shots 5,10 \
  --lambda_rcs 0.05,0.1 \
  --kernel_sets '8-16-32;10-20-40' \
  --extra '--batch_size 64 --num_of_channels 62' \
  > nohup_parallel_seed.out 2>&1 &
```

这样 GPU 0 上会同时存在 2 个训练进程。若显存不足出现 CUDA OOM，把 `--jobs_per_gpu` 调回 `1`，或者减小 batch size。

每个并行配置还会自动传入唯一的 `--run_tag`，把结果和日志放到独立目录，避免多个进程同时写同一个文件。

运行前只想检查会生成哪些命令，可以加：

```bash
--dry_run
```

---

## 13. 实验结果保存

结果目录默认格式：

```text
results/<数据集名称>_seed<seed>_<n>shot/
```

例如：

```text
results/SEED_seed42_5shot/
```


```text
```

`run_parallel.py` 还会自动附加唯一 `run_tag`，因此不同并行配置会写入不同目录，不会抢占同一个结果文件。

结果文件：

```text
result_SEED_5shot.txt
```

如果文件已经存在，则自动变成：

```text
result_SEED_5shot_1.txt
result_SEED_5shot_2.txt
...
```

不会覆盖之前的实验结果。

文件开头会记录主要实验参数，例如：

```text
dataset
data_path
target
epochs
finetune_epochs
batch_size
n_shot
seed
gpu_id
channels
time_segments
transformer_tokens
target_fs
lr
finetune_lr
lambda_rc
perturb_strength
temporal_kernels
effective_temporal_kernels
subject_perturbation
relational_consistency_lrc
run_tag
Transformer 参数
```

每完成一个被试，该折结果都会**立即写入并 flush 到 txt 文件**，而不是等待全部 LOSO 完成后再保存。

结果表包含：

```text
Fold
TestSubject
SourceTrainN
FewShotN
TestN
Accuracy
MacroF1
```

最终会统计所有被试的：

```text
Accuracy mean/std
MacroF1 mean/std
```

格式为：

```text
64.39/5.23
```

即：

```text
平均值/标准差
```

最后还会写入总运行时间。

---

## 14. 日志保存

日志目录与结果目录使用相同的 run name 规则：

```text
logs/<数据集名称>_seed<seed>_<n>shot>/
```

并行运行时还会附加唯一 `run_tag`，保证每个子进程拥有独立日志目录。

如果日志文件已经存在，也会自动递增：

```text
train_SEED_5shot.log
train_SEED_5shot_1.log
train_SEED_5shot_2.log
...
```

日志开头会记录实际执行命令，例如：

```text
COMMAND: train.py --dataset SEED --gpu_id 0 --n_shot 5 ...
```

训练过程中控制台的输出同时写入日志，包括：

```text
每个 Source epoch 的 loss / accuracy
每个 Few-shot epoch 的 total loss
classification loss
L_rc
每折 Accuracy
每折 MacroF1
最终结果文件位置
总运行时间
```

---

## 15. 默认数据路径

代码中已经按照提供的数据说明设置默认路径：

```text
Stroop:
/data2/member/Xu/work2/stroop

SEED-IV:
/data2/member/Xu/work2_cross/SEED_IV

DEAP:
/data2/member/Xu/work2/DEAP

DREAMER:
/data2/member/Xu/work2/DREAMER

SEED:
/data2/member/Xu/work2_cross/SEED

BCI2a:
/data2/member/Xu/work2_cross/BCI2a

BCI2b:
/data2/member/Xu/work2_cross/BCI2b

EEGMAT:
/data2/member/Xu/work2_cross/EEGMAT
```

如果以后数据路径变化，可以直接：

```bash
--data_path /新的数据路径
```

覆盖默认路径。

---

## 16. 工程文件说明

```text
eeg_fewshot_duala/
├── train.py
├── models.py
├── data.py
├── duala.py
├── utils.py
├── run_parallel.py
├── requirements.txt
└── README.md
```

### `train.py`

负责：

```text
LOSO
Source 训练
Few-shot 微调
测试
Accuracy / Macro-F1
结果保存
日志
```

### `models.py`

负责：

```text
多尺度时域 CNN
空间 CNN
Transformer
平均池化
分类头
```

### `data.py`

负责：

```text
各数据集读取
Z-score
重采样
2 秒无重叠切片
LOSO 划分
n-shot 样本选择
```

### `module.py`

负责：

```text
源被试统计量
S_ref
Subject-level Feature Perturbation
L_rc
```

### `run_parallel.py`

负责多组超参数并行实验。

---

## 17. 推荐的基础实验

建议先固定一个完整模型基准配置，例如直接使用偶数多尺度卷积核：

```bash
nohup python train.py \
  --dataset SEED \
  --gpu_id 0 \
  --batch_size 64 \
  --num_of_channels 62 \
  --time_segments 400 \
  --n_shot 5 \
  --epochs 50 \
  --finetune_epochs 50 \
  --lr 1e-3 \
  --finetune_lr 1e-4 \
  --lambda_rc 0.1 \
  --perturb_strength 1.0 \
  --temporal_kernels 8,16,32 \
  --embed_dim 128 \
  --transformer_layers 2 \
  --transformer_heads 4 \
  > nohup_seed_baseline.out 2>&1 &
```


---
