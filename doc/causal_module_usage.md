# Time-LLM 因果模块使用指南

## 概述

Time-LLM 集成了因果发现模块，支持自适应因果提示（Adaptive Causal Prompt）和因果一致性损失（Causal Consistency Loss）。本文档说明如何通过 bash 脚本配置不同的运行模式。

---

## 功能特性

### 1. **自适应因果提示（Adaptive Causal Prompt）**
- **功能**：根据每个样本的趋势动态生成因果关系提示
- **优势**：相比静态提示，能更好地捕捉数据的动态特征
- **启用条件**：只要启用 `--use_causal`，自适应提示就会自动生效

### 2. **因果一致性损失（Causal Consistency Loss）**
- **功能**：在训练过程中添加因果约束，确保预测符合因果关系
- **优势**：提高预测的可解释性和一致性
- **启用条件**：需要同时启用 `--use_causal` 和 `--use_causal_loss`

---

## 运行模式配置

### 模式 1：基础模式（无因果模块）

**特点**：标准的 Time-LLM 训练，不使用任何因果信息

**配置示例**：
```bash
# 不添加任何因果相关参数
accelerate launch --mixed_precision bf16 run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_96 \
  --model TimeLLM \
  --data ETTh1 \
  --features M \
  --seq_len 512 \
  --pred_len 96 \
  --batch_size 24 \
  --learning_rate 0.01 \
  --train_epochs 20
```

---

### 模式 2：自适应因果提示模式（推荐）

**特点**：
- ✅ 使用自适应因果提示
- ❌ 不使用因果一致性损失
- **适用场景**：想要利用因果信息但不希望损失函数过于复杂

**Bash 脚本配置**：
```bash
#!/bin/bash
cd "$(dirname "$0")/.." || exit 1

# 基础参数
model_name=TimeLLM
train_epochs=20
learning_rate=0.01
batch_size=24
d_model=32
d_ff=128

# === 因果模块参数 ===
use_causal="--use_causal"              # 启用因果模块（自动启用自适应提示）
causal_cache_dir="./causal_results"
causal_tau_max=4
causal_pc_alpha=0.05
causal_top_k=10

# 训练命令
accelerate launch --mixed_precision bf16 run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_96_AdaptiveCausal \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 512 \
  --pred_len 96 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  $use_causal \
  --causal_cache_dir $causal_cache_dir \
  --causal_tau_max $causal_tau_max \
  --causal_pc_alpha $causal_pc_alpha \
  --causal_top_k $causal_top_k
```

---

### 模式 3：完整因果模式（最强约束）

**特点**：
- ✅ 使用自适应因果提示
- ✅ 使用因果一致性损失
- **适用场景**：需要最强的因果约束和可解释性

**Bash 脚本配置**：
```bash
#!/bin/bash
cd "$(dirname "$0")/.." || exit 1

# 基础参数
model_name=TimeLLM
train_epochs=20
learning_rate=0.01
batch_size=24
d_model=32
d_ff=128

# === 因果模块参数 ===
use_causal="--use_causal"              # 启用因果模块
use_causal_loss="--use_causal_loss"    # 启用因果一致性损失
causal_loss_weight=0.1                 # 因果损失权重（可调节）
causal_cache_dir="./causal_results"
causal_tau_max=4
causal_pc_alpha=0.05
causal_top_k=10

# 训练命令
accelerate launch --mixed_precision bf16 run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_96_FullCausal \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 512 \
  --pred_len 96 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  $use_causal \
  $use_causal_loss \
  --causal_loss_weight $causal_loss_weight \
  --causal_cache_dir $causal_cache_dir \
  --causal_tau_max $causal_tau_max \
  --causal_pc_alpha $causal_pc_alpha \
  --causal_top_k $causal_top_k
```

---

## 参数说明

### 因果模块核心参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use_causal` | flag | False | 启用因果发现模块（自动启用自适应提示） |
| `--use_causal_loss` | flag | False | 启用因果一致性损失 |
| `--causal_loss_weight` | float | 0.1 | 因果损失的权重系数 |

### 因果发现参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--causal_cache_dir` | str | ./causal_results | 因果关系缓存目录 |
| `--causal_tau_max` | int | 4 | 因果发现的最大时间滞后 |
| `--causal_pc_alpha` | float | 0.05 | PCMCI 算法的显著性水平 |
| `--causal_top_k` | int | 10 | 使用前 k 个最强因果关系 |

---

## 模式切换快速指南

### 从基础模式切换到自适应因果模式

**添加以下变量和参数**：
```bash
# 在脚本开头添加
use_causal="--use_causal"
causal_cache_dir="./causal_results"
causal_tau_max=4
causal_pc_alpha=0.05
causal_top_k=10

# 在训练命令中添加
  $use_causal \
  --causal_cache_dir $causal_cache_dir \
  --causal_tau_max $causal_tau_max \
  --causal_pc_alpha $causal_pc_alpha \
  --causal_top_k $causal_top_k
```

### 从自适应因果模式切换到完整因果模式

**额外添加以下变量和参数**：
```bash
# 在脚本开头添加
use_causal_loss="--use_causal_loss"
causal_loss_weight=0.1

# 在训练命令中添加（在 $use_causal 后面）
  $use_causal_loss \
  --causal_loss_weight $causal_loss_weight \
```

### 禁用因果模块

**方法 1：注释掉因果相关参数**
```bash
# use_causal="--use_causal"
# use_causal_loss="--use_causal_loss"
```

**方法 2：在训练命令中移除因果参数**
```bash
# 移除以下行：
#   $use_causal \
#   $use_causal_loss \
#   --causal_loss_weight $causal_loss_weight \
#   --causal_cache_dir $causal_cache_dir \
#   --causal_tau_max $causal_tau_max \
#   --causal_pc_alpha $causal_pc_alpha \
#   --causal_top_k $causal_top_k
```

---

## 超参数调优建议

### `causal_loss_weight` 调优

因果损失权重决定了因果约束的强度：

- **0.05-0.1**：轻度约束，适合大多数场景（推荐）
- **0.1-0.3**：中度约束，适合因果关系明确的数据
- **0.3-0.5**：强约束，可能影响预测精度，谨慎使用

**调优示例**：
```bash
# 尝试不同权重
causal_loss_weight=0.05   # 轻度
# causal_loss_weight=0.1  # 中度（默认）
# causal_loss_weight=0.2  # 较强
```

### `causal_top_k` 调优

控制使用的因果关系数量：

- **5-10**：适合变量较少的数据集（推荐）
- **10-20**：适合变量较多的数据集
- **20+**：可能引入噪声，不推荐

### `causal_tau_max` 调优

控制因果发现的时间滞后范围：

- **2-4**：适合短期依赖（推荐）
- **5-10**：适合长期依赖
- **注意**：值越大，计算开销越大

---

## 实验对比建议

### 消融实验设计

为了评估因果模块的效果，建议进行以下对比实验：

```bash
# 实验 1：基础模型
bash scripts/baseline.sh

# 实验 2：仅自适应提示
bash scripts/adaptive_causal.sh

# 实验 3：完整因果模式
bash scripts/full_causal.sh
```

### 评估指标

- **预测精度**：MSE, MAE
- **因果一致性**：因果损失值
- **训练效率**：训练时间、收敛速度

---

## 常见问题

### Q1: 为什么没有静态提示模式？

**A**: 实验表明自适应提示在所有场景下都优于静态提示，因此移除了静态模式以简化使用。

### Q2: 因果损失会影响训练速度吗？

**A**: 会有轻微影响（约 5-10% 的额外时间），但通常可以通过更好的收敛来弥补。

### Q3: 如何判断是否需要使用因果损失？

**A**: 
- 如果数据集有明确的因果关系，建议使用
- 如果只关注预测精度，可以只用自适应提示
- 如果需要可解释性，强烈建议使用因果损失

### Q4: 因果缓存目录的作用？

**A**: 因果发现是耗时操作，缓存可以避免重复计算。首次运行会生成缓存，后续运行会自动加载。

---

## 完整示例脚本

参考 `scripts/TimeLLM_ETTh1_GPT2.sh` 获取完整的配置示例。

---

## 技术细节

### 自适应因果提示的工作原理

1. 计算每个样本的趋势特征
2. 根据趋势激活相关的因果关系
3. 动态生成针对该样本的因果提示
4. 将提示嵌入到 LLM 的输入中

### 因果一致性损失的计算

```python
# 伪代码
for each causal_relationship:
    delta_cause = past_change_of_cause_variable
    delta_effect_pred = predicted_change_of_effect_variable
    expected_change = delta_cause * causal_strength
    loss += MSE(delta_effect_pred, expected_change)
```

---

## 更新日志

- **2024-12**: 移除静态提示模式，简化配置
- **2024-12**: 添加独立的因果损失控制参数
- **2024-12**: 优化自适应提示生成逻辑

---

## 联系与反馈

如有问题或建议，请提交 Issue 或联系项目维护者。
