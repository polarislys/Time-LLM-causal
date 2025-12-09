# 因果模块Bug修复说明 (2024-12-07)

## 问题描述

用户报告启用自适应因果模块后，性能没有提升：
- **启用因果模块**: MSE = 0.391
- **不启用因果模块**: MSE = 0.392
- **差异**: 几乎没有改善

## 根本原因分析

经过深入分析，发现了**三个严重的bug**导致因果模块实际上没有正常工作：

### Bug 1: trends shape不匹配 ⚠️ **最严重**

**位置**: `models/TimeLLM.py` 第222-235行

**问题**:
```python
# 原代码
B, T, N = x_enc.size()
x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)  # reshape成 (B*N, T, 1)
trends = x_enc.diff(dim=1).sum(dim=1)  # shape: (B*N, 1)

# 传给 get_adaptive_causal_prompt
causal_prompt_list = self.causal_module.get_adaptive_causal_prompt(trends)
```

**期望**:
- `get_adaptive_causal_prompt` 期望输入 shape 为 `(batch_size, n_vars)`
- 即每个batch中每个变量的趋势

**实际**:
- 传入的 trends shape 是 `(B*N, 1)`
- 每个变量被当作独立的样本处理
- `trends_np[b, cause_idx]` 索引会越界或访问错误数据

**影响**: 自适应prompt生成逻辑完全失效

**修复**:
```python
# 修复后的代码
B, T, N = x_enc.size()

# 在reshape之前计算trends（需要保持(B, N)的形状）
trends_per_var = x_enc.diff(dim=1).sum(dim=1)  # (B, T-1, N) -> (B, N)

x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

# 使用正确shape的trends
if self.use_causal and hasattr(self, 'causal_module'):
    batch_prompts = self.causal_module.get_adaptive_causal_prompt(trends_per_var)
    # 将batch prompts扩展到 (B*N) 个
    for b_idx in range(B):
        for n_idx in range(N):
            causal_prompt_list[b_idx * N + n_idx] = batch_prompts[b_idx]
```

---

### Bug 2: 因果模块未初始化 ⚠️ **致命**

**位置**: `run_main.py` - 缺少初始化代码

**问题**:
- `CausalModule` 的 `initialize_causal_discovery()` 方法从未被调用
- 导致以下变量未设置:
  - `self.var_names = None`
  - `self.causal_discovery = None`
  - `self.text_generator = None`
  - `self.causal_relationships = None`

**影响**:
```python
def get_adaptive_causal_prompt(self, trends):
    if self.text_generator is None or not self.causal_relationships:
        return [""] * trends.shape[0]  # 直接返回空字符串！
```

因果prompt始终为空，模块完全不工作！

**修复**:
在 `run_main.py` 中模型创建后添加初始化:
```python
# 初始化因果模块（如果启用）
if args.use_causal and hasattr(model, 'causal_module'):
    # 获取变量名（从数据集）
    if args.data == 'ETTh1' or args.data == 'ETTh2':
        var_names = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    elif args.data == 'ETTm1' or args.data == 'ETTm2':
        var_names = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    elif args.data == 'ECL':
        var_names = [f'MT_{i}' for i in range(args.enc_in)]
    elif args.data == 'Weather':
        var_names = ['p', 'T', 'Tpot', 'Tdew', 'rh', 'VPmax', 'VPact', 'VPdef', 'sh', 'H2OC', 'rho', 'wv', 'max_wv', 'wd']
    else:
        var_names = [f'var_{i}' for i in range(args.enc_in)]
    
    model.causal_module.initialize_causal_discovery(var_names)
    print(f"Causal module initialized with {len(var_names)} variables")
```

---

### Bug 3: 冗余代码

**位置**: `models/TimeLLM.py` 第271-278行

**问题**:
```python
# 第271行：先拼接一次
llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)

# 第273-278行：if-else又重新赋值
if self.use_causal and hasattr(self, 'causal_module'):
    causal_tokens = self.causal_module.forward(B * N).to(x_enc.device).to(torch.bfloat16)
    llama_enc_out = torch.cat([prompt_embeddings, causal_tokens, enc_out], dim=1)
else:
    llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
```

**影响**: 
- 第271行是多余的，会被覆盖
- 不影响功能，但代码冗余

**修复**:
删除第271行，直接使用if-else块

---

## 修复总结

### 修改的文件

1. **`models/TimeLLM.py`**
   - 修复 trends 计算逻辑
   - 修复 causal_prompt_list 生成逻辑
   - 删除冗余代码

2. **`run_main.py`**
   - 添加因果模块初始化代码
   - 根据数据集类型设置变量名

### 预期效果

修复后，因果模块应该能够：
1. ✅ 正确计算每个变量的趋势
2. ✅ 根据趋势生成自适应因果prompt
3. ✅ 将因果知识正确注入到LLM
4. ✅ 提供可解释的预测

### 性能预期

修复后的预期改善：
- **轻度改善**: MSE降低 0.5-2%（因果关系较弱的数据集）
- **中度改善**: MSE降低 2-5%（因果关系明确的数据集）
- **显著改善**: MSE降低 5-10%（强因果依赖的数据集）

---

## 验证方法

### 1. 检查初始化日志

运行训练时应该看到：
```
Causal module initialized with 7 variables
Loaded 107 causal relationships from cache
```

### 2. 检查prompt内容

在 `models/TimeLLM.py` 的 `forecast` 方法中添加调试代码：
```python
if self.use_causal and hasattr(self, 'causal_module'):
    batch_prompts = self.causal_module.get_adaptive_causal_prompt(trends_per_var)
    print(f"Sample causal prompt: {batch_prompts[0][:200]}")  # 打印前200个字符
```

应该看到类似输出：
```
Sample causal prompt: Adaptive Causal: Since HUFL increases, HULL is likely to increase; Since MUFL decreases, LUFL is likely to decrease; ...
```

### 3. 对比实验

建议进行以下对比：
- **Baseline**: 不使用因果模块
- **Fixed Causal**: 使用修复后的因果模块
- **With Loss**: 使用因果模块 + 因果一致性loss

---

## 重新运行实验

### 清理旧结果

```bash
# 删除旧的checkpoint（可选）
rm -rf checkpoints/long_term_forecast_ETTh1_512_96_*causal*

# 保留因果缓存（避免重新计算）
# causal_results/ 目录保留
```

### 运行修复后的代码

```bash
cd /home/nl/disk_8T/lys/Time-LLM
bash scripts/TimeLLM_ETTh1_GPT2.sh
```

### 预期日志输出

```
Causal module initialized with 7 variables
Loaded 107 causal relationships from cache
Training iteration: 1/1
Epoch: 1 | Train Loss: 0.xxx
...
```

---

## 技术细节

### 为什么之前没有报错？

1. **Python的鸭子类型**: `trends_np[b, cause_idx]` 即使索引错误也可能返回值
2. **空prompt回退**: `get_adaptive_causal_prompt` 返回空字符串而不是抛出异常
3. **静默失败**: 因果模块未初始化时返回空prompt，不影响主流程

### 为什么性能几乎相同？

因为bug导致：
- 自适应prompt实际上是空字符串
- 因果soft tokens虽然被计算，但基于空的因果关系矩阵
- 模型实际上在没有因果信息的情况下训练

---

## 后续建议

### 1. 添加断言检查

在 `CausalModule` 中添加：
```python
def get_adaptive_causal_prompt(self, trends):
    assert self.text_generator is not None, "Call initialize_causal_discovery first!"
    assert self.causal_relationships, "No causal relationships found!"
    # ...
```

### 2. 添加日志

在关键位置添加日志输出：
```python
if self.use_causal:
    print(f"Using causal module with {len(self.causal_relationships)} relationships")
    print(f"Sample prompt length: {len(causal_prompt_list[0])}")
```

### 3. 单元测试

创建测试确保：
- trends shape正确
- 因果模块正确初始化
- prompt非空

---

## 联系

如有问题，请检查：
1. 是否看到 "Causal module initialized" 日志
2. 是否看到 "Loaded X causal relationships" 日志
3. prompt是否包含 "Adaptive Causal:" 字样

---

**修复日期**: 2024-12-07  
**修复版本**: v1.1  
**影响范围**: 所有使用 `--use_causal` 参数的实验
