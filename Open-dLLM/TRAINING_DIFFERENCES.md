# Open-dLLM 训练流程与普通 Transformer 训练的差异

本文档总结了 Open-dLLM（基于 Masked Diffusion Model）与标准因果语言模型训练的主要区别。

## 核心差异概览

Open-dLLM 实现了 **Masked Diffusion Model (MDM)** 训练范式，与传统的自回归因果语言模型训练在数据预处理、模型前向、损失计算和生成方式上都有显著不同。

---

## 1. 数据预处理阶段

### 普通 Transformer 训练
- **输入格式**：连续的 token 序列，使用因果掩码（causal mask）确保每个位置只能看到前面的 token
- **标签对齐**：`labels` 与 `input_ids` 左移一位对齐（token t 预测 token t+1）
- **掩码策略**：仅使用 padding mask 和因果掩码，不进行随机掩码

### Open-dLLM 训练
- **添加 Mask Token**：
  ```python
  # train_torch.py:68-70
  if tokenizer.mask_token is None:
      tokenizer.add_special_tokens({"mask_token": "<M>"})
  ```
- **随机掩码处理**：
  ```python
  # data_collator.py:156-159
  mask_ratio = torch.rand(1, device=input_ids.device).clamp(1/500, 1-1/500)
  mask_indices = torch.rand_like(input_ids.float()) < mask_ratio
  input_ids[mask_indices] = self.mask_token_id
  batch["mask_ratio"] = mask_ratio.repeat(input_ids.size(0))
  ```
- **标签掩码**：只有被 mask 的位置才计算损失
  ```python
  # data_collator.py:181
  batch["labels"][batch["input_ids"] != self.mask_token_id] = IGNORE_INDEX
  ```

**关键点**：每个样本随机生成一个 `mask_ratio`（范围 [1/500, 1-1/500]），并将该比例传递给模型用于损失加权。

---

## 2. 模型前向传播

### 普通 Transformer 训练
- **注意力模式**：严格因果（`is_causal=True`），每个位置只能看到前面的 token
- **前向参数**：标准的 `input_ids`, `attention_mask`, `labels` 等

### Open-dLLM 训练
- **双向注意力**：当传入 `mask_ratio` 时，关闭因果掩码
  ```python
  # modeling_qwen2.py:910-912
  if mask_ratio is not None:
      is_causal = False
      mask_ratio = mask_ratio[..., 1:].contiguous()
  ```
- **额外参数**：`mask_ratio` 张量，形状与序列长度匹配，表示每个位置的掩码比例

**关键点**：MDM 训练需要双向注意力，因为模型需要同时看到被 mask 位置前后的上下文来预测被掩盖的 token。

---

## 3. 损失计算

### 普通 Transformer 训练
- **标准交叉熵损失**：
  ```python
  loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
  loss = loss_fct(logits, labels)
  ```
- **损失范围**：所有非 padding 的 token 都参与计算

### Open-dLLM 训练
- **Diffusion-style 损失**：包含两个额外项
  1. **Path Loss**（路径一致性损失）：
     ```python
     # modeling_qwen2.py:949-950, 966-967
     path_loss = (-loss).exp().detach() * loss
     loss = loss + path_loss
     ```
  2. **Mask Ratio 加权**：
     ```python
     # modeling_qwen2.py:951-952, 968-969
     loss_mask = labels != IGNORE_INDEX
     loss = (loss * loss_mask * (1/mask_ratio)).sum() / (loss_mask.sum() + 1e-8)
     ```
- **损失范围**：只计算被 mask 的位置（`labels != IGNORE_INDEX`）

**关键点**：
- `path_loss` 是 diffusion 训练的核心，通过 `exp(-loss) * loss` 的形式强化低损失路径
- `1/mask_ratio` 加权确保掩码比例高的位置（更难预测）获得更大的权重

---

## 4. 生成/推理方式

### 普通 Transformer 训练
- **自回归生成**：逐 token 生成，每个新 token 基于之前所有 token
- **生成方法**：`model.generate()` 使用标准的采样策略（greedy, top-k, top-p 等）

### Open-dLLM 训练
- **多步扩散生成**：使用 `MDMGenerationMixin` 实现迭代去噪
  ```python
  # generation_utils.py:166-227
  def _mdm_sample(self, x, attention_mask, generation_config):
      # 多步迭代，逐步填充 mask token
      for i in range(steps):
          mask_index = (x == mask_token_id)
          outputs = self(input_ids=x, is_causal=False)  # 双向注意力
          # 采样并更新 mask 位置
  ```
- **生成参数**：
  - `steps`: 扩散步数（默认 100）
  - `temperature`: 采样温度
  - `alg`: 采样算法（"origin", "p2" 等）

**关键点**：MDM 生成是迭代式的，需要多步才能完成，而标准生成是单步自回归的。

---

## 5. 训练循环差异

### 普通 Transformer 训练
```python
# 标准训练循环
for batch in dataloader:
    outputs = model(**batch)  # batch 包含 input_ids, labels
    loss = outputs.loss
    loss.backward()
    optimizer.step()
```

### Open-dLLM 训练
```python
# train_icl.py:228-241
for micro_batch in micro_batches:
    micro_batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                   for k, v in micro_batch.items()}
    # micro_batch 包含 input_ids, labels, mask_ratio
    loss = model(**micro_batch, use_cache=False).loss.mean()
    loss.backward()
```

**关键点**：训练循环本身结构相似，但 `micro_batch` 中多了一个 `mask_ratio` 字段。

---

## 6. 配置差异

### 普通 Transformer 训练配置
```yaml
train:
  enable_masking: false  # 不使用掩码
  # 其他标准配置...
```

### Open-dLLM 训练配置
```yaml
train:
  enable_masking: true   # 启用随机掩码
  # 其他配置...
```

---

## 总结对比表

| 维度 | 普通 Transformer | Open-dLLM (MDM) |
|------|-----------------|-----------------|
| **注意力模式** | 因果（单向） | 双向（当有 mask_ratio 时） |
| **数据预处理** | 无随机掩码 | 随机掩码 + mask_ratio |
| **损失函数** | 标准交叉熵 | 交叉熵 + path_loss + mask_ratio 加权 |
| **损失范围** | 所有非 padding token | 仅被 mask 的 token |
| **生成方式** | 自回归（单步） | 扩散（多步迭代） |
| **特殊 Token** | 不需要 mask token | 需要 `<M>` mask token |
| **训练目标** | 预测下一个 token | 预测被 mask 的 token |

---

## 代码位置参考

- **数据掩码**：`veomni/data/data_collator.py:144-182`
- **模型前向**：`veomni/models/transformers/qwen2/modeling_qwen2.py:836-999`
- **损失计算**：`veomni/models/transformers/qwen2/modeling_qwen2.py:936-981`
- **生成逻辑**：`veomni/models/transformers/qwen2/generation_utils.py:91-339`
- **训练循环**：`tasks/train_icl.py:228-241`

---

## 为什么需要这些差异？

1. **双向注意力**：MDM 需要同时看到被 mask 位置前后的上下文，才能准确预测
2. **Path Loss**：这是 diffusion 模型的核心，通过路径一致性损失提高生成质量
3. **Mask Ratio 加权**：不同掩码比例的任务难度不同，加权确保模型学习平衡
4. **多步生成**：扩散模型需要迭代去噪，而不是一次性生成

这些差异使得 Open-dLLM 能够实现更灵活的生成方式（如代码补全、文本填充等），而不仅仅是自左向右的生成。

