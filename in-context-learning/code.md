# 代码修改指南

好——既然你决定用 **连续（Gaussian）扩散模型** 来建模线性回归任务，我把整个思路、数学公式、训练与采样流程、代码级修改建议（针对你当前基于 Transformer/GPT2 的实现）都给你一套可直接落地的方案。尽量把重点点到为止：**你可以把问题看成条件连续扩散（conditional DDPM）——模型条件化于 x，目标恢复 y（连续）**。

我会分成四部分：

1. 概念与公式（最重要的数学）
2. 训练目标与实现要点（如何构造数据、loss、噪声调度、t embedding）
3. 采样（从噪声恢复 y 的步骤）
4. 针对你代码的具体改动建议和 PyTorch 代码片段（可直接替换/插入）

---

## 1) 基本概念与数学（简洁版）

把 ($y\in\mathbb{R}^{p}$) 看作要建模的连续向量。标准的离散时间 DDPM 前向过程（添加高斯噪声）：

- 令时间步 ($t\in{1,\dots,T}$)。定义 noise schedule (${\beta_t}$)（例如线性或 cosine），并定义
    
    $$
    \alpha_t = 1-\beta_t,\quad \bar\alpha_t = \prod_{s=1}^{t}\alpha_s
    $$
    
- 前向采样（closed form）：
    
    $$
    q(y_t\mid y_0) = \mathcal{N}\big(y_t;\ \sqrt{\bar\alpha_t}, y_0,\ (1-\bar\alpha_t)\mathbf{I}\big)
    $$
    
    等价地，你可以从 (y_0) 直接采样任意 (t)：
    
    $$
    y_t = \sqrt{\bar\alpha_t},y_0 + \sqrt{1-\bar\alpha_t},\varepsilon,\quad \varepsilon\sim\mathcal{N}(0,I)
    $$
    

训练目标（简化的 DDPM objective，常用）——让模型预测噪声 (\varepsilon)：

$$
\mathcal{L}{\text{simple}}=\mathbb{E}{y_0,\varepsilon,t}\big[|\varepsilon - \varepsilon_\theta(y_t,t,x)|^2\big].
$$

这里 ($\varepsilon_\theta$) 是条件模型，输入为被噪声化的 ($y_t$)，条件 ($x$)（你的 context/feature），以及时间步 ($t$)。

（你也可以让模型直接预测 ($y_0$)，或预测噪声两者等价通过变换，但预测噪声是最常见且训练稳定的方法。）

---

## 2) 训练细节与实现要点（关键）

### 2.1 数据与归一化

- **标准化 ($y_0$)**：对每维做 zero-mean unit-variance 标准化（或 min-max），能够显著提升训练稳定性。训练后采样再反归一化回原尺度。
- ($x$)（context / features）也应归一化/标准化。

### 2.2 噪声调度 ($\beta_t$)

- 常用：线性 schedule（从 ($1e^{-1}$) 到 ($2e^{-2}$)），或 cosine schedule。
- 不需要太大 ($T$) 才能跑实验：可先用 ($T=200$) 或 ($T=500$) 进行快速试验，再扩到 1000。

### 2.3 时间步 embedding

- 需要给模型明确时间条件 (t)。常见做法：把 (t) 转为 **sinusoidal** 或 learned embedding，再加入到每个 token /每个位置的表示中（可广播到所有位置）。
- 对于你把 (x) 和 (y) 合并为序列的结构，把 time embedding 加到 **y 的 token embedding**（或直接加到所有位置的 embeds）是合理的。

### 2.4 条件化（如何把 x 条件放进去）

两种主流方式：

- **拼接（concatenate）**：把 x 的 embedding 与 (y_t) 的 embedding 交替/拼接，送到 Transformer，跟你现有的 `_combine` 思路一致。然后模型通过自注意力学习条件关系。
- **Cross-attention**：如果想更模块化，可把 backbone 设为 encoder-decoder：encoder 编码 x，decoder（带 cross-attention）对 (y_t) 进行 denoise 预测。对 GPT2 改造成带 cross-attention 比较复杂但更清晰。
    
    你的 current `_combine` 拼接方式是可行的。
    

### 2.5 Loss

- **MSE loss on predicted noise**：($|\varepsilon - \varepsilon_\theta(y_t,t,x)|^2$)。这等价于 Gaussian NLL（若使用等方差）。
- 不要用 CrossEntropy（除非你把 y 离散化成 bins）。

### 2.6 是否需要 mask／部分观测（in-context learning 场景）

- 如果你想模拟 In-Context Learning（给部分已知 input–output 对作为上下文，然后预测新点的 y），可以把已知的 (y) 不加噪（或加少量噪），而只对目标预测点加噪。即训练时随机选择一些索引作为“context”（保持 clean 或低噪），其余为被噪声化的目标点。训练目标仍然是预测被噪点的噪声。
- 但为了与标准 DDPM 一致先做简单版本（全部 y 加噪并学习还原），再做部分观测的 conditional 训练。

---

## 3) 采样（从 noise 恢复 y）

我们用经典一步步反向采样（DDPM）：

初始化： ($y_T\sim \mathcal{N}(0,I)$).

对 ($t=T$) 到 ($1$) 迭代：

1. 由模型估计 ($\hat\varepsilon=\varepsilon_\theta(y_t,t,x)$)。
2. 通过公式估计 ( $\hat y_0 = \frac{1}{\sqrt{\bar\alpha_t}} (y_t - \sqrt{1-\bar\alpha_t},\hat\varepsilon)$)（可选）。
3. 计算模型分布 ($p(y_{t-1}\mid y_t)$) 的均值 ($\mu_\theta$)（DDPM 有 closed form），然后采样：
    
    $$
    y_{t-1} = \mu_\theta(y_t,t,x) + \sigma_t z,\quad z\sim\mathcal{N}(0,I)
    $$
    
    其中 ($\sigma_t$) 是已知的方差（取决于 ($\beta_t$)）。
    
4. 最终 ($y_0$) 即为模型生成结果，最后再反归一化回原尺度。

（可用改进的采样如 DDIM 做更少步的确定性采样。）

---

## 4) 针对你代码的具体建议与示例

下面我给出 **可直接替换/插入** 的 PyTorch 风格代码片段，说明如何把你现有的 `DiffusionTransformerModel` 改造成 conditional continuous DDPM。

> 关键变动点：
> 
> - 前向过程不用 `mask_token_id`，而是用加高斯噪声得到 `y_t`（连续）。
> - 模型预测 noise ($\hat\varepsilon$)。 `loss=MSE(ε, ε_hat)`
> - 增加 time embedding；把 x 作为条件（你已有 combine 可继续使用）。
> - 标准化 y（用户代码外处理）。

### 4.1 噪声 schedule & 辅助函数

```python
import math
import torch
import torch.nn as nn

def make_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, timesteps)

class DiffusionSchedule:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=2e-2, device='cpu'):
        self.timesteps = timesteps
        betas = make_beta_schedule(timesteps, beta_start, beta_end).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        # Precompute posterior variance etc if needed

```

### 4.2 q_sample（从 y0 直接得到 y_t）：

```csharp
def q_sample(y0, t, schedule: DiffusionSchedule, noise=None):
    # y0: (b, p) or (b, p, d)
    if noise is None:
        noise = torch.randn_like(y0)
    sqrt_alphas_cumprod_t = schedule.sqrt_alphas_cumprod[t].view(-1, *([1] * (y0.dim()-1)))
    sqrt_one_minus_alphas_cumprod_t = schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, *([1] * (y0.dim()-1)))
    return sqrt_alphas_cumprod_t * y0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

```

`t` 可以对每个 batch 随机采样：`t = torch.randint(0, T, (b,))`。注意上面需要按 batch 索引 `schedule.*[t]`。

### 4.3 时间嵌入（sinusoidal）

```python
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    @staticmethod
    def sinusoidal_embedding(timesteps, dim):
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32) / half)
        args = timesteps[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if dim % 2 == 1:  # pad
            emb = torch.cat([emb, torch.zeros(len(timesteps), 1)], dim=-1)
        return emb

    def forward(self, t):
        # t: LongTensor shape (b,)
        emb = self.sinusoidal_embedding(t, self.dim).to(t.device)
        return self.proj(emb)

```

### 4.4 模型部分：把 time embedding 加入到 embeds；模型输出为 noise prediction

改动说明（基于你现有 `_combine` 与 GPT2 backbone）：

- `zs = self._combine(xs, y_t)` 生成序列输入。
- `embeds = self._read_in(zs)` 得到 embedding（ `shape (b, 2*points, n_embd)`）。
- 把 time embedding `t_emb` 加到 **所有 y 位置的 embedding**（或直接加到 embeds 的每个位置）：
    
    ```python
    # t_emb: (b, n_embd)
    embeds = embeds + t_emb[:, None, :]  # 广播到序列长度
    ```
    
- backbone 输出后，从 y 的位置抽出 hidden states 并用 `_read_out` 得到 noise prediction `eps_hat`（shape same as y_t）。
- Loss：`loss = ((eps_hat - eps) ** 2).mean()`（平均在 batch 与所有 y dims 上；如需要可按维度加权）

示例 `forward`（仅展示核心思路）：

```python
def forward(self, xs, y_t, t, cond_mask=None):
    # xs: (b, p, dim_x)
    # y_t: (b, p) or (b, p, d_y)
    # t: (b,)
    zs = self._combine(xs, y_t)  # as you had
    embeds = self._read_in(zs)   # (b, 2*p, n_embd)
    t_emb = self.time_embed(t)   # (b, n_embd)
    embeds = embeds + t_emb[:, None, :]  # broadcast
    out = self._backbone(inputs_embeds=embeds).last_hidden_state
    eps_hat = self._read_out(out[:, 1::2, :])  # mapping to noise shape
    return eps_hat.squeeze(-1)  # (b, p)

```

### 4.5 完整 compute_loss 示例（训练步骤）

```python
def compute_loss(self, xs, y0, schedule: DiffusionSchedule):
    b = y0.shape[0]
    t = torch.randint(0, schedule.timesteps, (b,), device=y0.device)
    y_t, noise = q_sample(y0, t, schedule)  # y_t and the true eps
    eps_hat = self.forward(xs, y_t, t)
    loss = torch.mean((noise - eps_hat) ** 2)
    return loss, eps_hat, noise

```

> 如果你要做部分观测（i.e., 给一部分 $(x_i,y_i)$ 作为 context 不加噪），那 q_sample 只对目标索引加噪，保持 context 索引的 $y_t = y_0$（或加更小噪）。模型仍以相同方式预测被噪位置的噪声；在 forward 中可以通过 cond_mask 标记哪些是 context，不参与噪声化和 loss。
> 

### 4.6 采样示例（简化版 DDPM step）

```python
@torch.no_grad()
def p_sample_loop(self, xs, schedule: DiffusionSchedule, shape):
    # shape = (b, p)
    device = xs.device
    y_t = torch.randn(shape, device=device)
    for t_ in reversed(range(schedule.timesteps)):
        t = torch.full((shape[0],), t_, device=device, dtype=torch.long)
        eps_hat = self.forward(xs, y_t, t)
        # compute predicted x0
        sqrt_recip_alphacum = 1.0 / schedule.sqrt_alphas_cumprod[t_]
        sqrt_one_minus = schedule.sqrt_one_minus_alphas_cumprod[t_]
        y0_pred = sqrt_recip_alphacum * (y_t - sqrt_one_minus * eps_hat)
        # compute mean of p(y_{t-1}|y_t)
        # using DDPM formula (mu_theta)
        beta_t = schedule.betas[t_]
        alpha_t = schedule.alphas[t_]
        alphas_cumprod_t = schedule.alphas_cumprod[t_]
        # simplified mu calculation
        coef1 = beta_t * torch.sqrt(alphas_cumprod_t) / (1.0 - alphas_cumprod_t)
        coef2 = (1.0 - alpha_t) * torch.sqrt(1.0 - alphas_cumprod_t) / (1.0 - alphas_cumprod_t)
        mu = coef1.unsqueeze(-1) * y0_pred + coef2.unsqueeze(-1) * y_t
        if t_ > 0:
            noise = torch.randn_like(y_t)
            sigma = torch.sqrt(beta_t).unsqueeze(-1)
            y_t = mu + sigma * noise
        else:
            y_t = mu
    return y_t  # final y0 (remember to denormalize)

```

（上面是示意；实际实现时请确保对 `t` 的 batch 索引取值形状正确，并注意 `.view()` 与 broadcasting。）

---

## 5) 对你原代码的具体改动建议（清单）

1. **移除 `mask_token_id` 的用法**：不要把掩码替换成一个大整数。
2. **把 forward_process 改成 q_sample：** 返回 `y_t, noise, t` 而不是 `masked_indices`。
3. **增加 `TimeEmbedding` 模块并在 forward 中加入**（如上）。
4. **_read_in 输入**：把原本直接把 scalar y（或向量）当做数值输入的逻辑改成 **先做一层小的 MLP 把 y（或 y_t）映射到 embedding space**，例如 `nn.Linear(1, n_embd)` 或 `nn.Sequential(nn.Linear(1,n_embd),nn.SiLU(),nn.Linear(n_embd,n_embd))`。
5. **_read_out 输出**：保持 `Linear(n_embd, out_dim)`，这里 out_dim 等于 y 的维度（或 1）。模型输出为噪声预测值（与 noise 同维度）。
6. **归一化/反归一化**：在 dataset 层实现 `y = (y - mean)/std`，采样后 `y = y * std + mean`。
7. **conditional training（ICL）**：如果你要在 training 中模拟 in-context learning，设计一个 `context_mask`：对于 context indices，把 `y_t = y0` 或 `y_t = y0 + small_noise`；目标 indices 才是真正的 `q_sample`。在 loss 上只计算目标 indices 的 MSE。

---

## 6) 超参数建议（起点）

- T = 200 ~ 1000（先 200 试验）
- beta linear from 1e-4 to 2e-2（或尝试 cosine）
- batch size 根据内存，先 64 或 128（小模型）
- optimizer: AdamW, lr=1e-4 或 5e-5，weight_decay 小量
- 标准化 y（非常重要）

---

## 7) 常见问题 & 排查建议

- **训练不收敛 / loss 不下降**：先确认 y 已标准化；确认模型输出的尺度匹配 noise（噪声方差是 1）；确认 time embedding 被正确加入；确认 `q_sample` 正确使用 `sqrt_alphas_cumprod[t]` 的 batch 索引。
- **模型输出数值爆炸**：可能是你之前把 mask token 用巨大整数导致的残留权重，需要重新初始化模型并清理 optimizer state。
- **若想把任务做成“离散化 + CE”**：那需要把 y 离散化为 bins（会改变问题性质），不推荐作为第一步。
- **如果要测 In-Context Learning**：先把 conditional setup 做正确（把 context 点作为 clean/轻噪），训练时随机采不同数量 context，让模型学到“看到几个 (x,y) 对后推断新点”。

---

如果你愿意，我可以：

- 帮你把上面示例整合成 **一个最小可运行的 PyTorch class（完整）**，把你原来类替换成这个版本并给出训练 loop；或者
- 根据你当前 repo/文件（你粘贴关键函数）来**逐行指出应改的代码并给出替换片段**（比如替换 `forward_process`, `compute_loss`, `forward` 中几行）。

你更希望我直接给出 **完整的可运行 class**，还是按你目前代码逐段修改？