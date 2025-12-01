import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tqdm  # 建议安装 tqdm: pip install tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. 辅助工具 (Schedule & Embedding)
# ==========================================
class DiffusionSchedule:
    def __init__(self, timesteps=100, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # 计算辅助变量 (为了代码清晰，提前算好)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

def extract(a, t, x_shape):
    """
    从张量 a 中提取 t 索引的值，并 reshape 到 x_shape 以便广播
    a: (T,)
    t: (B,)
    x_shape: (B, P, D, ...)
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# ==========================================
# 2. 清晰版 Linear Regression Diffusion
# ==========================================
class LinearRegressionDiffusion(nn.Module):
    def __init__(self, n_x_dims, n_embd=256, n_layer=6, n_head=8, timesteps=100):
        super().__init__()
        self.n_x_dims = n_x_dims
        self.n_y_dims = 1
        self.n_embd = n_embd
        self.timesteps = timesteps

        # 1. 投影层: [x, y] -> embedding
        self.input_proj = nn.Linear(n_x_dims + 1, n_embd)
        
        # 2. 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, n_embd),
            nn.SiLU(),
            nn.Linear(n_embd, n_embd),
        )

        # 3. 主干: Full Attention Transformer (Encoder)
        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=n_embd, 
        #     nhead=n_head, 
        #     dim_feedforward=n_embd * 4, 
        #     dropout=0.0, 
        #     activation="gelu", 
        #     batch_first=True, 
        #     norm_first=True
        # )
        # self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        # GPT2 Config
        config = GPT2Config(
            n_positions=2 * n_positions,  # 序列总长度依然是 2P
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.backbone = GPT2Model(config)

        # 4. 输出层: 预测噪声
        self.eps_head = nn.Linear(n_embd, 1)
        
        self.schedule = None

    def _get_schedule(self, device):
        if self.schedule is None or self.schedule.betas.device != device:
            self.schedule = DiffusionSchedule(self.timesteps, device=device)
        return self.schedule

    # ============================================
    # 核心过程 1: 前向传播 (Model Prediction)
    # ============================================
    def forward(self, xs, ys_current, t):
        """
        xs:         (B, P, Dx) 条件
        ys_current: (B, P, 1)  当前的 y (包含 Context 部分的 Clean 和 Target 部分的 Noisy)
        t:          (B,)       时间步
        """
        # 1. Input Projection
        # 拼接 x 和 y，形成 Transformer 的输入 Token
        inp = torch.cat([xs, ys_current], dim=-1) # (B, P, D+1)
        emb = self.input_proj(inp)

        # 2. Add Time Embedding
        # 将时间 t 归一化后映射，并加到所有 Token 上
        t_vec = t.float().unsqueeze(-1) / self.timesteps
        t_emb = self.time_embed(t_vec)            # (B, n_embd)
        emb = emb + t_emb[:, None, :]

        # 3. Backbone (Global Attention)
        # 此时所有位置 (Context 和 Target) 都能互相看到
        out = self.backbone(emb)

        # 4. Predict Noise
        return self.eps_head(out)

    # ============================================
    # 核心过程 2: 加噪过程 (Diffusion Forward)
    # ============================================
    def q_sample(self, y0, t, noise=None):
        """
        标准加噪公式: y_t = sqrt(alpha_bar) * y0 + sqrt(1-alpha_bar) * eps
        """
        if noise is None:
            noise = torch.randn_like(y0)
            
        schedule = self._get_schedule(y0.device)
        
        sqrt_alphas_cumprod_t = extract(schedule.sqrt_alphas_cumprod, t, y0.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(schedule.sqrt_one_minus_alphas_cumprod, t, y0.shape)
        
        y_t = sqrt_alphas_cumprod_t * y0 + sqrt_one_minus_alphas_cumprod_t * noise
        return y_t, noise

    # ============================================
    # 核心过程 3: 训练 Loss (Hybrid Strategy)
    # ============================================
    def compute_loss(self, xs, ys_gt):
        """
        混合训练策略:
        1. 随机切分序列为 Context (Clean) 和 Target (Noisy)。
        2. Context 帮助模型理解 x->y 的关系。
        3. Target 提供加噪样本供模型去噪。
        """
        B, P, _ = ys_gt.shape
        device = ys_gt.device
        schedule = self._get_schedule(device)

        # 1. 随机采样时间步 t
        t = torch.randint(0, schedule.timesteps, (B,), device=device)

        # 2. 生成 Context/Target 掩码
        # 为每个样本随机选一个切分点 k (k in [1, P-1])
        ks = torch.randint(1, P, (B,), device=device)
        indices = torch.arange(P, device=device).unsqueeze(0).expand(B, P)
        
        # mask_target: True 表示该位置是 Target (需要加噪)，False 表示 Context (保持干净)
        mask_target = (indices >= ks.unsqueeze(1)).unsqueeze(-1).float() # (B, P, 1)

        # 3. 构造输入 y_t (Hybrid)
        # 对整个序列计算加噪版本
        y_noisy_full, noise_true = self.q_sample(ys_gt, t)
        
        # 混合: 如果是 Target 用 Noisy，如果是 Context 用 Clean (GT)
        # y_input = mask * y_noisy + (1-mask) * y_clean
        ys_input = mask_target * y_noisy_full + (1.0 - mask_target) * ys_gt

        # 4. 模型预测
        # 注意：这里传入的 t 对应的是 Target 的噪声程度。Context 虽然是 Clean (t=0)，
        # 但我们统一传 t。模型因为能看到 Context 是 Clean 的，会自动学会利用它。
        eps_pred = self.forward(xs, ys_input, t)

        # 5. 计算 Loss
        # 只在 Target 部分计算 MSE
        loss = F.mse_loss(eps_pred, noise_true, reduction='none')
        loss = (loss * mask_target).sum() / mask_target.sum().clamp(min=1)

        return loss

    # ============================================
    # 核心过程 4: 单步去噪 (Inverse Step)
    # ============================================
    @torch.no_grad()
    def p_sample(self, xs, y_t, t, t_index):
        """
        标准 DDPM 采样步: y_{t-1} = 1/sqrt(alpha) * (y_t - ...) + sigma * z
        """
        schedule = self._get_schedule(xs.device)
        
        # 1. 预测噪声
        eps_pred = self.forward(xs, y_t, t)
        
        # 2. 提取系数
        betas_t = extract(schedule.betas, t, y_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(schedule.sqrt_one_minus_alphas_cumprod, t, y_t.shape)
        sqrt_recip_alphas_t = extract(schedule.sqrt_recip_alphas, t, y_t.shape)
        
        # 3. 计算均值 mu
        # mu = (1 / sqrt(alpha_t)) * (y_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps)
        model_mean = sqrt_recip_alphas_t * (
            y_t - (betas_t / sqrt_one_minus_alphas_cumprod_t) * eps_pred
        )

        # 4. 计算方差 (t > 0 时加噪声)
        if t_index > 0:
            noise = torch.randn_like(y_t)
            # 使用简单的方差 sigma = sqrt(beta)
            posterior_variance_t = extract(torch.sqrt(schedule.betas), t, y_t.shape)
            y_prev = model_mean + posterior_variance_t * noise
        else:
            y_prev = model_mean
            
        return y_prev

    # ============================================
    # 核心过程 5: 完整采样循环 (Inference Loop)
    # ============================================
    @torch.no_grad()
    def p_sample_loop(self, xs, ys_demo, n_query):
        """
        ICL 推理循环:
        xs: 全量 x
        ys_demo: 已知的 y (Clean)
        n_query: 需要预测的点数
        """
        device = xs.device
        B, P, _ = xs.shape
        schedule = self._get_schedule(device)
        n_demo = P - n_query

        # 1. 初始化: Context 是 Clean 的，Query 是纯噪声
        y_query_noisy = torch.randn(B, n_query, 1, device=device)
        ys_current = torch.cat([ys_demo, y_query_noisy], dim=1) # (B, P, 1)
        
        # 2. 迭代去噪
        for i in reversed(range(schedule.timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            
            # 执行一步去噪 (Update Whole Sequence)
            # 虽然我们只关心 Query，但为了利用 Transformer 的全局 Attention，
            # 我们通常把整个序列扔进去预测。
            y_prev_full = self.p_sample(xs, ys_current, t, i)
            
            # 3. 强制 In-painting (Context Replacement)
            # 关键步骤：无论模型对 Demo 部分预测变成什么样，
            # 在每一步结束后，强制把 Demo 部分重置回 Ground Truth。
            # 这样模型在下一步预测 Query 时，总能看到完美的 Context。
            
            # 提取 Query 的更新结果
            y_query_updated = y_prev_full[:, n_demo:, :]
            
            # 重新拼接: Demo (Clean) + Query (Updated)
            ys_current = torch.cat([ys_demo, y_query_updated], dim=1)
            
        return ys_current

# ==========================================
# 3. 大规模训练流程
# ==========================================
def train_large_scale():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- 配置: 扩大规模 ---
    D = 20          # 维度 (Scale Up!)
    P = 64          # 序列长度 (Scale Up!)
    BATCH_SIZE = 64
    STEPS = 10000   # 更多步数
    LR = 3e-4
    
    # 初始化模型
    model = LinearRegressionDiffusion(
        n_x_dims=D, 
        n_embd=256,   # 容量增加
        n_layer=6,    # 深度增加
        n_head=8,     # 头数增加
        timesteps=100
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    # Cosine Annealing 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS)
    
    print(f"--- Starting Large Scale Training (D={D}, P={P}) ---")
    model.train()
    
    pbar = tqdm.tqdm(range(STEPS))
    for step in pbar:
        optimizer.zero_grad()
        
        # 1. 生成高维数据
        xs = torch.randn(BATCH_SIZE, P, D, device=DEVICE)
        
        # 【关键】对 Weights 进行缩放，保持 y 的方差 ~ 1.0
        ws = torch.randn(BATCH_SIZE, D, 1, device=DEVICE) / math.sqrt(D)
        
        ys = torch.bmm(xs, ws) # (B, P, 1)
        
        # 2. Hybrid Loss
        loss = model.compute_hybrid_loss(xs, ys)
        
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        if step % 100 == 0:
            pbar.set_description(f"Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
            
    # ================= EVALUATION =================
    print("\n--- Evaluation on Test Set ---")
    model.eval()
    
    # 生成测试集 (更大 Batch 以求稳)
    TEST_BATCH = 100
    test_xs = torch.randn(TEST_BATCH, P, D, device=DEVICE)
    test_ws = torch.randn(TEST_BATCH, D, 1, device=DEVICE) / math.sqrt(D)
    test_ys = torch.bmm(test_xs, test_ws)
    
    # 设定 Query 数量
    n_query = 5 # 预测最后5个点
    ys_demo = test_ys[:, :-n_query, :]
    ys_truth = test_ys[:, -n_query:, :]
    
    # 采样
    ys_pred_full = model.sample_icl(test_xs, ys_demo, n_query)
    ys_pred_query = ys_pred_full[:, -n_query:, :]
    
    # 计算指标
    mse = F.mse_loss(ys_pred_query, ys_truth).item()
    var_ref = ys_truth.var().item()
    r2_score = 1.0 - (mse / var_ref)
    
    print(f"Baseline Variance: {var_ref:.4f}")
    print(f"Model MSE:         {mse:.4f}")
    print(f"R² Score:          {r2_score:.4f} (Closer to 1.0 is better)")
    
    if r2_score > 0.9:
        print("🚀 Excellent! High-dimensional ICL achieved.")
    elif r2_score > 0.5:
        print("✅ Good. Model is learning, but maybe needs more steps.")
    else:
        print("⚠️ Failed to generalize to high dimensions.")

    # Visual Check (First sample)
    print("\nSample 0 Visualization (Last 5 points):")
    # 只打印前3个维度如果是多维y，这里y是1维
    print("Truth:", ys_truth[0].flatten().detach().cpu().numpy().round(3))
    print("Pred :", ys_pred_query[0].flatten().detach().cpu().numpy().round(3))

if __name__ == "__main__":
    train_large_scale()