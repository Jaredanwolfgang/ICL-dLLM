import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm

# ==========================================
# 1. 简化的 Diffusion Schedule
# ==========================================
class DiffusionSchedule:
    def __init__(self, timesteps=100, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

# ==========================================
# 2. 强化的 Transformer Model
# ==========================================
class LinearRegressionDiffusion(nn.Module):
    def __init__(self, n_x_dims=1, n_embd=128, n_layer=4, n_head=4, timesteps=100):
        super().__init__()
        self.n_x_dims = n_x_dims
        self.n_y_dims = 1
        self.n_embd = n_embd
        self.timesteps = timesteps

        # 输入投影：直接把 x 和 y 拼起来映射
        # Input shape: (B, P, Dx + Dy) -> (B, P, n_embd)
        self.input_proj = nn.Linear(n_x_dims + 1, n_embd)
        
        # Time Embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, n_embd),
            nn.SiLU(),
            nn.Linear(n_embd, n_embd),
        )

        # Backbone: 使用原生的 Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd, nhead=n_head, dim_feedforward=n_embd*2, 
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True
        )
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        # 输出头
        self.eps_head = nn.Linear(n_embd, 1)
        
        self.schedule = None

    def get_schedule(self, device):
        if self.schedule is None:
            self.schedule = DiffusionSchedule(self.timesteps, device=device)
        return self.schedule

    def forward(self, xs, ys_noisy, t):
        """
        xs: (B, P, Dx)
        ys_noisy: (B, P, 1)
        t: (B,) 
        """
        B, P, _ = xs.shape
        
        # 1. 拼接 x 和 y -> (B, P, Dx+1)
        inp = torch.cat([xs, ys_noisy], dim=-1)
        emb = self.input_proj(inp) # (B, P, embd)

        # 2. Time Embedding
        # t: (B,) -> (B, 1) -> (B, embd)
        t_vec = t.float().unsqueeze(-1) / self.timesteps
        t_emb = self.time_embed(t_vec)
        emb = emb + t_emb[:, None, :] # Broadcast to sequence

        # 3. Transformer
        # 不需要 Mask，因为我们希望看到全文上下文
        out = self.backbone(emb) # (B, P, embd)

        # 4. Predict
        return self.eps_head(out)

    # ==========================================
    # 3. 关键修改：混合训练 Loss
    # ==========================================
    def compute_hybrid_loss(self, xs, ys):
        """
        训练时模拟 ICL：
        随机选择一部分作为 Context (Clean)，一部分作为 Target (Noisy)
        """
        schedule = self.get_schedule(xs.device)
        B, P, _ = ys.shape
        
        # 1. 随机生成 Mask: 1代表要加噪(Target), 0代表保持干净(Context)
        # 策略：对于每个 batch，随机选一个 split point，前面是 Demo，后面是 Query
        # 这样模型能学到任意长度的 Context
        loss_mask = torch.zeros((B, P, 1), device=xs.device)
        t = torch.randint(0, schedule.timesteps, (B,), device=xs.device)
        
        noise = torch.randn_like(ys)
        ys_input = ys.clone()
        
        # 构造混合输入
        for i in range(B):
            # 随机选前 k 个点作为 clean demo (k in [1, P-1])
            k = torch.randint(1, P, (1,)).item()
            
            # 前 k 个点：保持 Clean (t=0)
            # 后 P-k 个点：加噪 (t=t[i])
            
            # 计算加噪后的 target 部分
            # y_noisy = sqrt(alpha)*y0 + sqrt(1-alpha)*eps
            target_idx = slice(k, P)
            
            curr_t = t[i]
            sqrt_alpha = schedule.sqrt_alphas_cumprod[curr_t]
            sqrt_one_minus = schedule.sqrt_one_minus_alphas_cumprod[curr_t]
            
            y_target_noisy = sqrt_alpha * ys[i, target_idx] + sqrt_one_minus * noise[i, target_idx]
            
            # 更新输入：Context部分保持原样(Clean)，Target部分替换为噪声
            ys_input[i, target_idx] = y_target_noisy
            
            # 记录 Loss Mask: 只计算 Target 部分的 Loss
            loss_mask[i, target_idx] = 1.0

        # 2. Forward
        # 注意：这里我们传入的 t 对 Context 来说其实是不对的 (Context 是 t=0)，
        # 但因为 Context 是 Clean 的，模型会学会忽略 t 带来的影响，或者我们可以给 t 设为 0
        # 简单起见，我们统一传 t，模型会自己学会区分 "Clean Data" 和 "Noisy Data" 的特征
        eps_pred = self.forward(xs, ys_input, t)
        
        # 3. Loss (只计算 Target 部分)
        loss = F.mse_loss(eps_pred, noise, reduction='none')
        loss = (loss * loss_mask).sum() / loss_mask.sum().clamp(min=1)
        
        return loss

    @torch.no_grad()
    def sample_icl(self, xs, ys_demo, n_query):
        """
        ICL 推理
        xs: (B, P, Dx) 全部的 x
        ys_demo: (B, P-n_query, 1) 已知的 y (Clean)
        n_query: int 需要预测的最后几个点
        """
        device = xs.device
        schedule = self.get_schedule(device)
        B, P, _ = xs.shape
        n_demo = P - n_query
        
        # 初始化：Demo 部分是 Clean 的，Query 部分是 Pure Noise
        y_query_noisy = torch.randn(B, n_query, 1, device=device)
        
        # 拼接输入
        ys_current = torch.cat([ys_demo, y_query_noisy], dim=1)
        
        for i in reversed(range(schedule.timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            
            # 1. 预测
            eps_pred = self.forward(xs, ys_current, t)
            
            # 只关心 Query 部分的预测
            eps_pred_query = eps_pred[:, n_demo:, :]
            y_curr_query = ys_current[:, n_demo:, :]
            
            # 2. DDPM Step (只更新 Query)
            alpha = schedule.alphas[i]
            alpha_cum = schedule.alphas_cumprod[i]
            beta = schedule.betas[i]
            
            noise = torch.randn_like(y_curr_query) if i > 0 else 0
            
            # mu
            pred_mean = (1 / torch.sqrt(alpha)) * (y_curr_query - (beta / torch.sqrt(1 - alpha_cum)) * eps_pred_query)
            y_prev_query = pred_mean + torch.sqrt(beta) * noise
            
            # 3. 更新输入 (Demo 保持不变，Query 更新)
            ys_current = torch.cat([ys_demo, y_prev_query], dim=1)
            
        return ys_current

# ==========================================
# 4. 训练脚本
# ==========================================
def train_optimized():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 配置：先用 D=1 验证
    BATCH_SIZE = 128
    P = 16
    D = 1 
    STEPS = 3000
    
    model = LinearRegressionDiffusion(n_x_dims=D, n_embd=64, n_layer=3).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    print("--- Start Optimized Training (Hybrid Strategy) ---")
    model.train()
    
    with tqdm(range(STEPS)) as pbar:
        for step in pbar:
            optimizer.zero_grad()
            
            # 1. 生成数据 (D=1)
            # y = wx. w ~ N(0, 1), x ~ N(0, 1) -> y ~ N(0, 1)
            # 不需要额外的归一化，这正是 Diffusion 喜欢的分布
            xs = torch.randn(BATCH_SIZE, P, D, device=DEVICE)
            ws = torch.randn(BATCH_SIZE, D, 1, device=DEVICE)
            ys = torch.bmm(xs, ws)
            
            # 2. 混合 Loss
            loss = model.compute_hybrid_loss(xs, ys)
            
            loss.backward()
            optimizer.step()
            
            if step % 100 == 0:
                pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
            
    # ================= EVALUATION =================
    print("\n--- Evaluation ---")
    model.eval()
    
    # 测试数据
    xs = torch.randn(BATCH_SIZE, P, D, device=DEVICE)
    ws = torch.randn(BATCH_SIZE, D, 1, device=DEVICE)
    ys = torch.bmm(xs, ws)
    
    # 拆分 Demo 和 Query
    n_query = 1
    ys_demo = ys[:, :-n_query, :]
    ys_truth = ys[:, -n_query:, :]
    
    # 采样
    ys_pred_full = model.sample_icl(xs, ys_demo, n_query)
    ys_pred_query = ys_pred_full[:, -n_query:, :]
    
    # 计算指标
    mse = F.mse_loss(ys_pred_query, ys_truth).item()
    var = ys_truth.var().item()
    
    print(f"Baseline Variance: {var:.4f}")
    print(f"Model MSE:         {mse:.4f}")
    
    if mse < var * 0.1:
        print("✅ Success! Model learned the function.")
    else:
        print("⚠️ Still failing.")

    print("\nVis (Truth vs Pred):")
    print(ys_truth[0].detach().cpu().numpy(), " vs ", ys_pred_query[0].detach().cpu().numpy())

if __name__ == "__main__":
    train_optimized()