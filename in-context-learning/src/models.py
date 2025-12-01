import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel
from transformers import Qwen2Model, Qwen2Config
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Lasso
import warnings
from sklearn import tree
import xgboost as xgb
import math

from base_models import NeuralNetwork, ParallelNetworks


def build_model(conf):
    if conf.family == "gpt2":
        model = TransformerModel(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
        )
    elif conf.family == "qwen2.5":
        model = Qwen2TransformerModel(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
        )
    elif conf.family == "diffusion_gpt2":
        model = DiffusionTransformerModel(
            n_positions=conf.n_positions,
            n_x_dims=conf.n_dims,
            n_y_dims=1,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
            timesteps=getattr(conf, "timesteps", 100),
            beta_start=getattr(conf, "beta_start", 1e-4),
            beta_end=getattr(conf, "beta_end", 2e-2),
        )
    elif conf.family == "diffusion_qwen2":
        model = Qwen2DiffusionTransformerModel(
            n_x_dims=conf.n_dims,
            n_y_dims=1,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
            timesteps=getattr(conf, "timesteps", 1000),
            beta_start=getattr(conf, "beta_start", 1e-4),
            beta_end=getattr(conf, "beta_end", 2e-2),
        )
    else:
        raise NotImplementedError

    return model


def get_relevant_baselines(task_name):
    task_to_baselines = {
        "linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "linear_classification": [
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "sparse_linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ]
        + [(LassoModel, {"alpha": alpha}) for alpha in [1, 0.1, 0.01, 0.001, 0.0001]],
        "relu_2nn_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
            (
                GDModel,
                {
                    "model_class": NeuralNetwork,
                    "model_class_args": {
                        "in_size": 20,
                        "hidden_size": 100,
                        "out_size": 1,
                    },
                    "opt_alg": "adam",
                    "batch_size": 100,
                    "lr": 5e-3,
                    "num_steps": 100,
                },
            ),
        ],
        "decision_tree": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (DecisionTreeModel, {"max_depth": 4}),
            (DecisionTreeModel, {"max_depth": None}),
            (XGBoostModel, {}),
            (AveragingModel, {}),
        ],
    }

    models = [model_cls(**kwargs) for model_cls, kwargs in task_to_baselines[task_name]]
    return models


class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, 1)

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        zs = self._combine(xs, ys)
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        return prediction[:, ::2, 0][:, inds]  # predict only on xs

class Qwen2TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(Qwen2TransformerModel, self).__init__()
        configuration = Qwen2Config(
            vocab_size=1,  # using embeddings directly, not tokens
            hidden_size=n_embd,
            intermediate_size=n_embd * 4,  # Standard ratio for transformers(?)
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            num_key_value_heads=n_head,  
            max_position_embeddings=2 * n_positions,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            use_cache=False,
        )
        self.name = f"qwen2.5_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = Qwen2Model(configuration)
        self._read_out = nn.Linear(n_embd, 1)

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        zs = self._combine(xs, ys)
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        return prediction[:, ::2, 0][:, inds]  # predict only on xs


# --------- Diffusion 线性噪声表 ---------
class DiffusionSchedule:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=2e-2, device="cpu"):
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=device), alphas_cumprod[:-1]], dim=0
        )

        self.betas = betas                      # (T,)
        self.alphas = alphas                    # (T,)
        self.alphas_cumprod = alphas_cumprod    # (T,)
        self.alphas_cumprod_prev = alphas_cumprod_prev  # (T,)

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)

        # posterior variance
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).clamp(min=1e-20)  # (T,)


# --------- 时间嵌入：sinusoidal + MLP ---------
class TimeEmbedding(nn.Module):
    def __init__(self, n_embd, max_period=10000, timesteps=1000):
        super().__init__()
        self.n_embd = n_embd
        self.timesteps = timesteps
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.SiLU(),
            nn.Linear(n_embd, n_embd),
        )

    def forward(self, t):
        """
        t: (B,) int64 in [0, T-1]
        输出: (B, n_embd)
        """
        if t.dtype != torch.float32 and t.dtype != torch.float64:
            t = t.float()
        # 归一化到 [0,1] 有利于数值稳定
        t = t / (self.timesteps - 1)

        half_dim = self.n_embd // 2
        freqs = torch.exp(
            torch.arange(half_dim, device=t.device, dtype=t.dtype)
            * (-math.log(10000.0) / (half_dim - 1))
        )  # (half_dim,)

        # (B, 1) * (half_dim,) -> (B, half_dim)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, n_embd or n_embd-1)

        if emb.shape[-1] < self.n_embd:
            # pad if n_embd is odd
            pad = self.n_embd - emb.shape[-1]
            emb = F.pad(emb, (0, pad))

        return self.mlp(emb)  # (B, n_embd)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import GPT2Config, GPT2Model

class DiffusionTransformerModel(nn.Module):
    def __init__(
        self,
        n_positions=1024,
        n_x_dims=20,
        n_y_dims=1,
        n_embd=768,
        n_layer=12,
        n_head=12,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
    ):
        super().__init__()
        self.n_positions = n_positions
        self.n_x_dims = n_x_dims
        self.n_y_dims = n_y_dims
        self.timesteps = timesteps

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

        # 投影层
        self.x_proj = nn.Linear(n_x_dims, n_embd)
        self.y_proj = nn.Linear(n_y_dims, n_embd)
        
        # Token Type: 0 for x, 1 for y
        self.token_type_embed = nn.Embedding(2, n_embd)
        self.time_embed = TimeEmbedding(n_embd, timesteps=timesteps)

        # 输出头
        self.eps_head = nn.Linear(n_embd, n_y_dims)
        
        self._beta_start = beta_start
        self._beta_end = beta_end
        self._schedule = None

    def _get_schedule(self, device):
        if self._schedule is None or self._schedule.betas.device != device:
            self._schedule = DiffusionSchedule(
                timesteps=self.timesteps,
                beta_start=self._beta_start,
                beta_end=self._beta_end,
                device=device,
            )
        return self._schedule

    # ========== q_sample (保持不变) ==========
    @staticmethod
    def q_sample(y0, t, schedule, noise=None):
        if noise is None:
            noise = torch.randn_like(y0)
        
        sqrt_alphas_cumprod_t = schedule.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = schedule.sqrt_one_minus_alphas_cumprod[t]

        while sqrt_alphas_cumprod_t.dim() < y0.dim():
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)

        y_t = sqrt_alphas_cumprod_t * y0 + sqrt_one_minus_alphas_cumprod_t * noise
        return y_t, noise

    # ========== forward (重点修改：Interleaved Layout) ==========
    def forward(self, xs, y_t, t):
        """
        Input Layout:
            xs:  [x1, x2, ..., xP]
            y_t: [y1, y2, ..., yP]
        
        Target Internal Layout (Interleaved):
            [x1, y1, x2, y2, ..., xP, yP]
        """
        if y_t.dim() == 2:
            y_t = y_t.unsqueeze(-1)

        B, P, _ = xs.shape
        device = xs.device

        # 1. 单独投影
        x_emb = self.x_proj(xs)   # (B, P, n_embd)
        y_emb = self.y_proj(y_t)  # (B, P, n_embd)

        # 2. 交织 (Interleave)
        # 技巧：先 stack 成 (B, P, 2, D)，然后 flatten 成 (B, 2P, D)
        # 结果顺序: x1, y1, x2, y2, ...
        combined_emb = torch.stack([x_emb, y_emb], dim=2) # (B, P, 2, n_embd)
        tok_emb = combined_emb.view(B, P * 2, -1)         # (B, 2P, n_embd)

        # 3. Token Type Embeddings (交替 0, 1, 0, 1...)
        # 构造 [0, 1] 重复 P 次
        type_ids = torch.tensor([0, 1], device=device).repeat(P) # (2P,)
        type_ids = type_ids.unsqueeze(0).expand(B, -1)           # (B, 2P)
        
        tok_emb = tok_emb + self.token_type_embed(type_ids)

        # 4. Time Embeddings (广播到所有 token)
        # t_emb = self.time_embed(t) # (B, n_embd)
        # tok_emb = tok_emb + t_emb[:, None, :]

        # 5. Backbone (GPT2 Causal Attention)
        # 此时的 Causal Mask 允许 y_i 看到 x_i 以及所有之前的 x, y
        out = self.backbone(inputs_embeds=tok_emb).last_hidden_state # (B, 2P, n_embd)

        # 6. 提取 y 对应的输出
        # y 位于索引 1, 3, 5, ... (从 0 开始计数)
        # slicing: [start:stop:step] -> [1::2]
        y_hidden = out[:, 1::2, :] # (B, P, n_embd)

        eps_hat = self.eps_head(y_hidden) # (B, P, n_y_dims)
        return eps_hat

    # ========== compute_loss (逻辑不变，自动适配 forward) ==========
    def compute_loss(self, xs, y0):
        if y0.dim() == 2:
            y0 = y0.unsqueeze(-1)
        device = y0.device
        schedule = self._get_schedule(device)
        B, P, _ = y0.shape

        t = torch.randint(0, schedule.timesteps, (B,), device=device)
        y_t, noise = self.q_sample(y0, t, schedule)
        
        # forward 内部会自动交织 x 和 y
        eps_hat = self.forward(xs, y_t, t)
        
        loss = F.mse_loss(eps_hat, noise)
        return loss, eps_hat, y_t, t

    # ========== p_sample (逻辑不变) ==========
    @torch.no_grad()
    def p_sample(self, xs, y_t, t, schedule):
        # ... 代码与之前完全相同，此处省略 ...
        # (请直接复制之前的 p_sample 实现，因为它是基于 y_t 形状计算的，不受 forward 内部布局影响)
        
        # 为了完整性，这里简写核心逻辑：
        eps_hat = self.forward(xs, y_t, t)
        
        # 获取系数...
        sqrt_recip_alphas_cumprod_t = schedule.sqrt_recip_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = schedule.sqrt_one_minus_alphas_cumprod[t]
        
        def broadcast(arr, target):
            while arr.dim() < target.dim():
                arr = arr.unsqueeze(-1)
            return arr

        x0_pred = broadcast(sqrt_recip_alphas_cumprod_t, y_t) * (
            y_t - broadcast(sqrt_one_minus_alphas_cumprod_t, y_t) * eps_hat
        )
        
        alphas_cumprod_prev_t = schedule.alphas_cumprod_prev[t]
        betas_t = schedule.betas[t]
        alphas_cumprod_t = schedule.alphas_cumprod[t]
        alphas_t = schedule.alphas[t]

        coef_x0 = torch.sqrt(alphas_cumprod_prev_t) * betas_t / (1.0 - alphas_cumprod_t)
        coef_xt = torch.sqrt(alphas_t) * (1.0 - alphas_cumprod_prev_t) / (1.0 - alphas_cumprod_t)
        
        mu = broadcast(coef_x0, y_t) * x0_pred + broadcast(coef_xt, y_t) * y_t
        
        noise = torch.randn_like(y_t)
        posterior_variance_t = schedule.posterior_variance[t]
        nonzero_mask = (t > 0).float().view(-1, *([1]*(y_t.dim()-1)))
        
        y_prev = mu + nonzero_mask * torch.sqrt(broadcast(posterior_variance_t, y_t)) * noise
        return y_prev

    # ========== p_sample_loop (ICL 逻辑不变) ==========
    @torch.no_grad()
    def p_sample_loop(self, xs, shape=None, y_gt_icl=None, mask_icl=None):
        """
        接口不变，用于 ICL 推理
        """
        device = xs.device
        schedule = self._get_schedule(device)

        if shape is None:
            B, P, _ = xs.shape
            shape = (B, P, self.n_y_dims)
        else:
            B, P, _ = shape

        y_t = torch.randn(shape, device=device)

        for t_step in reversed(range(schedule.timesteps)):
            t = torch.full((B,), t_step, device=device, dtype=torch.long)
            
            y_prev = self.p_sample(xs, y_t, t, schedule)
            
            if y_gt_icl is not None and mask_icl is not None:
                if t_step > 0:
                    t_prev = torch.full((B,), t_step - 1, device=device, dtype=torch.long)
                    y_gt_noisy, _ = self.q_sample(y_gt_icl, t_prev, schedule)
                else:
                    y_gt_noisy = y_gt_icl
                
                y_prev = y_gt_noisy * mask_icl + y_prev * (~mask_icl)

            y_t = y_prev

        return y_t

# class DiffusionTransformerModel(nn.Module):
#     """
#     Continuous Diffusion Transformer Model based on GPT2.
    
#     Implements DDPM-style continuous diffusion for regression tasks.
#     Model predicts noise epsilon instead of directly predicting y values.
#     """
    
#     def __init__(
#         self,
#         n_positions=1024,
#         n_dims=1,
#         n_embd=768,
#         n_layer=12,
#         n_head=12,
#         timesteps=1000,
#         beta_start=1e-4,
#         beta_end=2e-2,
#     ):
#         super(DiffusionTransformerModel, self).__init__()
#         self.n_positions = n_positions
#         self.n_dims = n_dims
#         self.timesteps = timesteps
        
#         configuration = GPT2Config(
#             n_positions=2 * n_positions,
#             n_embd=n_embd,
#             n_layer=n_layer,
#             n_head=n_head,
#             resid_pdrop=0.0,
#             embd_pdrop=0.0,
#             attn_pdrop=0.0,
#             use_cache=False,
#         )
        
#         # Use GPT2Model (not LMHeadModel) since we'll add custom read_in/read_out for continuous values
#         self._read_in = nn.Linear(n_dims, n_embd)
#         self._backbone = GPT2Model(configuration)
        
#         # Time embedding for diffusion
#         self.time_embed = TimeEmbedding(n_embd)
        
#         # Output layer: predicts noise (epsilon) instead of y directly
#         self._read_out = nn.Linear(n_embd, 1)  # Output dim is 1 (for noise prediction)
#         # Initialize output layer with reasonable scale
#         nn.init.normal_(self._read_out.weight, mean=0.0, std=0.02)
#         nn.init.zeros_(self._read_out.bias)
        
#         self.name = f"diffusion_gpt2_embd={n_embd}_layer={n_layer}_head={n_head}_T={timesteps}"
        
#         # Initialize schedule (will be moved to device when needed)
#         self._schedule = None
#         self._beta_start = beta_start
#         self._beta_end = beta_end
    
#     @staticmethod
#     def _combine(xs_b, ys_b):
#         """
#         xs_b: continuous input (b, points, dims)
#         ys_b: continuous input (b, points) -> (b, points, 1) -> (b, points, dims)
#         zs: combined input (b, 2 * points, dims)
#         """
#         bsize, points, dim = xs_b.shape
#         ys_b_wide = torch.cat(
#             (
#                 ys_b.view(bsize, points, 1),
#                 torch.zeros(bsize, points, dim - 1, device=ys_b.device),
#             ),
#             axis=2,
#         )
#         zs = torch.stack((xs_b, ys_b_wide), dim=2)
#         zs = zs.view(bsize, 2 * points, dim)
#         return zs
    
#     def _get_schedule(self, device):
#         """Get or create diffusion schedule on the specified device."""
#         if self._schedule is None or self._schedule.betas.device != device:
#             self._schedule = DiffusionSchedule(
#                 timesteps=self.timesteps,
#                 beta_start=self._beta_start,
#                 beta_end=self._beta_end,
#                 device=device
#             )
#         return self._schedule
    
#     @staticmethod
#     def q_sample(y0, t, schedule, noise=None):
#         """
#         Sample y_t from y_0 using the forward diffusion process.
        
#         Args:
#             y0: Clean y values (b, points) or (b, points, d)
#             t: Time steps (b,) - each element in [0, timesteps-1]
#             schedule: DiffusionSchedule instance
#             noise: Optional pre-sampled noise (same shape as y0)
        
#         Returns:
#             y_t: Noisy y values (same shape as y0)
#             noise: The noise that was added (same shape as y0)
#         """
#         if noise is None:
#             noise = torch.randn_like(y0)
        
#         # Get schedule values for each batch element
#         # t is (b,), we need to index into schedule arrays
#         sqrt_alphas_cumprod_t = schedule.sqrt_alphas_cumprod[t].view(-1, *([1] * (y0.dim() - 1)))
#         sqrt_one_minus_alphas_cumprod_t = schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, *([1] * (y0.dim() - 1)))
        
#         y_t = sqrt_alphas_cumprod_t * y0 + sqrt_one_minus_alphas_cumprod_t * noise
#         return y_t, noise
    
#     def forward(self, xs, y_t, t, inds=None):
#         """
#         Forward pass: predict noise epsilon given noisy y_t and time step t.
        
#         Args:
#             xs: continuous input (b, points, dims) - context/features
#             y_t: noisy y values (b, points) - y values at time step t
#             t: time steps (b,) - each element in [0, timesteps-1]
#             inds: Optional indices to return predictions for (default: all y positions)
        
#         Returns:
#             eps_hat: (b, points) - predicted noise for all y positions (or subset if inds provided)
#         """
#         if inds is None:
#             inds = torch.arange(y_t.shape[1])
#         else:
#             inds = torch.tensor(inds)
#             if max(inds) >= y_t.shape[1] or min(inds) < 0:
#                 raise ValueError("inds contain indices where xs and y_t are not defined")
        
#         zs = self._combine(xs, y_t)  # (b, 2 * points, dims)
#         embeds = self._read_in(zs)  # (b, 2*points, n_embd)
        
#         # Add time embedding to all positions
#         # t_emb = self.time_embed(t)  # (b, n_embd)
#         # embeds = embeds + t_emb[:, None, :]  # Broadcast to sequence length
        
#         # Process through backbone
#         output = self._backbone(inputs_embeds=embeds).last_hidden_state  # (b, 2*points, n_embd)
        
#         # Extract predictions for y positions (indices 1, 3, 5, ...)
#         # Model predicts noise (epsilon), not y directly
#         eps_hat = self._read_out(output)  # (b, 2*points, 1)
#         return eps_hat[:, ::2, 0][:, inds]  # (b, len(inds))
    
#     def compute_loss(self, xs, y0):
#         """
#         Compute DDPM loss: predict noise epsilon.
        
#         Process:
#         1. Sample random time steps t ~ Uniform(0, timesteps-1)
#         2. Sample noise epsilon ~ N(0, I)
#         3. Compute y_t = sqrt(alpha_bar_t) * y0 + sqrt(1 - alpha_bar_t) * epsilon
#         4. Predict epsilon_hat = model(xs, y_t, t)
#         5. Compute MSE loss: ||epsilon - epsilon_hat||^2
        
#         Args:
#             xs: Continuous input tensor of shape (b, points, dims)
#             y0: Ground truth y values (b, points)
        
#         Returns:
#             loss: Scalar loss value (MSE on noise prediction)
#             eps_hat: Model predictions of noise (b, points)
#             y_t: Noisy y values used (b, points)
#             t: Time steps used (b,)
#         """
#         device = y0.device
#         schedule = self._get_schedule(device)
#         b, points = y0.shape
        
#         # Sample random time steps for each batch element
#         t = torch.randint(0, schedule.timesteps, (b,), device=device)
        
#         # Sample noise and compute y_t
#         y_t, noise = self.q_sample(y0, t, schedule)
        
#         # Predict noise
#         eps_hat = self.forward(xs, y_t, t)  # (b, points)
        
#         # Compute MSE loss on noise prediction
#         loss = torch.mean((noise - eps_hat) ** 2)
        
#         # Debug output (occasionally)
#         if torch.rand(1).item() < 0.001:  # Sample 1% of batches for debugging
#             noise_std = noise.std().item()
#             eps_hat_std = eps_hat.std().item()
#             error = (noise - eps_hat).abs().mean().item()
#             print(f"[DEBUG] Noise prediction error: {error:.4f}, "
#                   f"Noise std: {noise_std:.4f}, Predicted noise std: {eps_hat_std:.4f}, "
#                   f"Y0 range: [{y0.min().item():.4f}, {y0.max().item():.4f}], "
#                   f"Y_t range: [{y_t.min().item():.4f}, {y_t.max().item():.4f}]")
#             print(f"[DEBUG] sample t[0:4]: {t[:4]}")
#             print(f"[DEBUG] sample noise[0, :4]: {noise[0, :4]}")
#             print(f"[DEBUG] sample eps_hat[0, :4]: {eps_hat[0, :4]}")
        
#         return loss, eps_hat, y_t, t
    
#     @torch.no_grad()
#     def p_sample(self, xs, y_t, t, schedule):
#         """
#         Single step of reverse diffusion process (DDPM sampling step).
        
#         Args:
#             xs: Context/features (b, points, dims)
#             y_t: Noisy y at time t (b, points)
#             t: Time step (b,) - each element in [0, timesteps-1]
#             schedule: DiffusionSchedule instance
        
#         Returns:
#             y_{t-1}: Denoised y at time t-1 (b, points)
#         """
#         # Predict noise
#         eps_hat = self.forward(xs, y_t, t)  # (b, points)
        
#         # Compute predicted y0
#         sqrt_recip_alphas_cumprod_t = schedule.sqrt_recip_alphas_cumprod[t].view(-1, 1)
#         sqrt_one_minus_alphas_cumprod_t = schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
#         y0_pred = sqrt_recip_alphas_cumprod_t * (y_t - sqrt_one_minus_alphas_cumprod_t * eps_hat)
        
#         # Compute mean of p(y_{t-1}|y_t, y0_pred)
#         # Using DDPM formula: mu_theta = coef1 * y0_pred + coef2 * y_t
#         alphas_cumprod_t = schedule.alphas_cumprod[t].view(-1, 1)
#         # Handle t=0 case: use alpha_bar_0 = 1.0
#         t_prev = torch.clamp(t - 1, min=0)
#         alphas_cumprod_tm1 = schedule.alphas_cumprod[t_prev].view(-1, 1)
#         betas_t = schedule.betas[t].view(-1, 1)
#         alpha_t = schedule.alphas[t].view(-1, 1)
        
#         # Posterior mean coefficients
#         coef1 = betas_t * torch.sqrt(alphas_cumprod_tm1) / (1.0 - alphas_cumprod_t + 1e-8)
#         coef2 = (1.0 - alpha_t) * torch.sqrt(1.0 - alphas_cumprod_tm1 + 1e-8) / (1.0 - alphas_cumprod_t + 1e-8)
#         mu = coef1 * y0_pred + coef2 * y_t
        
#         # Sample
#         is_not_final = (t > 0).float().view(-1, 1)
#         # Posterior variance
#         sigma = torch.sqrt(betas_t * (1.0 - alphas_cumprod_tm1 + 1e-8) / (1.0 - alphas_cumprod_t + 1e-8))
#         noise = torch.randn_like(y_t)
#         y_prev = mu + is_not_final * sigma * noise
        
#         return y_prev
    
#     @torch.no_grad()
#     def p_sample_loop(self, xs, shape=None):
#         """
#         Full reverse diffusion sampling loop (DDPM).
        
#         Args:
#             xs: Context/features (b, points, dims)
#             shape: Optional shape for y (b, points). If None, inferred from xs.
        
#         Returns:
#             y0: Generated y values (b, points)
#         """
#         device = xs.device
#         schedule = self._get_schedule(device)
        
#         if shape is None:
#             b, points, _ = xs.shape
#             shape = (b, points)
#         else:
#             b, points = shape
        
#         # Start from pure noise
#         y_t = torch.randn(shape, device=device)
        
#         # Iterate backwards through time steps
#         for t_step in reversed(range(schedule.timesteps)):
#             t = torch.full((b,), t_step, device=device, dtype=torch.long)
#             y_t = self.p_sample(xs, y_t, t, schedule)
        
#         return y_t

class Qwen2DiffusionTransformerModel(nn.Module):
    """
    Continuous Diffusion Transformer Model based on GPT2.
    
    Implements DDPM-style continuous diffusion for regression tasks.
    Model predicts noise epsilon instead of directly predicting y values.
    """
    
    def __init__(
        self,
        n_positions=1024,
        n_dims=1,
        n_embd=768,
        n_layer=12,
        n_head=12,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
    ):
        super(Qwen2DiffusionTransformerModel, self).__init__()
        self.n_positions = n_positions
        self.n_dims = n_dims
        self.timesteps = timesteps
        
        configuration = Qwen2Config(
            vocab_size=1,
            hidden_size=n_embd,
            intermediate_size=n_embd * 4,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            num_key_value_heads=n_head,  
            max_position_embeddings=2 * n_positions,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            use_cache=False,
        )
        
        # Use GPT2Model (not LMHeadModel) since we'll add custom read_in/read_out for continuous values
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = Qwen2Model(configuration)
        
        # Time embedding for diffusion
        self.time_embed = TimeEmbedding(n_embd)
        
        # Output layer: predicts noise (epsilon) instead of y directly
        self._read_out = nn.Linear(n_embd, 1)  # Output dim is 1 (for noise prediction)
        # Initialize output layer with reasonable scale
        nn.init.normal_(self._read_out.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self._read_out.bias)
        
        self.name = f"diffusion_qwen2_embd={n_embd}_layer={n_layer}_head={n_head}_T={timesteps}"
        
        # Initialize schedule (will be moved to device when needed)
        self._schedule = None
        self._beta_start = beta_start
        self._beta_end = beta_end
    
    @staticmethod
    def _combine(xs_b, ys_b):
        """
        xs_b: continuous input (b, points, dims)
        ys_b: continuous input (b, points) -> (b, points, 1) -> (b, points, dims)
        zs: combined input (b, 2 * points, dims)
        """
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs
    
    def _get_schedule(self, device):
        """Get or create diffusion schedule on the specified device."""
        if self._schedule is None or self._schedule.betas.device != device:
            self._schedule = DiffusionSchedule(
                timesteps=self.timesteps,
                beta_start=self._beta_start,
                beta_end=self._beta_end,
                device=device
            )
        return self._schedule
    
    @staticmethod
    def q_sample(y0, t, schedule, noise=None):
        """
        Sample y_t from y_0 using the forward diffusion process.
        
        Args:
            y0: Clean y values (b, points) or (b, points, d)
            t: Time steps (b,) - each element in [0, timesteps-1]
            schedule: DiffusionSchedule instance
            noise: Optional pre-sampled noise (same shape as y0)
        
        Returns:
            y_t: Noisy y values (same shape as y0)
            noise: The noise that was added (same shape as y0)
        """
        if noise is None:
            noise = torch.randn_like(y0)
        
        # Get schedule values for each batch element
        # t is (b,), we need to index into schedule arrays
        sqrt_alphas_cumprod_t = schedule.sqrt_alphas_cumprod[t].view(-1, *([1] * (y0.dim() - 1)))
        sqrt_one_minus_alphas_cumprod_t = schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, *([1] * (y0.dim() - 1)))
        
        y_t = sqrt_alphas_cumprod_t * y0 + sqrt_one_minus_alphas_cumprod_t * noise
        return y_t, noise
    
    def forward(self, xs, y_t, t, inds=None):
        """
        Forward pass: predict noise epsilon given noisy y_t and time step t.
        
        Args:
            xs: continuous input (b, points, dims) - context/features
            y_t: noisy y values (b, points) - y values at time step t
            t: time steps (b,) - each element in [0, timesteps-1]
            inds: Optional indices to return predictions for (default: all y positions)
        
        Returns:
            eps_hat: (b, points) - predicted noise for all y positions (or subset if inds provided)
        """
        if inds is None:
            inds = torch.arange(y_t.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= y_t.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and y_t are not defined")
        
        zs = self._combine(xs, y_t)  # (b, 2 * points, dims)
        embeds = self._read_in(zs)  # (b, 2*points, n_embd)
        
        # Add time embedding to all positions
        t_emb = self.time_embed(t)  # (b, n_embd)
        embeds = embeds + t_emb[:, None, :]  # Broadcast to sequence length
        
        # Process through backbone
        output = self._backbone(inputs_embeds=embeds).last_hidden_state  # (b, 2*points, n_embd)
        
        # Extract predictions for y positions (indices 1, 3, 5, ...)
        # Model predicts noise (epsilon), not y directly
        eps_hat = self._read_out(output[:, 1::2, :])  # (b, points, 1)
        return eps_hat[:, :, 0][:, inds]  # (b, len(inds))
    
    def compute_loss(self, xs, y0):
        """
        Compute DDPM loss: predict noise epsilon.
        
        Process:
        1. Sample random time steps t ~ Uniform(0, timesteps-1)
        2. Sample noise epsilon ~ N(0, I)
        3. Compute y_t = sqrt(alpha_bar_t) * y0 + sqrt(1 - alpha_bar_t) * epsilon
        4. Predict epsilon_hat = model(xs, y_t, t)
        5. Compute MSE loss: ||epsilon - epsilon_hat||^2
        
        Args:
            xs: Continuous input tensor of shape (b, points, dims)
            y0: Ground truth y values (b, points)
        
        Returns:
            loss: Scalar loss value (MSE on noise prediction)
            eps_hat: Model predictions of noise (b, points)
            y_t: Noisy y values used (b, points)
            t: Time steps used (b,)
        """
        device = y0.device
        schedule = self._get_schedule(device)
        b, points = y0.shape
        
        # Sample random time steps for each batch element
        t = torch.randint(0, schedule.timesteps, (b,), device=device)
        
        # Sample noise and compute y_t
        y_t, noise = self.q_sample(y0, t, schedule)
        
        # Predict noise
        eps_hat = self.forward(xs, y_t, t)  # (b, points)
        
        # Compute MSE loss on noise prediction
        loss = torch.mean((noise - eps_hat) ** 2)
        
        # Debug output (occasionally)
        if torch.rand(1).item() < 0.01:  # Sample 1% of batches for debugging
            noise_std = noise.std().item()
            eps_hat_std = eps_hat.std().item()
            error = (noise - eps_hat).abs().mean().item()
            print(f"[DEBUG] Noise prediction error: {error:.4f}, "
                  f"Noise std: {noise_std:.4f}, Predicted noise std: {eps_hat_std:.4f}, "
                  f"Y0 range: [{y0.min().item():.4f}, {y0.max().item():.4f}], "
                  f"Y_t range: [{y_t.min().item():.4f}, {y_t.max().item():.4f}]")
            print(f"[DEBUG] sample t[0:4]: {t[:4]}")
            print(f"[DEBUG] sample noise[0, :4]: {noise[0, :4]}")
            print(f"[DEBUG] sample eps_hat[0, :4]: {eps_hat[0, :4]}")
        
        return loss, eps_hat, y_t, t
    
    @torch.no_grad()
    def p_sample(self, xs, y_t, t, schedule):
        """
        Single step of reverse diffusion process (DDPM sampling step).
        
        Args:
            xs: Context/features (b, points, dims)
            y_t: Noisy y at time t (b, points)
            t: Time step (b,) - each element in [0, timesteps-1]
            schedule: DiffusionSchedule instance
        
        Returns:
            y_{t-1}: Denoised y at time t-1 (b, points)
        """
        # Predict noise
        eps_hat = self.forward(xs, y_t, t)  # (b, points)
        
        # Compute predicted y0
        sqrt_recip_alphas_cumprod_t = schedule.sqrt_recip_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        y0_pred = sqrt_recip_alphas_cumprod_t * (y_t - sqrt_one_minus_alphas_cumprod_t * eps_hat)
        
        # Compute mean of p(y_{t-1}|y_t, y0_pred)
        # Using DDPM formula: mu_theta = coef1 * y0_pred + coef2 * y_t
        alphas_cumprod_t = schedule.alphas_cumprod[t].view(-1, 1)
        # Handle t=0 case: use alpha_bar_0 = 1.0
        t_prev = torch.clamp(t - 1, min=0)
        alphas_cumprod_tm1 = schedule.alphas_cumprod[t_prev].view(-1, 1)
        betas_t = schedule.betas[t].view(-1, 1)
        alpha_t = schedule.alphas[t].view(-1, 1)
        
        # Posterior mean coefficients
        coef1 = betas_t * torch.sqrt(alphas_cumprod_tm1) / (1.0 - alphas_cumprod_t + 1e-8)
        coef2 = (1.0 - alpha_t) * torch.sqrt(1.0 - alphas_cumprod_tm1 + 1e-8) / (1.0 - alphas_cumprod_t + 1e-8)
        mu = coef1 * y0_pred + coef2 * y_t
        
        # Sample
        is_not_final = (t > 0).float().view(-1, 1)
        # Posterior variance
        sigma = torch.sqrt(betas_t * (1.0 - alphas_cumprod_tm1 + 1e-8) / (1.0 - alphas_cumprod_t + 1e-8))
        noise = torch.randn_like(y_t)
        y_prev = mu + is_not_final * sigma * noise
        
        return y_prev
    
    @torch.no_grad()
    def p_sample_loop(self, xs, shape=None):
        """
        Full reverse diffusion sampling loop (DDPM).
        
        Args:
            xs: Context/features (b, points, dims)
            shape: Optional shape for y (b, points). If None, inferred from xs.
        
        Returns:
            y0: Generated y values (b, points)
        """
        device = xs.device
        schedule = self._get_schedule(device)
        
        if shape is None:
            b, points, _ = xs.shape
            shape = (b, points)
        else:
            b, points = shape
        
        # Start from pure noise
        y_t = torch.randn(shape, device=device)
        
        # Iterate backwards through time steps
        for t_step in reversed(range(schedule.timesteps)):
            t = torch.full((b,), t_step, device=device, dtype=torch.long)
            y_t = self.p_sample(xs, y_t, t, schedule)
        
        return y_t

# class Qwen2DiffusionTransformerModel(nn.Module):
#     """
#     Diffusion Language Model based on Qwen2.5 for discrete token diffusion.
    
#     Implements a forward process that masks tokens with probability eps,
#     and trains the model to denoise/predict masked tokens.
#     """
    
#     def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, mask_token_id=126336, eps=1e-3):
#         super(Qwen2DiffusionTransformerModel, self).__init__()
#         self.n_positions = n_positions
#         self.n_dims = n_dims
#         # For continuous values, use 0.0 as default mask token instead of large integer
#         self.mask_token_id = mask_token_id
#         self.eps = eps
        
#         configuration = Qwen2Config(
#             vocab_size=1,
#             hidden_size=n_embd,
#             intermediate_size=n_embd * 4,
#             num_hidden_layers=n_layer,
#             num_attention_heads=n_head,
#             num_key_value_heads=n_head,  
#             max_position_embeddings=2 * n_positions,
#             attention_dropout=0.0,
#             hidden_dropout=0.0,
#             use_cache=False,
#         )
#         self._read_in = nn.Linear(n_dims, n_embd)
#         self._backbone = Qwen2Model(configuration)
#         self._read_out = nn.Linear(n_embd, 1)
        
#         # Initialize output layer with larger scale to handle wide value ranges
#         # Use larger std to help model learn to predict values in the correct range
#         # The default std=0.02 is too small for regression tasks with wide value ranges
#         nn.init.normal_(self._read_out.weight, mean=0.0, std=0.1)  # Increased from 0.02 to 0.1
#         nn.init.zeros_(self._read_out.bias)
        
#         self.name = f"qwen2_diffusion_embd={n_embd}_layer={n_layer}_head={n_head}"
    
#     @staticmethod
#     def _combine(xs_b, ys_b):
#         """
#         xs_b: continuous input (b, points, dims)
#         ys_b: continuous input (b, points) -> (b, points, 1) -> (b, points, dims)
#         zs: combined input (b, 2 * points, dims)
#         """
#         bsize, points, dim = xs_b.shape
#         ys_b_wide = torch.cat(
#             (
#                 ys_b.view(bsize, points, 1),
#                 torch.zeros(bsize, points, dim - 1, device=ys_b.device),
#             ),
#             axis=2,
#         )
#         zs = torch.stack((xs_b, ys_b_wide), dim=2)
#         zs = zs.view(bsize, 2 * points, dim)
#         return zs
    
#     @staticmethod
#     def forward_process(ys_b, eps=1e-3, mask_token_id=126336, device=None):
#         """
#         ys_b: (b, points)
#         """
#         if device is None:
#             device = ys_b.device
#         b, points= ys_b.shape
#         t = torch.rand(b, device=device) # Sample: t ~ Uniform(0, 1)
#         p_mask = (1 - eps) * t + eps
#         p_mask = p_mask[:, None].repeat(1, points) # Expand to sequence length
#         masked_indices = torch.rand((b, points), device=device) < p_mask
#         noisy_batch = torch.where(masked_indices, mask_token_id, ys_b)
#         return noisy_batch, masked_indices, p_mask
    
#     def forward(self, xs, ys, inds=None, masked_indices=None):
#         """
#         Forward pass: combine xs and ys, process through backbone, and predict y values.
        
#         Args:
#             xs: continuous input (b, points, dims)
#             ys: continuous input (b, points) - can be noisy/masked ys
#             inds: Optional indices to return predictions for (default: all y positions)
#             masked_indices: Optional boolean mask (b, points) indicating which y positions are masked.
#                            If provided, creates bidirectional attention mask allowing model to see
#                            all x positions and unmasked y positions.
        
#         Returns:
#             predictions: (b, points) - predicted y values for all y positions
#         """
#         if inds is None:
#             inds = torch.arange(ys.shape[1])
#         else:
#             inds = torch.tensor(inds)
#             if max(inds) >= ys.shape[1] or min(inds) < 0:
#                 raise ValueError("inds contain indices where xs and ys are not defined")
        
#         zs = self._combine(xs, ys) # (b, 2 * points, dims)
#         embeds = self._read_in(zs)  # (b, 2*points, n_embd)
        
#         # Create bidirectional attention mask for diffusion training
#         # IMPORTANT: For GPT2Model, attention_mask is ADDED to causal mask, not replacing it
#         # This means GPT2 still has causal limitations even with attention_mask
#         # For Qwen2Model, check if attention_mask properly prevents seeing masked positions
#         attention_mask = None
#         if masked_indices is not None:
#             b, points = masked_indices.shape
#             seq_len = 2 * points
#             n_heads = self._backbone.config.num_attention_heads
            
#             # Create full bidirectional mask: all positions can see all positions
#             # Shape: (b, n_heads, seq_len, seq_len)
#             # 0.0 = can attend, -inf = cannot attend
#             attention_mask = torch.zeros((b, n_heads, seq_len, seq_len), device=zs.device, dtype=torch.float)
            
#             for i in range(points):
#                 y_idx = 2 * i + 1
#                 for batch_idx in range(b):
#                     if masked_indices[batch_idx, i]:
#                         # Prevent all positions from attending to this masked y position
#                         attention_mask[batch_idx, :, :, y_idx] = float('-inf')
        
#         output = self._backbone(inputs_embeds=embeds, attention_mask=attention_mask).last_hidden_state  # (b, 2*points, n_embd)
#         prediction = self._read_out(output[:, 1::2, :])  # (b, points, 1)
#         return prediction[:, :, 0][:, inds]  # (b, len(inds))
    
#     def compute_loss(self, xs, ys, eps=None, random_length_prob=0.01):
#         """
#         Compute diffusion loss following the forward process.
        
#         Process:
#         1. Apply forward_process to ys to get noisy_ys (masked y values)
#         2. Combine xs and noisy_ys to get zs
#         3. Forward pass through model to predict y values
#         4. Compute MSE loss only on masked y positions
        
#         Args:
#             xs: Continuous input tensor of shape (b, points, dims)
#             ys: Continuous input tensor of shape (b, points) - ground truth y values
#             eps: Masking probability (default: self.eps)
#             random_length_prob: Probability of using random sequence length (default: 0.01)
        
#         Returns:
#             loss: Scalar loss value (MSE on masked positions)
#             predictions: Model predictions (b, points)
#             masked_indices: Boolean mask of masked positions (b, points)
#         """
#         if eps is None:
#             eps = self.eps
        
#         b, points, _ = xs.shape
        
#         # if torch.rand(1).item() < random_length_prob:
#         #     random_length = torch.randint(1, points + 1, (1,)).item()
#         #     xs = xs[:, :random_length, :]
#         #     ys = ys[:, :random_length]
#         #     points = random_length
        
#         noisy_ys, masked_indices, p_mask = Qwen2DiffusionTransformerModel.forward_process(
#             ys, eps=eps, mask_token_id=self.mask_token_id, device=ys.device
#         )
#         # Pass masked_indices to forward to enable bidirectional attention
#         predictions = self.forward(xs, noisy_ys, masked_indices=masked_indices)  # (b, points)
        
#         # Compute loss with proper weighting for diffusion training
#         squared_error = (predictions - ys).square()  # (b, points)
        
#         if masked_indices.sum() > 0:
#             # Weighted loss: weight by inverse of masking probability (importance sampling)
#             masked_errors = squared_error[masked_indices]  # (num_masked,)
#             weighted_errors = masked_errors / p_mask[masked_indices]  # (num_masked,)
#             # Normalize by total number of positions (not just masked ones)
#             loss = weighted_errors.sum() / (b * points)
            
#             # Debug: Check if model is "cheating" by predicting unmasked values
#             # If predictions on masked positions are too close to true values, model might be seeing them
#             if torch.rand(1).item() < 0.01:  # Sample 1% of batches for debugging
#                 unmasked_errors = squared_error[~masked_indices].mean() if (~masked_indices).sum() > 0 else torch.tensor(0.0)
#                 masked_errors_mean = masked_errors.mean()
#                 # Check if predictions are in reasonable range compared to ys
#                 pred_std = predictions.std().item()
#                 ys_std = ys.std().item()
#                 scale_ratio = pred_std / (ys_std + 1e-8)  # Ratio of prediction scale to y scale
#                 print(f"[DEBUG] Masked error: {masked_errors_mean.item():.4f}, Unmasked error: {unmasked_errors.item():.4f}, "
#                       f"Predictions range: [{predictions.min().item():.4f}, {predictions.max().item():.4f}], "
#                       f"YS range: [{ys.min().item():.4f}, {ys.max().item():.4f}], "
#                       f"Scale ratio (pred_std/ys_std): {scale_ratio:.4f}")
#         else:
#             # Fallback: if no positions are masked, compute loss on all positions
#             # This should rarely happen if eps > 0, but provides stability
#             loss = squared_error.mean()
        
#         return loss, predictions, masked_indices

class NNModel:
    def __init__(self, n_neighbors, weights="uniform"):
        # should we be picking k optimally
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.name = f"NN_n={n_neighbors}_{weights}"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]
            dist = (train_xs - test_x).square().sum(dim=2).sqrt()

            if self.weights == "uniform":
                weights = torch.ones_like(dist)
            else:
                weights = 1.0 / dist
                inf_mask = torch.isinf(weights).float()  # deal with exact match
                inf_row = torch.any(inf_mask, axis=1)
                weights[inf_row] = inf_mask[inf_row]

            pred = []
            k = min(i, self.n_neighbors)
            ranks = dist.argsort()[:, :k]
            for y, w, n in zip(train_ys, weights, ranks):
                y, w = y[n], w[n]
                pred.append((w * y).sum() / w.sum())
            preds.append(torch.stack(pred))

        return torch.stack(preds, dim=1)


# xs and ys should be on cpu for this method. Otherwise the output maybe off in case when train_xs is not full rank due to the implementation of torch.linalg.lstsq.
class LeastSquaresModel:
    def __init__(self, driver=None):
        self.driver = driver
        self.name = f"OLS_driver={driver}"

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            ws, _, _, _ = torch.linalg.lstsq(
                train_xs, train_ys.unsqueeze(2), driver=self.driver
            )

            pred = test_x @ ws
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


class AveragingModel:
    def __init__(self):
        self.name = "averaging"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            train_zs = train_xs * train_ys.unsqueeze(dim=-1)
            w_p = train_zs.mean(dim=1).unsqueeze(dim=-1)
            pred = test_x @ w_p
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


# Lasso regression (for sparse linear regression).
# Seems to take more time as we decrease alpha.
class LassoModel:
    def __init__(self, alpha, max_iter=100000):
        # the l1 regularizer gets multiplied by alpha.
        self.alpha = alpha
        self.max_iter = max_iter
        self.name = f"lasso_alpha={alpha}_max_iter={max_iter}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    # If all points till now have the same label, predict that label.

                    clf = Lasso(
                        alpha=self.alpha, fit_intercept=False, max_iter=self.max_iter
                    )

                    # Check for convergence.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error")
                        try:
                            clf.fit(train_xs, train_ys)
                        except Warning:
                            print(f"lasso convergence warning at i={i}, j={j}.")
                            raise

                    w_pred = torch.from_numpy(clf.coef_).unsqueeze(1)

                    test_x = xs[j, i : i + 1]
                    y_pred = (test_x @ w_pred.float()).squeeze(1)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


# Gradient Descent and variants.
# Example usage: gd_model = GDModel(NeuralNetwork, {'in_size': 50, 'hidden_size':400, 'out_size' :1}, opt_alg = 'adam', batch_size = 100, lr = 5e-3, num_steps = 200)
class GDModel:
    def __init__(
        self,
        model_class,
        model_class_args,
        opt_alg="sgd",
        batch_size=1,
        num_steps=1000,
        lr=1e-3,
        loss_name="squared",
    ):
        # model_class: torch.nn model class
        # model_class_args: a dict containing arguments for model_class
        # opt_alg can be 'sgd' or 'adam'
        # verbose: whether to print the progress or not
        # batch_size: batch size for sgd
        self.model_class = model_class
        self.model_class_args = model_class_args
        self.opt_alg = opt_alg
        self.lr = lr
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.loss_name = loss_name

        self.name = f"gd_model_class={model_class}_model_class_args={model_class_args}_opt_alg={opt_alg}_lr={lr}_batch_size={batch_size}_num_steps={num_steps}_loss_name={loss_name}"

    def __call__(self, xs, ys, inds=None, verbose=False, print_step=100):
        # inds is a list containing indices where we want the prediction.
        # prediction made at all indices by default.
        # xs: bsize X npoints X ndim.
        # ys: bsize X npoints.
        xs, ys = xs.cuda(), ys.cuda()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            model = ParallelNetworks(
                ys.shape[0], self.model_class, **self.model_class_args
            )
            model.cuda()
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])

                train_xs, train_ys = xs[:, :i], ys[:, :i]
                test_xs, test_ys = xs[:, i : i + 1], ys[:, i : i + 1]

                if self.opt_alg == "sgd":
                    optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
                elif self.opt_alg == "adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
                else:
                    raise NotImplementedError(f"{self.opt_alg} not implemented.")

                if self.loss_name == "squared":
                    loss_criterion = nn.MSELoss()
                else:
                    raise NotImplementedError(f"{self.loss_name} not implemented.")

                # Training loop
                for j in range(self.num_steps):

                    # Prepare batch
                    mask = torch.zeros(i).bool()
                    perm = torch.randperm(i)
                    mask[perm[: self.batch_size]] = True
                    train_xs_cur, train_ys_cur = train_xs[:, mask, :], train_ys[:, mask]

                    if verbose and j % print_step == 0:
                        model.eval()
                        with torch.no_grad():
                            outputs = model(train_xs_cur)
                            loss = loss_criterion(
                                outputs[:, :, 0], train_ys_cur
                            ).detach()
                            outputs_test = model(test_xs)
                            test_loss = loss_criterion(
                                outputs_test[:, :, 0], test_ys
                            ).detach()
                            print(
                                f"ind:{i},step:{j}, train_loss:{loss.item()}, test_loss:{test_loss.item()}"
                            )

                    optimizer.zero_grad()

                    model.train()
                    outputs = model(train_xs_cur)
                    loss = loss_criterion(outputs[:, :, 0], train_ys_cur)
                    loss.backward()
                    optimizer.step()

                model.eval()
                pred = model(test_xs).detach()

                assert pred.shape[1] == 1 and pred.shape[2] == 1
                pred = pred[:, 0, 0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class DecisionTreeModel:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.name = f"decision_tree_max_depth={max_depth}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = tree.DecisionTreeRegressor(max_depth=self.max_depth)
                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class XGBoostModel:
    def __init__(self):
        self.name = "xgboost"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = xgb.XGBRegressor()

                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0].item()

            preds.append(pred)

        return torch.stack(preds, dim=1)
