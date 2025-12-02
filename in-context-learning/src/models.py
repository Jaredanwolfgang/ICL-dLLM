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
    elif conf.family == "diffusion_encoder":
        model = DiffusionEncoderModel(
            n_positions=conf.n_positions,
            n_dims=conf.n_dims,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
            timesteps=getattr(conf, "timesteps", 100),
            beta_start=getattr(conf, "beta_start", 1e-4),
            beta_end=getattr(conf, "beta_end", 2e-2),
        )
    elif conf.family == "diffusion_decoder":
        model = DiffusionDecoderModel(
            n_positions=conf.n_positions,
            n_dims=conf.n_dims,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
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

class DiffusionSchedule:
    def __init__(self, timesteps=100, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class DiffusionEncoderModel(nn.Module):
    def __init__(self, n_dims, n_embd=256, n_layer=6, n_head=8, timesteps=100, n_positions=1024, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.n_dims = n_dims
        self.n_embd = n_embd
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self._read_in = nn.Linear(n_dims + 1, n_embd)
        # Time Embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, n_embd),
            nn.SiLU(),
            nn.Linear(n_embd, n_embd),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd, 
            nhead=n_head, 
            dim_feedforward=n_embd * 4, 
            dropout=0.0, 
            activation="gelu", 
            batch_first=True, 
            norm_first=True
        )
        self._backbone = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self._read_out = nn.Linear(n_embd, 1)
        self.schedule = None
        self.name = f"diffusion_encoder_embd={n_embd}_layer={n_layer}_head={n_head}_timesteps={timesteps}"

    def _get_schedule(self, device):
        if self.schedule is None or self.schedule.betas.device != device:
            self.schedule = DiffusionSchedule(self.timesteps, beta_start=self.beta_start, beta_end=self.beta_end, device=device)
        return self.schedule
    
    def forward(self, xs, ys_current, t):   
        """
        xs:         (B, P, Dx) 
        ys_current: (B, P, 1)
        t:          (B,)
        """
        # Input Projection
        inp = torch.cat([xs, ys_current], dim=-1) # (B, P, D+1)
        emb = self._read_in(inp)

        # Time Embedding
        t_vec = t.float().unsqueeze(-1) / self.timesteps
        t_emb = self.time_embed(t_vec)            # (B, n_embd)
        emb = emb + t_emb[:, None, :]

        # Backbone
        out = self._backbone(emb)
        # out = self.backbone(inputs_embeds=emb).last_hidden_state

        # Predict Noise
        return self._read_out(out)

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

    def compute_loss(self, xs, ys_gt):
        if ys_gt.dim() == 2:
            ys_gt = ys_gt.unsqueeze(-1)
        B, P, _ = ys_gt.shape
        device = ys_gt.device
        schedule = self._get_schedule(device)

        t = torch.randint(0, schedule.timesteps, (B,), device=device)

        ks = torch.randint(1, P, (B,), device=device)
        indices = torch.arange(P, device=device).unsqueeze(0).expand(B, P)
        mask_target = (indices >= ks.unsqueeze(1)).unsqueeze(-1).float() # (B, P, 1)

        y_noisy_full, noise_true = self.q_sample(ys_gt, t)
        ys_input = mask_target * y_noisy_full + (1.0 - mask_target) * ys_gt

        eps_pred = self.forward(xs, ys_input, t)

        loss = F.mse_loss(eps_pred, noise_true, reduction='none')
        loss = (loss * mask_target).sum() / mask_target.sum().clamp(min=1)

        return loss, eps_pred, y_noisy_full, t

    @torch.no_grad()
    def p_sample(self, xs, y_t, t, t_index):
        """
        y_{t-1} = 1/sqrt(alpha) * (y_t - ...) + sigma * z
        """
        schedule = self._get_schedule(xs.device)
        
        eps_pred = self.forward(xs, y_t, t)  # (B, P, 1)
        
        betas_t = extract(schedule.betas, t, y_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(schedule.sqrt_one_minus_alphas_cumprod, t, y_t.shape)
        sqrt_recip_alphas_t = extract(schedule.sqrt_recip_alphas, t, y_t.shape)
        
        # mu = (1 / sqrt(alpha_t)) * (y_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps)
        model_mean = sqrt_recip_alphas_t * (
            y_t - (betas_t / sqrt_one_minus_alphas_cumprod_t) * eps_pred
        )

        if t_index > 0:
            noise = torch.randn_like(y_t)
            posterior_variance_t = extract(torch.sqrt(schedule.betas), t, y_t.shape)
            y_prev = model_mean + posterior_variance_t * noise
        else:
            y_prev = model_mean
            
        return y_prev

    @torch.no_grad()
    def p_sample_loop(self, xs, ys_demo, n_query):
        """
        ICL 推理循环:
        xs: 全量 x (B, P, D) 包含所有点（demo + query）
        ys_demo: 已知的 y (Clean) (B, n_demo, 1) 只包含 demo 点
        n_query: 需要预测的点数
        """
        device = xs.device
        B, P, _ = xs.shape
        schedule = self._get_schedule(device)
        n_demo = P - n_query
        
        y_query_noisy = torch.randn(B, n_query, 1, device=device)
        ys_current = torch.cat([ys_demo, y_query_noisy], dim=1) # (B, P, 1)
        
        for i in reversed(range(schedule.timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            y_prev_full = self.p_sample(xs, ys_current, t, i)
            y_query_updated = y_prev_full[:, n_demo:, :]
            ys_current = torch.cat([ys_demo, y_query_updated], dim=1)
        return ys_current

class DiffusionDecoderModel(nn.Module):
    def __init__(self, n_dims, n_embd=256, n_layer=6, n_head=8, timesteps=100, n_positions=1024, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.n_dims = n_dims
        self.n_embd = n_embd
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self._read_in = nn.Linear(n_dims + 1, n_embd)
        # Time Embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, n_embd),
            nn.SiLU(),
            nn.Linear(n_embd, n_embd),
        )
        config = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self._backbone = GPT2Model(config)
        self._read_out = nn.Linear(n_embd, 1)
        self.schedule = None
        self.name = f"diffusion_encoder_embd={n_embd}_layer={n_layer}_head={n_head}_timesteps={timesteps}"

    def _get_schedule(self, device):
        if self.schedule is None or self.schedule.betas.device != device:
            self.schedule = DiffusionSchedule(self.timesteps, beta_start=self.beta_start, beta_end=self.beta_end, device=device)
        return self.schedule
    
    def forward(self, xs, ys_current, t):   
        """
        xs:         (B, P, Dx) 
        ys_current: (B, P, 1)
        t:          (B,)
        """
        # Input Projection
        inp = torch.cat([xs, ys_current], dim=-1) # (B, P, D+1)
        emb = self._read_in(inp)

        # Time Embedding
        t_vec = t.float().unsqueeze(-1) / self.timesteps
        t_emb = self.time_embed(t_vec)            # (B, n_embd)
        emb = emb + t_emb[:, None, :]

        # Backbone
        # out = self._backbone(emb)
        out = self._backbone(inputs_embeds=emb).last_hidden_state

        # Predict Noise
        return self._read_out(out)

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

    def compute_loss(self, xs, ys_gt):
        if ys_gt.dim() == 2:
            ys_gt = ys_gt.unsqueeze(-1)
        B, P, _ = ys_gt.shape
        device = ys_gt.device
        schedule = self._get_schedule(device)

        t = torch.randint(0, schedule.timesteps, (B,), device=device)

        ks = torch.randint(1, P, (B,), device=device)
        indices = torch.arange(P, device=device).unsqueeze(0).expand(B, P)
        mask_target = (indices >= ks.unsqueeze(1)).unsqueeze(-1).float() # (B, P, 1)

        y_noisy_full, noise_true = self.q_sample(ys_gt, t)
        ys_input = mask_target * y_noisy_full + (1.0 - mask_target) * ys_gt

        eps_pred = self.forward(xs, ys_input, t)

        loss = F.mse_loss(eps_pred, noise_true, reduction='none')
        loss = (loss * mask_target).sum() / mask_target.sum().clamp(min=1)

        return loss, eps_pred, y_noisy_full, t

    @torch.no_grad()
    def p_sample(self, xs, y_t, t, t_index):
        """
        y_{t-1} = 1/sqrt(alpha) * (y_t - ...) + sigma * z
        """
        schedule = self._get_schedule(xs.device)
        
        eps_pred = self.forward(xs, y_t, t)  # (B, P, 1)
        
        betas_t = extract(schedule.betas, t, y_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(schedule.sqrt_one_minus_alphas_cumprod, t, y_t.shape)
        sqrt_recip_alphas_t = extract(schedule.sqrt_recip_alphas, t, y_t.shape)
        
        # mu = (1 / sqrt(alpha_t)) * (y_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps)
        model_mean = sqrt_recip_alphas_t * (
            y_t - (betas_t / sqrt_one_minus_alphas_cumprod_t) * eps_pred
        )

        if t_index > 0:
            noise = torch.randn_like(y_t)
            posterior_variance_t = extract(torch.sqrt(schedule.betas), t, y_t.shape)
            y_prev = model_mean + posterior_variance_t * noise
        else:
            y_prev = model_mean
            
        return y_prev

    @torch.no_grad()
    def p_sample_loop(self, xs, ys_demo, n_query):
        """
        ICL 推理循环:
        xs: 全量 x (B, P, D) 包含所有点（demo + query）
        ys_demo: 已知的 y (Clean) (B, n_demo, 1) 只包含 demo 点
        n_query: 需要预测的点数
        """
        device = xs.device
        B, P, _ = xs.shape
        schedule = self._get_schedule(device)
        n_demo = P - n_query
        
        y_query_noisy = torch.randn(B, n_query, 1, device=device)
        ys_current = torch.cat([ys_demo, y_query_noisy], dim=1) # (B, P, 1)
        
        for i in reversed(range(schedule.timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            y_prev_full = self.p_sample(xs, ys_current, t, i)
            y_query_updated = y_prev_full[:, n_demo:, :]
            ys_current = torch.cat([ys_demo, y_query_updated], dim=1)
        return ys_current

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
