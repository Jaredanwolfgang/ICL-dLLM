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
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
            mask_token_id=getattr(conf, "mask_token_id", 126336),
            eps=1e-2,
        )
    elif conf.family == "diffusion_qwen2":
        model = Qwen2DiffusionTransformerModel(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
            mask_token_id=getattr(conf, "mask_token_id", 126336),
            eps=1e-2,
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


class DiffusionTransformerModel(nn.Module):
    """
    Diffusion Language Model based on GPT2 for discrete token diffusion.
    
    Implements a forward process that masks tokens with probability eps,
    and trains the model to denoise/predict masked tokens.
    """
    
    def __init__(
        self,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        mask_token_id=126336,
        eps=1e-3,
    ):
        super(DiffusionTransformerModel, self).__init__()
        self.n_positions = n_positions
        self.n_dims = 1  # Compatibility with existing code (not used for diffusion)
        self.mask_token_id = mask_token_id
        self.eps = eps
        
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
        
        # Use GPT2Model (not LMHeadModel) since we'll add custom read_in/read_out for continuous values
        # Note: _read_in needs n_dims but we don't know it at init, so we'll use a placeholder
        # We'll set it properly in forward based on actual input dims
        self._read_in = None  # Will be initialized on first forward pass
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, 1)  # Output dim is 1 (for y predictions)
        self.name = f"diffusion_gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"
    
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
    
    @staticmethod
    def forward_process(ys_b, eps=1e-3, mask_token_id=126336, device=None):
        """
        ys_b: (b, points)
        """
        if device is None:
            device = ys_b.device
        b, points= ys_b.shape
        t = torch.rand(b, device=device) # Sample: t ~ Uniform(0, 1)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, points) # Expand to sequence length
        masked_indices = torch.rand((b, points), device=device) < p_mask
        noisy_batch = torch.where(masked_indices, mask_token_id, ys_b)
        return noisy_batch, masked_indices, p_mask
    
    def forward(self, xs, ys, inds=None, masked_indices=None):
        """
        Forward pass: combine xs and ys, process through backbone, and predict y values.
        
        Args:
            xs: continuous input (b, points, dims)
            ys: continuous input (b, points) - can be noisy/masked ys
            inds: Optional indices to return predictions for (default: all y positions)
            masked_indices: Optional boolean mask (b, points) indicating which y positions are masked.
                           If provided, creates bidirectional attention mask allowing model to see
                           all x positions and unmasked y positions.
        
        Returns:
            predictions: (b, points) - predicted y values for all y positions
        """
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        
        zs = self._combine(xs, ys) # (b, 2 * points, dims)
        if self._read_in is None:
            dim = zs.shape[2]
            n_embd = self._backbone.config.n_embd
            self._read_in = nn.Linear(dim, n_embd).to(zs.device)
        embeds = self._read_in(zs)  # (b, 2*points, n_embd)
        
        # Create bidirectional attention mask for diffusion training
        attention_mask = None
        if masked_indices is not None:
            b, points = masked_indices.shape
            seq_len = 2 * points
            n_heads = self._backbone.config.num_attention_heads
            
            # Create full bidirectional mask: all positions can see all positions
            # Shape: (b, n_heads, seq_len, seq_len)
            # 0.0 = can attend, -inf = cannot attend
            attention_mask = torch.zeros((b, n_heads, seq_len, seq_len), device=zs.device, dtype=torch.float)
            
            for i in range(points):
                y_idx = 2 * i + 1
                for batch_idx in range(b):
                    if masked_indices[batch_idx, i]:
                        attention_mask[batch_idx, :, :, y_idx] = float('-inf')
        
        output = self._backbone(inputs_embeds=embeds, attention_mask=attention_mask).last_hidden_state  # (b, 2*points, n_embd)
        prediction = self._read_out(output[:, 1::2, :])  # (b, points, 1)
        return prediction[:, :, 0][:, inds]  # (b, len(inds))
    
    def compute_loss(self, xs, ys, eps=None, random_length_prob=0.01):
        """
        Compute diffusion loss following the forward process.
        
        Process:
        1. Apply forward_process to ys to get noisy_ys (masked y values)
        2. Combine xs and noisy_ys to get zs
        3. Forward pass through model to predict y values
        4. Compute MSE loss only on masked y positions
        
        Args:
            xs: Continuous input tensor of shape (b, points, dims)
            ys: Continuous input tensor of shape (b, points) - ground truth y values
            eps: Masking probability (default: self.eps)
            random_length_prob: Probability of using random sequence length (default: 0.01)
        
        Returns:
            loss: Scalar loss value (MSE on masked positions)
            predictions: Model predictions (b, points)
            masked_indices: Boolean mask of masked positions (b, points)
        """
        if eps is None:
            eps = self.eps
        
        b, points, _ = xs.shape
        
        # if torch.rand(1).item() < random_length_prob:
        #     random_length = torch.randint(1, points + 1, (1,)).item()
        #     xs = xs[:, :random_length, :]
        #     ys = ys[:, :random_length]
        #     points = random_length
        
        noisy_ys, masked_indices, p_mask = DiffusionTransformerModel.forward_process(
            ys, eps=eps, mask_token_id=self.mask_token_id, device=ys.device
        )
        # Pass masked_indices to forward to enable bidirectional attention
        predictions = self.forward(xs, noisy_ys, masked_indices=masked_indices)  # (b, points)
        
        # Compute loss with proper weighting for diffusion training
        squared_error = (predictions - ys).square()  # (b, points)
        
        if masked_indices.sum() > 0:
            # Weighted loss: weight by inverse of masking probability (importance sampling)
            masked_errors = squared_error[masked_indices]  # (num_masked,)
            weighted_errors = masked_errors / p_mask[masked_indices]  # (num_masked,)
            # Normalize by total number of positions (not just masked ones)
            loss = weighted_errors.sum() / (b * points)
        else:
            # Fallback: if no positions are masked, compute loss on all positions
            # This should rarely happen if eps > 0, but provides stability
            loss = squared_error.mean()
        
        return loss, predictions, masked_indices



class Qwen2DiffusionTransformerModel(nn.Module):
    """
    Diffusion Language Model based on Qwen2.5 for discrete token diffusion.
    
    Implements a forward process that masks tokens with probability eps,
    and trains the model to denoise/predict masked tokens.
    """
    
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, mask_token_id=126336, eps=1e-3):
        super(Qwen2DiffusionTransformerModel, self).__init__()
        self.n_positions = n_positions
        self.n_dims = 1  # Compatibility with existing code (not used for diffusion)
        self.mask_token_id = mask_token_id
        self.eps = eps
        
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
        self._read_in = None
        self._backbone = Qwen2Model(configuration)
        self._read_out = nn.Linear(n_embd, 1)
        self.name = f"qwen2_diffusion_embd={n_embd}_layer={n_layer}_head={n_head}"
    
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
    
    @staticmethod
    def forward_process(ys_b, eps=1e-3, mask_token_id=126336, device=None):
        """
        ys_b: (b, points)
        """
        if device is None:
            device = ys_b.device
        b, points= ys_b.shape
        t = torch.rand(b, device=device) # Sample: t ~ Uniform(0, 1)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, points) # Expand to sequence length
        masked_indices = torch.rand((b, points), device=device) < p_mask
        noisy_batch = torch.where(masked_indices, mask_token_id, ys_b)
        return noisy_batch, masked_indices, p_mask
    
    def forward(self, xs, ys, inds=None, masked_indices=None):
        """
        Forward pass: combine xs and ys, process through backbone, and predict y values.
        
        Args:
            xs: continuous input (b, points, dims)
            ys: continuous input (b, points) - can be noisy/masked ys
            inds: Optional indices to return predictions for (default: all y positions)
            masked_indices: Optional boolean mask (b, points) indicating which y positions are masked.
                           If provided, creates bidirectional attention mask allowing model to see
                           all x positions and unmasked y positions.
        
        Returns:
            predictions: (b, points) - predicted y values for all y positions
        """
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        
        zs = self._combine(xs, ys) # (b, 2 * points, dims)
        if self._read_in is None:
            dim = zs.shape[2]
            n_embd = self._backbone.config.hidden_size
            self._read_in = nn.Linear(dim, n_embd).to(zs.device)
        embeds = self._read_in(zs)  # (b, 2*points, n_embd)
        
        # Create bidirectional attention mask for diffusion training
        attention_mask = None
        if masked_indices is not None:
            b, points = masked_indices.shape
            seq_len = 2 * points
            n_heads = self._backbone.config.num_attention_heads
            
            # Create full bidirectional mask: all positions can see all positions
            # Shape: (b, n_heads, seq_len, seq_len)
            # 0.0 = can attend, -inf = cannot attend
            attention_mask = torch.zeros((b, n_heads, seq_len, seq_len), device=zs.device, dtype=torch.float)
            
            for i in range(points):
                y_idx = 2 * i + 1
                for batch_idx in range(b):
                    if masked_indices[batch_idx, i]:
                        attention_mask[batch_idx, :, :, y_idx] = float('-inf')
        
        output = self._backbone(inputs_embeds=embeds, attention_mask=attention_mask).last_hidden_state  # (b, 2*points, n_embd)
        prediction = self._read_out(output[:, 1::2, :])  # (b, points, 1)
        return prediction[:, :, 0][:, inds]  # (b, len(inds))
    
    def compute_loss(self, xs, ys, eps=None, random_length_prob=0.01):
        """
        Compute diffusion loss following the forward process.
        
        Process:
        1. Apply forward_process to ys to get noisy_ys (masked y values)
        2. Combine xs and noisy_ys to get zs
        3. Forward pass through model to predict y values
        4. Compute MSE loss only on masked y positions
        
        Args:
            xs: Continuous input tensor of shape (b, points, dims)
            ys: Continuous input tensor of shape (b, points) - ground truth y values
            eps: Masking probability (default: self.eps)
            random_length_prob: Probability of using random sequence length (default: 0.01)
        
        Returns:
            loss: Scalar loss value (MSE on masked positions)
            predictions: Model predictions (b, points)
            masked_indices: Boolean mask of masked positions (b, points)
        """
        if eps is None:
            eps = self.eps
        
        b, points, _ = xs.shape
        
        # if torch.rand(1).item() < random_length_prob:
        #     random_length = torch.randint(1, points + 1, (1,)).item()
        #     xs = xs[:, :random_length, :]
        #     ys = ys[:, :random_length]
        #     points = random_length
        
        noisy_ys, masked_indices, p_mask = Qwen2DiffusionTransformerModel.forward_process(
            ys, eps=eps, mask_token_id=self.mask_token_id, device=ys.device
        )
        # Pass masked_indices to forward to enable bidirectional attention
        predictions = self.forward(xs, noisy_ys, masked_indices=masked_indices)  # (b, points)
        
        # Compute loss with proper weighting for diffusion training
        squared_error = (predictions - ys).square()  # (b, points)
        
        if masked_indices.sum() > 0:
            # Weighted loss: weight by inverse of masking probability (importance sampling)
            masked_errors = squared_error[masked_indices]  # (num_masked,)
            weighted_errors = masked_errors / p_mask[masked_indices]  # (num_masked,)
            # Normalize by total number of positions (not just masked ones)
            loss = weighted_errors.sum() / (b * points)
        else:
            # Fallback: if no positions are masked, compute loss on all positions
            # This should rarely happen if eps > 0, but provides stability
            loss = squared_error.mean()
        
        return loss, predictions, masked_indices

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
