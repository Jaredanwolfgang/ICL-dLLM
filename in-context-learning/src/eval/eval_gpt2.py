# Add src to path
import sys
import os
sys.path.insert(0, './src')

# Import required modules
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm
from munch import Munch

# Import project modules
from eval import (
    get_model_from_run,
    get_run_metrics,
    eval_model,
    build_evals,
    baseline_names
)
from models import build_model, get_relevant_baselines
from tasks import get_task_sampler
from samplers import get_data_sampler
from plot_utils import *

# Set plot style (matching eval.ipynb)
sns.set_theme('notebook', 'darkgrid')
palette = sns.color_palette('colorblind')

print("✓ All modules imported successfully")

# ========================================
# UPDATE THIS WITH YOUR RUN ID!
# ========================================
decoder_run_id = "db9417c7-1b49-40b1-bc85-303f597d930f"  # Replace with your run_id from training
# ========================================

decoder_run_path = os.path.join("/workspace/models/decision_tree_gpt2/", decoder_run_id)

print(f"Loading Diffusion Decoder model from: {decoder_run_path}\n")

# Load model and config
decoder_model, decoder_conf = get_model_from_run(decoder_run_path, step=-1)  # step=-1 loads final checkpoint

# Move model to GPU if available
if torch.cuda.is_available():
    decoder_model = decoder_model.cuda()
    print(f"✓ Model loaded on GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  Running on CPU (slower)")

decoder_model.eval()

# Display model configuration
print(f"\n{'='*60}")
print("Diffusion Decoder Model Configuration:")
print(f"{'='*60}")
print(f"  Model Family: {decoder_conf.model.family}")
print(f"  Task: {decoder_conf.training.task}")
print(f"  n_dims: {decoder_conf.model.n_dims}")
print(f"  n_positions: {decoder_conf.model.n_positions}")
print(f"  n_embd: {decoder_conf.model.n_embd}")
print(f"  n_layer: {decoder_conf.model.n_layer}")
print(f"  n_head: {decoder_conf.model.n_head}")
print(f"  Training points: {decoder_conf.training.curriculum.points.end}")
print(f"{'='*60}")
print(f"\n✓ Model ready for evaluation")

_ = get_run_metrics(decoder_run_path, skip_baselines=False)