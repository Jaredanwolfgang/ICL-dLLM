# Add src to path
import sys
import os
sys.path.insert(0, './src')

# Import project modules
from eval import (
    get_run_metrics,
)
# ========================================
# UPDATE THIS WITH YOUR RUN ID!
# ========================================
run_id = "diffusion_encoder_wotim"  # Replace with your run_id from training
# ========================================

run_path = os.path.join("/workspace/in-context-learning/logs/in-context-learning/", run_id)

_ = get_run_metrics(run_path, skip_baselines=True)