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
run_id = "7133f891-9051-47e7-9373-801b594c02a3"  # Replace with your run_id from training
# ========================================

run_path = os.path.join("/workspace/in-context-learning/models/models/relu2nn_encoder/", run_id)

_ = get_run_metrics(run_path, skip_baselines=True)