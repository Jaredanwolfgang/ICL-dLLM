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
run_id = "a1259393-e9ad-400c-800e-e0a85e73f72d"  # Replace with your run_id from training
# ========================================

run_path = os.path.join("/workspace/in-context-learning/models/models/relu2nn_decoder/", run_id)

_ = get_run_metrics(run_path, skip_baselines=True)