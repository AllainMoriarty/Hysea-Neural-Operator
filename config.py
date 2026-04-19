import os
import torch

# Dataset paths
DATA_PATHS = {
    "lf": "./datasets/dataset_5km.h5",
    "mf": "./datasets/dataset_2km.h5",
    "hf": "./datasets/dataset_1km.h5",
}

# MLflow
MLFLOW_TRACKING_URI    = os.getenv("MLFLOW_TRACKING_URI", "https://allainverse-mlflow.hf.space")
MLFLOW_EXPERIMENT_BASE = "tsunami_mf"

# Training schedule (epochs per MF stage)
EPOCHS        = {"lf": 300, "mf": 150, "hf": 75}
OPTUNA_TRIALS = 20
RANDOM_SEED   = 42

# Compute device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Memory management ─────────────────────────────────────────────────────────
# BATCH_SIZE: micro-batch fed each forward pass.  Effective batch =
#   BATCH_SIZE * GRAD_ACCUM_STEPS (gradient accumulation compensates).
#   4 is safe at 1 km grid resolution; increase if VRAM headroom allows.
BATCH_SIZE       = 4

# NUM_WORKERS: DataLoader parallel prefetch workers.
#   H5FieldView is spawn-safe (implements __getstate__/__setstate__).
#   Set 0 to disable if HDF5 files were not opened with SWMR.
NUM_WORKERS      = 2

# PIN_MEMORY: pinning large 2-D spatial batches wastes locked RAM pages.
PIN_MEMORY       = False

# GRAD_ACCUM_STEPS: accumulate this many micro-batches before an optimizer
#   step.  Effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS = 32.
GRAD_ACCUM_STEPS = 8

# GRAD_CKPT: recompute FNO backbone activations during backward instead of
#   storing them.  Saves ~60 % activation VRAM at ~25 % speed cost.
GRAD_CKPT        = True

# USE_AMP: BFloat16 automatic mixed precision.  RTX 5090 supports BF16
#   natively; halves activation memory with no loss in accuracy.
USE_AMP          = True

# PREFETCH_FACTOR: batches to pre-load per worker (None if NUM_WORKERS=0).
PREFETCH_FACTOR  = 2
