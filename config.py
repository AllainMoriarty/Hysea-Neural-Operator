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
