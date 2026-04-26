import sys
import runpy
import config
import data

# 1. We drastically reduce the epochs and number of trials.
config.OPTUNA_TRIALS = 1
config.EPOCHS = {"lf": 1, "mf": 1, "hf": 1}

# 2. Change the MLflow experiment name so you don't overwrite your actual data records
config.MLFLOW_EXPERIMENT_BASE = "tsunami_mf_TEST_DRYRUN"

# 3. We monkeypatch the data loader so that it only loads 4 samples per dataset.
# The dataloader handles numpy arrays exactly the same as the custom HDF5 views.
original_load_dataset = data.load_dataset

def mock_load_dataset(*args, **kwargs):
    print(">>> DRY RUN: Loading dataset and shrinking it to 4 samples...")
    db = original_load_dataset(*args, **kwargs)

    # Slice train/val/test splits down to tiny sample counts across all targets.
    split_fields = ["X", "ymh", "ymh2d", "yat", "yat2d", "yeta", "yeta2d"]
    per_fidelity_n = {"lf": 4, "mf": 2, "hf": 2}

    for fid, n_keep in per_fidelity_n.items():
        for stem in split_fields:
            field_name = f"{stem}_{fid}"
            split_map = getattr(db, field_name, None)
            if split_map is None:
                continue
            for k in ["tr", "va", "te"]:
                if k in split_map:
                    split_map[k] = split_map[k][:n_keep]
        
    return db

data.load_dataset = mock_load_dataset

print("="*60)
print("STARTING DRY RUN OF max_height.py")
print("This will execute 1 trial, 1 epoch per fidelity, on 4 samples.")
print("It should only take 1-3 minutes in total.")
print("="*60)

# 4. Execute the max_height.py file exactly as if we ran `python max_height.py`
runpy.run_path("max_height.py", run_name="__main__")
