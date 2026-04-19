import runpy
import config
import data

# 1. Drastically reduce epochs and number of Optuna trials.
config.OPTUNA_TRIALS = 1
config.EPOCHS = {"lf": 1, "mf": 1, "hf": 1}

# 2. Use a dry-run MLflow experiment prefix.
config.MLFLOW_EXPERIMENT_BASE = "tsunami_mf_TEST_DRYRUN"

# 3. Monkeypatch dataset loading to keep only a tiny subset.
original_load_dataset = data.load_dataset


def mock_load_dataset(*args, **kwargs):
    print(">>> DRY RUN: Loading dataset and shrinking it to tiny subsets...")
    db = original_load_dataset(*args, **kwargs)

    split_fields = ["X", "ymh", "ymh2d", "yat", "yat2d", "yeta", "yeta2d"]
    per_fidelity_n = {"lf": 4, "mf": 2, "hf": 2}

    for fid, n_keep in per_fidelity_n.items():
        for stem in split_fields:
            field_name = f"{stem}_{fid}"
            split_map = getattr(db, field_name, None)
            if split_map is None:
                continue
            for split in ["tr", "va", "te"]:
                if split in split_map:
                    split_map[split] = split_map[split][:n_keep]

    return db


data.load_dataset = mock_load_dataset

print("=" * 60)
print("STARTING DRY RUN OF arrival_times.py")
print("This will execute 1 trial, 1 epoch per fidelity, on tiny subsets.")
print("It should only take 1-3 minutes in total.")
print("=" * 60)

# 4. Execute arrival_times.py exactly as if run directly.
runpy.run_path("arrival_times.py", run_name="__main__")
