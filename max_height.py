"""
Train and evaluate multi-fidelity models for MAXIMUM WAVE HEIGHT prediction.

Models : MF-DeepONet · MF-FNO · MF-PINO
Target : max_height   (log1p-normalised; inverse = np.expm1)

Run
---
    uv run max_height.py
    MLFLOW_TRACKING_URI=http://my-server:5000 uv run max_height.py

Memory design
-------------
• Only ONE model lives on GPU at a time (train → save → del → gc → next).
• Evaluation reloads each model from its saved .pt file, evaluates, then
  deletes it before loading the next — never holding 2+ models in VRAM.
• Training is done via AMP + gradient accumulation in training.py.
"""
import gc
import numpy as np
import pandas as pd
import torch
import optuna
import mlflow

from config import (
    device, DATA_PATHS,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_BASE,
    EPOCHS, OPTUNA_TRIALS, RANDOM_SEED,
)
from data import load_dataset, load_h5, build_grid_info
from models import MFDeepONet, MFFno, MFPino
from training import (
    make_objective_deeponet, make_objective_fno, make_objective_pino,
    train_mf, train_mf_pino,
    evaluate, plot_maps,
)

TARGET     = "max_height"
EXPERIMENT = f"{MLFLOW_EXPERIMENT_BASE}_{TARGET}"
INVERSE_FN = np.expm1        # undo log1p applied in preprocess()


# ─────────────────────────────────────────────────────────────────────────────
# MLflow helpers
# ─────────────────────────────────────────────────────────────────────────────

def _log_history(hist: dict) -> None:
    """Log per-epoch train/val RMSE for every MF stage to the active run."""
    for stage in ["lf", "mf", "hf"]:
        for i, (tr, va) in enumerate(
            zip(hist.get(f"{stage}_train", []),
                hist.get(f"{stage}_val",   []))
        ):
            mlflow.log_metrics(
                {f"{stage}_train_rmse": tr, f"{stage}_val_rmse": va}, step=i
            )


def _log_results(results: list) -> None:
    """Log evaluation metrics for all model × fidelity pairs."""
    for r in results:
        prefix = r["model"].replace(" ", "_").lower() + f"_{r['fidelity']}"
        mlflow.log_metrics({
            f"{prefix}_mae":     r["mae"],
            f"{prefix}_rmse":    r["rmse"],
            f"{prefix}_nrmse":   r["nrmse"],
            f"{prefix}_r2":      r["r2"],
            f"{prefix}_rel_err": r["rel_err"],
        })


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Global torch settings ─────────────────────────────────────────────────
    torch.backends.cudnn.benchmark = True          # auto-tune CUDNN kernels
    torch.set_float32_matmul_precision("high")     # enable TF32 on RTX 5090

    print(f"Device : {device}")
    print(f"MLflow : {MLFLOW_TRACKING_URI}  |  experiment: {EXPERIMENT}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    # ── Load data ─────────────────────────────────────────────────────────────
    db   = load_dataset(target="max_height", lazy=True)
    nlat, nlon = db.NLAT, db.NLON

    # Build SWE grid info for PINO physics loss
    lf_raw              = load_h5(DATA_PATHS["lf"], keys=["lon", "lat", "bathymetry"])
    grid_info, H_raw    = build_grid_info(lf_raw, db)
    del lf_raw          # free raw metadata — lon/lat/bathymetry no longer needed

    # =========================================================================
    # HYPERPARAMETER TUNING
    # =========================================================================

    # ── DeepONet ──────────────────────────────────────────────────────────────
    print("\n" + "="*60 + "\nOptuna Tuning — DeepONet")
    with mlflow.start_run(run_name="tune_deeponet"):
        mlflow.set_tags({"target": TARGET, "phase": "tuning", "model": "deeponet"})
        study_don = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        )
        study_don.optimize(
            make_objective_deeponet(
                db.X_lf["tr"], db.ymh_lf["tr"],
                db.X_lf["va"], db.ymh_lf["va"],
                db.q_sp, trunk_dim=2,
            ),
            n_trials=OPTUNA_TRIALS, show_progress_bar=True,
        )
        best_don = study_don.best_params
        mlflow.log_params(best_don)
        mlflow.log_metric("best_val_rmse", study_don.best_value)
    print(f"Best DeepONet : {best_don}")

    # ── FNO ───────────────────────────────────────────────────────────────────
    print("\n" + "="*60 + "\nOptuna Tuning — FNO")
    with mlflow.start_run(run_name="tune_fno"):
        mlflow.set_tags({"target": TARGET, "phase": "tuning", "model": "fno"})
        study_fno = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        )
        study_fno.optimize(
            make_objective_fno(
                db.X_lf["tr"], db.ymh2d_lf["tr"],
                db.X_lf["va"], db.ymh2d_lf["va"],
                db.bathy_t, nlat, nlon,
            ),
            n_trials=OPTUNA_TRIALS, show_progress_bar=True,
        )
        best_fno = study_fno.best_params
        mlflow.log_params(best_fno)
        mlflow.log_metric("best_val_rmse", study_fno.best_value)
    print(f"Best FNO : {best_fno}")

    # ── PINO ──────────────────────────────────────────────────────────────────
    print("\n" + "="*60 + "\nOptuna Tuning — PINO")
    with mlflow.start_run(run_name="tune_pino"):
        mlflow.set_tags({"target": TARGET, "phase": "tuning", "model": "pino"})
        study_pino = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        )
        study_pino.optimize(
            make_objective_pino(
                db.X_lf["tr"], db.ymh2d_lf["tr"],
                db.X_lf["va"], db.ymh2d_lf["va"],
                db.bathy_t, nlat, nlon,
            ),
            n_trials=OPTUNA_TRIALS, show_progress_bar=True,
        )
        best_pino = study_pino.best_params
        mlflow.log_params(best_pino)
        mlflow.log_metric("best_val_rmse", study_pino.best_value)
    print(f"Best PINO : {best_pino}")

    # =========================================================================
    # TRAINING — one model on GPU at a time
    # Lifecycle: build → train → save → del → gc → empty_cache → next model
    # =========================================================================

    # ── DeepONet ──────────────────────────────────────────────────────────────
    print("\n" + "="*60 + "\nTraining MF-DeepONet: max_height")
    don_kwargs = dict(p=best_don["p"], hidden=best_don["hidden"], trunk_dim=2)
    with mlflow.start_run(run_name="train_deeponet"):
        mlflow.set_tags({"target": TARGET, "phase": "training",
                         "model": "deeponet", "device": str(device)})
        mlflow.log_params(best_don)

        don_mh = MFDeepONet(**don_kwargs).to(device)
        hist_don = train_mf(
            don_mh, "deeponet",
            (db.X_lf["tr"], db.X_lf["va"], db.ymh_lf["tr"], db.ymh_lf["va"]),
            (db.X_mf["tr"], db.X_mf["va"], db.ymh_mf["tr"], db.ymh_mf["va"]),
            (db.X_hf["tr"], db.X_hf["va"], db.ymh_hf["tr"], db.ymh_hf["va"]),
            db.q_sp, best_don,
            epochs_lf=EPOCHS["lf"], epochs_mf=EPOCHS["mf"], epochs_hf=EPOCHS["hf"],
        )
        _log_history(hist_don)
        torch.save(don_mh.state_dict(), "don_maxheight.pt")
        mlflow.log_artifact("don_maxheight.pt")

    # Free GPU before building the next model
    del don_mh
    gc.collect()
    torch.cuda.empty_cache()

    # ── FNO ───────────────────────────────────────────────────────────────────
    print("\n" + "="*60 + "\nTraining MF-FNO: max_height")
    fno_kwargs = dict(
        nlat=nlat, nlon=nlon,
        latent_channels=best_fno["latent_channels"],
        num_fno_layers=best_fno["num_fno_layers"],
        num_fno_modes=best_fno["num_fno_modes"],
        decoder_layer_size=best_fno["decoder_layer_size"],
        out_channels=1,
    )
    with mlflow.start_run(run_name="train_fno"):
        mlflow.set_tags({"target": TARGET, "phase": "training",
                         "model": "fno", "device": str(device)})
        mlflow.log_params(best_fno)

        fno_mh = MFFno(**fno_kwargs).to(device)
        hist_fno = train_mf(
            fno_mh, "fno",
            (db.X_lf["tr"], db.X_lf["va"], db.ymh2d_lf["tr"], db.ymh2d_lf["va"]),
            (db.X_mf["tr"], db.X_mf["va"], db.ymh2d_mf["tr"], db.ymh2d_mf["va"]),
            (db.X_hf["tr"], db.X_hf["va"], db.ymh2d_hf["tr"], db.ymh2d_hf["va"]),
            db.bathy_t, best_fno,
            epochs_lf=EPOCHS["lf"], epochs_mf=EPOCHS["mf"], epochs_hf=EPOCHS["hf"],
        )
        _log_history(hist_fno)
        torch.save(fno_mh.state_dict(), "fno_maxheight.pt")
        mlflow.log_artifact("fno_maxheight.pt")

    del fno_mh
    gc.collect()
    torch.cuda.empty_cache()

    # ── PINO ──────────────────────────────────────────────────────────────────
    print("\n" + "="*60 + "\nTraining MF-PINO: max_height")
    pino_kwargs = dict(
        nlat=nlat, nlon=nlon,
        latent_channels=best_pino["latent_channels"],
        num_fno_layers=best_pino["num_fno_layers"],
        num_fno_modes=best_pino["num_fno_modes"],
        decoder_layer_size=best_pino["decoder_layer_size"],
        out_channels=1,
        task="max_height",
        H_raw=H_raw,
        grid_info=grid_info,
    )
    with mlflow.start_run(run_name="train_pino"):
        mlflow.set_tags({"target": TARGET, "phase": "training",
                         "model": "pino", "device": str(device)})
        mlflow.log_params(best_pino)

        pino_mh = MFPino(**pino_kwargs).to(device)
        hist_pino = train_mf_pino(
            pino_mh,
            (db.X_lf["tr"], db.X_lf["va"], db.ymh2d_lf["tr"], db.ymh2d_lf["va"]),
            (db.X_mf["tr"], db.X_mf["va"], db.ymh2d_mf["tr"], db.ymh2d_mf["va"]),
            (db.X_hf["tr"], db.X_hf["va"], db.ymh2d_hf["tr"], db.ymh2d_hf["va"]),
            db.bathy_t, best_pino,
            epochs_lf=EPOCHS["lf"], epochs_mf=EPOCHS["mf"], epochs_hf=EPOCHS["hf"],
        )
        _log_history(hist_pino)
        torch.save(pino_mh.state_dict(), "pino_maxheight.pt")
        mlflow.log_artifact("pino_maxheight.pt")

    del pino_mh
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # EVALUATION — reload models from disk one at a time
    # Never hold more than one model in VRAM simultaneously.
    # =========================================================================
    print("\n" + "="*60 + "\nEvaluation")
    with mlflow.start_run(run_name="evaluation"):
        mlflow.set_tags({"target": TARGET, "phase": "evaluation"})

        te_X      = {"lf": db.X_lf["te"], "mf": db.X_mf["te"], "hf": db.X_hf["te"]}
        te_y_flat = {"lf": db.ymh_lf["te"],  "mf": db.ymh_mf["te"],  "hf": db.ymh_hf["te"]}
        te_y_2d   = {"lf": db.ymh2d_lf["te"], "mf": db.ymh2d_mf["te"], "hf": db.ymh2d_hf["te"]}

        results = []

        # ── Evaluate DeepONet ─────────────────────────────────────────────────
        don_mh = MFDeepONet(**don_kwargs).to(device)
        don_mh.load_state_dict(
            torch.load("don_maxheight.pt", map_location=device, weights_only=True)
        )
        for fid in ["lf", "mf", "hf"]:
            results.append(evaluate(
                don_mh, "deeponet", te_X[fid], te_y_flat[fid],
                db.q_sp, "MF-DeepONet max_height", fid, INVERSE_FN,
            ))
        del don_mh
        gc.collect()
        torch.cuda.empty_cache()

        # ── Evaluate FNO ──────────────────────────────────────────────────────
        fno_mh = MFFno(**fno_kwargs).to(device)
        fno_mh.load_state_dict(
            torch.load("fno_maxheight.pt", map_location=device, weights_only=True)
        )
        for fid in ["lf", "mf", "hf"]:
            results.append(evaluate(
                fno_mh, "fno", te_X[fid], te_y_2d[fid],
                db.bathy_t, "MF-FNO max_height", fid, INVERSE_FN,
            ))
        del fno_mh
        gc.collect()
        torch.cuda.empty_cache()

        # ── Evaluate PINO ─────────────────────────────────────────────────────
        pino_mh = MFPino(**pino_kwargs).to(device)
        pino_mh.load_state_dict(
            torch.load("pino_maxheight.pt", map_location=device, weights_only=True)
        )
        for fid in ["lf", "mf", "hf"]:
            results.append(evaluate(
                pino_mh, "fno", te_X[fid], te_y_2d[fid],
                db.bathy_t, "MF-PINO max_height", fid, INVERSE_FN,
            ))
        del pino_mh
        gc.collect()
        torch.cuda.empty_cache()

        _log_results(results)

        # ── Comparison table ──────────────────────────────────────────────────
        df = pd.DataFrame([
            {k: v for k, v in r.items() if k not in ("pred", "true")}
            for r in results
        ])
        for col in ["mae", "rmse", "nrmse", "r2"]:
            df[col] = df[col].round(4)
        df["rel_err"] = df["rel_err"].round(2)
        print("\n" + "="*60 + "\nFINAL COMPARISON\n" + "="*60)
        print(df.to_string(index=False))
        df.to_csv("results_max_height.csv", index=False)
        mlflow.log_artifact("results_max_height.csv")

        # ── Loss curves ───────────────────────────────────────────────────────
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(21, 5))
        for hist, label, ax in [
            (hist_don,  "MF-DeepONet", axes[0]),
            (hist_fno,  "MF-FNO",      axes[1]),
            (hist_pino, "MF-PINO",     axes[2]),
        ]:
            for stage in ["lf", "mf", "hf"]:
                ax.semilogy(hist[f"{stage}_train"], label=f"{stage.upper()} train")
                ax.semilogy(hist[f"{stage}_val"],   label=f"{stage.upper()} val", ls="--")
            ax.set_title(f"{label} — max_height loss")
            ax.set_xlabel("Epoch"); ax.set_ylabel("RMSE"); ax.legend()
        plt.tight_layout()
        plt.savefig("loss_curves_max_height.png", dpi=150)
        plt.close(fig)
        mlflow.log_artifact("loss_curves_max_height.png")

        # ── Spatial maps ──────────────────────────────────────────────────────
        for r in results:
            fname = f"map_{r['model'].replace(' ', '_')}_{r['fidelity']}.png"
            plot_maps(r, db.lon, db.lat, nlat, nlon, out_path=fname)
            mlflow.log_artifact(fname)

    print("\n✓ max_height complete.")
    print("  Models   : don_maxheight.pt  fno_maxheight.pt  pino_maxheight.pt")
    print("  Results  : results_max_height.csv  loss_curves_max_height.png")
