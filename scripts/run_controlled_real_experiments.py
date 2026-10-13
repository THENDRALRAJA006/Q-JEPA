"""
Controlled Multi-Seed Multi-Horizon Experiments on METR-LA (Phase 2.1).

Evaluates:
1. Persistence
2. Historical Average
3. Spatial-Temporal LSTM
4. STGCN
5. JEPAGraph (Full JEPA Objective)
6. JEPAGraph Direct Regression (Ablation)

Protocols & Integrity:
- Chronological 70% Train / 10% Val / 20% Held-Out Test split.
- Normalization parameters (mean, std) fit exclusively on training split (no leakage).
- Checkpoint selection based strictly on validation MSE (early stopping/best val).
- Evaluation at step 3 (15m), step 6 (30m), step 12 (60m), and overall (steps 1..12).
- Multiple seeds (default: 42, 43, 44) with mean, std, and confidence intervals reported.
- Relative improvement over STGCN: 100 * (STGCN_MAE - JEPAGraph_MAE) / STGCN_MAE.
- Explicit logging of dataset size, sample count, and hardware limits.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.common.logging import get_logger
from src.data.metr_la_loader import (
    DatasetSplits,
    METRLALoader,
    TrafficWindowDataset,
    evaluate_metrics,
)
from src.forecasting.real_graph_models import (
    HistoricalAverageBaseline,
    JEPAGraphModel,
    PersistenceBaseline,
    SpatialTemporalLSTM,
    STGCNModel,
)

logger = get_logger(__name__)


def set_seed(seed: int):
    """Ensure reproducibility for random initialization."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_model_at_horizons(
    model_fn,
    test_loader: DataLoader,
    mean: float,
    std: float,
    adj_matrix: Optional[torch.Tensor] = None,
    device: str = "cpu",
    is_pytorch: bool = True,
) -> Tuple[Dict[str, Dict[str, float]], float, int]:
    """Evaluate predictions at steps 3 (15m), 6 (30m), 12 (60m) and average over steps 1..12.

    Returns:
        metrics_by_horizon: dict
        mean_latency_ms: float
        num_predictions_evaluated: int
    """
    all_preds = []
    all_targets = []
    latencies = []

    if is_pytorch:
        model_fn.eval()

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_np = x_batch.numpy()
            y_np = y_batch.numpy()

            t0 = time.perf_counter()
            if is_pytorch:
                x_t = x_batch.to(device)
                if adj_matrix is not None:
                    out = model_fn(x_t, adj_matrix.to(device))
                else:
                    out = model_fn(x_t)

                if isinstance(out, dict):
                    pred_norm = out["speed_pred"].cpu().numpy()
                else:
                    pred_norm = out.cpu().numpy()

                pred_mph = pred_norm * std + mean
            else:
                pred_mph = model_fn.predict(x_np, num_horizons=12)

            latency = (time.perf_counter() - t0) * 1000.0
            latencies.append(latency)

            all_preds.append(pred_mph)
            all_targets.append(y_np)

    y_pred_all = np.concatenate(all_preds, axis=0)  # [N_test, 12, num_nodes]
    y_true_all = np.concatenate(all_targets, axis=0)  # [N_test, 12, num_nodes]

    horizons = {
        "15_min_step3": 2,     # 0-indexed: index 2 is step 3 (15 min)
        "30_min_step6": 5,     # index 5 is step 6 (30 min)
        "60_min_step12": 11,   # index 11 is step 12 (60 min)
    }

    results = {}
    for h_name, h_idx in horizons.items():
        results[h_name] = evaluate_metrics(y_true_all[:, h_idx, :], y_pred_all[:, h_idx, :])

    # Distinctly label the average across all future steps 1..12
    results["average_steps_1_to_12"] = evaluate_metrics(y_true_all, y_pred_all)

    mean_latency = float(np.mean(latencies)) if latencies else 0.0
    total_valid_evals = int(np.count_nonzero((y_true_all > 0.0) & np.isfinite(y_true_all)))

    return results, mean_latency, total_valid_evals


def train_single_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    adj_norm: torch.Tensor,
    mean: float,
    std: float,
    epochs: int = 2,
    lr: float = 1e-3,
    is_jepa: bool = False,
    rep_loss_weight: float = 0.1,
    device: str = "cpu",
) -> Tuple[nn.Module, float]:
    """Train PyTorch model with early stopping / checkpoint selection on validation loss."""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_acc = 0.0
        n_batches = 0

        for x_b, y_b in train_loader:
            x_b = x_b.to(device)
            y_b_norm = (y_b.to(device) - mean) / std

            optimizer.zero_grad()
            if is_jepa:
                res = model(x_b, adj_norm.to(device), y_future=y_b_norm)
                speed_loss = F.mse_loss(res["speed_pred"], y_b_norm)
                if rep_loss_weight > 0.0 and "z_target" in res:
                    rep_loss = F.smooth_l1_loss(res["z_pred"], res["z_target"])
                    loss = speed_loss + rep_loss_weight * rep_loss
                else:
                    loss = speed_loss

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                model.update_target_ema()
            else:
                if isinstance(model, STGCNModel):
                    pred = model(x_b, adj_norm.to(device))
                else:
                    pred = model(x_b)
                loss = F.mse_loss(pred, y_b_norm)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            train_loss_acc += loss.item()
            n_batches += 1

        # Validation evaluation (strictly for checkpoint selection)
        model.eval()
        val_loss_acc = 0.0
        n_val = 0
        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v = x_v.to(device)
                y_v_norm = (y_v.to(device) - mean) / std
                if is_jepa:
                    res = model(x_v, adj_norm.to(device))
                    val_loss = F.mse_loss(res["speed_pred"], y_v_norm)
                elif isinstance(model, STGCNModel):
                    pred = model(x_v, adj_norm.to(device))
                    val_loss = F.mse_loss(pred, y_v_norm)
                else:
                    pred = model(x_v)
                    val_loss = F.mse_loss(pred, y_v_norm)
                val_loss_acc += val_loss.item()
                n_val += 1

        avg_val = val_loss_acc / max(n_val, 1)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_weights = copy.deepcopy(model.state_dict())

    train_time_s = time.time() - t0
    model.load_state_dict(best_weights)
    return model, train_time_s


def run_controlled_seed_experiment(
    splits: DatasetSplits,
    adj_norm: torch.Tensor,
    seed: int,
    epochs: int = 2,
    batch_size: int = 64,
    device: str = "cpu",
) -> Dict:
    """Run all 6 required models for one seed."""
    set_seed(seed)
    num_nodes = splits.train_x.shape[2]

    train_ds = TrafficWindowDataset(splits.train_x, splits.train_y)
    val_ds = TrafficWindowDataset(splits.val_x, splits.val_y)
    test_ds = TrafficWindowDataset(splits.test_x, splits.test_y)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    seed_results = {}
    models_dir = Path("models/metr_la")
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Persistence
    pers = PersistenceBaseline(mean=splits.mean, std=splits.std)
    m_pers, lat_pers, n_eval = evaluate_model_at_horizons(
        pers, test_loader, splits.mean, splits.std, is_pytorch=False
    )
    seed_results["Persistence"] = {
        "params": 0, "train_s": 0.0, "latency_ms": lat_pers, "metrics": m_pers, "n_eval": n_eval
    }

    # 2. Historical Average
    ha = HistoricalAverageBaseline(mean=splits.mean, std=splits.std)
    t0 = time.time()
    ha.fit(splits.train_x, splits.train_y)
    ha_t = time.time() - t0
    m_ha, lat_ha, _ = evaluate_model_at_horizons(
        ha, test_loader, splits.mean, splits.std, is_pytorch=False
    )
    seed_results["Historical_Average"] = {
        "params": int(ha.table.size) if ha.table is not None else 0,
        "train_s": ha_t, "latency_ms": lat_ha, "metrics": m_ha, "n_eval": n_eval
    }

    # 3. Spatial-Temporal LSTM
    lstm_m = SpatialTemporalLSTM(num_nodes=num_nodes, in_features=2, hidden_dim=64, num_layers=2)
    p_lstm = count_parameters(lstm_m)
    lstm_m, t_lstm = train_single_model(
        lstm_m, train_loader, val_loader, adj_norm, splits.mean, splits.std,
        epochs=epochs, lr=1e-3, is_jepa=False, device=device
    )
    m_lstm, lat_lstm, _ = evaluate_model_at_horizons(
        lstm_m, test_loader, splits.mean, splits.std, adj_matrix=None, device=device
    )
    torch.save(lstm_m.state_dict(), models_dir / f"lstm_seed_{seed}.pt")
    seed_results["SpatialTemporal_LSTM"] = {
        "params": p_lstm, "train_s": t_lstm, "latency_ms": lat_lstm, "metrics": m_lstm, "n_eval": n_eval
    }

    # 4. STGCN
    stgcn_m = STGCNModel(num_nodes=num_nodes, in_channels=2, hidden_dim=32, output_window=12)
    p_stgcn = count_parameters(stgcn_m)
    stgcn_m, t_stgcn = train_single_model(
        stgcn_m, train_loader, val_loader, adj_norm, splits.mean, splits.std,
        epochs=epochs, lr=1e-3, is_jepa=False, device=device
    )
    m_stgcn, lat_stgcn, _ = evaluate_model_at_horizons(
        stgcn_m, test_loader, splits.mean, splits.std, adj_matrix=adj_norm, device=device
    )
    torch.save(stgcn_m.state_dict(), models_dir / f"stgcn_seed_{seed}.pt")
    seed_results["STGCN"] = {
        "params": p_stgcn, "train_s": t_stgcn, "latency_ms": lat_stgcn, "metrics": m_stgcn, "n_eval": n_eval
    }

    # 5. JEPAGraph (Full)
    jepa_m = JEPAGraphModel(
        num_nodes=num_nodes, in_channels=2, latent_dim=32, output_window=12,
        use_graph=True, use_ema=True, ema_decay=0.99
    )
    p_jepa = count_parameters(jepa_m)
    jepa_m, t_jepa = train_single_model(
        jepa_m, train_loader, val_loader, adj_norm, splits.mean, splits.std,
        epochs=epochs, lr=1e-3, is_jepa=True, rep_loss_weight=0.1, device=device
    )
    m_jepa, lat_jepa, _ = evaluate_model_at_horizons(
        jepa_m, test_loader, splits.mean, splits.std, adj_matrix=adj_norm, device=device
    )
    torch.save(jepa_m.state_dict(), models_dir / f"jepagraph_seed_{seed}.pt")
    seed_results["JEPAGraph"] = {
        "params": p_jepa, "train_s": t_jepa, "latency_ms": lat_jepa, "metrics": m_jepa, "n_eval": n_eval
    }

    # 6. JEPAGraph Direct Regression Ablation
    jepa_reg = JEPAGraphModel(
        num_nodes=num_nodes, in_channels=2, latent_dim=32, output_window=12,
        use_graph=True, use_ema=False
    )
    p_reg = count_parameters(jepa_reg)
    jepa_reg, t_reg = train_single_model(
        jepa_reg, train_loader, val_loader, adj_norm, splits.mean, splits.std,
        epochs=epochs, lr=1e-3, is_jepa=True, rep_loss_weight=0.0, device=device
    )
    m_reg, lat_reg, _ = evaluate_model_at_horizons(
        jepa_reg, test_loader, splits.mean, splits.std, adj_matrix=adj_norm, device=device
    )
    torch.save(jepa_reg.state_dict(), models_dir / f"jepa_direct_reg_seed_{seed}.pt")
    seed_results["JEPAGraph_DirectRegression"] = {
        "params": p_reg, "train_s": t_reg, "latency_ms": lat_reg, "metrics": m_reg, "n_eval": n_eval
    }

    return seed_results


def run_multi_seed_benchmark(
    seeds: List[int] = [42, 43, 44],
    timesteps: Optional[int] = 3000,
    epochs: int = 2,
    batch_size: int = 64,
) -> Dict:
    """Orchestrate multi-seed experiments across models and compile statistics."""
    logger.info("=" * 70)
    logger.info("PHASE 2.1 — Controlled Multi-Seed METR-LA Forecasting Benchmark")
    logger.info("Seeds: %s | Timesteps: %s | Epochs: %d | Batch Size: %d",
                seeds, str(timesteps), epochs, batch_size)
    logger.info("=" * 70)

    loader = METRLALoader(data_dir="data/metr_la")
    adj_norm = loader.compute_normalized_adj(adj_type="random_walk")
    splits = loader.prepare_dataset(max_timesteps=timesteps, include_time_of_day=True)

    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_seed_runs = {}
    for s_idx, seed in enumerate(seeds, 1):
        logger.info("\n>>> Running Seed %d/%d (seed=%d) <<<", s_idx, len(seeds), seed)
        t_seed_start = time.time()
        s_res = run_controlled_seed_experiment(
            splits, adj_norm, seed=seed, epochs=epochs, batch_size=batch_size
        )
        elapsed = time.time() - t_seed_start
        logger.info("Seed %d completed in %.1fs", seed, elapsed)
        all_seed_runs[str(seed)] = s_res

        # Save individual seed run
        with open(out_dir / f"real_exp_seed_{seed}.json", "w") as f:
            json.dump(s_res, f, indent=2)

    # Compile multi-seed summary statistics
    model_names = list(all_seed_runs[str(seeds[0])].keys())
    horizons = ["15_min_step3", "30_min_step6", "60_min_step12", "average_steps_1_to_12"]

    summary = {
        "dataset_name": "METR-LA (Los Angeles Highway Traffic Sensors)",
        "source": "Zenodo (10.5281/zenodo.5724362, CC BY 4.0)",
        "total_dataset_timesteps": 34272,
        "timesteps_evaluated": timesteps if timesteps is not None else 34272,
        "usable_samples": {
            "train_windows": len(splits.train_x),
            "val_windows": len(splits.val_x),
            "test_windows": len(splits.test_x),
        },
        "num_sensors": splits.train_x.shape[2],
        "train_mean_mph": splits.mean,
        "train_std_mph": splits.std,
        "seeds": seeds,
        "epochs_per_seed": epochs,
        "batch_size": batch_size,
        "prediction_counts": {
            "per_seed": {
                "raw_predictions": len(splits.test_x) * 12 * splits.train_x.shape[2],
                "valid_unmasked_predictions": int(np.count_nonzero((splits.test_y > 0.0) & np.isfinite(splits.test_y))),
                "masked_missing_predictions": int(len(splits.test_x) * 12 * splits.train_x.shape[2] - np.count_nonzero((splits.test_y > 0.0) & np.isfinite(splits.test_y))),
                "valid_unmasked_percentage": float(100.0 * np.count_nonzero((splits.test_y > 0.0) & np.isfinite(splits.test_y)) / (len(splits.test_x) * 12 * splits.train_x.shape[2])),
            },
            "across_all_seeds": {
                "num_seeds": len(seeds),
                "raw_predictions_total": len(splits.test_x) * 12 * splits.train_x.shape[2] * len(seeds),
                "valid_unmasked_predictions_total": int(np.count_nonzero((splits.test_y > 0.0) & np.isfinite(splits.test_y))) * len(seeds),
                "masked_missing_predictions_total": int(len(splits.test_x) * 12 * splits.train_x.shape[2] - np.count_nonzero((splits.test_y > 0.0) & np.isfinite(splits.test_y))) * len(seeds),
            },
        },
        "models": {},
        "comparison_metrics": {},
    }

    n_raw_seed = summary["prediction_counts"]["per_seed"]["raw_predictions"]
    n_valid_seed = summary["prediction_counts"]["per_seed"]["valid_unmasked_predictions"]

    for m in model_names:
        m_params = all_seed_runs[str(seeds[0])][m]["params"]
        m_durations = [all_seed_runs[str(s)][m]["train_s"] for s in seeds]
        m_latencies = [all_seed_runs[str(s)][m]["latency_ms"] for s in seeds]

        model_summary = {
            "parameters": m_params,
            "mean_train_s": float(np.mean(m_durations)),
            "mean_latency_ms": float(np.mean(m_latencies)),
            "prediction_counts": {
                "raw_per_seed": n_raw_seed,
                "valid_unmasked_per_seed": n_valid_seed,
                "valid_unmasked_total": n_valid_seed * len(seeds),
            },
            "horizons": {},
        }

        for h in horizons:
            maes = [all_seed_runs[str(s)][m]["metrics"][h]["MAE"] for s in seeds]
            rmses = [all_seed_runs[str(s)][m]["metrics"][h]["RMSE"] for s in seeds]

            mae_mean = float(np.mean(maes))
            mae_std = float(np.std(maes, ddof=1)) if len(seeds) > 1 else 0.0
            rmse_mean = float(np.mean(rmses))
            rmse_std = float(np.std(rmses, ddof=1)) if len(seeds) > 1 else 0.0

            # 95% Student-t confidence interval
            if len(seeds) > 1 and mae_std > 0:
                t_crit = float(stats.t.ppf(0.975, df=len(seeds) - 1))
                ci_half = t_crit * mae_std / np.sqrt(len(seeds))
                ci_95 = [mae_mean - ci_half, mae_mean + ci_half]
            else:
                ci_95 = [mae_mean, mae_mean]

            model_summary["horizons"][h] = {
                "mae_mean": mae_mean,
                "mae_std": mae_std,
                "mae_ci_95": ci_95,
                "rmse_mean": rmse_mean,
                "rmse_std": rmse_std,
                "seed_maes": maes,
            }

        summary["models"][m] = model_summary

    # Calculate Relative MAE improvements over STGCN and Direct Regression
    for h in horizons:
        stgcn_mae = summary["models"]["STGCN"]["horizons"][h]["mae_mean"]
        jepa_mae = summary["models"]["JEPAGraph"]["horizons"][h]["mae_mean"]
        reg_mae = summary["models"]["JEPAGraph_DirectRegression"]["horizons"][h]["mae_mean"]

        imp_vs_stgcn = 100.0 * (stgcn_mae - jepa_mae) / max(stgcn_mae, 1e-5)
        imp_vs_reg = 100.0 * (reg_mae - jepa_mae) / max(reg_mae, 1e-5)

        summary["comparison_metrics"][h] = {
            "jepa_improvement_over_stgcn_pct": imp_vs_stgcn,
            "jepa_improvement_over_direct_regression_pct": imp_vs_reg,
        }

    # Save summary JSON
    summary_path = out_dir / "controlled_real_experiments_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("\nSaved controlled experiment summary to: %s", summary_path)

    # Distinctly save smoke test or full data result files
    is_full_data = (timesteps is None or timesteps >= 34272)
    tier_filename = "metr_la_full_data_benchmark_results.json" if is_full_data else "metr_la_smoke_test_results.json"
    tier_path = out_dir / tier_filename
    with open(tier_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved tier-specific experiment summary to: %s", tier_path)

    # Print Formatted Table
    print("\n" + "=" * 105)
    print(f"METR-LA BENCHMARK SUMMARY (N_seeds = {len(seeds)}, Timesteps = {summary['timesteps_evaluated']})")
    print("=" * 105)
    print(f"{'Model':<28} {'Params':>8} {'15m MAE (step 3)':>18} {'30m MAE (step 6)':>18} {'60m MAE (step 12)':>18}")
    print("-" * 105)
    for m, mdata in summary["models"].items():
        p = mdata["parameters"]
        h15 = mdata["horizons"]["15_min_step3"]
        h30 = mdata["horizons"]["30_min_step6"]
        h60 = mdata["horizons"]["60_min_step12"]
        str_15 = f"{h15['mae_mean']:.2f} ± {h15['mae_std']:.2f}"
        str_30 = f"{h30['mae_mean']:.2f} ± {h30['mae_std']:.2f}"
        str_60 = f"{h60['mae_mean']:.2f} ± {h60['mae_std']:.2f}"
        print(f"{m:<28} {p:>8} {str_15:>18} {str_30:>18} {str_60:>18}")
    print("=" * 105)

    print("\nPrediction Counts:")
    print(f"  Raw predictions per seed:             {n_raw_seed:,}")
    print(f"  Valid unmasked predictions per seed:  {n_valid_seed:,} ({summary['prediction_counts']['per_seed']['valid_unmasked_percentage']:.2f}%)")
    print(f"  Total raw predictions ({len(seeds)} seeds):        {summary['prediction_counts']['across_all_seeds']['raw_predictions_total']:,}")
    print(f"  Total valid unmasked ({len(seeds)} seeds):         {summary['prediction_counts']['across_all_seeds']['valid_unmasked_predictions_total']:,}")

    print("\nRelative JEPAGraph Gains:")
    for h in ["15_min_step3", "30_min_step6", "60_min_step12"]:
        g_stgcn = summary["comparison_metrics"][h]["jepa_improvement_over_stgcn_pct"]
        g_reg = summary["comparison_metrics"][h]["jepa_improvement_over_direct_regression_pct"]
        print(f"  {h:<15}: vs STGCN: {g_stgcn:+.2f}%  |  vs Direct Regression (Ablation): {g_reg:+.2f}%")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=3000, help="Timesteps to evaluate (default: 3000)")
    parser.add_argument("--epochs", type=int, default=2, help="Epochs per seed (default: 2)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Random seeds")
    parser.add_argument("--full-data", action="store_true", help="Run full 34,272 dataset evaluation")
    args = parser.parse_args()

    t_steps = None if args.full_data else args.timesteps

    run_multi_seed_benchmark(
        seeds=args.seeds,
        timesteps=t_steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

