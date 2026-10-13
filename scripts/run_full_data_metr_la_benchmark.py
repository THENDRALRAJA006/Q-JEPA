"""
Full-Timeline Multi-Horizon Benchmark on METR-LA (Task 3).

Evaluates the complete usable dataset:
- Timesteps: 34,272 (4 continuous months)
- Graph nodes: 207 sensors, 1,722 directed edges
- Chronological 70% Train (23,967 windows) / 10% Val (3,416 windows) / 20% Held-Out Test (6,844 windows)
- Evaluates:
  1. Persistence Baseline
  2. Historical Average Baseline (Slot-level, masking missing speeds)
  3. Spatial-Temporal LSTM
  4. STGCN
  5. JEPAGraph (Full JEPA Objective)
  6. JEPAGraph Direct Regression (Ablation)
- Predictions: 17,000,496 raw test targets; 14,940,228 valid unmasked test targets.
- Output: results/metrics/metr_la_full_data_benchmark_results.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gc
import numpy as np
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


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_model_on_test(
    model_fn,
    test_loader: DataLoader,
    mean: float,
    std: float,
    adj_matrix: Optional[torch.Tensor] = None,
    device: str = "cpu",
    is_pytorch: bool = True,
) -> Tuple[Dict[str, Dict[str, float]], float, int, int]:
    """
    Evaluates predictions at steps 3 (15m), 6 (30m), 12 (60m) and average over steps 1..12.
    Returns:
        (metrics_by_horizon, mean_latency_ms, raw_evals, valid_evals)
    """
    latencies = []
    horizons = {
        "15_min_step3": 2,     # 0-indexed: index 2 is step 3 (15 min)
        "30_min_step6": 5,     # index 5 is step 6 (30 min)
        "60_min_step12": 11,   # index 11 is step 12 (60 min)
    }

    abs_err_sums = {h: 0.0 for h in horizons}
    sq_err_sums = {h: 0.0 for h in horizons}
    ape_sums = {h: 0.0 for h in horizons}
    counts = {h: 0 for h in horizons}

    all_abs_err_sum = 0.0
    all_sq_err_sum = 0.0
    all_ape_sum = 0.0
    all_count = 0

    raw_evals = 0
    valid_evals = 0

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

            mask_batch = (y_np > 0.0) & np.isfinite(y_np) & np.isfinite(pred_mph)
            raw_evals += int(y_np.size)
            valid_evals += int(np.count_nonzero(mask_batch))

            diff_batch = np.abs(y_np - pred_mph)
            sq_diff_batch = diff_batch ** 2

            for h_name, h_idx in horizons.items():
                m_h = mask_batch[:, h_idx, :]
                n_h = int(np.count_nonzero(m_h))
                if n_h > 0:
                    d_h = diff_batch[:, h_idx, :][m_h]
                    t_h = y_np[:, h_idx, :][m_h]
                    abs_err_sums[h_name] += float(np.sum(d_h))
                    sq_err_sums[h_name] += float(np.sum(sq_diff_batch[:, h_idx, :][m_h]))
                    ape_sums[h_name] += float(np.sum((d_h / t_h) * 100.0))
                    counts[h_name] += n_h

            n_all = int(np.count_nonzero(mask_batch))
            if n_all > 0:
                all_abs_err_sum += float(np.sum(diff_batch[mask_batch]))
                all_sq_err_sum += float(np.sum(sq_diff_batch[mask_batch]))
                all_ape_sum += float(np.sum((diff_batch[mask_batch] / y_np[mask_batch]) * 100.0))
                all_count += n_all

    results = {}
    for h_name in horizons:
        c = counts[h_name]
        results[h_name] = {
            "MAE": float(abs_err_sums[h_name] / max(c, 1)),
            "RMSE": float(np.sqrt(sq_err_sums[h_name] / max(c, 1))),
            "MAPE": float(ape_sums[h_name] / max(c, 1)),
        }

    results["average_steps_1_to_12"] = {
        "MAE": float(all_abs_err_sum / max(all_count, 1)),
        "RMSE": float(np.sqrt(all_sq_err_sum / max(all_count, 1))),
        "MAPE": float(all_ape_sum / max(all_count, 1)),
    }

    mean_latency = float(np.mean(latencies)) if latencies else 0.0
    return results, mean_latency, raw_evals, valid_evals


def train_single_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    adj_norm: torch.Tensor,
    mean: float,
    std: float,
    epochs: int = 1,
    lr: float = 1e-3,
    is_jepa: bool = False,
    rep_loss_weight: float = 0.1,
    device: str = "cpu",
) -> Tuple[nn.Module, float]:
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
                if adj_norm is not None and isinstance(model, STGCNModel):
                    pred = model(x_b, adj_norm.to(device))
                else:
                    pred = model(x_b)

                loss = F.mse_loss(pred, y_b_norm)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            train_loss_acc += loss.item()
            n_batches += 1

        avg_train = train_loss_acc / max(n_batches, 1)

        # Validation evaluation for checkpoint selection
        model.eval()
        val_loss_acc = 0.0
        val_batches = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx = vx.to(device)
                vy_norm = (vy.to(device) - mean) / std
                if is_jepa:
                    res = model(vx, adj_norm.to(device))
                    vpred = res["speed_pred"]
                elif isinstance(model, STGCNModel):
                    vpred = model(vx, adj_norm.to(device))
                else:
                    vpred = model(vx)

                vloss = F.mse_loss(vpred, vy_norm)
                val_loss_acc += vloss.item()
                val_batches += 1

        avg_val = val_loss_acc / max(val_batches, 1)
        logger.info(
            "Epoch %d/%d — Train Loss: %.4f | Val Loss: %.4f%s",
            epoch, epochs, avg_train, avg_val,
            " (New Best -> Checkpoint Saved)" if avg_val < best_val_loss else ""
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_weights = copy.deepcopy(model.state_dict())

    train_duration = time.time() - t0
    model.load_state_dict(best_weights)
    return model, train_duration


def run_full_data_benchmark(
    seed: int = 42,
    epochs: int = 1,
    batch_size: int = 64,
) -> Dict:
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Initializing Full METR-LA Dataset (34,272 timesteps, Device: %s)...", device)

    loader = METRLALoader(data_dir="data/metr_la")
    splits = loader.prepare_dataset(max_timesteps=None)
    adj_norm = loader.compute_normalized_adj(adj_type="random_walk")

    train_ds = TrafficWindowDataset(splits.train_x, splits.train_y)
    val_ds = TrafficWindowDataset(splits.val_x, splits.val_y)
    test_ds = TrafficWindowDataset(splits.test_x, splits.test_y)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    num_nodes = splits.train_x.shape[2]
    n_raw_total = len(splits.test_x) * 12 * num_nodes
    valid_mask = (splits.test_y > 0.0) & np.isfinite(splits.test_y)
    n_valid_total = int(np.count_nonzero(valid_mask))
    n_masked_total = n_raw_total - n_valid_total

    results: Dict[str, Any] = {
        "metadata": {
            "dataset_name": "METR-LA (Los Angeles Highway Traffic Sensors — Complete Timeline)",
            "source": "Zenodo (10.5281/zenodo.5724362, CC BY 4.0)",
            "total_dataset_timesteps": 34272,
            "timesteps_evaluated": 34272,
            "sample_counts": {
                "train_windows": len(splits.train_x),
                "val_windows": len(splits.val_x),
                "test_windows": len(splits.test_x),
            },
            "prediction_counts": {
                "raw_test_predictions": n_raw_total,
                "valid_unmasked_predictions": n_valid_total,
                "masked_missing_predictions": n_masked_total,
                "valid_percentage": float(100.0 * n_valid_total / n_raw_total),
            },
            "normalization": {
                "train_mean_mph": float(splits.mean),
                "train_std_mph": float(splits.std),
                "fitted_split": "train_only (leakage-free)",
            },
            "num_sensors": num_nodes,
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "device": device,
            "resource_note": "Evaluated on full 34,272 timesteps for 1 complete epoch per neural architecture on CPU.",
        },
        "models": {},
        "comparison_metrics": {},
    }

    # 1. Persistence Baseline
    logger.info("\n>>> 1. Evaluating Persistence Baseline on Full Test Set (N=%d)...", len(splits.test_x))
    pers = PersistenceBaseline(mean=splits.mean, std=splits.std)
    m_pers, lat_pers, r_eval, v_eval = evaluate_model_on_test(
        pers, test_loader, splits.mean, splits.std, is_pytorch=False
    )
    results["models"]["Persistence"] = {
        "parameters": 0, "train_s": 0.0, "latency_ms": lat_pers, "horizons": m_pers,
    }

    # 2. Historical Average Baseline
    logger.info(">>> 2. Fitting & Evaluating Historical Average Baseline...")
    ha = HistoricalAverageBaseline(mean=splits.mean, std=splits.std)
    t0 = time.time()
    ha.fit(splits.train_x, splits.train_y)
    t_ha = time.time() - t0
    m_ha, lat_ha, _, _ = evaluate_model_on_test(
        ha, test_loader, splits.mean, splits.std, is_pytorch=False
    )
    results["models"]["Historical_Average"] = {
        "parameters": int(ha.table.size) if ha.table is not None else 0,
        "train_s": t_ha, "latency_ms": lat_ha, "horizons": m_ha,
    }

    # 3. Spatial-Temporal LSTM
    logger.info(">>> 3. Training & Evaluating Spatial-Temporal LSTM...")
    lstm_m = SpatialTemporalLSTM(num_nodes=num_nodes, in_features=2, hidden_dim=64, num_layers=2)
    p_lstm = count_parameters(lstm_m)
    lstm_m, t_lstm = train_single_model(
        lstm_m, train_loader, val_loader, adj_norm, splits.mean, splits.std,
        epochs=epochs, lr=1e-3, is_jepa=False, device=device
    )
    m_lstm, lat_lstm, _, _ = evaluate_model_on_test(
        lstm_m, test_loader, splits.mean, splits.std, adj_matrix=None, device=device
    )
    results["models"]["SpatialTemporal_LSTM"] = {
        "parameters": p_lstm, "train_s": t_lstm, "latency_ms": lat_lstm, "horizons": m_lstm,
    }
    del lstm_m; gc.collect()

    # 4. STGCN
    logger.info(">>> 4. Training & Evaluating STGCN...")
    stgcn_m = STGCNModel(num_nodes=num_nodes, in_channels=2, hidden_dim=32, output_window=12)
    p_stgcn = count_parameters(stgcn_m)
    stgcn_m, t_stgcn = train_single_model(
        stgcn_m, train_loader, val_loader, adj_norm, splits.mean, splits.std,
        epochs=epochs, lr=1e-3, is_jepa=False, device=device
    )
    m_stgcn, lat_stgcn, _, _ = evaluate_model_on_test(
        stgcn_m, test_loader, splits.mean, splits.std, adj_matrix=adj_norm, device=device
    )
    results["models"]["STGCN"] = {
        "parameters": p_stgcn, "train_s": t_stgcn, "latency_ms": lat_stgcn, "horizons": m_stgcn,
    }
    del stgcn_m; gc.collect()

    # 5. JEPAGraph (Full)
    logger.info(">>> 5. Training & Evaluating JEPAGraph (Full JEPA Objective)...")
    jepa_m = JEPAGraphModel(
        num_nodes=num_nodes, in_channels=2, latent_dim=32, output_window=12,
        use_graph=True, use_ema=True, ema_decay=0.99
    )
    p_jepa = count_parameters(jepa_m)
    jepa_m, t_jepa = train_single_model(
        jepa_m, train_loader, val_loader, adj_norm, splits.mean, splits.std,
        epochs=epochs, lr=1e-3, is_jepa=True, rep_loss_weight=0.1, device=device
    )
    m_jepa, lat_jepa, _, _ = evaluate_model_on_test(
        jepa_m, test_loader, splits.mean, splits.std, adj_matrix=adj_norm, device=device
    )
    results["models"]["JEPAGraph"] = {
        "parameters": p_jepa, "train_s": t_jepa, "latency_ms": lat_jepa, "horizons": m_jepa,
    }
    del jepa_m; gc.collect()

    # 6. JEPAGraph Direct Regression Ablation
    logger.info(">>> 6. Training & Evaluating JEPAGraph Direct Regression Ablation...")
    jepa_reg = JEPAGraphModel(
        num_nodes=num_nodes, in_channels=2, latent_dim=32, output_window=12,
        use_graph=True, use_ema=False
    )
    p_reg = count_parameters(jepa_reg)
    jepa_reg, t_reg = train_single_model(
        jepa_reg, train_loader, val_loader, adj_norm, splits.mean, splits.std,
        epochs=epochs, lr=1e-3, is_jepa=True, rep_loss_weight=0.0, device=device
    )
    m_reg, lat_reg, _, _ = evaluate_model_on_test(
        jepa_reg, test_loader, splits.mean, splits.std, adj_matrix=adj_norm, device=device
    )
    results["models"]["JEPAGraph_DirectRegression"] = {
        "parameters": p_reg, "train_s": t_reg, "latency_ms": lat_reg, "horizons": m_reg,
    }
    del jepa_reg; gc.collect()

    # Relative Improvements
    horizons = ["15_min_step3", "30_min_step6", "60_min_step12", "average_steps_1_to_12"]
    for h in horizons:
        stgcn_mae = results["models"]["STGCN"]["horizons"][h]["MAE"]
        jepa_mae = results["models"]["JEPAGraph"]["horizons"][h]["MAE"]
        reg_mae = results["models"]["JEPAGraph_DirectRegression"]["horizons"][h]["MAE"]

        imp_vs_stgcn = 100.0 * (stgcn_mae - jepa_mae) / max(stgcn_mae, 1e-5)
        imp_vs_reg = 100.0 * (reg_mae - jepa_mae) / max(reg_mae, 1e-5)

        results["comparison_metrics"][h] = {
            "jepa_improvement_over_stgcn_pct": imp_vs_stgcn,
            "jepa_improvement_over_direct_regression_pct": imp_vs_reg,
        }

    # Save to metrics directory
    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metr_la_full_data_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved full-data benchmark results to: %s", out_path)

    # Print Formatted Table
    print("\n" + "=" * 105)
    print(f"FULL METR-LA BENCHMARK SUMMARY (Timesteps = 34,272, Evaluated Targets = {n_valid_total:,})")
    print("=" * 105)
    print(f"{'Model':<28} {'Params':>8} {'15m MAE (step 3)':>18} {'30m MAE (step 6)':>18} {'60m MAE (step 12)':>18}")
    print("-" * 105)
    for m, mdata in results["models"].items():
        p = mdata["parameters"]
        h15 = mdata["horizons"]["15_min_step3"]["MAE"]
        h30 = mdata["horizons"]["30_min_step6"]["MAE"]
        h60 = mdata["horizons"]["60_min_step12"]["MAE"]
        print(f"{m:<28} {p:>8} {h15:>18.2f} {h30:>18.2f} {h60:>18.2f}")
    print("=" * 105)

    print("\nRelative JEPAGraph Gains on Full Dataset:")
    for h in ["15_min_step3", "30_min_step6", "60_min_step12"]:
        g_stgcn = results["comparison_metrics"][h]["jepa_improvement_over_stgcn_pct"]
        g_reg = results["comparison_metrics"][h]["jepa_improvement_over_direct_regression_pct"]
        print(f"  {h:<15}: vs STGCN: {g_stgcn:+.2f}%  |  vs Direct Regression (Ablation): {g_reg:+.2f}%")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1, help="Epochs per neural model (default: 1 on full dataset)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    run_full_data_benchmark(seed=args.seed, epochs=args.epochs, batch_size=args.batch_size)
