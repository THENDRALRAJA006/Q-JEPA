"""
Standardized Local State-of-the-Art Benchmark Suite on METR-LA.

Executes a fair, reproducible, apples-to-apples comparison among:
1. Persistence
2. Historical Average (Time-of-day slots)
3. Spatial-Temporal LSTM
4. STGCN (Spatio-Temporal Graph Convolutional Network)
5. Graph WaveNet (Dilated Gated TCN + Spatial GCN)
6. Spatial-Temporal Transformer (Spatial Multihead Attention + Temporal Multihead Attention)
7. JEPAGraph (Full JEPA Latent Representation Learning)
8. JEPAGraph Direct Regression Ablation (Identical encoder/decoder, rep_loss_weight=0.0)

Protocols & Integrity:
- Chronological 70% Train / 10% Val / 20% Held-Out Test split.
- Strict train-only Z-score normalization (no leakage).
- Missing-value masking on speed <= 0.0 and NaNs (consistent across all models).
- Validation loss checkpoint selection (early stopping / best weights restored).
- Horizons: 15m (step 3), 30m (step 6), 60m (step 12), and intermediate steps 1..12.
- Relative MAE formula: 100 * (baseline_MAE - JEPAGraph_MAE) / baseline_MAE.
- Produces: results/metrics/sota_local_benchmark.json
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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
    GraphWaveNetModel,
    HistoricalAverageBaseline,
    JEPAGraphModel,
    PersistenceBaseline,
    SpatialTemporalLSTM,
    SpatialTemporalTransformerModel,
    STGCNModel,
)

logger = get_logger(__name__)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: Any) -> int:
    if isinstance(model, nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    elif hasattr(model, "table") and model.table is not None:
        return int(model.table.size)
    return 0


def evaluate_model_on_test(
    model: Any,
    test_loader: DataLoader,
    mean: float,
    std: float,
    adj_matrix: Optional[torch.Tensor] = None,
    device: str = "cpu",
    is_pytorch: bool = True,
) -> Tuple[Dict[str, Dict[str, float]], float, int, int]:
    """
    Evaluates predictions across all horizons (steps 1..12), with specific focus on
    15m (step 3), 30m (step 6), 60m (step 12), and the 12-step average.
    """
    latencies = []
    horizons = {
        "15_min_step3": 2,     # 0-indexed: index 2 is step 3 (15 min)
        "30_min_step6": 5,     # index 5 is step 6 (30 min)
        "60_min_step12": 11,   # index 11 is step 12 (60 min)
    }

    all_preds = []
    all_targets = []

    if is_pytorch:
        model.eval()

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_np = x_batch.numpy()
            y_np = y_batch.numpy()

            t0 = time.perf_counter()
            if is_pytorch:
                x_t = x_batch.to(device)
                if isinstance(model, (STGCNModel, GraphWaveNetModel)):
                    out = model(x_t, adj_matrix.to(device))
                elif isinstance(model, JEPAGraphModel):
                    res = model(x_t, adj_matrix.to(device))
                    out = res["speed_pred"]
                elif isinstance(model, SpatialTemporalTransformerModel):
                    out = model(x_t)
                elif isinstance(model, SpatialTemporalLSTM):
                    out = model(x_t)
                else:
                    out = model(x_t)

                if isinstance(out, dict):
                    pred_norm = out["speed_pred"].cpu().numpy()
                else:
                    pred_norm = out.cpu().numpy()

                pred_mph = pred_norm * std + mean
            else:
                pred_mph = model.predict(x_np, num_horizons=12)

            latency_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(latency_ms)

            all_preds.append(pred_mph)
            all_targets.append(y_np)

    y_pred_all = np.concatenate(all_preds, axis=0)  # [N_test, 12, N_nodes]
    y_true_all = np.concatenate(all_targets, axis=0)  # [N_test, 12, N_nodes]

    results_by_horizon: Dict[str, Dict[str, float]] = {}
    for h_name, h_idx in horizons.items():
        results_by_horizon[h_name] = evaluate_metrics(y_true_all[:, h_idx, :], y_pred_all[:, h_idx, :])

    # Also record individual step metrics from 1 to 12
    for step in range(1, 13):
        results_by_horizon[f"step_{step}"] = evaluate_metrics(y_true_all[:, step - 1, :], y_pred_all[:, step - 1, :])

    # Overall average across all 12 horizons
    results_by_horizon["average_steps_1_to_12"] = evaluate_metrics(y_true_all, y_pred_all)

    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    raw_evals = int(y_true_all.size)
    valid_evals = int(np.count_nonzero((y_true_all > 0.0) & np.isfinite(y_true_all)))

    return results_by_horizon, mean_lat, raw_evals, valid_evals


def train_single_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    adj_norm: torch.Tensor,
    mean: float,
    std: float,
    epochs: int = 3,
    lr: float = 1e-3,
    is_jepa: bool = False,
    rep_loss_weight: float = 0.1,
    device: str = "cpu",
) -> Tuple[nn.Module, float]:
    """Train model with validation-loss based checkpoint selection."""
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
                if isinstance(model, (STGCNModel, GraphWaveNetModel)):
                    pred = model(x_b, adj_norm.to(device))
                elif isinstance(model, SpatialTemporalTransformerModel):
                    pred = model(x_b)
                elif isinstance(model, SpatialTemporalLSTM):
                    pred = model(x_b)
                else:
                    pred = model(x_b)

                loss = F.mse_loss(pred, y_b_norm)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            train_loss_acc += loss.item()
            n_batches += 1

        # Validation evaluation
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
                elif isinstance(model, (STGCNModel, GraphWaveNetModel)):
                    vpred = model(vx, adj_norm.to(device))
                elif isinstance(model, SpatialTemporalTransformerModel):
                    vpred = model(vx)
                elif isinstance(model, SpatialTemporalLSTM):
                    vpred = model(vx)
                else:
                    vpred = model(vx)

                vloss = F.mse_loss(vpred, vy_norm)
                val_loss_acc += vloss.item()
                val_batches += 1

        avg_val = val_loss_acc / max(val_batches, 1)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_weights = copy.deepcopy(model.state_dict())

    train_duration = time.time() - t0
    model.load_state_dict(best_weights)
    return model, train_duration


def run_benchmark(
    timesteps: int = 3000,
    epochs: int = 2,
    seeds: List[int] = [42, 43, 44],
    batch_size: int = 64,
) -> Dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Initializing Standardized Local Benchmark (Timesteps: %d, Device: %s, Seeds: %s)...", timesteps, device, seeds)

    loader = METRLALoader(data_dir="data/metr_la")
    splits = loader.prepare_dataset(max_timesteps=timesteps)
    adj_norm = loader.compute_normalized_adj(adj_type="random_walk")

    num_nodes = splits.train_x.shape[2]
    val_ds = TrafficWindowDataset(splits.val_x, splits.val_y)
    test_ds = TrafficWindowDataset(splits.test_x, splits.test_y)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model_names = [
        "Persistence",
        "Historical_Average",
        "SpatialTemporal_LSTM",
        "STGCN",
        "Graph_WaveNet",
        "SpatialTemporal_Transformer",
        "JEPAGraph",
        "JEPAGraph_DirectRegression",
    ]

    seed_results: Dict[str, List[Dict[str, Any]]] = {m: [] for m in model_names}
    durations: Dict[str, List[float]] = {m: [] for m in model_names}
    latencies: Dict[str, List[float]] = {m: [] for m in model_names}

    for seed_idx, seed in enumerate(seeds):
        logger.info("\n========== RUNNING SEED %d (%d/%d) ==========", seed, seed_idx + 1, len(seeds))
        set_seed(seed)

        train_ds = TrafficWindowDataset(splits.train_x, splits.train_y)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        # 1. Persistence
        pers = PersistenceBaseline(mean=splits.mean, std=splits.std)
        m_pers, lat_pers, r_eval, v_eval = evaluate_model_on_test(
            pers, test_loader, splits.mean, splits.std, is_pytorch=False
        )
        seed_results["Persistence"].append(m_pers)
        durations["Persistence"].append(0.0)
        latencies["Persistence"].append(lat_pers)

        # 2. Historical Average
        ha = HistoricalAverageBaseline(mean=splits.mean, std=splits.std)
        t0 = time.time()
        ha.fit(splits.train_x, splits.train_y)
        t_ha = time.time() - t0
        m_ha, lat_ha, _, _ = evaluate_model_on_test(
            ha, test_loader, splits.mean, splits.std, is_pytorch=False
        )
        seed_results["Historical_Average"].append(m_ha)
        durations["Historical_Average"].append(t_ha)
        latencies["Historical_Average"].append(lat_ha)

        # 3. Spatial-Temporal LSTM
        lstm_m = SpatialTemporalLSTM(num_nodes=num_nodes, in_features=2, hidden_dim=64, num_layers=2)
        lstm_m, t_lstm = train_single_model(
            lstm_m, train_loader, val_loader, adj_norm, splits.mean, splits.std,
            epochs=epochs, lr=1e-3, is_jepa=False, device=device
        )
        m_lstm, lat_lstm, _, _ = evaluate_model_on_test(
            lstm_m, test_loader, splits.mean, splits.std, adj_matrix=None, device=device
        )
        seed_results["SpatialTemporal_LSTM"].append(m_lstm)
        durations["SpatialTemporal_LSTM"].append(t_lstm)
        latencies["SpatialTemporal_LSTM"].append(lat_lstm)
        del lstm_m; gc.collect()

        # 4. STGCN
        stgcn_m = STGCNModel(num_nodes=num_nodes, in_channels=2, hidden_dim=32, output_window=12)
        stgcn_m, t_stgcn = train_single_model(
            stgcn_m, train_loader, val_loader, adj_norm, splits.mean, splits.std,
            epochs=epochs, lr=1e-3, is_jepa=False, device=device
        )
        m_stgcn, lat_stgcn, _, _ = evaluate_model_on_test(
            stgcn_m, test_loader, splits.mean, splits.std, adj_matrix=adj_norm, device=device
        )
        seed_results["STGCN"].append(m_stgcn)
        durations["STGCN"].append(t_stgcn)
        latencies["STGCN"].append(lat_stgcn)
        del stgcn_m; gc.collect()

        # 5. Graph WaveNet
        gwn_m = GraphWaveNetModel(num_nodes=num_nodes, in_channels=2, hidden_dim=32, output_window=12)
        gwn_m, t_gwn = train_single_model(
            gwn_m, train_loader, val_loader, adj_norm, splits.mean, splits.std,
            epochs=epochs, lr=1e-3, is_jepa=False, device=device
        )
        m_gwn, lat_gwn, _, _ = evaluate_model_on_test(
            gwn_m, test_loader, splits.mean, splits.std, adj_matrix=adj_norm, device=device
        )
        seed_results["Graph_WaveNet"].append(m_gwn)
        durations["Graph_WaveNet"].append(t_gwn)
        latencies["Graph_WaveNet"].append(lat_gwn)
        del gwn_m; gc.collect()

        # 6. Spatial-Temporal Transformer
        stt_m = SpatialTemporalTransformerModel(num_nodes=num_nodes, in_channels=2, hidden_dim=32, num_heads=2, output_window=12)
        stt_m, t_stt = train_single_model(
            stt_m, train_loader, val_loader, adj_norm, splits.mean, splits.std,
            epochs=epochs, lr=1e-3, is_jepa=False, device=device
        )
        m_stt, lat_stt, _, _ = evaluate_model_on_test(
            stt_m, test_loader, splits.mean, splits.std, adj_matrix=None, device=device
        )
        seed_results["SpatialTemporal_Transformer"].append(m_stt)
        durations["SpatialTemporal_Transformer"].append(t_stt)
        latencies["SpatialTemporal_Transformer"].append(lat_stt)
        del stt_m; gc.collect()

        # 7. JEPAGraph (Full JEPA Objective)
        jepa_m = JEPAGraphModel(
            num_nodes=num_nodes, in_channels=2, latent_dim=32, output_window=12,
            use_graph=True, use_ema=True, ema_decay=0.99
        )
        jepa_m, t_jepa = train_single_model(
            jepa_m, train_loader, val_loader, adj_norm, splits.mean, splits.std,
            epochs=epochs, lr=1e-3, is_jepa=True, rep_loss_weight=0.1, device=device
        )
        m_jepa, lat_jepa, _, _ = evaluate_model_on_test(
            jepa_m, test_loader, splits.mean, splits.std, adj_matrix=adj_norm, device=device
        )
        seed_results["JEPAGraph"].append(m_jepa)
        durations["JEPAGraph"].append(t_jepa)
        latencies["JEPAGraph"].append(lat_jepa)
        del jepa_m; gc.collect()

        # 8. JEPAGraph Direct Regression Ablation
        jepa_reg = JEPAGraphModel(
            num_nodes=num_nodes, in_channels=2, latent_dim=32, output_window=12,
            use_graph=True, use_ema=False
        )
        jepa_reg, t_reg = train_single_model(
            jepa_reg, train_loader, val_loader, adj_norm, splits.mean, splits.std,
            epochs=epochs, lr=1e-3, is_jepa=True, rep_loss_weight=0.0, device=device
        )
        m_reg, lat_reg, _, _ = evaluate_model_on_test(
            jepa_reg, test_loader, splits.mean, splits.std, adj_matrix=adj_norm, device=device
        )
        seed_results["JEPAGraph_DirectRegression"].append(m_reg)
        durations["JEPAGraph_DirectRegression"].append(t_reg)
        latencies["JEPAGraph_DirectRegression"].append(lat_reg)
        del jepa_reg; gc.collect()

    # Aggregate Multi-Seed Statistics
    out_models: Dict[str, Any] = {}
    key_horizons = ["15_min_step3", "30_min_step6", "60_min_step12", "average_steps_1_to_12"]

    # Parameter counts
    param_counts = {
        "Persistence": 0,
        "Historical_Average": int(ha.table.size) if ha.table is not None else 0,
        "SpatialTemporal_LSTM": count_parameters(SpatialTemporalLSTM(num_nodes=num_nodes, in_features=2, hidden_dim=64, num_layers=2)),
        "STGCN": count_parameters(STGCNModel(num_nodes=num_nodes, in_channels=2, hidden_dim=32, output_window=12)),
        "Graph_WaveNet": count_parameters(GraphWaveNetModel(num_nodes=num_nodes, in_channels=2, hidden_dim=32, output_window=12)),
        "SpatialTemporal_Transformer": count_parameters(SpatialTemporalTransformerModel(num_nodes=num_nodes, in_channels=2, hidden_dim=32, num_heads=2, output_window=12)),
        "JEPAGraph": count_parameters(JEPAGraphModel(num_nodes=num_nodes, in_channels=2, latent_dim=32, output_window=12, use_graph=True, use_ema=True)),
        "JEPAGraph_DirectRegression": count_parameters(JEPAGraphModel(num_nodes=num_nodes, in_channels=2, latent_dim=32, output_window=12, use_graph=True, use_ema=False)),
    }

    for m in model_names:
        m_dict: Dict[str, Any] = {
            "parameters": param_counts[m],
            "mean_train_s": float(np.mean(durations[m])),
            "mean_latency_ms": float(np.mean(latencies[m])),
            "horizons": {},
        }
        for h in key_horizons:
            maes = [seed_results[m][s_idx][h]["MAE"] for s_idx in range(len(seeds))]
            rmses = [seed_results[m][s_idx][h]["RMSE"] for s_idx in range(len(seeds))]
            mapes = [seed_results[m][s_idx][h]["MAPE"] for s_idx in range(len(seeds))]

            mae_m = float(np.mean(maes))
            mae_s = float(np.std(maes))
            rmse_m = float(np.mean(rmses))
            rmse_s = float(np.std(rmses))
            mape_m = float(np.mean(mapes))

            # 95% Confidence Interval
            if len(seeds) > 1 and mae_s > 0:
                h_ci = 1.96 * mae_s / np.sqrt(len(seeds))
                ci_95 = [mae_m - h_ci, mae_m + h_ci]
            else:
                ci_95 = [mae_m, mae_m]

            m_dict["horizons"][h] = {
                "mae_mean": mae_m,
                "mae_std": mae_s,
                "mae_ci_95": ci_95,
                "rmse_mean": rmse_m,
                "rmse_std": rmse_s,
                "mape_mean": mape_m,
                "seed_maes": maes,
            }
        out_models[m] = m_dict

    # Compute Relative Improvements of JEPAGraph over Baselines
    comparison_metrics: Dict[str, Any] = {}
    for h in key_horizons:
        jepa_mae = out_models["JEPAGraph"]["horizons"][h]["mae_mean"]
        comp_h: Dict[str, float] = {}
        for b in model_names:
            if b == "JEPAGraph":
                continue
            b_mae = out_models[b]["horizons"][h]["mae_mean"]
            rel_imp = 100.0 * (b_mae - jepa_mae) / max(b_mae, 1e-5)
            comp_h[f"jepa_improvement_over_{b.lower()}_pct"] = rel_imp
        comparison_metrics[h] = comp_h

    # Find the best eligible baseline automatically for each horizon
    best_baselines: Dict[str, Any] = {}
    for h in key_horizons:
        eligible_baselines = [b for b in model_names if b not in ("JEPAGraph", "JEPAGraph_DirectRegression")]
        best_b = min(eligible_baselines, key=lambda b: out_models[b]["horizons"][h]["mae_mean"])
        best_b_mae = out_models[best_b]["horizons"][h]["mae_mean"]
        jepa_mae = out_models["JEPAGraph"]["horizons"][h]["mae_mean"]
        rel_gain = 100.0 * (best_b_mae - jepa_mae) / max(best_b_mae, 1e-5)
        best_baselines[h] = {
            "best_baseline_name": best_b,
            "best_baseline_mae": best_b_mae,
            "jepa_mae": jepa_mae,
            "relative_gain_pct": rel_gain,
        }

    benchmark_summary: Dict[str, Any] = {
        "metadata": {
            "title": "Standardized Multi-Seed Local Benchmark Suite",
            "dataset_name": "METR-LA (Los Angeles Highway Traffic Sensors)",
            "timesteps_evaluated": timesteps,
            "sample_counts": {
                "train_windows": len(splits.train_x),
                "val_windows": len(splits.val_x),
                "test_windows": len(splits.test_x),
            },
            "num_sensors": num_nodes,
            "seeds": seeds,
            "epochs_per_seed": epochs,
            "batch_size": batch_size,
            "device": device,
            "normalization": {
                "train_mean_mph": float(splits.mean),
                "train_std_mph": float(splits.std),
                "fitted_split": "train_only (leakage-free)",
            },
            "zero_handling_policy": "Speed readings <= 0.0 mph and NaNs are masked strictly during metric evaluation.",
            "hardware_platform": "Windows CPU (Intel/AMD x86_64, PyTorch CPU)",
        },
        "models": out_models,
        "comparison_metrics": comparison_metrics,
        "best_eligible_baselines": best_baselines,
    }

    out_path = Path("results/metrics/sota_local_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(benchmark_summary, f, indent=2)
    logger.info("Saved local benchmark results to: %s", out_path)

    # Format Local Benchmark Results
    print("\n" + "=" * 115)
    print(f"LOCAL STANDARDIZED BENCHMARK SUMMARY (Timesteps={timesteps}, Seeds={seeds}, Epochs={epochs})")
    print("=" * 115)
    print(f"{'Model':<30} {'Params':>8} {'15m MAE (mph)':>20} {'30m MAE (mph)':>20} {'60m MAE (mph)':>20} {'Latency':>10}")
    print("-" * 115)
    for m, mdata in out_models.items():
        p = mdata["parameters"]
        h15 = f"{mdata['horizons']['15_min_step3']['mae_mean']:.2f} +/- {mdata['horizons']['15_min_step3']['mae_std']:.2f}"
        h30 = f"{mdata['horizons']['30_min_step6']['mae_mean']:.2f} +/- {mdata['horizons']['30_min_step6']['mae_std']:.2f}"
        h60 = f"{mdata['horizons']['60_min_step12']['mae_mean']:.2f} +/- {mdata['horizons']['60_min_step12']['mae_std']:.2f}"
        lat = f"{mdata['mean_latency_ms']:.2f}ms"
        print(f"{m:<30} {p:>8} {h15:>20} {h30:>20} {h60:>20} {lat:>10}")
    print("=" * 115)

    # Check if full timeline results exist, and include them as the second tier
    full_path = Path("results/metrics/metr_la_full_data_benchmark_results.json")
    full_data_tier = None
    if full_path.exists():
        try:
            with open(full_path) as ff:
                full_data_tier = json.load(ff)
        except Exception as e:
            logger.warning("Could not load full timeline results: %s", e)

    combined_benchmark = {
        "metadata": {
            "title": "Standardized Local State-of-the-Art Benchmark Suite",
            "dataset_name": "METR-LA (Los Angeles Highway Traffic Sensors)",
            "benchmark_date": "2026-10-13",
            "device": device,
            "zero_handling_policy": "Speed readings <= 0.0 mph and NaNs masked strictly during metric evaluation.",
            "hardware_platform": "Windows CPU (Intel/AMD x86_64, PyTorch CPU)",
            "notes": "Separates controlled multi-seed convergent experiments from full-timeline single-epoch runs.",
        },
        "controlled_multi_seed_tier": benchmark_summary,
        "full_timeline_tier": full_data_tier,
    }

    out_path = Path("results/metrics/sota_local_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(combined_benchmark, f, indent=2)
    logger.info("Saved combined local benchmark results to: %s", out_path)

    return combined_benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=3000, help="Timesteps to evaluate (default: 3000)")
    parser.add_argument("--epochs", type=int, default=2, help="Epochs per seed (default: 2)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44], help="Random seeds (default: 42 43 44)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    args = parser.parse_args()

    run_benchmark(
        timesteps=args.timesteps,
        epochs=args.epochs,
        seeds=args.seeds,
        batch_size=args.batch_size,
    )
