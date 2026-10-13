"""
Complete Training and Evaluation Pipeline for METR-LA Graph Traffic Forecasting.

Executes:
1. Data loading and chronological splitting (70% train / 10% val / 20% test)
2. Overfitting verification on tiny subset (Task 4)
3. Baselines: Persistence, Historical Average, ST-LSTM, STGCN
4. Proposed Model: JEPAGraph (Joint Embedding Predictive Architecture)
5. Ablation Study (Task 7):
   - Ablation 1: JEPAGraph Direct Regression (no latent representation loss)
   - Ablation 2: JEPAGraph No Graph (identity adjacency)
   - Ablation 3: JEPAGraph No EMA (target directly tracks online)
6. Multi-Horizon Metrics:
   - 15 min (horizon 3)
   - 30 min (horizon 6)
   - 60 min (horizon 12)
   - Overall (all 12 horizons)
7. Resource accounting (params, training time, inference latency, memory)
8. Output persistence to results/metrics/metr_la_forecasting_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import copy
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


def masked_mae_loss(y_pred: torch.Tensor, y_true: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    """Masked L1 loss ignoring sensor dropout / zeros."""
    mask = (y_true > null_val) & torch.isfinite(y_true) & torch.isfinite(y_pred)
    if not torch.any(mask):
        return torch.tensor(0.0, device=y_pred.device, requires_grad=True)
    return torch.mean(torch.abs(y_pred[mask] - y_true[mask]))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_model_multi_horizon(
    model_fn,
    test_loader: DataLoader,
    mean: float,
    std: float,
    adj_matrix: Optional[torch.Tensor] = None,
    device: str = "cpu",
    is_pytorch: bool = True,
) -> Tuple[Dict[str, Dict[str, float]], float]:
    """Evaluate multi-horizon performance at horizons 3 (15m), 6 (30m), 12 (60m), and overall.

    Returns:
        metrics_dict: dict of horizon -> {MAE, RMSE, MAPE}
        latency_ms: inference latency per batch of windows in ms
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

            latency = (time.perf_counter() - t0) * 1000.0  # ms
            latencies.append(latency)

            all_preds.append(pred_mph)
            all_targets.append(y_np)

    y_pred_all = np.concatenate(all_preds, axis=0)  # [N, 12, num_nodes]
    y_true_all = np.concatenate(all_targets, axis=0)  # [N, 12, num_nodes]

    horizons = {
        "15_min_step3": 2,    # 0-indexed: index 2 is step 3
        "30_min_step6": 5,    # index 5 is step 6
        "60_min_step12": 11,  # index 11 is step 12
    }

    results = {}
    for h_name, h_idx in horizons.items():
        m = evaluate_metrics(y_true_all[:, h_idx, :], y_pred_all[:, h_idx, :])
        results[h_name] = m

    # Overall across all 12 steps
    results["overall_12steps"] = evaluate_metrics(y_true_all, y_pred_all)

    mean_latency_ms = float(np.mean(latencies)) if latencies else 0.0
    return results, mean_latency_ms


def run_tiny_overfitting_check(splits: DatasetSplits, adj_norm: torch.Tensor) -> bool:
    """Task 4 verification: verify training pipeline can overfit a tiny subset."""
    logger.info("=" * 60)
    logger.info("Task 4 Verification: Tiny subset overfitting check")
    logger.info("=" * 60)

    # Use first 8 windows
    tiny_x = splits.train_x[:8]
    tiny_y = splits.train_y[:8]
    tiny_y_norm = (tiny_y - splits.mean) / splits.std

    dataset = TrafficWindowDataset(tiny_x, tiny_y_norm)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    num_nodes = splits.train_x.shape[2]
    model = JEPAGraphModel(num_nodes=num_nodes, in_channels=2, latent_dim=16, output_window=12)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    initial_loss = None
    final_loss = None

    for epoch in range(25):
        model.train()
        epoch_loss = 0.0
        for x_b, y_b in loader:
            optimizer.zero_grad()
            res = model(x_b, adj_norm, y_future=y_b)
            loss_speed = F.mse_loss(res["speed_pred"], y_b)
            loss_rep = F.smooth_l1_loss(res["z_pred"], res["z_target"])
            loss = loss_speed + 0.1 * loss_rep
            loss.backward()
            optimizer.step()
            model.update_target_ema()
            epoch_loss += loss.item()

        if epoch == 0:
            initial_loss = epoch_loss
        final_loss = epoch_loss

    logger.info("Tiny overfit check: initial_loss=%.4f -> final_loss=%.4f", initial_loss, final_loss)
    assert final_loss < initial_loss, "Model failed to reduce loss on tiny set!"
    logger.info("Overfitting check PASSED (loss reduced by %.1f%%)", (initial_loss - final_loss) / initial_loss * 100)
    return True


def train_pytorch_model(
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
    """Train a PyTorch forecasting model with early stopping on validation loss."""
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
            # y_b in physical units; convert to normalized for loss calculation
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

        # Validation step
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

        avg_train = train_loss_acc / max(n_batches, 1)
        avg_val = val_loss_acc / max(n_val, 1)
        logger.info("Epoch %d/%d — Train MSE: %.4f | Val MSE: %.4f", epoch, epochs, avg_train, avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_weights = copy.deepcopy(model.state_dict())

    train_time_s = time.time() - t0
    model.load_state_dict(best_weights)
    return model, train_time_s


def run_full_metr_la_benchmark(
    max_timesteps: Optional[int] = 3000,
    epochs: int = 3,
    batch_size: int = 32,
    device: str = "cpu",
) -> Dict:
    """Run full benchmarking pipeline across Baselines, JEPAGraph, and Ablations."""
    logger.info("=" * 70)
    logger.info("SURAKSHIT BHARAT — METR-LA Real Graph Traffic Forecasting Benchmark")
    logger.info("Max timesteps: %s | Epochs: %d | Batch size: %d | Device: %s",
                str(max_timesteps), epochs, batch_size, device)
    logger.info("=" * 70)

    loader = METRLALoader(data_dir="data/metr_la")
    adj_norm = loader.compute_normalized_adj(adj_type="random_walk")
    splits = loader.prepare_dataset(max_timesteps=max_timesteps, include_time_of_day=True)

    # 1. Verification of Overfitting on Tiny Subset (Task 4)
    run_tiny_overfitting_check(splits, adj_norm)

    # Prepare DataLoaders
    train_ds = TrafficWindowDataset(splits.train_x, splits.train_y)
    val_ds = TrafficWindowDataset(splits.val_x, splits.val_y)
    test_ds = TrafficWindowDataset(splits.test_x, splits.test_y)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    num_nodes = splits.train_x.shape[2]
    all_results = {}
    saved_checkpoints = {}
    models_dir = Path("models/metr_la")
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── BASELINE 1: Persistence ───────────────────────────────────────────────
    logger.info("\n--- Evaluating Baseline: Persistence ---")
    pers = PersistenceBaseline(mean=splits.mean, std=splits.std)
    m_pers, lat_pers = evaluate_model_multi_horizon(
        pers, test_loader, splits.mean, splits.std, is_pytorch=False
    )
    all_results["Persistence"] = {
        "model_type": "Heuristic Baseline",
        "parameters": 0,
        "train_time_s": 0.0,
        "latency_ms_per_batch": lat_pers,
        "metrics": m_pers,
    }
    logger.info("Persistence 15m MAE: %.2f | 30m MAE: %.2f | 60m MAE: %.2f",
                m_pers["15_min_step3"]["MAE"], m_pers["30_min_step6"]["MAE"], m_pers["60_min_step12"]["MAE"])

    # ── BASELINE 2: Historical Average ─────────────────────────────────────────
    logger.info("\n--- Evaluating Baseline: Historical Average ---")
    ha = HistoricalAverageBaseline(mean=splits.mean, std=splits.std)
    t0 = time.time()
    ha.fit(splits.train_x, splits.train_y)
    ha_fit_time = time.time() - t0
    m_ha, lat_ha = evaluate_model_multi_horizon(
        ha, test_loader, splits.mean, splits.std, is_pytorch=False
    )
    all_results["Historical_Average"] = {
        "model_type": "Seasonal Heuristic Baseline",
        "parameters": int(ha.table.size) if ha.table is not None else 0,
        "train_time_s": ha_fit_time,
        "latency_ms_per_batch": lat_ha,
        "metrics": m_ha,
    }
    logger.info("Historical Average 15m MAE: %.2f | 30m MAE: %.2f | 60m MAE: %.2f",
                m_ha["15_min_step3"]["MAE"], m_ha["30_min_step6"]["MAE"], m_ha["60_min_step12"]["MAE"])

    # ── BASELINE 3: Spatial-Temporal LSTM ──────────────────────────────────────
    logger.info("\n--- Training Baseline: Spatial-Temporal LSTM ---")
    lstm_model = SpatialTemporalLSTM(num_nodes=num_nodes, in_features=2, hidden_dim=64, num_layers=2)
    p_lstm = count_parameters(lstm_model)
    lstm_model, t_lstm = train_pytorch_model(
        lstm_model, train_loader, val_loader, adj_norm, splits.mean, splits.std,
        epochs=epochs, lr=1e-3, is_jepa=False, device=device
    )
    m_lstm, lat_lstm = evaluate_model_multi_horizon(
        lstm_model, test_loader, splits.mean, splits.std, adj_matrix=None, device=device
    )
    all_results["SpatialTemporal_LSTM"] = {
        "model_type": "Temporal Recurrent Neural Network",
        "parameters": p_lstm,
        "train_time_s": t_lstm,
        "latency_ms_per_batch": lat_lstm,
        "metrics": m_lstm,
    }
    torch.save(lstm_model.state_dict(), models_dir / "st_lstm_metr_la.pt")
    logger.info("ST-LSTM 15m MAE: %.2f | 30m MAE: %.2f | 60m MAE: %.2f",
                m_lstm["15_min_step3"]["MAE"], m_lstm["30_min_step6"]["MAE"], m_lstm["60_min_step12"]["MAE"])

    # ── BASELINE 4: STGCN (Graph-Temporal) ─────────────────────────────────────
    logger.info("\n--- Training Baseline: STGCN (Graph Convolution) ---")
    stgcn_model = STGCNModel(num_nodes=num_nodes, in_channels=2, hidden_dim=32, output_window=12)
    p_stgcn = count_parameters(stgcn_model)
    stgcn_model, t_stgcn = train_pytorch_model(
        stgcn_model, train_loader, val_loader, adj_norm, splits.mean, splits.std,
        epochs=epochs, lr=1e-3, is_jepa=False, device=device
    )
    m_stgcn, lat_stgcn = evaluate_model_multi_horizon(
        stgcn_model, test_loader, splits.mean, splits.std, adj_matrix=adj_norm, device=device
    )
    all_results["STGCN_Graph"] = {
        "model_type": "Spatial-Temporal Graph Convolutional Network",
        "parameters": p_stgcn,
        "train_time_s": t_stgcn,
        "latency_ms_per_batch": lat_stgcn,
        "metrics": m_stgcn,
    }
    torch.save(stgcn_model.state_dict(), models_dir / "stgcn_metr_la.pt")
    logger.info("STGCN 15m MAE: %.2f | 30m MAE: %.2f | 60m MAE: %.2f",
                m_stgcn["15_min_step3"]["MAE"], m_stgcn["30_min_step6"]["MAE"], m_stgcn["60_min_step12"]["MAE"])

    # ── PROPOSED MODEL: JEPAGraph (Full JEPA) ─────────────────────────────────
    logger.info("\n--- Training Proposed: JEPAGraph (Full JEPA Architecture) ---")
    jepa_model = JEPAGraphModel(
        num_nodes=num_nodes, in_channels=2, latent_dim=32, output_window=12,
        use_graph=True, use_ema=True, ema_decay=0.99
    )
    p_jepa = count_parameters(jepa_model)
    jepa_model, t_jepa = train_pytorch_model(
        jepa_model, train_loader, val_loader, adj_norm, splits.mean, splits.std,
        epochs=epochs, lr=1e-3, is_jepa=True, rep_loss_weight=0.1, device=device
    )
    m_jepa, lat_jepa = evaluate_model_multi_horizon(
        jepa_model, test_loader, splits.mean, splits.std, adj_matrix=adj_norm, device=device
    )
    all_results["JEPAGraph_Full"] = {
        "model_type": "Joint Embedding Predictive Architecture + GCN",
        "parameters": p_jepa,
        "train_time_s": t_jepa,
        "latency_ms_per_batch": lat_jepa,
        "metrics": m_jepa,
    }
    torch.save(jepa_model.state_dict(), models_dir / "jepagraph_full_metr_la.pt")
    logger.info("JEPAGraph 15m MAE: %.2f | 30m MAE: %.2f | 60m MAE: %.2f",
                m_jepa["15_min_step3"]["MAE"], m_jepa["30_min_step6"]["MAE"], m_jepa["60_min_step12"]["MAE"])

    # ── ABLATION 1: JEPAGraph Direct Regression (No Representation Loss) ──────
    logger.info("\n--- Training Ablation 1: JEPAGraph Direct Regression (rep_weight=0.0) ---")
    jepa_reg = JEPAGraphModel(
        num_nodes=num_nodes, in_channels=2, latent_dim=32, output_window=12,
        use_graph=True, use_ema=False
    )
    p_reg = count_parameters(jepa_reg)
    jepa_reg, t_reg = train_pytorch_model(
        jepa_reg, train_loader, val_loader, adj_norm, splits.mean, splits.std,
        epochs=epochs, lr=1e-3, is_jepa=True, rep_loss_weight=0.0, device=device
    )
    m_reg, lat_reg = evaluate_model_multi_horizon(
        jepa_reg, test_loader, splits.mean, splits.std, adj_matrix=adj_norm, device=device
    )
    all_results["Ablation_DirectRegression"] = {
        "model_type": "Ablation: JEPAGraph without Representation Loss",
        "parameters": p_reg,
        "train_time_s": t_reg,
        "latency_ms_per_batch": lat_reg,
        "metrics": m_reg,
    }
    logger.info("Ablation-DirectReg 15m MAE: %.2f | 30m MAE: %.2f | 60m MAE: %.2f",
                m_reg["15_min_step3"]["MAE"], m_reg["30_min_step6"]["MAE"], m_reg["60_min_step12"]["MAE"])

    # ── ABLATION 2: JEPAGraph No Graph (Identity Adjacency) ────────────────────
    logger.info("\n--- Training Ablation 2: JEPAGraph No Graph (Identity Adj) ---")
    jepa_nograph = JEPAGraphModel(
        num_nodes=num_nodes, in_channels=2, latent_dim=32, output_window=12,
        use_graph=False, use_ema=True, ema_decay=0.99
    )
    p_nograph = count_parameters(jepa_nograph)
    jepa_nograph, t_nograph = train_pytorch_model(
        jepa_nograph, train_loader, val_loader, adj_norm, splits.mean, splits.std,
        epochs=epochs, lr=1e-3, is_jepa=True, rep_loss_weight=0.1, device=device
    )
    m_nograph, lat_nograph = evaluate_model_multi_horizon(
        jepa_nograph, test_loader, splits.mean, splits.std, adj_matrix=adj_norm, device=device
    )
    all_results["Ablation_NoGraph"] = {
        "model_type": "Ablation: JEPAGraph without Spatial Graph Propagation",
        "parameters": p_nograph,
        "train_time_s": t_nograph,
        "latency_ms_per_batch": lat_nograph,
        "metrics": m_nograph,
    }
    logger.info("Ablation-NoGraph 15m MAE: %.2f | 30m MAE: %.2f | 60m MAE: %.2f",
                m_nograph["15_min_step3"]["MAE"], m_nograph["30_min_step6"]["MAE"], m_nograph["60_min_step12"]["MAE"])

    # ── ABLATION 3: JEPAGraph No EMA Target (Instant Online Sync) ──────────────
    logger.info("\n--- Training Ablation 3: JEPAGraph No EMA Target ---")
    jepa_noema = JEPAGraphModel(
        num_nodes=num_nodes, in_channels=2, latent_dim=32, output_window=12,
        use_graph=True, use_ema=False
    )
    p_noema = count_parameters(jepa_noema)
    jepa_noema, t_noema = train_pytorch_model(
        jepa_noema, train_loader, val_loader, adj_norm, splits.mean, splits.std,
        epochs=epochs, lr=1e-3, is_jepa=True, rep_loss_weight=0.1, device=device
    )
    m_noema, lat_noema = evaluate_model_multi_horizon(
        jepa_noema, test_loader, splits.mean, splits.std, adj_matrix=adj_norm, device=device
    )
    all_results["Ablation_NoEMA"] = {
        "model_type": "Ablation: JEPAGraph without EMA Target Updates",
        "parameters": p_noema,
        "train_time_s": t_noema,
        "latency_ms_per_batch": lat_noema,
        "metrics": m_noema,
    }
    logger.info("Ablation-NoEMA 15m MAE: %.2f | 30m MAE: %.2f | 60m MAE: %.2f",
                m_noema["15_min_step3"]["MAE"], m_noema["30_min_step6"]["MAE"], m_noema["60_min_step12"]["MAE"])

    # ── Save Full Results JSON ────────────────────────────────────────────────
    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "metr_la_forecasting_results.json"

    final_payload = {
        "dataset": "METR-LA (Los Angeles Highway Traffic Sensors)",
        "source": "Zenodo (10.5281/zenodo.5724362, CC BY 4.0)",
        "num_sensors": num_nodes,
        "timesteps_evaluated": max_timesteps if max_timesteps is not None else "full (34272)",
        "sampling_rate": "5 minutes",
        "horizons": {
            "15_min": "3 steps",
            "30_min": "6 steps",
            "60_min": "12 steps",
        },
        "train_mean_mph": splits.mean,
        "train_std_mph": splits.std,
        "split_ratios": {"train": 0.70, "val": 0.10, "test": 0.20},
        "models": all_results,
    }

    with open(out_json, "w") as f:
        json.dump(final_payload, f, indent=2)
    logger.info("\nSaved all METR-LA forecasting benchmark results to: %s", out_json)

    # Print Summary Table
    print("\n" + "=" * 90)
    print("SURAKSHIT BHARAT — METR-LA Multi-Horizon Forecasting Evaluation")
    print("=" * 90)
    print(f"{'Model':<25} {'Params':>8} {'Train(s)':>8} {'15m MAE':>9} {'30m MAE':>9} {'60m MAE':>9} {'Overall MAE':>11}")
    print("-" * 90)
    for mname, mdata in all_results.items():
        p = mdata.get("parameters", 0)
        t = mdata.get("train_time_s", 0.0)
        m = mdata["metrics"]
        mae15 = m["15_min_step3"]["MAE"]
        mae30 = m["30_min_step6"]["MAE"]
        mae60 = m["60_min_step12"]["MAE"]
        mae_all = m["overall_12steps"]["MAE"]
        print(f"{mname:<25} {p:>8} {t:>7.1f}s {mae15:>8.2f}  {mae30:>8.2f}  {mae60:>8.2f}   {mae_all:>8.2f}")
    print("=" * 90)

    return final_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=3000, help="Number of timesteps to load (default: 3000 for fast CPU execution)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    run_full_metr_la_benchmark(
        max_timesteps=args.timesteps,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
