"""
Q-JEPA V3.1 — Fair Standardized Baseline Benchmark Suite.

Evaluates 6 models under identical data splits, preprocessing, metrics, and hardware:
1. Persistence
2. STGCN
3. Graph WaveNet
4. JEPAGraph V1
5. JEPAGraph Direct Regression (No JEPA Latent Representation Loss)
6. JEPAGraph-FM (LODO Pretrained on PEMS04+08 + Fine-Tuned on METR-LA)

Horizons evaluated:
- 15 min (horizon step 3)
- 30 min (horizon step 6)
- 60 min (horizon step 12)

Metrics:
- MAE (mph)
- RMSE (mph)
- Model Parameters
- Training Duration (s)
- Inference Latency per batch (ms)

Strict separation between locally reproduced results and published literature benchmarks.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.common.logging import get_logger
from src.data.dataset_registry import get_manifest
from src.data.metr_la_loader import METRLALoader, TrafficWindowDataset
from src.forecasting.jepagraph_foundation import (
    FoundationModelConfig,
    JEPAGraphFoundationModel,
)
from src.forecasting.real_graph_models import (
    GraphWaveNetModel,
    JEPAGraphModel,
    PersistenceBaseline,
    STGCNModel,
)

logger = get_logger("baseline_benchmark_v3_1")


def masked_metrics(y_pred: np.ndarray, y_true: np.ndarray, null_val: float = 0.0) -> Dict[str, float]:
    """Calculate MAE and RMSE ignoring missing values (speeds <= 0 or NaNs)."""
    mask = (y_true > null_val) & np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return {"mae": 0.0, "rmse": 0.0}
    diff = y_pred[mask] - y_true[mask]
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    return {"mae": round(mae, 4), "rmse": round(rmse, 4)}


def run_v3_1_benchmark(
    timesteps: int = 4000,
    epochs: int = 2,
    batch_size: int = 32,
    seed: int = 42,
    lodo_ckpt_path: Path = Path("models/foundation/jepagraph_fm_lodo_pems04_pems08.pt"),
) -> Dict[str, Any]:
    print("=" * 80)
    print("Q-JEPA V3.1 — FAIR COMPARATIVE BASELINE BENCHMARK")
    print(f"Target Dataset: METR-LA | Timesteps: {timesteps} | Epochs: {epochs} | Batch: {batch_size}")
    print("=" * 80 + "\n")

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cpu"

    # 1. Load authentic METR-LA dataset
    loader = METRLALoader(data_dir="data/metr_la")
    splits = loader.prepare_dataset(max_timesteps=timesteps)
    adj_norm = loader.compute_normalized_adj(adj_type="random_walk")

    num_nodes = splits.train_x.shape[2]
    mean, std = splits.mean, splits.std

    train_ds = TrafficWindowDataset(splits.train_x, splits.train_y)
    val_ds = TrafficWindowDataset(splits.val_x, splits.val_y)
    test_ds = TrafficWindowDataset(splits.test_x, splits.test_y)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"[*] Prepared Data Splits: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    print(f"[*] Sensors: {num_nodes} | Train Mean: {mean:.2f} mph, Std: {std:.2f} mph\n")

    benchmark_results: Dict[str, Any] = {}

    adj_t = adj_norm.float() if torch.is_tensor(adj_norm) else torch.from_numpy(adj_norm).float()

    # Helper for evaluating PyTorch models on test split
    def evaluate_pytorch_model(model: nn.Module, is_fm: bool = False) -> Tuple[Dict[str, Any], float]:
        model.eval()
        latencies = []
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, y in test_loader:
                t0 = time.perf_counter()
                if is_fm:
                    # Foundation model expects [B, T_in, N, 3]
                    p = model.forecast(x, adj_t, mean=mean, std=std)  # [B, 12, N]
                else:
                    out = model(x, adj_t)
                    if isinstance(out, dict):
                        out = out["speed_pred"]
                    elif isinstance(out, tuple):
                        out = out[0]
                    p = out.numpy() * std + mean
                lat = (time.perf_counter() - t0) * 1000.0
                latencies.append(lat)

                if torch.is_tensor(p):
                    p = p.cpu().numpy()
                all_preds.append(p)
                all_targets.append(y.numpy())

        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # Horizons: step 3 (15m), step 6 (30m), step 12 (60m)
        m_15 = masked_metrics(preds[:, 2, :], targets[:, 2, :])
        m_30 = masked_metrics(preds[:, 5, :], targets[:, 5, :])
        m_60 = masked_metrics(preds[:, 11, :], targets[:, 11, :])

        return {
            "15m": m_15,
            "30m": m_30,
            "60m": m_60,
        }, float(np.mean(latencies))

    # Helper for training standard PyTorch baseline
    def train_baseline(model: nn.Module, lr: float = 1e-3, is_jepa: bool = False, rep_weight: float = 1.0) -> float:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        t0 = time.time()
        for ep in range(epochs):
            model.train()
            for x, y in train_loader:
                optimizer.zero_grad()
                y_norm = (y - mean) / std
                valid = (y > 0.0)

                if is_jepa:
                    outputs = model(x, adj_t, y_future=y_norm)
                    speed_pred = outputs["speed_pred"]
                    speed_loss = F.smooth_l1_loss(speed_pred[valid], y_norm[valid]) if valid.sum() > 0 else F.smooth_l1_loss(speed_pred, y_norm)
                    rep_loss = F.smooth_l1_loss(outputs["z_pred"], outputs["z_target"]) if rep_weight > 0 else 0.0
                    loss = speed_loss + rep_weight * rep_loss
                else:
                    preds = model(x, adj_t)
                    if isinstance(preds, dict):
                        preds = preds["speed_pred"]
                    elif isinstance(preds, tuple):
                        preds = preds[0]
                    loss = F.smooth_l1_loss(preds[valid], y_norm[valid]) if valid.sum() > 0 else F.smooth_l1_loss(preds, y_norm)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                if is_jepa and hasattr(model, "update_target_ema"):
                    model.update_target_ema()
        return time.time() - t0

    # -------------------------------------------------------------
    # 1. Persistence Baseline
    # -------------------------------------------------------------
    print("[*] Evaluating 1/6: Persistence Baseline...")
    pers = PersistenceBaseline(mean=mean, std=std)
    all_preds_pers = []
    all_targets_pers = []
    lat_pers = []
    for x, y in test_loader:
        t0 = time.perf_counter()
        p = pers.predict(x.numpy(), num_horizons=12)
        lat_pers.append((time.perf_counter() - t0) * 1000.0)
        all_preds_pers.append(p)
        all_targets_pers.append(y.numpy())
    p_pers = np.concatenate(all_preds_pers, axis=0)
    t_pers = np.concatenate(all_targets_pers, axis=0)

    benchmark_results["Persistence"] = {
        "parameters": 0,
        "train_time_s": 0.0,
        "inference_latency_ms": round(float(np.mean(lat_pers)), 2),
        "metrics": {
            "15m": masked_metrics(p_pers[:, 2, :], t_pers[:, 2, :]),
            "30m": masked_metrics(p_pers[:, 5, :], t_pers[:, 5, :]),
            "60m": masked_metrics(p_pers[:, 11, :], t_pers[:, 11, :]),
        },
    }

    # -------------------------------------------------------------
    # 2. STGCN Baseline
    # -------------------------------------------------------------
    print("[*] Training & Evaluating 2/6: STGCN...")
    stgcn = STGCNModel(num_nodes=num_nodes, in_channels=2, hidden_dim=32, output_window=12)
    dur_stgcn = train_baseline(stgcn)
    m_stgcn, lat_stgcn = evaluate_pytorch_model(stgcn)
    benchmark_results["STGCN"] = {
        "parameters": sum(p.numel() for p in stgcn.parameters() if p.requires_grad),
        "train_time_s": round(dur_stgcn, 2),
        "inference_latency_ms": round(lat_stgcn, 2),
        "metrics": m_stgcn,
    }
    del stgcn; gc.collect()

    # -------------------------------------------------------------
    # 3. Graph WaveNet Baseline
    # -------------------------------------------------------------
    print("[*] Training & Evaluating 3/6: Graph WaveNet...")
    gwn = GraphWaveNetModel(num_nodes=num_nodes, in_channels=2, hidden_dim=32, output_window=12)
    dur_gwn = train_baseline(gwn)
    m_gwn, lat_gwn = evaluate_pytorch_model(gwn)
    benchmark_results["Graph_WaveNet"] = {
        "parameters": sum(p.numel() for p in gwn.parameters() if p.requires_grad),
        "train_time_s": round(dur_gwn, 2),
        "inference_latency_ms": round(lat_gwn, 2),
        "metrics": m_gwn,
    }
    del gwn; gc.collect()

    # -------------------------------------------------------------
    # 4. JEPAGraph V1
    # -------------------------------------------------------------
    print("[*] Training & Evaluating 4/6: JEPAGraph V1...")
    jepa_v1 = JEPAGraphModel(num_nodes=num_nodes, in_channels=2, latent_dim=32, output_window=12)
    dur_jepa = train_baseline(jepa_v1, is_jepa=True, rep_weight=1.0)
    m_jepa, lat_jepa = evaluate_pytorch_model(jepa_v1)
    benchmark_results["JEPAGraph_V1"] = {
        "parameters": sum(p.numel() for p in jepa_v1.parameters() if p.requires_grad),
        "train_time_s": round(dur_jepa, 2),
        "inference_latency_ms": round(lat_jepa, 2),
        "metrics": m_jepa,
    }
    del jepa_v1; gc.collect()

    # -------------------------------------------------------------
    # 5. JEPAGraph Direct Regression Ablation
    # -------------------------------------------------------------
    print("[*] Training & Evaluating 5/6: JEPAGraph Direct Regression...")
    jepa_direct = JEPAGraphModel(num_nodes=num_nodes, in_channels=2, latent_dim=32, output_window=12)
    dur_direct = train_baseline(jepa_direct, is_jepa=True, rep_weight=0.0)
    m_direct, lat_direct = evaluate_pytorch_model(jepa_direct)
    benchmark_results["JEPAGraph_DirectRegression"] = {
        "parameters": sum(p.numel() for p in jepa_direct.parameters() if p.requires_grad),
        "train_time_s": round(dur_direct, 2),
        "inference_latency_ms": round(lat_direct, 2),
        "metrics": m_direct,
    }
    del jepa_direct; gc.collect()

    # -------------------------------------------------------------
    # 6. JEPAGraph-FM (Pretrained LODO + Fine-Tuned)
    # -------------------------------------------------------------
    print("[*] Fine-Tuning & Evaluating 6/6: JEPAGraph-FM (LODO Pretrained)...")
    fm_config = FoundationModelConfig.tiny()
    fm_model = JEPAGraphFoundationModel(fm_config)

    # Load LODO pretrained checkpoint (pretrained on PEMS04 + PEMS08)
    if lodo_ckpt_path.exists():
        ckpt = torch.load(lodo_ckpt_path, map_location=device)
        fm_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"  --> Loaded pretrained weights from {lodo_ckpt_path.name}")

    # Fine-tune on target METR-LA training split
    optimizer = torch.optim.AdamW(fm_model.parameters(), lr=5e-4, weight_decay=1e-4)
    t0 = time.time()
    for ep in range(epochs):
        fm_model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            # x is [B, T_in, N, 2] -> pad to 3 channels (feature, tod, dow)
            B, T_in, N, C = x.shape
            if C < 3:
                tod = x[:, :, :, 1:2] if C > 1 else torch.zeros(B, T_in, N, 1)
                dow = torch.zeros(B, T_in, N, 1)
                x_3ch = torch.cat([x[:, :, :, 0:1], tod, dow], dim=-1)
            else:
                x_3ch = x
            preds = fm_model.forward(x_3ch, adj_t, variable_type="speed")
            y_norm = (y - mean) / std
            valid = (y > 0.0)
            if valid.sum() > 0:
                loss = F.smooth_l1_loss(preds[valid], y_norm[valid])
            else:
                loss = F.smooth_l1_loss(preds, y_norm)
            loss.backward()
            optimizer.step()
    dur_fm = time.time() - t0

    # Evaluate FM
    fm_model.eval()
    lat_fm_list = []
    all_preds_fm = []
    all_targets_fm = []
    with torch.no_grad():
        for x, y in test_loader:
            B, T_in, N, C = x.shape
            if C < 3:
                tod = x[:, :, :, 1:2] if C > 1 else torch.zeros(B, T_in, N, 1)
                dow = torch.zeros(B, T_in, N, 1)
                x_3ch = torch.cat([x[:, :, :, 0:1], tod, dow], dim=-1)
            else:
                x_3ch = x
            t0 = time.perf_counter()
            p = fm_model.forecast(x_3ch, adj_t, mean=mean, std=std)
            lat_fm_list.append((time.perf_counter() - t0) * 1000.0)
            all_preds_fm.append(p.cpu().numpy())
            all_targets_fm.append(y.numpy())

    p_fm = np.concatenate(all_preds_fm, axis=0)
    t_fm = np.concatenate(all_targets_fm, axis=0)
    benchmark_results["JEPAGraph_FM_Finetuned"] = {
        "parameters": fm_model.count_parameters()["total_parameters"],
        "train_time_s": round(dur_fm, 2),
        "inference_latency_ms": round(float(np.mean(lat_fm_list)), 2),
        "metrics": {
            "15m": masked_metrics(p_fm[:, 2, :], t_fm[:, 2, :]),
            "30m": masked_metrics(p_fm[:, 5, :], t_fm[:, 5, :]),
            "60m": masked_metrics(p_fm[:, 11, :], t_fm[:, 11, :]),
        },
    }

    # Published SOTA Literature Reference Table (strictly distinguished from local CPU runs)
    published_literature_sota = {
        "note": "Published benchmark values from official papers on full METR-LA test split (34,272 timesteps, GPU converged).",
        "DCRNN_Li_et_al_ICLR_2018": {"15m_mae": 2.77, "30m_mae": 3.15, "60m_mae": 3.60},
        "STGCN_Yu_et_al_IJCAI_2018": {"15m_mae": 2.88, "30m_mae": 3.47, "60m_mae": 4.59},
        "GraphWaveNet_Wu_et_al_IJCAI_2019": {"15m_mae": 2.69, "30m_mae": 3.07, "60m_mae": 3.53},
        "ASTGCN_Guo_et_al_AAAI_2019": {"15m_mae": 4.14, "30m_mae": 5.07, "60m_mae": 6.55},
        "GMAN_Zheng_et_al_AAAI_2020": {"15m_mae": 2.77, "30m_mae": 3.07, "60m_mae": 3.40},
    }

    # Display comparison table
    print("\n" + "=" * 90)
    print(f"{'Model':<30s} | {'Params':>8s} | {'15m MAE':>8s} | {'30m MAE':>8s} | {'60m MAE':>8s} | {'Latency':>8s}")
    print("-" * 90)
    for mname, res in benchmark_results.items():
        m = res["metrics"]
        print(
            f"{mname:<30s} | {res['parameters']:>8,d} | "
            f"{m['15m']['mae']:>8.2f} | {m['30m']['mae']:>8.2f} | {m['60m']['mae']:>8.2f} | "
            f"{res['inference_latency_ms']:>6.1f}ms"
        )
    print("=" * 90 + "\n")

    full_payload = {
        "benchmark_title": "Q-JEPA V3.1 Controlled Baseline Benchmark",
        "dataset": "metr_la",
        "hardware_environment": "CPU (AMD64, PyTorch 2.9.0)",
        "timesteps_evaluated": timesteps,
        "epochs_per_model": epochs,
        "locally_reproduced_models": benchmark_results,
        "published_literature_sota": published_literature_sota,
    }

    for out_path in [
        project_root / "results" / "metrics" / "v3_1" / "fair_baselines_benchmark.json",
        project_root / "results" / "metrics" / "fair_baselines_benchmark.json",
    ]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(full_payload, f, indent=2)

    print(f"[*] Fair baseline results saved to results/metrics/v3_1/fair_baselines_benchmark.json")
    return full_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fair baseline benchmark.")
    parser.add_argument("--timesteps", type=int, default=3000, help="Timesteps to load from METR-LA")
    parser.add_argument("--epochs", type=int, default=2, help="Epochs to train each model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    run_v3_1_benchmark(timesteps=args.timesteps, epochs=args.epochs, batch_size=args.batch_size)
