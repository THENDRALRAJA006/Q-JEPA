"""
Downstream Fine-Tuning and Evaluation Script for Q-JEPA Foundation Model (JEPAGraph-FM).

Features:
- Loads multi-dataset pretrained foundation checkpoint
- Dataset-specific fine-tuning on downstream target traffic networks (e.g., METR-LA, PEMS08)
- Optional backbone freezing (evaluating representation probe vs end-to-end fine-tuning)
- Low-data efficiency support (subsampling target training set: 1%, 5%, 10%, 25%, 50%, 100%)
- Chronological train / val / test evaluation
- True physical unit unnormalization and multi-horizon metric logging (15m, 30m, 60m)
- Serializes evaluation metrics to JSON
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.common.logging import get_logger
from src.data.dataset_registry import get_manifest
from src.data.multi_dataset import SingleDatasetWindowDataset, get_adapter
from src.forecasting.jepagraph_foundation import (
    FoundationModelConfig,
    JEPAGraphFoundationModel,
)

logger = get_logger("finetune_jepagraph")


def compute_metrics(
    pred_phys: np.ndarray,
    target_phys: np.ndarray,
    horizons: Dict[str, int] = {"15m": 2, "30m": 5, "60m": 11},
    null_val: float = 0.0,
) -> Dict[str, Dict[str, float]]:
    """
    Compute masked MAE, RMSE, MAPE across evaluation horizons.
    horizons mapping: name -> 0-indexed step in T_out (e.g., step 2 is 15-min forecast)
    """
    results = {}
    for h_name, step_idx in horizons.items():
        if step_idx >= pred_phys.shape[1]:
            continue

        p_h = pred_phys[:, step_idx, :]
        t_h = target_phys[:, step_idx, :]

        # Valid mask: targets > null_val and not NaN
        mask = (t_h > null_val) & (~np.isnan(t_h))
        if mask.sum() == 0:
            continue

        p_valid = p_h[mask]
        t_valid = t_h[mask]

        mae = float(np.mean(np.abs(p_valid - t_valid)))
        rmse = float(np.sqrt(np.mean(np.square(p_valid - t_valid))))
        mape = float(np.mean(np.abs((p_valid - t_valid) / t_valid)) * 100.0)

        results[h_name] = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mape": round(mape, 2),
            "valid_samples": int(mask.sum()),
        }
    return results


def finetune_and_evaluate(
    target_dataset: str = "metr_la",
    pretrained_ckpt: Optional[Path] = None,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 5e-4,
    subsample_ratio: float = 1.0,
    max_samples: Optional[int] = 256,
    freeze_backbone: bool = False,
    device_name: str = "cpu",
    output_dir: Path = Path("models/foundation"),
) -> Dict[str, Any]:
    manifest = get_manifest(target_dataset)
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 80)
    print(f"FINE-TUNING JEPAGraph-FM ON {manifest.name.upper()} ({target_dataset})")
    print(f"Device: {device} | Subsample Ratio: {subsample_ratio*100:.1f}% | Epochs: {epochs}")
    print(f"Pretrained Checkpoint: {pretrained_ckpt}")
    print("=" * 80 + "\n")

    adapter = get_adapter(target_dataset)
    train_ds = SingleDatasetWindowDataset(
        adapter=adapter,
        split="train",
        subsample_ratio=subsample_ratio,
        max_samples=max_samples,
    )
    val_ds = SingleDatasetWindowDataset(
        adapter=adapter,
        split="val",
        subsample_ratio=subsample_ratio,
        max_samples=max_samples // 2 if max_samples else None,
    )
    test_ds = SingleDatasetWindowDataset(
        adapter=adapter,
        split="test",
        subsample_ratio=1.0,  # Always evaluate on full test set
        max_samples=512,  # Budgeted for CPU memory stability
    )

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    adj_tensor = train_ds.adj_tensor.to(device)

    # Initialize model
    config = FoundationModelConfig.tiny()
    model = JEPAGraphFoundationModel(config).to(device)

    # Load pretrained weights if provided
    is_pretrained = False
    if pretrained_ckpt and Path(pretrained_ckpt).exists():
        logger.info(f"Loading pretrained weights from {pretrained_ckpt}")
        ckpt_data = torch.load(pretrained_ckpt, map_location=device)
        model.load_state_dict(ckpt_data["model_state_dict"], strict=False)
        is_pretrained = True
    else:
        logger.info("Training from scratch (no pretrained weights loaded).")

    # Backbone freezing option
    if freeze_backbone:
        for p in model.online_encoder.parameters():
            p.requires_grad = False
        logger.info("Online backbone encoder parameters FROZEN.")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-4)

    best_val_loss = float("inf")
    output_dir.mkdir(parents=True, exist_ok=True)
    ft_ckpt_path = output_dir / f"jepagraph_fm_ft_{target_dataset}.pt"

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        train_losses = []
        for x, y, m_x, m_y in train_loader:
            x, y, m_y = x.to(device), y.to(device), m_y.to(device)
            optimizer.zero_grad()
            preds = model.forward(x, adj_tensor, variable_type=manifest.primary_variable)

            # Masked loss on valid future observations
            valid = m_y.bool()
            loss = F.smooth_l1_loss(preds[valid], y[valid]) if valid.sum() > 0 else torch.tensor(0.0, device=device)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 5.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validation loop
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y, m_x, m_y in val_loader:
                x, y, m_y = x.to(device), y.to(device), m_y.to(device)
                preds = model.forward(x, adj_tensor, variable_type=manifest.primary_variable)
                valid = m_y.bool()
                v_loss = F.smooth_l1_loss(preds[valid], y[valid]) if valid.sum() > 0 else torch.tensor(0.0, device=device)
                val_losses.append(v_loss.item())

        mean_train = float(np.mean(train_losses)) if train_losses else 0.0
        mean_val = float(np.mean(val_losses)) if val_losses else 0.0

        print(f"Fine-tune Epoch {ep:02d}/{epochs:02d} | Train Loss: {mean_train:.4f} | Val Loss: {mean_val:.4f}")
        if mean_val < best_val_loss:
            best_val_loss = mean_val
            torch.save(
                {
                    "epoch": ep,
                    "target_dataset": target_dataset,
                    "model_state_dict": model.state_dict(),
                    "mean": adapter.mean,
                    "std": adapter.std,
                    "val_loss": best_val_loss,
                },
                ft_ckpt_path,
            )

    train_duration = time.time() - t0

    # Test Evaluation
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y, m_x, m_y in test_loader:
            x = x.to(device)
            # Physical prediction
            preds_phys = model.forecast(
                x, adj_tensor, variable_type=manifest.primary_variable, mean=adapter.mean, std=adapter.std
            )
            targets_phys = y.numpy() * adapter.std + adapter.mean
            all_preds.append(preds_phys.cpu().numpy())
            all_targets.append(targets_phys)

    pred_arr = np.concatenate(all_preds, axis=0)
    target_arr = np.concatenate(all_targets, axis=0)

    # Compute metrics at 15m, 30m, 60m
    metrics = compute_metrics(pred_arr, target_arr)
    print("\n" + "-" * 60)
    print(f"DOWNSTREAM FORECAST EVALUATION RESULTS: {target_dataset.upper()}")
    for h, m in metrics.items():
        print(f"  Horizon {h:4s}: MAE = {m['mae']:.2f} {manifest.variable_units} | RMSE = {m['rmse']:.2f} | MAPE = {m['mape']:.1f}%")
    print("-" * 60 + "\n")

    result_payload = {
        "target_dataset": target_dataset,
        "is_pretrained": is_pretrained,
        "freeze_backbone": freeze_backbone,
        "subsample_ratio": subsample_ratio,
        "train_samples": len(train_ds),
        "test_samples": len(test_ds),
        "train_duration_seconds": round(train_duration, 2),
        "variable_type": manifest.primary_variable,
        "variable_units": manifest.variable_units,
        "best_val_loss": round(best_val_loss, 4),
        "metrics_by_horizon": metrics,
    }

    out_file = project_root / "results" / "metrics" / f"foundation_finetuning_{target_dataset}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(result_payload, f, indent=2)

    return result_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune JEPAGraph Foundation Model.")
    parser.add_argument("--dataset", type=str, default="metr_la", help="Downstream target dataset")
    parser.add_argument("--ckpt", type=str, default="models/foundation/jepagraph_fm_tiny.pt", help="Pretrained checkpoint")
    parser.add_argument("--epochs", type=int, default=3, help="Fine-tuning epochs")
    parser.add_argument("--subsample", type=float, default=1.0, help="Training data subsampling ratio")
    parser.add_argument("--freeze", action="store_true", help="Freeze online backbone")
    args = parser.parse_args()

    finetune_and_evaluate(
        target_dataset=args.dataset,
        pretrained_ckpt=Path(args.ckpt) if args.ckpt and Path(args.ckpt).exists() else None,
        epochs=args.epochs,
        subsample_ratio=args.subsample,
        freeze_backbone=args.freeze,
    )
