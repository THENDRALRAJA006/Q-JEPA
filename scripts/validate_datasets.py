"""
Dataset Validation and Integrity Audit for Q-JEPA Foundation Model.

Checks:
- Dataset manifest completeness and provenance.
- File existence or verified profile availability.
- Tensor dimensions (T, N) and sensor counts.
- Graph connectivity (density, isolated nodes, symmetry).
- Missing-value and zero-value statistics.
- Train-only normalization mean and std.
- Sliding window tensor generation test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.common.logging import get_logger
from src.data.dataset_registry import (
    DATASET_REGISTRY,
    analyze_missing_values,
    diagnose_graph_connectivity,
    list_registered_datasets,
)
from src.data.multi_dataset import get_adapter

logger = get_logger("validate_datasets")


def validate_all_datasets() -> Dict[str, Any]:
    print("=" * 80)
    print("SURAKSHIT BHARAT Q-JEPA V3.1 — DATASET PROVENANCE & REAL-DATA AUDIT")
    print("=" * 80)

    results = {}
    datasets = list_registered_datasets()

    for did in datasets:
        manifest = DATASET_REGISTRY[did]
        print(f"\n[*] Validating Dataset: {manifest.name} ({did})")
        print(f"    - Variable: {manifest.primary_variable} ({manifest.variable_units})")
        print(f"    - Expected Nodes: {manifest.num_nodes}, Expected Timesteps: {manifest.num_timesteps_expected}")
        print(f"    - Provenance: {manifest.provenance_repo}")
        print(f"    - License: {manifest.license}")

        # Check raw files on disk
        data_dir = project_root / manifest.data_dir
        primary_file = data_dir / manifest.data_file
        adj_file = data_dir / manifest.adj_file

        file_exists = primary_file.exists()
        file_size_bytes = primary_file.stat().st_size if file_exists else 0
        sha256_hash = None
        if file_exists and file_size_bytes > 0:
            import hashlib
            sha = hashlib.sha256()
            with open(primary_file, "rb") as f:
                while chunk := f.read(1024 * 1024):
                    sha.update(chunk)
            sha256_hash = sha.hexdigest()

        adapter = get_adapter(did, input_window=12, output_window=12, use_cache=False)
        adapter.prepare()

        raw = adapter.raw_data
        adj = adapter.adj_matrix
        T, N = raw.shape
        is_real = adapter.is_real_data

        # Verify dimensions
        node_match = N == manifest.num_nodes
        status = "PASSED" if node_match and is_real else ("SYNTHETIC_FALLBACK" if not is_real else "DIM_MISMATCH")
        print(f"    - Provenance Status: {'REAL PHYSICAL DATA ON DISK' if is_real else 'SYNTHETIC TEST FALLBACK (NOT FOR PRETRAINING)'}")
        print(f"    - Verified Dimensions: T={T}, N={N} [Match: {node_match}]")
        print(f"    - File Size: {file_size_bytes / (1024*1024):.2f} MB | SHA-256: {sha256_hash[:16] + '...' if sha256_hash else 'N/A'}")
        print(f"    - Train Normalization: Mean={adapter.mean:.2f}, Std={adapter.std:.2f}")

        # Graph Diagnostics
        graph_diag = diagnose_graph_connectivity(adj)
        print(
            f"    - Graph Diagnostics: Edges={graph_diag['num_edges']}, "
            f"Density={graph_diag['density']:.4f}, Isolated Nodes={graph_diag['isolated_nodes']}"
        )

        # Missingness Audit
        missing_diag = analyze_missing_values(raw)
        print(
            f"    - Data Quality: Missing Ratio={missing_diag['missing_ratio']*100:.2f}%, "
            f"Zero Ratio={missing_diag['zero_ratio']*100:.2f}%, NaN Ratio={missing_diag['nan_ratio']*100:.2f}%"
        )

        # Sliding window audit
        train_end = int(T * adapter.train_ratio)
        window_size = adapter.input_window + adapter.output_window
        total_train_windows = train_end - window_size + 1
        x_w, y_w, m_x, m_y = adapter.create_sliding_windows("val", max_samples=100)
        print(f"    - Total Valid Train Windows: {total_train_windows:,}")
        print(f"    - Validation Sliding Windows (Sample=100): X={x_w.shape}, Y={y_w.shape}")

        results[did] = {
            "status": status,
            "is_real_physical_data": is_real,
            "eligible_for_pretraining": is_real,
            "file_on_disk": str(primary_file) if file_exists else None,
            "file_size_bytes": file_size_bytes,
            "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
            "sha256": sha256_hash,
            "manifest": {
                "name": manifest.name,
                "primary_variable": manifest.primary_variable,
                "variable_units": manifest.variable_units,
                "sampling_interval_minutes": manifest.sampling_interval_minutes,
                "license": manifest.license,
                "provenance_repo": manifest.provenance_repo,
            },
            "shape": {"T": T, "N": N},
            "total_valid_train_windows": total_train_windows,
            "graph_diagnostics": graph_diag,
            "missingness": missing_diag,
            "train_statistics": {"mean": float(adapter.mean), "std": float(adapter.std)},
            "val_windows_tested": int(len(x_w)),
        }

    # Write to both V3.1 results and default metrics directory
    for out_path in [
        project_root / "results" / "metrics" / "v3_1" / "dataset_validation_audit.json",
        project_root / "results" / "metrics" / "dataset_validation_audit.json",
    ]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Dataset audit successfully completed. Reports written to results/metrics/v3_1/dataset_validation_audit.json")
    print("=" * 80)
    return results


if __name__ == "__main__":
    validate_all_datasets()
