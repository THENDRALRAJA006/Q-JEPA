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
    print("SURAKSHIT BHARAT Q-JEPA V2.0 — DATASET AUDIT & INTEGRITY CHECK")
    print("=" * 80)

    results = {}
    datasets = list_registered_datasets()

    for did in datasets:
        manifest = DATASET_REGISTRY[did]
        print(f"\n[*] Validating Dataset: {manifest.name} ({did})")
        print(f"    - Variable: {manifest.primary_variable} ({manifest.variable_units})")
        print(f"    - Expected Nodes: {manifest.num_nodes}, Timesteps: {manifest.num_timesteps_expected}")
        print(f"    - Provenance: {manifest.provenance_repo}")
        print(f"    - License: {manifest.license}")

        adapter = get_adapter(did, input_window=12, output_window=12)
        adapter.prepare()

        raw = adapter.raw_data
        adj = adapter.adj_matrix
        T, N = raw.shape

        # Verify dimensions
        node_match = N == manifest.num_nodes
        status = "PASSED" if node_match else "FAILED"
        print(f"    - Verified Dimensions: T={T}, N={N} [Match: {node_match}]")
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

        # Quick Window generation test
        x_w, y_w, m_x, m_y = adapter.create_sliding_windows("val", max_samples=100)
        print(f"    - Validation Sliding Windows (Sample=100): X={x_w.shape}, Y={y_w.shape}, MaskX={m_x.shape}")

        results[did] = {
            "status": status,
            "manifest": {
                "name": manifest.name,
                "primary_variable": manifest.primary_variable,
                "variable_units": manifest.variable_units,
                "license": manifest.license,
            },
            "shape": {"T": T, "N": N},
            "graph_diagnostics": graph_diag,
            "missingness": missing_diag,
            "train_statistics": {"mean": adapter.mean, "std": adapter.std},
            "val_windows": int(len(x_w)),
        }

    out_file = project_root / "results" / "metrics" / "dataset_validation_audit.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Dataset audit successfully completed. Report written to {out_file}")
    print("=" * 80)
    return results


if __name__ == "__main__":
    validate_all_datasets()
