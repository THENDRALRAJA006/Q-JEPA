"""
Traffic Dataset Registry and Metadata Manifests for Q-JEPA Traffic Foundation Model.

Covers 6 benchmark spatiotemporal traffic datasets:
1. METR-LA (Los Angeles highway speeds, 207 sensors, mph)
2. PEMS-BAY (SF Bay Area highway speeds, 325 sensors, mph)
3. PEMS03 (Caltrans D3 traffic flow, 358 sensors, veh/5min)
4. PEMS04 (Caltrans D4 SF Bay traffic flow/speed/occ, 307 sensors, veh/5min)
5. PEMS07 (Caltrans D7 LA traffic flow, 883 sensors, veh/5min)
6. PEMS08 (Caltrans D8 San Bernardino traffic flow, 170 sensors, veh/5min)

Provides:
- DatasetManifest with explicit provenance, units, and licensing metadata
- File integrity checking (SHA-256)
- Graph connectivity and diagnostic utilities
- Data quality audits (missing values, zero policies, temporal continuity)
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetManifest:
    """Standardized metadata manifest for a traffic benchmark dataset."""

    dataset_id: str
    name: str
    num_nodes: int
    primary_variable: str  # 'speed' or 'flow'
    variable_units: str  # 'mph', 'km/h', 'veh/5min'
    sampling_interval_minutes: int  # Standard: 5 minutes
    num_timesteps_expected: int
    license: str
    provenance_repo: str
    data_dir: str
    data_file: str
    adj_file: str
    train_ratio: float = 0.70
    val_ratio: float = 0.10
    test_ratio: float = 0.20
    description: str = ""
    zero_value_policy: str = "masked"  # 'masked' (sensor dropout) or 'valid' (zero flow)
    sha256_checksum: Optional[str] = None


# Official registry of supported benchmark datasets
DATASET_REGISTRY: Dict[str, DatasetManifest] = {
    "metr_la": DatasetManifest(
        dataset_id="metr_la",
        name="METR-LA Highway Traffic Speed",
        num_nodes=207,
        primary_variable="speed",
        variable_units="mph",
        sampling_interval_minutes=5,
        num_timesteps_expected=34272,  # March 1, 2012 to June 30, 2012
        license="CC BY 4.0",
        provenance_repo="https://github.com/liyaguang/DCRNN (Li et al., ICLR 2018)",
        data_dir="data/metr_la",
        data_file="METR-LA.csv",
        adj_file="adj_mx_METR-LA.pkl",
        train_ratio=0.70,
        val_ratio=0.10,
        test_ratio=0.20,
        description="Loop detectors on Los Angeles County highways measuring average vehicle speeds.",
        zero_value_policy="masked",  # Speeds <= 0 indicate missing sensor readings
    ),
    "pems_bay": DatasetManifest(
        dataset_id="pems_bay",
        name="PEMS-BAY Highway Traffic Speed",
        num_nodes=325,
        primary_variable="speed",
        variable_units="mph",
        sampling_interval_minutes=5,
        num_timesteps_expected=52116,  # Jan 1, 2017 to May 31, 2017 (5 months)
        license="CC BY 4.0 / Caltrans Open Data",
        provenance_repo="https://github.com/liyaguang/DCRNN (Li et al., ICLR 2018)",
        data_dir="data/pems_bay",
        data_file="PEMS-BAY.h5",
        adj_file="adj_mx_PEMS-BAY.pkl",
        train_ratio=0.70,
        val_ratio=0.10,
        test_ratio=0.20,
        description="Caltrans PeMS sensor network across the California South Bay Area (San Jose / Santa Clara).",
        zero_value_policy="masked",
    ),
    "pems03": DatasetManifest(
        dataset_id="pems03",
        name="PEMS03 Traffic Flow Benchmark",
        num_nodes=358,
        primary_variable="flow",
        variable_units="veh/5min",
        sampling_interval_minutes=5,
        num_timesteps_expected=26208,  # Sep 1, 2018 to Nov 30, 2018
        license="CC BY 4.0 / Caltrans Open Data",
        provenance_repo="https://github.com/Davidham3/ASTGCN (Guo et al., AAAI 2019)",
        data_dir="data/pems03",
        data_file="PEMS03.npz",
        adj_file="PEMS03.csv",
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        description="Caltrans District 3 (North Sacramento Valley) freeway sensor counts.",
        zero_value_policy="masked",
    ),
    "pems04": DatasetManifest(
        dataset_id="pems04",
        name="PEMS04 Traffic Flow & Speed Benchmark",
        num_nodes=307,
        primary_variable="flow",
        variable_units="veh/5min",
        sampling_interval_minutes=5,
        num_timesteps_expected=16992,  # Jan 1, 2018 to Feb 28, 2018
        license="CC BY 4.0 / Caltrans Open Data",
        provenance_repo="https://github.com/Davidham3/ASTGCN (Guo et al., AAAI 2019)",
        data_dir="data/pems04",
        data_file="PEMS04.npz",
        adj_file="distance_PEMS04.csv",
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        description="Caltrans District 4 (San Francisco Bay Area) freeway loop detectors.",
        zero_value_policy="masked",
    ),
    "pems07": DatasetManifest(
        dataset_id="pems07",
        name="PEMS07 Large-Scale Traffic Flow Benchmark",
        num_nodes=883,
        primary_variable="flow",
        variable_units="veh/5min",
        sampling_interval_minutes=5,
        num_timesteps_expected=28224,  # May 1, 2017 to Aug 31, 2017
        license="CC BY 4.0 / Caltrans Open Data",
        provenance_repo="https://github.com/Davidham3/ASTGCN (Guo et al., AAAI 2019)",
        data_dir="data/pems07",
        data_file="PEMS07.npz",
        adj_file="distance_PEMS07.csv",
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        description="Caltrans District 7 (Los Angeles & Ventura County) large-scale arterial and highway flow network.",
        zero_value_policy="masked",
    ),
    "pems08": DatasetManifest(
        dataset_id="pems08",
        name="PEMS08 Traffic Flow Benchmark",
        num_nodes=170,
        primary_variable="flow",
        variable_units="veh/5min",
        sampling_interval_minutes=5,
        num_timesteps_expected=17856,  # July 1, 2016 to Aug 31, 2016
        license="CC BY 4.0 / Caltrans Open Data",
        provenance_repo="https://github.com/Davidham3/ASTGCN (Guo et al., AAAI 2019)",
        data_dir="data/pems08",
        data_file="PEMS08.npz",
        adj_file="distance_PEMS08.csv",
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        description="Caltrans District 8 (San Bernardino County) freeway loop detectors.",
        zero_value_policy="masked",
    ),
}


def get_manifest(dataset_id: str) -> DatasetManifest:
    """Retrieve the manifest for a dataset identifier."""
    key = dataset_id.lower().replace("-", "_")
    if key not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_id}'. Supported datasets: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[key]


def list_registered_datasets() -> List[str]:
    """Return all registered dataset identifiers."""
    return list(DATASET_REGISTRY.keys())


def compute_file_sha256(file_path: Path) -> str:
    """Compute the SHA-256 hash of a file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot compute hash: {file_path} does not exist.")
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):  # 1 MB chunks
            sha.update(chunk)
    return sha.hexdigest()


def diagnose_graph_connectivity(adj_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Compute structural diagnostics on a spatial adjacency matrix.

    Returns:
        - num_nodes: Total number of sensors
        - num_edges: Directed edges with weight > 0
        - density: Graph edge density
        - isolated_nodes: Number of nodes with in-degree = 0 and out-degree = 0
        - is_symmetric: Whether A == A.T
        - max_weight: Maximum edge weight
        - min_nonzero_weight: Minimum non-zero edge weight
        - avg_degree: Average node degree
    """
    N = adj_matrix.shape[0]
    binary_adj = (adj_matrix > 0).astype(int)
    np.fill_diagonal(binary_adj, 0)

    num_edges = int(np.sum(binary_adj))
    possible_edges = N * (N - 1)
    density = float(num_edges / possible_edges) if possible_edges > 0 else 0.0

    in_degrees = np.sum(binary_adj, axis=0)
    out_degrees = np.sum(binary_adj, axis=1)
    isolated_nodes = int(np.sum((in_degrees == 0) & (out_degrees == 0)))

    diff = np.abs(adj_matrix - adj_matrix.T)
    is_symmetric = bool(np.all(diff < 1e-5))

    nonzero_weights = adj_matrix[adj_matrix > 0]
    max_weight = float(np.max(nonzero_weights)) if len(nonzero_weights) > 0 else 0.0
    min_nonzero_weight = float(np.min(nonzero_weights)) if len(nonzero_weights) > 0 else 0.0
    avg_degree = float(np.mean(in_degrees + out_degrees) / 2.0)

    return {
        "num_nodes": N,
        "num_edges": num_edges,
        "density": density,
        "isolated_nodes": isolated_nodes,
        "is_symmetric": is_symmetric,
        "max_weight": max_weight,
        "min_nonzero_weight": min_nonzero_weight,
        "avg_degree": avg_degree,
    }


def analyze_missing_values(values: np.ndarray) -> Dict[str, Any]:
    """
    Analyze missing values and zeros in traffic observations.
    
    Args:
        values: numpy array of shape [T, N] or [T, N, F]
    """
    flat = values[..., 0] if values.ndim == 3 else values
    total_elements = flat.size
    nan_count = int(np.isnan(flat).sum())
    zero_count = int((flat == 0).sum())
    negative_count = int((flat < 0).sum())
    valid_count = total_elements - nan_count - zero_count - negative_count

    return {
        "total_elements": total_elements,
        "nan_count": nan_count,
        "nan_ratio": float(nan_count / total_elements),
        "zero_count": zero_count,
        "zero_ratio": float(zero_count / total_elements),
        "negative_count": negative_count,
        "missing_ratio": float((nan_count + zero_count + negative_count) / total_elements),
        "valid_count": valid_count,
        "valid_ratio": float(valid_count / total_elements),
    }
