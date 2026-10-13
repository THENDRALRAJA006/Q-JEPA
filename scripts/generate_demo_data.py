"""
Synthetic data generation script for Surakshit Bharat Q-JEPA.
Runs the traffic simulator, collects observations, and saves to CSV/Parquet.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import numpy as np
import pandas as pd

from src.simulation.road_network import RoadNetwork
from src.simulation.traffic_simulator import TrafficSimulator, ScenarioType
from src.common.config import config, abs_path
from src.common.reproducibility import set_seed
from src.common.logging import get_logger

logger = get_logger(__name__)


def generate_traffic_dataset(
    time_steps: int = 288,
    seed: int = 42,
    scenario: ScenarioType = ScenarioType.NORMAL,
    rows: int = 5,
    cols: int = 6,
) -> pd.DataFrame:
    """
    Generate synthetic traffic time series.
    Returns a DataFrame with one row per (step, edge) observation.
    """
    set_seed(seed)
    network = RoadNetwork.create_synthetic_grid(rows=rows, cols=cols, seed=seed)
    sim = TrafficSimulator(
        network=network,
        scenario=scenario,
        seed=seed,
        missing_obs_rate=0.05,
        noise_std_kmh=2.0,
    )

    records = []
    logger.info("Generating %d time steps, scenario=%s ...", time_steps, scenario.value)
    for _ in range(time_steps):
        state = sim.step_forward()
        hour = (state.time_minutes / 60.0) % 24.0
        for obs in state.observations:
            d = network.graph.get_edge_data(obs.u, obs.v, default={})
            records.append({
                "step": state.step,
                "time_minutes": state.time_minutes,
                "hour_of_day": round(hour, 3),
                "day_of_week": 0,  # synthetic — all same day
                "edge_id": obs.edge_id,
                "u": obs.u,
                "v": obs.v,
                "length_m": d.get("length_m", 0),
                "baseline_speed_kmh": d.get("baseline_speed_kmh", 0),
                "gt_speed_kmh": d.get("gt_speed_kmh", 0),
                "observed_speed_kmh": obs.observed_speed_kmh,
                "flow_veh_per_hour": obs.flow_veh_per_hour,
                "congestion_level": obs.congestion_level,
                "is_missing": obs.is_missing,
                "is_noisy": obs.is_noisy,
                "has_incident": d.get("has_incident", False),
                "is_closed": d.get("is_closed", False),
            })

    df = pd.DataFrame(records)
    logger.info("Generated dataset: %d rows, %d unique edges", len(df), df["edge_id"].nunique())
    return df


def save_dataset(df: pd.DataFrame, out_dir: Path, name: str = "synthetic_traffic") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    parquet_path = out_dir / f"{name}.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    logger.info("Saved: %s, %s", csv_path, parquet_path)

    # Save metadata
    meta = {
        "source": "synthetic",
        "generator": "TrafficSimulator (Surakshit Bharat Q-JEPA)",
        "rows": len(df),
        "unique_edges": int(df["edge_id"].nunique()),
        "time_steps": int(df["step"].max()),
        "time_step_minutes": 5,
        "features": list(df.columns),
        "note": "All data is synthetically generated. NOT real traffic data.",
    }
    with open(out_dir / f"{name}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Metadata saved.")


def main():
    cfg = config()
    out_dir = abs_path(cfg.data.synthetic_dir)

    logger.info("=== Surakshit Bharat Q-JEPA: Demo Data Generator ===")
    logger.info("Output directory: %s", out_dir)

    # Normal scenario (main dataset)
    df_normal = generate_traffic_dataset(
        time_steps=cfg.simulation.time_steps,
        seed=cfg.simulation.seed,
        scenario=ScenarioType.NORMAL,
    )
    save_dataset(df_normal, out_dir, "synthetic_traffic_normal")

    # Rush-hour scenario
    df_rush = generate_traffic_dataset(
        time_steps=96,
        seed=cfg.simulation.seed + 1,
        scenario=ScenarioType.RUSH_HOUR,
    )
    save_dataset(df_rush, out_dir, "synthetic_traffic_rush_hour")

    # Road blockage scenario
    df_block = generate_traffic_dataset(
        time_steps=96,
        seed=cfg.simulation.seed + 2,
        scenario=ScenarioType.ROAD_BLOCKAGE,
    )
    save_dataset(df_block, out_dir, "synthetic_traffic_road_blockage")

    logger.info("=== Data generation complete ===")
    logger.info("Files in %s:", out_dir)
    for f in sorted(out_dir.iterdir()):
        logger.info("  %s (%.1f KB)", f.name, f.stat().st_size / 1024)


if __name__ == "__main__":
    main()
