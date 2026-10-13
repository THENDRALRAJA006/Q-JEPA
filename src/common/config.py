"""
Centralized configuration management for Surakshit Bharat Q-JEPA.
Loads from YAML files and environment variables with Pydantic validation.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict, Any

import yaml
from pydantic import BaseModel, Field

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


class DataConfig(BaseModel):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    synthetic_dir: str = "data/synthetic"
    default_seed: int = 42
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


class SimulationConfig(BaseModel):
    num_nodes: int = 30
    num_edges: int = 60
    time_steps: int = 288          # 24 h at 5-min intervals
    time_step_minutes: int = 5
    seed: int = 42
    max_speed_kmh: float = 60.0
    min_speed_kmh: float = 5.0
    incident_probability: float = 0.02
    closure_probability: float = 0.005


class ForecastingConfig(BaseModel):
    horizons: list[int] = [1, 3, 6]   # steps -> 5, 15, 30 min
    lstm_hidden: int = 64
    lstm_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_epochs: int = 30
    lstm_lr: float = 1e-3
    lstm_batch: int = 32
    window_size: int = 12              # look-back steps
    jepa_embed_dim: int = 64
    jepa_pred_dim: int = 64
    jepa_ema_tau: float = 0.99
    jepa_epochs: int = 30
    jepa_lr: float = 1e-3
    jepa_batch: int = 32


class RoutingConfig(BaseModel):
    risk_penalty: float = 0.3         # for uncertainty-aware routing
    reroute_threshold_pct: float = 10.0
    reroute_min_interval_s: int = 60
    sa_temperature: float = 100.0
    sa_cooling: float = 0.98
    sa_iterations: int = 500


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    log_level: str = "info"


class AppConfig(BaseModel):
    project_name: str = "Surakshit Bharat Q-JEPA Traffic"
    version: str = "0.1.0"
    data: DataConfig = Field(default_factory=DataConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    forecasting: ForecastingConfig = Field(default_factory=ForecastingConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    models_dir: str = "models"
    results_dir: str = "results"
    checkpoints_dir: str = "models/checkpoints"


def load_yaml(path: Path) -> Dict[str, Any]:
    if path.exists():
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def get_config(config_path: Optional[str] = None) -> AppConfig:
    """Load configuration, merging YAML and environment overrides."""
    base = load_yaml(PROJECT_ROOT / "configs" / "default.yaml")
    if config_path:
        extra = load_yaml(Path(config_path))
        base.update(extra)
    return AppConfig(**base)


# Singleton
_config: Optional[AppConfig] = None


def config() -> AppConfig:
    global _config
    if _config is None:
        _config = get_config()
    return _config


def abs_path(relative: str) -> Path:
    """Convert a config relative path to an absolute path."""
    return PROJECT_ROOT / relative
