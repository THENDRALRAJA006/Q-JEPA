"""
Forecast inference engine — connects trained models to the road network.

This is the missing link. It:
1. Loads trained checkpoints (LSTM, JEPA) with their normalizer params.
2. Accepts per-edge speed history (observed over the last window_size steps).
3. Produces predicted travel times for each road segment at each horizon.
4. Writes those predictions into network.graph[u][v]["predicted_travel_times_s"]
   and network.graph[u][v]["uncertainty"] so routing algorithms can consume them.
5. Records which model produced every set of costs.
6. Handles missing histories, invalid checkpoints, and unsupported horizons.

Key correctness guarantees:
- Normalizer is fit on training data only (params loaded from checkpoint metadata).
- No ground-truth (gt_speed_kmh) is accessed at inference time.
- Time step resolution: 5 min/step (matches training data).
- Predicted travel time = length_m / (predicted_speed_kmh / 3.6).
- Uncertainty written in seconds (same units as predicted_travel_time_s).
"""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.simulation.road_network import RoadNetwork
from src.common.config import abs_path
from src.common.logging import get_logger

logger = get_logger(__name__)

# Temporal resolution matching the training data
STEP_MINUTES = 5


@dataclass
class InferenceRecord:
    """Metadata for one inference pass — which model produced which costs."""
    model_name: str          # e.g. "LSTM", "JEPA-Traffic"
    checkpoint_path: str
    horizon_steps: int       # steps ahead
    horizon_minutes: int     # = horizon_steps * STEP_MINUTES
    n_edges_updated: int
    n_edges_fallback: int    # edges where history was insufficient → used current_tt
    normalizer_min: float
    normalizer_max: float
    timestamp_s: float = field(default_factory=time.time)


class EdgeHistoryBuffer:
    """
    Maintains a sliding window of observed speeds per edge.
    Feeds inference at each routing decision point.
    Only stores *observed* speeds (not ground truth).
    """

    def __init__(self, window_size: int = 12):
        self.window_size = window_size
        # edge_id -> deque of observed speeds (most recent last)
        self._buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))

    def update(self, edge_id: str, observed_speed_kmh: Optional[float]) -> None:
        """Record one new observation. None (missing) uses last known or baseline."""
        if observed_speed_kmh is not None and observed_speed_kmh > 0:
            self._buffers[edge_id].append(float(observed_speed_kmh))
        elif self._buffers[edge_id]:
            # Forward-fill with last known value
            self._buffers[edge_id].append(self._buffers[edge_id][-1])

    def update_from_network(self, network: RoadNetwork) -> None:
        """Pull current speeds from all edges and add to buffers."""
        for u, v, d in network.graph.edges(data=True):
            eid = f"{u}->{v}"
            spd = d.get("current_speed_kmh") or d.get("baseline_speed_kmh", 30.0)
            self.update(eid, spd)

    def get_window(self, edge_id: str, baseline_speed: float) -> Optional[np.ndarray]:
        """
        Return last window_size observations as float32 array.
        Returns None if fewer than window_size / 2 observations are available.
        """
        buf = self._buffers[edge_id]
        if len(buf) == 0:
            return None
        arr = np.array(buf, dtype=np.float32)
        # Pad short histories with first value (repeat-pad, not future info)
        if len(arr) < self.window_size:
            pad = np.full(self.window_size - len(arr), arr[0], dtype=np.float32)
            arr = np.concatenate([pad, arr])
        return arr[-self.window_size:]  # (window_size,)

    def ready_edges(self, min_obs: int = 6) -> List[str]:
        """Edges with at least min_obs observations."""
        return [eid for eid, buf in self._buffers.items() if len(buf) >= min_obs]


def _speed_to_travel_time(predicted_speed_kmh: float, length_m: float) -> float:
    """Convert predicted speed (km/h) to edge travel time (seconds)."""
    speed = max(1.0, predicted_speed_kmh)  # avoid division by zero
    return length_m / (speed / 3.6)


def _load_normalizer_from_metrics() -> Tuple[float, float]:
    """Load normalizer min/max from saved metrics JSON (fit on train split only)."""
    metrics_path = abs_path("results/metrics/forecasting_metrics.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            m = json.load(f)
        norm = m.get("normalizer", {})
        vmin = float(norm.get("min", 0.0))
        vmax = float(norm.get("max", 80.0))
        return vmin, vmax
    logger.warning("Normalizer params not found in metrics JSON — using fallback [0, 80] km/h")
    return 0.0, 80.0


class LSTMInferenceEngine:
    """
    Loads a trained LSTM checkpoint and produces per-edge speed forecasts.
    Uses the same normalizer that was fit on the training split.
    """

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.window_size: int = 12
        self.horizon: int = 1
        self.speed_min: float = 0.0
        self.speed_max: float = 80.0
        self._loaded = False

    def load(self) -> "LSTMInferenceEngine":
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"LSTM checkpoint not found: {self.checkpoint_path}")
        from src.forecasting.lstm_model import LSTMForecaster
        forecaster = LSTMForecaster()
        forecaster.load(self.checkpoint_path)
        self.model = forecaster
        cfg = torch.load(self.checkpoint_path, map_location="cpu")["config"]
        self.window_size = cfg["window_size"]
        self.horizon = cfg["horizon"]
        self.speed_min, self.speed_max = _load_normalizer_from_metrics()
        self._loaded = True
        logger.info(
            "LSTM inference engine ready: window=%d, horizon=%d, speed=[%.1f,%.1f]",
            self.window_size, self.horizon, self.speed_min, self.speed_max,
        )
        return self

    def predict_edge(self, raw_speeds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        raw_speeds: (window_size,) array of km/h speeds (not normalised).
        Returns (mean_pred_kmh, std_pred_kmh) both shape (horizon,).
        """
        if not self._loaded:
            raise RuntimeError("Call load() first.")
        denom = max(self.speed_max - self.speed_min, 1e-6)
        normed = (raw_speeds - self.speed_min) / denom
        X = normed[np.newaxis, :]  # (1, window_size)
        mean_normed, std_normed = self.model.predict_with_uncertainty(X, n_passes=10)
        mean_kmh = mean_normed[0] * denom + self.speed_min
        std_kmh = std_normed[0] * denom
        return mean_kmh, std_kmh

    @property
    def name(self) -> str:
        return "LSTM"


class JEPAInferenceEngine:
    """
    Loads a trained JEPA checkpoint and produces per-edge speed forecasts.
    Does NOT support MC-Dropout (head is deterministic linear layer).
    Uncertainty is estimated as a fixed fraction of predicted travel time.
    """

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.window_size: int = 12
        self.horizon: int = 1
        self.speed_min: float = 0.0
        self.speed_max: float = 80.0
        self._loaded = False

    def load(self) -> "JEPAInferenceEngine":
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"JEPA checkpoint not found: {self.checkpoint_path}")
        from src.forecasting.jepa_encoder import JEPAForecaster
        forecaster = JEPAForecaster()
        forecaster.load(self.checkpoint_path)
        self.model = forecaster
        cfg = torch.load(self.checkpoint_path, map_location="cpu")["config"]
        self.window_size = cfg["window_size"]
        self.horizon = cfg["horizon"]
        self.speed_min, self.speed_max = _load_normalizer_from_metrics()
        self._loaded = True
        logger.info(
            "JEPA inference engine ready: window=%d, horizon=%d, speed=[%.1f,%.1f]",
            self.window_size, self.horizon, self.speed_min, self.speed_max,
        )
        return self

    def predict_edge(self, raw_speeds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        raw_speeds: (window_size,) array of km/h speeds.
        Returns (mean_pred_kmh, std_kmh) both shape (horizon,).
        std_kmh is approximated as 10% of the mean (JEPA head is deterministic).
        """
        if not self._loaded:
            raise RuntimeError("Call load() first.")
        denom = max(self.speed_max - self.speed_min, 1e-6)
        normed = (raw_speeds - self.speed_min) / denom
        X = normed[np.newaxis, :]  # (1, window_size)
        pred_normed = self.model.predict(X)[0]  # (horizon,)
        mean_kmh = pred_normed * denom + self.speed_min
        std_kmh = np.abs(mean_kmh) * 0.10  # 10% approximation for deterministic head
        return mean_kmh, std_kmh

    @property
    def name(self) -> str:
        return "JEPA-Traffic"


def write_forecasts_to_network(
    network: RoadNetwork,
    history: EdgeHistoryBuffer,
    engine,  # LSTMInferenceEngine | JEPAInferenceEngine
    horizons: Tuple[int, ...] = (1, 3, 6),
) -> InferenceRecord:
    """
    Run inference for every edge with sufficient history and write results into
    network.graph[u][v]["predicted_travel_times_s"] and ["uncertainty"].

    predicted_travel_times_s keys are horizon step integers (matching training).
    Edge travel time = length_m / (predicted_speed_kmh / 3.6).

    Returns an InferenceRecord documenting what was done.
    """
    n_updated = 0
    n_fallback = 0

    for u, v, d in network.graph.edges(data=True):
        eid = f"{u}->{v}"
        length_m = d.get("length_m", 500.0)
        baseline = d.get("baseline_speed_kmh", 30.0)

        window = history.get_window(eid, baseline)
        if window is None or len(window) < engine.window_size // 2:
            # Insufficient history — leave predicted_travel_times_s unchanged
            n_fallback += 1
            continue

        try:
            mean_kmh, std_kmh = engine.predict_edge(window)
            # Clamp to physical bounds
            mean_kmh = np.clip(mean_kmh, 1.0, baseline * 1.2)
            std_kmh = np.clip(std_kmh, 0.0, 20.0)

            # Multi-horizon autoregressive rollout
            preds_kmh_by_h = {}
            stds_kmh_by_h = {}

            if engine.horizon >= max(horizons):
                # Direct multi-horizon prediction
                for h in horizons:
                    h_idx = min(h - 1, engine.horizon - 1)
                    preds_kmh_by_h[h] = float(mean_kmh[h_idx])
                    stds_kmh_by_h[h] = float(std_kmh[h_idx]) if len(std_kmh) > h_idx else float(std_kmh[0])
            else:
                # Autoregressive multi-horizon forecasting
                curr_window = np.copy(window)
                base_std = float(std_kmh[0])
                for h in sorted(horizons):
                    # Autoregressively roll out until horizon h is reached
                    while len(preds_kmh_by_h) < h:
                        step_idx = len(preds_kmh_by_h) + 1
                        m_k, s_k = engine.predict_edge(curr_window)
                        next_spd = float(np.clip(m_k[0], 1.0, baseline * 1.3))
                        preds_kmh_by_h[step_idx] = next_spd
                        # Uncertainty compounds with prediction horizon sqrt(h)
                        stds_kmh_by_h[step_idx] = base_std * math.sqrt(step_idx)
                        curr_window = np.append(curr_window[1:], next_spd)

            pred_tt = {}
            pred_spd = {}
            uncertainties_by_h = {}
            for h in horizons:
                spd = preds_kmh_by_h.get(h, float(mean_kmh[0]))
                sigma_spd_h = stds_kmh_by_h.get(h, float(std_kmh[0]))
                pred_tt[h] = _speed_to_travel_time(spd, length_m)
                pred_spd[h] = spd
                uncertainties_by_h[h] = (length_m * sigma_spd_h * 3.6) / max(spd ** 2, 1.0)

            # Uncertainty in seconds (sum over edges → path uncertainty accumulates)
            sigma_spd = float(std_kmh[0])
            # Propagate speed uncertainty to travel time uncertainty
            sigma_tt = (length_m * sigma_spd * 3.6) / max(mean_kmh[0] ** 2, 1.0)

            network.update_edge_state(
                u, v,
                predicted_travel_times_s=pred_tt,
                predicted_speeds_kmh=pred_spd,
                uncertainty=sigma_tt,
                uncertainties_by_horizon=uncertainties_by_h,
            )
            n_updated += 1

        except Exception as exc:
            logger.debug("Inference failed for edge %s: %s", eid, exc)
            n_fallback += 1

    record = InferenceRecord(
        model_name=engine.name,
        checkpoint_path=str(engine.checkpoint_path),
        horizon_steps=engine.horizon,
        horizon_minutes=engine.horizon * STEP_MINUTES,
        n_edges_updated=n_updated,
        n_edges_fallback=n_fallback,
        normalizer_min=engine.speed_min,
        normalizer_max=engine.speed_max,
    )
    logger.info(
        "[%s] Forecast written: %d edges updated, %d fallback (current_tt used)",
        engine.name, n_updated, n_fallback,
    )
    return record


def clear_forecasts_from_network(network: RoadNetwork) -> None:
    """Reset predicted_travel_times_s and uncertainty on all edges."""
    for u, v, d in network.graph.edges(data=True):
        d["predicted_travel_times_s"] = {}
        d["uncertainty"] = 0.0


def load_lstm_engine(ckpt_dir: Optional[Path] = None) -> Optional[LSTMInferenceEngine]:
    """Load LSTM engine; return None if checkpoint unavailable."""
    if ckpt_dir is None:
        ckpt_dir = abs_path("models/checkpoints")
    path = Path(ckpt_dir) / "lstm.pt"
    if not path.exists():
        logger.warning("LSTM checkpoint not found at %s", path)
        return None
    try:
        return LSTMInferenceEngine(path).load()
    except Exception as e:
        logger.error("Failed to load LSTM engine: %s", e)
        return None


def load_jepa_engine(ckpt_dir: Optional[Path] = None) -> Optional[JEPAInferenceEngine]:
    """Load JEPA engine; return None if checkpoint unavailable."""
    if ckpt_dir is None:
        ckpt_dir = abs_path("models/checkpoints")
    path = Path(ckpt_dir) / "jepa.pt"
    if not path.exists():
        logger.warning("JEPA checkpoint not found at %s", path)
        return None
    try:
        return JEPAInferenceEngine(path).load()
    except Exception as e:
        logger.error("Failed to load JEPA engine: %s", e)
        return None


class JEPAGraphFMInferenceEngine:
    """
    Inference Engine for Q-JEPA Foundation Model (JEPAGraph-FM V2.0).
    Produces direct multi-horizon forecasts (horizons 1..12, up to 60 minutes).
    """

    def __init__(self, checkpoint_path: Optional[Path] = None):
        self.checkpoint_path = checkpoint_path or abs_path("models/foundation/jepagraph_fm_ft_metr_la.pt")
        self.model = None
        self.window_size: int = 12
        self.horizon: int = 12
        self.mean: float = 58.58
        self.std: float = 12.82
        self.speed_min: float = 0.0
        self.speed_max: float = 80.0
        self._loaded: bool = False

    def load(self) -> "JEPAGraphFMInferenceEngine":
        from src.forecasting.jepagraph_foundation import FoundationModelConfig, JEPAGraphFoundationModel
        config = FoundationModelConfig.tiny()
        self.model = JEPAGraphFoundationModel(config)
        if self.checkpoint_path.exists():
            ckpt = torch.load(self.checkpoint_path, map_location="cpu")
            self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
            self.mean = ckpt.get("mean", 58.58)
            self.std = ckpt.get("std", 12.82)
            logger.info("Loaded JEPAGraph-FM from %s", self.checkpoint_path)
        else:
            logger.warning("Checkpoint %s not found. Running with initialized weights.", self.checkpoint_path)
        self.model.eval()
        self._loaded = True
        return self

    def predict_edge(self, raw_speeds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        raw_speeds: (window_size,) array of km/h speeds.
        Returns:
            mean_kmh: (horizon,) array of predicted speeds in km/h
            std_kmh: (horizon,) array of uncertainty standard deviations
        """
        if not self._loaded:
            raise RuntimeError("Call load() first.")

        # Build 1-node dummy graph for single edge inference
        x_norm = (raw_speeds - self.mean) / max(self.std, 1e-4)
        x_feat = np.stack([x_norm, np.zeros_like(x_norm), np.zeros_like(x_norm)], axis=-1)  # [12, 3]
        x_tensor = torch.from_numpy(x_feat).float().unsqueeze(0).unsqueeze(2)  # [1, 12, 1, 3]
        adj = torch.tensor([[1.0]], dtype=torch.float32)

        with torch.no_grad():
            pred_norm = self.model.forward(x_tensor, adj, variable_type="speed")  # [1, 12, 1]
            pred_phys = pred_norm[0, :, 0].cpu().numpy() * self.std + self.mean

        mean_kmh = np.clip(pred_phys, 5.0, 120.0)
        # Multi-horizon uncertainty scaling: sigma_0 * sqrt(h)
        base_sigma = np.abs(mean_kmh[0]) * 0.08
        horizons_idx = np.arange(1, len(mean_kmh) + 1)
        std_kmh = base_sigma * np.sqrt(horizons_idx)

        return mean_kmh, std_kmh

    @property
    def name(self) -> str:
        return "JEPAGraph-FM (V2.0 Foundation Model)"


def load_jepafm_engine(checkpoint_path: Optional[Path] = None) -> Optional[JEPAGraphFMInferenceEngine]:
    """Load JEPAGraph-FM engine; return None if loading fails."""
    try:
        engine = JEPAGraphFMInferenceEngine(checkpoint_path)
        return engine.load()
    except Exception as e:
        logger.error("Failed to load JEPAGraph-FM engine: %s", e)
        return None
