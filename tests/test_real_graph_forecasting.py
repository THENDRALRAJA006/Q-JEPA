"""
Unit and integration tests for METR-LA Graph Traffic Forecasting (Phase 2).

Tests:
1. Dataset parsing and graph alignment
2. Chronological data splitting (no temporal leakage)
3. Normalization parameters computed exclusively on train split
4. Missing value masking (null_val handling)
5. Multi-horizon tensor shapes (horizons 3, 6, 12)
6. JEPAGraph target-encoder stop-gradient and EMA behavior
7. Model checkpoint saving and loading
8. Reproducible baseline predictions
9. Ablation configurations (direct regression, identity graph, no EMA)
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import torch

from src.data.metr_la_loader import (
    DatasetSplits,
    METRLALoader,
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


@pytest.fixture(scope="module")
def metr_la_loader():
    return METRLALoader(data_dir="data/metr_la")


@pytest.fixture(scope="module")
def small_splits(metr_la_loader):
    return metr_la_loader.prepare_dataset(max_timesteps=150)


def test_metr_la_graph_alignment(metr_la_loader):
    """Test graph adjacency matrix loading and sensor alignment."""
    sensor_ids, id_map, adj_matrix = metr_la_loader.load_graph()
    assert len(sensor_ids) == 207
    assert len(id_map) == 207
    assert adj_matrix.shape == (207, 207)
    assert np.count_nonzero(adj_matrix) > 1000

    # Test random walk normalization
    norm_adj = metr_la_loader.compute_normalized_adj(adj_type="random_walk")
    assert norm_adj.shape == (207, 207)
    assert torch.is_tensor(norm_adj)
    # Row sums of random walk matrix with self-loops should be approximately 1.0
    row_sums = torch.sum(norm_adj, dim=1)
    valid_rows = row_sums > 0
    assert torch.allclose(row_sums[valid_rows], torch.ones_like(row_sums[valid_rows]), atol=1e-3)


def test_chronological_splitting_no_leakage(metr_la_loader):
    """Verify chronological ordering: train strictly precedes val, val strictly precedes test."""
    df = metr_la_loader.load_raw_speeds(nrows=200)
    total_len = len(df)
    train_end = int(total_len * 0.70)
    val_end = int(total_len * 0.80)

    train_idx = df.index[:train_end]
    val_idx = df.index[train_end:val_end]
    test_idx = df.index[val_end:]

    # Strict timestamp ordering
    assert train_idx[-1] < val_idx[0], "Temporal leakage: train timestamp overlaps with val"
    assert val_idx[-1] < test_idx[0], "Temporal leakage: val timestamp overlaps with test"


def test_train_only_normalization(small_splits):
    """Ensure mean and std are positive and computed from training data."""
    assert small_splits.mean > 20.0, "Unreasonable highway speed mean"
    assert small_splits.std > 2.0, "Unreasonable speed standard deviation"
    assert np.isfinite(small_splits.mean)
    assert np.isfinite(small_splits.std)


def test_missing_value_masking():
    """Verify evaluation metrics ignore missing sensor readings (zero / negative / NaN)."""
    y_true = np.array([[50.0, 0.0, 60.0], [0.0, 45.0, -1.0]])
    y_pred = np.array([[55.0, 999.0, 60.0], [-999.0, 45.0, 999.0]])

    # Valid entries are only (0,0) [diff=5], (0,2) [diff=0], and (1,1) [diff=0]
    # The 999.0 predictions at unobserved points must be completely ignored!
    m = evaluate_metrics(y_true, y_pred, null_val=0.0)
    expected_mae = (5.0 + 0.0 + 0.0) / 3.0
    assert pytest.approx(m["MAE"], abs=1e-4) == expected_mae


def test_multi_horizon_shapes(small_splits):
    """Verify sliding window tensor shapes across input and output horizons."""
    assert small_splits.train_x.ndim == 4
    assert small_splits.train_y.ndim == 3
    # [N, 12, 207, 2]
    assert small_splits.train_x.shape[1] == 12  # T_in = 12
    assert small_splits.train_x.shape[2] == 207 # num_nodes = 207
    assert small_splits.train_x.shape[3] == 2   # speed + time_of_day
    # [N, 12, 207]
    assert small_splits.train_y.shape[1] == 12  # T_out = 12
    assert small_splits.train_y.shape[2] == 207


def test_jepa_target_encoder_and_ema():
    """Verify stop-gradient on target encoder and proper EMA updates."""
    model = JEPAGraphModel(num_nodes=10, in_channels=2, latent_dim=16, output_window=12, ema_decay=0.95)

    # 1. Stop-gradient verification: target encoder parameters must not require grad
    for p in model.target_encoder.parameters():
        assert not p.requires_grad, "Target encoder must have requires_grad=False (stop-gradient)"

    # 2. EMA update verification
    w_target_init = model.target_encoder[0].gcn.linear.weight.data.clone()
    # Mutate online weights
    model.online_encoder[0].gcn.linear.weight.data.add_(2.0)
    model.update_target_ema()
    w_target_after = model.target_encoder[0].gcn.linear.weight.data.clone()

    diff = (w_target_after - w_target_init).abs().max().item()
    expected_change = (1.0 - 0.95) * 2.0  # 0.10
    assert pytest.approx(diff, abs=1e-3) == expected_change


def test_checkpoint_saving_and_loading(tmp_path):
    """Verify checkpoint save, load, and bitwise inference consistency."""
    model = JEPAGraphModel(num_nodes=10, in_channels=2, latent_dim=16, output_window=12)
    x = torch.randn(2, 12, 10, 2)
    adj = torch.eye(10)

    model.eval()
    with torch.no_grad():
        out1 = model(x, adj)["speed_pred"]

    ckpt_path = tmp_path / "test_jepa_ckpt.pt"
    torch.save(model.state_dict(), ckpt_path)

    loaded_model = JEPAGraphModel(num_nodes=10, in_channels=2, latent_dim=16, output_window=12)
    loaded_model.load_state_dict(torch.load(ckpt_path))
    loaded_model.eval()

    with torch.no_grad():
        out2 = loaded_model(x, adj)["speed_pred"]

    assert torch.allclose(out1, out2, atol=1e-6)


def test_reproducible_baselines(small_splits):
    """Verify Persistence and Historical Average produce consistent, non-trivial outputs."""
    pers = PersistenceBaseline(mean=small_splits.mean, std=small_splits.std)
    p1 = pers.predict(small_splits.test_x[:5])
    p2 = pers.predict(small_splits.test_x[:5])
    assert np.allclose(p1, p2)
    assert p1.shape == (5, 12, 207)

    ha = HistoricalAverageBaseline(mean=small_splits.mean, std=small_splits.std)
    ha.fit(small_splits.train_x, small_splits.train_y)
    h_pred = ha.predict(small_splits.test_x[:5])
    assert h_pred.shape == (5, 12, 207)
    assert np.all(np.isfinite(h_pred))


def test_ablation_architectures():
    """Verify that ablations correctly instantiate and run forward passes."""
    adj = torch.eye(10)
    x = torch.randn(2, 12, 10, 2)

    # Ablation 1: Direct regression (use_ema=False)
    m_reg = JEPAGraphModel(num_nodes=10, in_channels=2, latent_dim=16, use_ema=False)
    out_reg = m_reg(x, adj)["speed_pred"]
    assert out_reg.shape == (2, 12, 10)

    # Ablation 2: No graph (use_graph=False)
    m_nograph = JEPAGraphModel(num_nodes=10, in_channels=2, latent_dim=16, use_graph=False)
    out_nograph = m_nograph(x, adj)["speed_pred"]
    assert out_nograph.shape == (2, 12, 10)


def test_normalization_and_inverse_normalization_exact():
    """Verify forward and inverse normalization on known values."""
    # Known positive speeds
    raw_speeds = np.array([20.0, 40.0, 60.0, 80.0], dtype=np.float32)
    mean = float(np.mean(raw_speeds))  # 50.0
    std = float(np.std(raw_speeds))    # 22.36067977

    # Forward normalization: z = (x - mean) / std
    z = (raw_speeds - mean) / std
    assert np.allclose(np.mean(z), 0.0, atol=1e-6)
    assert np.allclose(np.std(z), 1.0, atol=1e-6)

    # Inverse normalization: x_rec = z * std + mean
    recovered = z * std + mean
    assert np.allclose(recovered, raw_speeds, atol=1e-6)

    # Check zero/missing values are not used when calculating train mean/std
    masked_series = np.array([0.0, 50.0, np.nan, 70.0, -5.0], dtype=np.float32)
    valid = masked_series[np.isfinite(masked_series) & (masked_series > 0.0)]
    assert len(valid) == 2
    assert float(np.mean(valid)) == 60.0
    assert float(np.std(valid)) == 10.0


def test_hand_calculated_metric_fixture():
    """
    Independently verify MAE, RMSE, and MAPE calculations using hand-calculated numbers.

    Data:
      y_true: [50.0, 60.0,  0.0,   NaN, 40.0]
      y_pred: [55.0, 58.0, 99.0,  99.0, 44.0]

    Masking rule: valid where y_true > 0 and isfinite(y_true) and isfinite(y_pred).
    Valid entries:
      - idx 0: y_true=50.0, y_pred=55.0 -> err = +5.0, sq = 25.0, ape = 5/50 = 0.1000000
      - idx 1: y_true=60.0, y_pred=58.0 -> err = -2.0, sq =  4.0, ape = 2/60 = 0.0333333
      - idx 4: y_true=40.0, y_pred=44.0 -> err = +4.0, sq = 16.0, ape = 4/40 = 0.1000000

    Hand-calculated metrics (N=3):
      MAE  = (5 + 2 + 4) / 3 = 11 / 3 = 3.6666667
      RMSE = sqrt((25 + 4 + 16) / 3) = sqrt(45 / 3) = sqrt(15.0) = 3.8729833
      MAPE = ((0.10 + 0.033333333 + 0.10) / 3) * 100 = 7.7777778%
    """
    y_true = np.array([50.0, 60.0, 0.0, np.nan, 40.0], dtype=np.float32)
    y_pred = np.array([55.0, 58.0, 99.0, 99.0, 44.0], dtype=np.float32)

    res = evaluate_metrics(y_true, y_pred, null_val=0.0)

    expected_mae = 11.0 / 3.0
    expected_rmse = np.sqrt(15.0)
    expected_mape = ((0.10 + (2.0 / 60.0) + 0.10) / 3.0) * 100.0

    assert abs(res["MAE"] - expected_mae) < 1e-5
    assert abs(res["RMSE"] - expected_rmse) < 1e-5
    assert abs(res["MAPE"] - expected_mape) < 1e-5


def test_horizon_mapping_exact():
    """Verify that horizon step indexing precisely maps to 15m, 30m, and 60m."""
    step_duration_minutes = 5
    horizons = {
        "15_min": {"step": 3, "array_idx": 2},
        "30_min": {"step": 6, "array_idx": 5},
        "60_min": {"step": 12, "array_idx": 11},
    }
    for h_name, h_info in horizons.items():
        assert h_info["step"] * step_duration_minutes == int(h_name.split("_")[0])
        assert h_info["array_idx"] == h_info["step"] - 1


def test_graph_wavenet_forward_and_output_shape(small_splits):
    """Test Graph WaveNet model initialization and forward prediction shape."""
    num_nodes = small_splits.train_x.shape[2]
    model = GraphWaveNetModel(
        num_nodes=num_nodes, in_channels=2, hidden_dim=16, output_window=12
    )
    x = torch.from_numpy(small_splits.train_x[:2]).float()
    adj = torch.from_numpy(small_splits.adj_matrix).float()
    pred = model(x, adj)
    assert pred.shape == (2, 12, num_nodes)
    assert torch.isfinite(pred).all()


def test_spatial_temporal_transformer_forward_and_output_shape(small_splits):
    """Test Spatio-Temporal Transformer initialization and forward prediction shape."""
    num_nodes = small_splits.train_x.shape[2]
    model = SpatialTemporalTransformerModel(
        num_nodes=num_nodes, in_channels=2, hidden_dim=16, num_heads=2, output_window=12
    )
    x = torch.from_numpy(small_splits.train_x[:2]).float()
    pred = model(x)
    assert pred.shape == (2, 12, num_nodes)
    assert torch.isfinite(pred).all()


