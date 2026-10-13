"""
Comprehensive Unit and Integration Tests for Q-JEPA Traffic Foundation Model V2.0.

Verifies:
1. Heterogeneous graph support (N=207, N=325, N=170)
2. Target encoder gradients disabled and EMA updates verified
3. Multi-objective loss suite (JEPA latent matching, masked reconstruction, supervised forecast)
4. Variable-specific heads (Speed vs Flow) and physical unit unnormalization
5. Multi-dataset mixture sampling and batch generation
6. Checkpoint save and restoration integrity
7. Train-only normalization (no leakage of val/test sets)
8. Multi-horizon forecasting shapes [B, T_out, N]
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import torch

from src.data.dataset_registry import (
    DATASET_REGISTRY,
    analyze_missing_values,
    diagnose_graph_connectivity,
    get_manifest,
    list_registered_datasets,
)
from src.data.multi_dataset import (
    MultiDatasetMixtureLoader,
    SingleDatasetWindowDataset,
    TrafficBatch,
    get_adapter,
)
from src.forecasting.graph_temporal_encoder import (
    GatedDilatedTemporalConv,
    GraphDiffusionConvolution,
    SpatiotemporalBlock,
)
from src.forecasting.jepagraph_foundation import (
    FoundationModelConfig,
    JEPAGraphFoundationModel,
)
from src.forecasting.representation_objectives import (
    LossWeightsConfig,
    RepresentationObjectives,
)


class TestDatasetRegistryAndAdapters:
    """Verifies dataset manifest integrity and multi-dataset adapters."""

    def test_registered_datasets_exist(self):
        datasets = list_registered_datasets()
        expected = ["metr_la", "pems_bay", "pems03", "pems04", "pems07", "pems08"]
        for exp in expected:
            assert exp in datasets

    def test_manifest_metadata_integrity(self):
        for did in list_registered_datasets():
            manifest = get_manifest(did)
            assert manifest.num_nodes > 0
            assert manifest.primary_variable in ["speed", "flow"]
            assert manifest.sampling_interval_minutes == 5
            assert manifest.license != ""
            assert manifest.train_ratio + manifest.val_ratio + manifest.test_ratio == pytest.approx(1.0)

    def test_adapter_heterogeneous_nodes(self):
        adapter_la = get_adapter("metr_la")
        adapter_bay = get_adapter("pems_bay")
        adapter_08 = get_adapter("pems08")

        adapter_la.prepare()
        adapter_bay.prepare()
        adapter_08.prepare()

        assert adapter_la.raw_data.shape[1] == 207
        assert adapter_bay.raw_data.shape[1] == 325
        assert adapter_08.raw_data.shape[1] == 170

    def test_train_only_normalization_no_leakage(self):
        adapter = get_adapter("pems08")
        adapter.prepare()

        T, N = adapter.raw_data.shape
        train_end = int(T * adapter.train_ratio)
        train_data = adapter.raw_data[:train_end]
        valid_mask = (train_data > 0) & (~np.isnan(train_data))

        expected_mean = float(np.mean(train_data[valid_mask]))
        assert adapter.mean == pytest.approx(expected_mean, rel=1e-3)


class TestFoundationModelArchitecture:
    """Tests JEPAGraph-FM modules and forward dynamics."""

    def test_gated_dilated_tcn(self):
        tcn = GatedDilatedTemporalConv(in_channels=16, out_channels=16, kernel_size=2, dilation=2)
        x = torch.randn(4, 16, 12)  # [B, C, T]
        out = tcn(x)
        assert out.shape == (4, 16, 12)

    def test_graph_diffusion_variable_nodes(self):
        conv = GraphDiffusionConvolution(in_channels=16, out_channels=16, max_diffusion_step=2)
        # Test with N=10 nodes
        x10 = torch.randn(2, 6, 10, 16)
        adj10 = torch.eye(10)
        out10 = conv(x10, adj10)
        assert out10.shape == (2, 6, 10, 16)

        # Test with N=50 nodes on same module
        x50 = torch.randn(2, 6, 50, 16)
        adj50 = torch.eye(50)
        out50 = conv(x50, adj50)
        assert out50.shape == (2, 6, 50, 16)

    def test_target_encoder_gradient_isolation(self):
        config = FoundationModelConfig.tiny()
        model = JEPAGraphFoundationModel(config)

        # Ensure all target encoder parameters have requires_grad = False
        for param in model.target_encoder.parameters():
            assert not param.requires_grad

        # Ensure online encoder has trainable parameters
        for param in model.online_encoder.parameters():
            assert param.requires_grad

    def test_ema_target_encoder_update(self):
        config = FoundationModelConfig.tiny()
        model = JEPAGraphFoundationModel(config)

        # Get initial online and target params
        online_p = next(model.online_encoder.parameters())
        target_p = next(model.target_encoder.parameters())

        # Modify online parameter
        with torch.no_grad():
            online_p.add_(1.0)
            target_before = target_p.clone()

        # Update EMA with momentum 0.90
        model.update_target_encoder(momentum=0.90)

        # Target should change towards online
        target_after = target_p.clone()
        expected = 0.90 * target_before + 0.10 * online_p
        assert torch.allclose(target_after, expected, atol=1e-5)

    def test_forward_pretrain_shapes(self):
        config = FoundationModelConfig.tiny()
        model = JEPAGraphFoundationModel(config)

        B, T_in, T_out, N = 2, 12, 12, 15
        x = torch.randn(B, T_in, N, 3)
        y = torch.randn(B, T_out, N)
        adj = torch.eye(N)

        outputs = model.forward_pretrain(x, y, adj, variable_type="speed")
        assert outputs["pred_latent"].shape == (B, T_out, N, config.hidden_dim)
        assert outputs["target_latent"].shape == (B, T_out, N, config.hidden_dim)
        assert outputs["recon_x"].shape == (B, T_in, N, 1)
        assert outputs["mask_patch"].shape == (B, T_in, N)
        assert outputs["pred_forecast"].shape == (B, T_out, N)

    def test_forward_inference_and_physical_unnormalization(self):
        config = FoundationModelConfig.tiny()
        model = JEPAGraphFoundationModel(config)

        B, T_in, N = 2, 12, 20
        x = torch.randn(B, T_in, N, 3)
        adj = torch.eye(N)

        pred_phys = model.forecast(x, adj, variable_type="speed", mean=55.0, std=10.0)
        assert pred_phys.shape == (B, 12, N)


class TestRepresentationObjectives:
    """Tests multi-objective loss computation."""

    def test_loss_suite_computation(self):
        loss_suite = RepresentationObjectives(LossWeightsConfig())
        B, T_in, T_out, N, D = 2, 12, 12, 10, 32

        model_outputs = {
            "pred_latent": torch.randn(B, T_out, N, D, requires_grad=True),
            "target_latent": torch.randn(B, T_out, N, D),
            "recon_x": torch.randn(B, T_in, N, 1, requires_grad=True),
            "mask_patch": torch.ones(B, T_in, N),
            "pred_forecast": torch.randn(B, T_out, N, requires_grad=True),
        }
        batch_x = torch.randn(B, T_in, N, 3)
        batch_y = torch.randn(B, T_out, N)
        valid_mask_x = torch.ones(B, T_in, N)
        valid_mask_y = torch.ones(B, T_out, N)

        total_loss, metrics = loss_suite.compute_total_loss(
            model_outputs=model_outputs,
            batch_x=batch_x,
            batch_y=batch_y,
            valid_mask_x=valid_mask_x,
            valid_mask_y=valid_mask_y,
        )

        assert total_loss.item() > 0.0
        assert "loss_total" in metrics
        assert "loss_jepa" in metrics
        assert "loss_mask" in metrics
        assert "loss_sup" in metrics

        # Verify backpropagation produces gradients
        total_loss.backward()
        assert model_outputs["pred_latent"].grad is not None
        assert model_outputs["recon_x"].grad is not None
        assert model_outputs["pred_forecast"].grad is not None


class TestMultiDatasetMixtureAndCheckpoints:
    """Tests multi-dataset batch generation and checkpoint serialization."""

    def test_mixture_loader_iteration(self):
        loader = MultiDatasetMixtureLoader(
            dataset_ids=["metr_la", "pems08"],
            split="val",
            batch_size=4,
            max_samples_per_dataset=16,
            shuffle=False,
        )

        batches_seen = 0
        datasets_seen = set()
        for batch in loader:
            batches_seen += 1
            datasets_seen.add(batch.dataset_id)
            assert batch.x.dim() == 4
            assert batch.y.dim() == 3
            assert batch.adj_matrix.shape[0] == batch.x.shape[2]
            if batches_seen >= 4:
                break

        assert batches_seen > 0
        assert len(datasets_seen) > 0

    def test_checkpoint_save_and_resume(self):
        config = FoundationModelConfig.tiny()
        model = JEPAGraphFoundationModel(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "test_foundation_ckpt.pt"
            state = {
                "epoch": 2,
                "config": config.__dict__,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": 0.42,
            }
            torch.save(state, ckpt_path)
            assert ckpt_path.exists()

            loaded = torch.load(ckpt_path, map_location="cpu")
            new_model = JEPAGraphFoundationModel(config)
            new_model.load_state_dict(loaded["model_state_dict"])
            assert loaded["epoch"] == 2
            assert loaded["val_loss"] == 0.42


class TestAmbulanceRoutingIntegration:
    """Tests integration of JEPAGraph-FM with time-dependent ambulance routing."""

    def test_jepafm_inference_engine(self):
        from src.forecasting.inference import JEPAGraphFMInferenceEngine
        engine = JEPAGraphFMInferenceEngine()
        engine.load()
        assert engine._loaded
        assert engine.horizon == 12

        raw_speeds = np.full(12, 45.0, dtype=np.float32)
        mean_kmh, std_kmh = engine.predict_edge(raw_speeds)
        assert len(mean_kmh) == 12
        assert len(std_kmh) == 12
        assert np.all(mean_kmh > 0)
        assert np.all(std_kmh > 0)

    def test_routing_five_strategies_execution(self):
        from src.simulation.road_network import RoadNetwork
        from src.routing.time_dependent import compare_five_strategies, FallbackMode

        network = RoadNetwork.create_synthetic_grid(rows=3, cols=3, seed=42)
        nodes = list(network.graph.nodes())
        comp = compare_five_strategies(
            network=network,
            origin=nodes[0],
            destination=nodes[-1],
            departure_time_s=100.0,
            risk_penalty=0.3,
            fallback_mode=FallbackMode.CLAMP_MAX_HORIZON,
        )
        assert len(comp) == 5
        assert "1_shortest_distance" in comp
        assert "4_time_dependent_predicted" in comp
        assert comp["1_shortest_distance"].feasible
        assert comp["4_time_dependent_predicted"].feasible
