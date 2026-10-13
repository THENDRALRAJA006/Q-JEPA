"""
Generate Publication-Quality Comparison Figures for SOTA vs Q-JEPA Benchmark.

Outputs:
1. results/figures/sota_literature_comparison.png
2. results/figures/sota_local_reproduced_comparison.png
3. results/figures/jepa_relative_gains_by_horizon.png
4. results/figures/pareto_accuracy_vs_latency.png
5. results/figures/routing_strategy_comparison.png
"""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Theme colors
NAVY_BG = "#0a0f1e"
CARD_BG = "#0d1a3a"
CYAN = "#4fc3f7"
ORANGE = "#f39c12"
RED = "#e74c3c"
GREEN = "#27ae60"
PURPLE = "#9b59b6"
BLUE = "#3498db"
GRAY = "#7a9bc7"
WHITE = "#e8edf5"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["text.color"] = WHITE
plt.rcParams["axes.labelcolor"] = WHITE
plt.rcParams["xtick.color"] = WHITE
plt.rcParams["ytick.color"] = WHITE
plt.rcParams["axes.edgecolor"] = "#1e3a6e"
plt.rcParams["figure.facecolor"] = NAVY_BG
plt.rcParams["axes.facecolor"] = CARD_BG


def plot_literature_comparison():
    lit_path = Path("results/metrics/sota_literature_reference.json")
    if not lit_path.exists():
        print("Literature metrics not found.")
        return

    with open(lit_path) as f:
        data = json.load(f)

    methods = data.get("candidate_methods", {})
    # Filter out models without reported metrics dict
    valid_models = []
    for k, v in methods.items():
        if "reported_metrics" in v and "15_min_step3" in v["reported_metrics"]:
            valid_models.append((k, v))

    names = [m[1]["model_name"].split(" (")[0] for m in valid_models]
    mae15 = [m[1]["reported_metrics"]["15_min_step3"]["MAE"] for m in valid_models]
    mae30 = [m[1]["reported_metrics"]["30_min_step6"]["MAE"] for m in valid_models]
    mae60 = [m[1]["reported_metrics"]["60_min_step12"]["MAE"] for m in valid_models]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    r1 = ax.bar(x - width, mae15, width, label="15 min (Step 3)", color=CYAN, edgecolor="#1e3a6e")
    r2 = ax.bar(x, mae30, width, label="30 min (Step 6)", color=ORANGE, edgecolor="#1e3a6e")
    r3 = ax.bar(x + width, mae60, width, label="60 min (Step 12)", color=RED, edgecolor="#1e3a6e")

    ax.set_ylabel("Reported Test MAE (mph) [Lower is Better]", fontsize=12, fontweight="bold")
    ax.set_title("Published State-of-the-Art Literature Results on METR-LA (As Reported in Primary Papers)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=10)
    ax.legend(facecolor=NAVY_BG, edgecolor="#1e3a6e", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#1e3a6e")
    ax.set_ylim(0, 5.5)

    plt.tight_layout()
    out_file = FIGURES_DIR / "sota_literature_comparison.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")


def plot_local_reproduced_comparison():
    local_path = Path("results/metrics/sota_local_benchmark.json")
    if not local_path.exists():
        # Fallback to controlled real experiments summary if running
        local_path = Path("results/metrics/controlled_real_experiments_summary.json")

    if not local_path.exists():
        print("Local benchmark metrics not found.")
        return

    with open(local_path) as f:
        data = json.load(f)

    if "controlled_multi_seed_tier" in data:
        data = data["controlled_multi_seed_tier"]

    models_data = data.get("models", {})
    names = []
    mae15, std15 = [], []
    mae30, std30 = [], []
    mae60, std60 = [], []

    for k, v in models_data.items():
        names.append(k.replace("_", " "))
        h15 = v.get("horizons", {}).get("15_min_step3", {})
        h30 = v.get("horizons", {}).get("30_min_step6", {})
        h60 = v.get("horizons", {}).get("60_min_step12", {})

        mae15.append(h15.get("mae_mean", h15.get("MAE", 0.0)))
        std15.append(h15.get("mae_std", 0.0))
        mae30.append(h30.get("mae_mean", h30.get("MAE", 0.0)))
        std30.append(h30.get("mae_std", 0.0))
        mae60.append(h60.get("mae_mean", h60.get("MAE", 0.0)))
        std60.append(h60.get("mae_std", 0.0))

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, mae15, width, yerr=std15, capsize=4, label="15 min (Step 3)", color=CYAN, edgecolor="#1e3a6e")
    ax.bar(x, mae30, width, yerr=std30, capsize=4, label="30 min (Step 6)", color=ORANGE, edgecolor="#1e3a6e")
    ax.bar(x + width, mae60, width, yerr=std60, capsize=4, label="60 min (Step 12)", color=RED, edgecolor="#1e3a6e")

    ax.set_ylabel("Locally Measured Test MAE (mph) [Lower is Better]", fontsize=12, fontweight="bold")
    ax.set_title("Locally Reproduced Standardized Benchmark on METR-LA (Multi-Seed Convergent Evaluation)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=10)
    ax.legend(facecolor=NAVY_BG, edgecolor="#1e3a6e", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#1e3a6e")

    plt.tight_layout()
    out_file = FIGURES_DIR / "sota_local_reproduced_comparison.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")


def plot_relative_gains():
    local_path = Path("results/metrics/sota_local_benchmark.json")
    if not local_path.exists():
        local_path = Path("results/metrics/controlled_real_experiments_summary.json")

    if not local_path.exists():
        return

    with open(local_path) as f:
        data = json.load(f)

    if "controlled_multi_seed_tier" in data:
        data = data["controlled_multi_seed_tier"]

    models_data = data.get("models", {})
    if "JEPAGraph" not in models_data:
        return

    jepa = models_data["JEPAGraph"]["horizons"]
    horizons = ["15_min_step3", "30_min_step6", "60_min_step12", "average_steps_1_to_12"]
    horizon_labels = ["15 min", "30 min", "60 min", "12-Step Avg"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(horizons))
    width = 0.2

    baselines_to_compare = [
        ("STGCN", "vs STGCN", CYAN),
        ("JEPAGraph_DirectRegression", "vs Direct Regression (Ablation)", PURPLE),
        ("SpatialTemporal_LSTM", "vs ST-LSTM", ORANGE),
    ]

    for idx, (b_key, b_label, color) in enumerate(baselines_to_compare):
        if b_key not in models_data:
            continue
        gains = []
        for h in horizons:
            b_mae = models_data[b_key]["horizons"][h].get("mae_mean", models_data[b_key]["horizons"][h].get("MAE", 1.0))
            j_mae = jepa[h].get("mae_mean", jepa[h].get("MAE", 1.0))
            gain = 100.0 * (b_mae - j_mae) / max(b_mae, 1e-5)
            gains.append(gain)

        ax.bar(x + (idx - 1) * width, gains, width, label=b_label, color=color, edgecolor="#1e3a6e")

    ax.axhline(0, color="#ffffff", linestyle="-", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("Relative MAE Improvement (%) [Higher is Better]", fontsize=11, fontweight="bold")
    ax.set_title("Relative Forecasting Improvement of JEPAGraph by Horizon", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(horizon_labels, fontsize=11)
    ax.legend(facecolor=NAVY_BG, edgecolor="#1e3a6e", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#1e3a6e")

    plt.tight_layout()
    out_file = FIGURES_DIR / "jepa_relative_gains_by_horizon.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")


def plot_pareto_accuracy_vs_latency():
    local_path = Path("results/metrics/sota_local_benchmark.json")
    if not local_path.exists():
        local_path = Path("results/metrics/controlled_real_experiments_summary.json")

    if not local_path.exists():
        return

    with open(local_path) as f:
        data = json.load(f)

    if "controlled_multi_seed_tier" in data:
        data = data["controlled_multi_seed_tier"]

    models_data = data.get("models", {})

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [CYAN, ORANGE, RED, GREEN, PURPLE, BLUE, "#f1c40f", "#1abc9c"]

    for idx, (mname, mval) in enumerate(models_data.items()):
        params = mval.get("parameters", 0)
        lat = mval.get("mean_latency_ms", mval.get("latency_ms", 1.0))
        mae = mval.get("horizons", {}).get("15_min_step3", {}).get("mae_mean", mval.get("horizons", {}).get("15_min_step3", {}).get("MAE", 5.0))

        color = colors[idx % len(colors)]
        ax.scatter(lat, mae, s=max(params / 1000.0 * 20.0, 80), color=color, alpha=0.85, edgecolors="#ffffff", linewidths=1.5, zorder=3)
        ax.annotate(
            f"{mname.replace('_', ' ')}\n({params:,} params)",
            (lat, mae),
            xytext=(10, 5), textcoords="offset points",
            fontsize=9, color=WHITE,
            bbox=dict(boxstyle="round,pad=0.2", facecolor=NAVY_BG, edgecolor="#1e3a6e", alpha=0.8)
        )

    ax.set_xlabel("Inference Latency per Batch (ms) [Log Scale]", fontsize=11, fontweight="bold")
    ax.set_ylabel("15-Minute Forecast MAE (mph) [Lower is Better]", fontsize=11, fontweight="bold")
    ax.set_xscale("log")
    ax.set_title("Pareto Efficiency: Inference Latency vs Forecast Accuracy (Circle Size ~ Model Parameters)", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, linestyle="--", alpha=0.3, color="#1e3a6e")

    plt.tight_layout()
    out_file = FIGURES_DIR / "pareto_accuracy_vs_latency.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")


def plot_routing_comparison():
    routing_path = Path("results/metrics/time_dependent_routing_results.json")
    if not routing_path.exists():
        print("Routing metrics not found.")
        return

    with open(routing_path) as f:
        data = json.load(f)

    strategies = data.get("strategies", {})
    names, mean_times, p90_times = [], [], []

    for k, v in strategies.items():
        name = v.get("name", k)
        names.append(name.split(": ")[-1] if ": " in name else name)
        mean_times.append(v.get("mean_journey_time_s", 0.0))
        p90_times.append(v.get("p90_journey_time_s", 0.0))

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width/2, mean_times, width, label="Mean Journey Time (s)", color=CYAN, edgecolor="#1e3a6e")
    ax.bar(x + width/2, p90_times, width, label="90th Percentile Tail Time (s)", color=RED, edgecolor="#1e3a6e")

    ax.set_ylabel("Simulated Ambulance Journey Time (seconds)", fontsize=11, fontweight="bold")
    ax.set_title("Ambulance Routing Benchmark Across 5 Strategies (Evaluated Separately on Urban Grid)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
    ax.legend(facecolor=NAVY_BG, edgecolor="#1e3a6e", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#1e3a6e")

    plt.tight_layout()
    out_file = FIGURES_DIR / "routing_strategy_comparison.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    plot_literature_comparison()
    plot_local_reproduced_comparison()
    plot_relative_gains()
    plot_pareto_accuracy_vs_latency()
    plot_routing_comparison()
