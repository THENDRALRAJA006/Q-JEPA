"""
Re-aggregate existing benchmark results with new assertions, CI, and p-values,
and save to integrated_benchmark_results.json and benchmark_summary.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_integrated_benchmark import JourneyRecord, aggregate_results, STRATEGY_NAMES

json_path = Path("results/metrics/integrated_benchmark_results.json")
with open(json_path) as f:
    data = json.load(f)

# Reconstruct JourneyRecords
records = []
for r in data["journey_records"]:
    records.append(JourneyRecord(**r))

# Re-aggregate
strat_results = aggregate_results(records)
data["strategies"] = strat_results

with open(json_path, "w") as f:
    json.dump(data, f, indent=2, default=str)
print("Updated integrated_benchmark_results.json")

# Update benchmark_summary.json
summary_path = Path("results/metrics/benchmark_summary.json")
with open(summary_path) as f:
    summary = json.load(f)

summary["strategies"] = strat_results
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2, default=str)
print("Updated benchmark_summary.json")
