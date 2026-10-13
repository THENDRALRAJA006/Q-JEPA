"""
Reconciliation and statistical significance calculations for Task 1.
"""
import json
import numpy as np
from scipy import stats

with open('results/metrics/integrated_benchmark_results.json') as f:
    d = json.load(f)

recs = d['journey_records']
strategies = sorted(set(r['strategy'] for r in recs))
baseline = 'A_shortest_distance'

# Check assertions: no duplicate (scenario_id, seed) per strategy
for s in strategies:
    s_recs = [r for r in recs if r['strategy'] == s]
    keys = [(r['scenario_id'], r['seed']) for r in s_recs]
    assert len(keys) == len(set(keys)), f'Duplicate keys in strategy {s}'
    assert len(keys) == 35, f'Expected 35 records, got {len(keys)} in {s}'

base_recs = {(r['scenario_id'], r['seed']): r['travel_time_s'] for r in recs if r['strategy'] == baseline}
base_times = [r['travel_time_s'] for r in recs if r['strategy'] == baseline]
base_mean = float(np.mean(base_times))

print(f"Baseline A Mean: {base_mean:.2f}s (N=35)")
print(f"{'Strategy':<25} {'Mean(s)':>8} {'Agg%':>8} {'Paired%':>8} {'Std%':>8} {'95% CI (t)':>18} {'p-val':>12}")
print("-" * 95)

results_summary = {}

for s in strategies:
    s_recs = [r for r in recs if r['strategy'] == s]
    times = [r['travel_time_s'] for r in s_recs]
    m = float(np.mean(times))
    agg_imp = (base_mean - m) / base_mean * 100.0
    paired_imp = [(base_recs[(r['scenario_id'], r['seed'])] - r['travel_time_s']) / base_recs[(r['scenario_id'], r['seed'])] * 100.0 for r in s_recs]
    p_mean = float(np.mean(paired_imp))
    p_std = float(np.std(paired_imp, ddof=1))
    
    # 95% Student-t confidence interval (df=34)
    t_crit = float(stats.t.ppf(0.975, df=len(paired_imp) - 1))
    ci_half = t_crit * p_std / np.sqrt(len(paired_imp))
    ci_str = f"[{p_mean - ci_half:+.2f}%, {p_mean + ci_half:+.2f}%]"
    
    # Paired t-test against baseline
    diffs = [base_recs[(r['scenario_id'], r['seed'])] - r['travel_time_s'] for r in s_recs]
    if s == baseline:
        pval = 1.0
    else:
        tt_res = stats.ttest_1samp(diffs, 0.0)
        pval = float(tt_res.pvalue)
        
    print(f"{s:<25} {m:8.1f} {agg_imp:+8.2f}% {p_mean:+8.2f}% {p_std:8.2f}% {ci_str:>18} {pval:12.4e}")
    results_summary[s] = {
        "mean_s": m,
        "aggregate_pct": agg_imp,
        "paired_pct": p_mean,
        "paired_std": p_std,
        "ci_95_lower": p_mean - ci_half,
        "ci_95_upper": p_mean + ci_half,
        "p_value": pval,
    }

with open('scratch_audit_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)
