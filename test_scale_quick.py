# -*- coding: utf-8 -*-
"""Quick validation test for customer scale experiment"""
import sys
sys.path.append('.')
from run_customer_scale_experiment import load_solomon_data, subset_solomon_data, patch_time_windows

# Test data loading and subsetting
data = load_solomon_data('data/1 Solomon Benchmark/c1/c101.txt')
print(f"Original data: {len(data['customers'])} nodes (with depot)")

for n in [15, 25, 50, 75, 100]:
    sub = subset_solomon_data(data, n)
    patch_time_windows(sub)
    print(f"  Subset n={n}: {len(sub['customers'])} nodes, ID range: 0-{sub['customers'][-1]['id']}")

print("\nData subsetting OK!")

# Quick single run test (15 customers, low iterations)
print("\n--- Running quick test: 15 customers, 100 iterations ---")
from run_customer_scale_experiment import run_single_experiment

result = run_single_experiment(
    'data/1 Solomon Benchmark/c1/c101.txt',
    n_customers=15,
    max_iters=100,
    seed=42
)

if result['success']:
    print(f"SUCCESS!")
    print(f"  Cost: {result['best_cost']:.2f}")
    print(f"  Routes: {result['num_routes']}")
    print(f"  Cost/Customer: {result['cost_per_customer']:.2f}")
    print(f"  Time: {result['elapsed_time']:.1f}s")
else:
    print(f"FAILED!")
    print(result['output'][:500])
