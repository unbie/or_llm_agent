# -*- coding: utf-8 -*-
"""Quick test: verify baseline solver + metrics collection works"""
import sys
sys.path.append('.')
from run_llm_optimize import load_solomon_data, run_solver, BASELINE_OPERATOR_CODE

data = load_solomon_data('data/1 Solomon Benchmark/c1/c101.txt')
n_cust = len([c for c in data['customers'] if c['id'] != 0])
print(f"Loaded {n_cust} customers")

sol, cost, metrics = run_solver(data, BASELINE_OPERATOR_CODE, max_iter=50, seed=42, verbose=False)
if metrics:
    print(f"Cost: {cost:.2f}")
    print(f"Routes: {metrics['num_routes']}")
    print(f"Time: {metrics['elapsed_time']:.1f}s")
    print(f"Last improve: iter {metrics['last_improve_iter']}")
    print(f"Operator stats:")
    print(metrics['operator_stats'])
    print(f"C11={metrics['c11']:.0f}  C12={metrics['c12']:.0f}  C13={metrics['c13']:.0f}  C2={metrics['c2']:.0f}  C3={metrics['c3']:.0f}")
    print("Quick test PASSED")
else:
    print("FAILED")
