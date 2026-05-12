# -*- coding: utf-8 -*-
"""
run_baseline_alns.py - 标准 ALNS 对比实验启动器
================================================
功能:
  1. 对 Solomon Benchmark 数据集批量运行 标准ALNS 基线
  2. 输出与 run_batch_no_llm.py / run_baseline_ga.py / run_baseline_tabu.py 格式完全对齐
  3. 实验结果自动保存到 experiments_alns_baseline/ 目录

用法:
  # 快速单个算例测试:
  python run_baseline_alns.py --instance c101 --type c1

  # 全套对比实验:
  python run_baseline_alns.py --all

  # 调整参数:
  python run_baseline_alns.py --all --maxiter 2000
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Windows GBK 终端 UTF-8 修复
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from baseline_alns import ALNSVRPSolver

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# ──────────────────────────────────────────────────
# 数据加载 (与其他 runner 完全一致)
# ──────────────────────────────────────────────────

def load_solomon_data(file_path: str) -> dict:
    """加载 Solomon Benchmark 数据集为标准 data 字典。"""
    data = {}
    customers = []
    vehicle_capacity = None

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 解析车辆容量
    for line in lines:
        line = line.strip()
        if not line or any(line.startswith(kw) for kw in ["C", "VEHICLE", "NUMBER", "CUSTOMER"]):
            continue
        parts = line.split()
        if len(parts) == 2:
            try:
                vehicle_capacity = int(parts[1])
                break
            except ValueError:
                continue

    if vehicle_capacity is None:
        vehicle_capacity = 200

    # 解析客户节点
    for line in lines:
        line = line.strip()
        if not line or any(line.startswith(kw) for kw in ["C", "VEHICLE", "NUMBER", "CUSTOMER", "CUST"]):
            continue
        parts = line.split()
        if parts[0].isdigit() and len(parts) >= 7:
            cust_id = int(parts[0])
            customers.append({
                "id":           cust_id,
                "demand":       int(parts[3]),
                "x":            float(parts[1]),
                "y":            float(parts[2]),
                "ready_time":   float(parts[4]),
                "due_date":     float(parts[5]),
                "service_time": float(parts[6]),
            })

    # 补充 E_i / L_i 字段
    if customers:
        depot = customers[0]
        for cust in customers:
            if cust['id'] != 0:
                cust['E_i'] = max(0, cust['ready_time'])
                cust['L_i'] = min(depot['due_date'], cust['due_date'])
            else:
                cust['E_i'] = 0
                cust['L_i'] = depot['due_date']

    data['vehicle_capacity'] = vehicle_capacity
    data['customers'] = customers
    return data


# ──────────────────────────────────────────────────
# 单次实验执行
# ──────────────────────────────────────────────────

def run_single_alns_experiment(
    dataset_path: str,
    max_iter: int = 1000,
    seed: int = None,
    verbose: bool = True,
) -> dict:
    """
    对单个数据集运行标准 ALNS 实验。
    返回 dict 格式与 GA / Tabu runner 完全兼容。
    """
    try:
        import random
        if seed is not None:
            random.seed(seed)

        data = load_solomon_data(dataset_path)
        n_cust = len([c for c in data['customers'] if c['id'] != 0])

        solver = ALNSVRPSolver(
            data=data,
            max_iter=max_iter,
            seed=seed,
            verbose=verbose,
            time_budget_sec=300,
        )

        t0 = time.time()
        best_solution, best_cost = solver.solve()
        elapsed = time.time() - t0

        # 构造对齐输出
        output_lines = [
            f"BEST_COST: {best_cost:.4f}",
            f"NUM_ROUTES: {len(best_solution)}",
            f"CUSTOMERS: {n_cust}",
            f"ELAPSED: {elapsed:.2f}s",
        ]
        for i, route in enumerate(best_solution):
            output_lines.append(f"Route {i+1}: {route}")

        return {
            'success':           True,
            'best_cost':         best_cost,
            'num_routes':        len(best_solution),
            'elapsed_time':      elapsed,
            'n_customers':       n_cust,
            'total_iterations':  len(solver.best_cost_history),
            'best_cost_history': solver.best_cost_history,
            'output':            '\n'.join(output_lines),
        }

    except Exception as e:
        import traceback
        return {
            'success':           False,
            'best_cost':         None,
            'num_routes':        0,
            'elapsed_time':      0,
            'n_customers':       0,
            'total_iterations':  0,
            'best_cost_history': [],
            'output':            f"ERROR: {e}\n{traceback.format_exc()}",
        }


# ──────────────────────────────────────────────────
# 批量实验 (与 GA 使用相同算例集)
# ──────────────────────────────────────────────────

EXPERIMENT_SUITE = [
    ('c1',  'c101.txt', 1),
    ('c1',  'c102.txt', 1),
    ('c1',  'c103.txt', 1),
    ('c1',  'c104.txt', 1),
    ('c1',  'c105.txt', 1),
    ('c2',  'c201.txt', 1),
    ('c2',  'c202.txt', 1),
    ('c2',  'c203.txt', 1),
    ('c2',  'c204.txt', 1),
    ('c2',  'c205.txt', 1),
    ('r1',  'r101.txt', 1),
    ('r1',  'r102.txt', 1),
    ('r1',  'r103.txt', 1),
    ('r1',  'r104.txt', 1),
    ('r1',  'r105.txt', 1),
    ('r2',  'r201.txt', 1),
    ('r2',  'r202.txt', 1),
    ('r2',  'r203.txt', 1),
    ('r2',  'r204.txt', 1),
    ('r2',  'r205.txt', 1),
    ('rc1', 'rc101.txt', 1),
    ('rc1', 'rc102.txt', 1),
    ('rc1', 'rc103.txt', 1),
    ('rc1', 'rc104.txt', 1),
    ('rc1', 'rc105.txt', 1),
    ('rc2', 'rc201.txt', 1),
    ('rc2', 'rc202.txt', 1),
    ('rc2', 'rc203.txt', 1),
    ('rc2', 'rc204.txt', 1),
    ('rc2', 'rc205.txt', 1),
]


def run_batch_alns_experiments(
    max_iter: int = 500,
    output_dir: str = "experiments_alns_baseline",
):
    """
    批量运行所有 ALNS 基线对比实验。
    """
    print("=" * 70)
    print("ALNS 标准基线对比实验 (手工算子, 无 LLM)")
    print(f"最大迭代: {max_iter}")
    print("=" * 70)
    print()

    out_path = Path(output_dir)
    results_dir = out_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset_base = Path("data/1 Solomon Benchmark")

    total_runs = sum(runs for _, _, runs in EXPERIMENT_SUITE)
    print(f"总计: {len(EXPERIMENT_SUITE)} 个数据集, {total_runs} 次运行")
    print()

    all_results = []

    run_tasks = []
    for dataset_type, instance, num_runs in EXPERIMENT_SUITE:
        instance_name = instance.replace('.txt', '')
        for run_id in range(1, num_runs + 1):
            run_tasks.append((dataset_type, instance, instance_name, run_id))

    if tqdm is not None:
        task_iter = tqdm(run_tasks, total=len(run_tasks), desc="ALNS Batch", ncols=100)
    else:
        task_iter = run_tasks

    exp_idx = 0

    for dataset_type, instance, instance_name, run_id in task_iter:
        dataset_path = dataset_base / dataset_type / instance
        if not dataset_path.exists():
            print(f"[SKIP] Dataset not found: {dataset_path}")
            continue

        exp_idx += 1
        exp_name = f"{dataset_type}_{instance_name}_run{run_id}"
        seed = run_id * 42 + hash(instance_name) % 1000
        result_file = results_dir / f"{exp_name}.json"

        if result_file.exists():
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    cached = json.load(f)
                all_results.append(cached)
                msg = f"[{exp_idx:2d}/{total_runs}] {exp_name} 已存在结果，跳过运行"
                if tqdm is not None:
                    tqdm.write(msg)
                else:
                    print(msg)
                continue
            except Exception:
                msg = f"[{exp_idx:2d}/{total_runs}] {exp_name} 已存在结果文件但读取失败，将重新运行"
                if tqdm is not None:
                    tqdm.write(msg)
                else:
                    print(msg)

        if tqdm is None:
            print(f"[{exp_idx:2d}/{total_runs}] {exp_name}")

        result = run_single_alns_experiment(
            dataset_path=str(dataset_path),
            max_iter=max_iter,
            seed=seed,
            verbose=True,
        )

        if result['success']:
            print(f"  OK  Cost={result['best_cost']:.2f} | "
                  f"Routes={result['num_routes']} | "
                  f"Time={result['elapsed_time']:.1f}s | "
                  f"Iters={result['total_iterations']}")
        else:
            print(f"  FAIL  {result['output'][:200]}")

        result_data = {
            'exp_name':         exp_name,
            'dataset_type':     dataset_type,
            'instance':         instance_name,
            'run_id':           run_id,
            'seed':             seed,
            'algorithm':        'ALNS_Baseline',
            'max_iter':         max_iter,
            'timestamp':        datetime.now().isoformat(),
            'success':          result['success'],
            'best_cost':        result['best_cost'],
            'num_routes':       result['num_routes'],
            'elapsed_time':     result['elapsed_time'],
            'n_customers':      result['n_customers'],
            'total_iterations': result['total_iterations'],
        }

        all_results.append(result_data)

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

    # 汇总
    summary_file = out_path / "all_results.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    success_count = sum(1 for r in all_results if r['success'])
    valid_costs = [r['best_cost'] for r in all_results if r['success'] and r['best_cost']]

    print()
    print("=" * 70)
    print("ALNS Baseline Batch Complete!")
    print(f"  Success:    {success_count}/{len(all_results)}")
    if valid_costs:
        print(f"  Avg cost:   {sum(valid_costs)/len(valid_costs):.2f}")
        print(f"  Best cost:  {min(valid_costs):.2f}")
        print(f"  Worst cost: {max(valid_costs):.2f}")
    print(f"  Results:    {results_dir}")
    print()
    print("Comparison outputs:")
    print("  ALNS(LLM)      -> experiments_batch/all_results.json")
    print("  ALNS(Baseline)  -> experiments_alns_baseline/all_results.json")
    print("  ACO             -> experiments_aco_baseline/all_results.json")
    print("  GA              -> experiments_ga_baseline/all_results.json")
    print("=" * 70)

    return all_results


# ──────────────────────────────────────────────────
# 快速单算例测试入口
# ──────────────────────────────────────────────────

def quick_test(
    dataset_type: str = 'c1',
    instance: str = 'c101',
    max_iter: int = 300,
    output_dir: str = 'experiments_alns_baseline',
):
    """快速测试单个算例，打印详细日志。"""
    dataset_path = Path(f"data/1 Solomon Benchmark/{dataset_type}/{instance}.txt")
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}")
        return

    print(f"\n{'='*60}")
    print(f"ALNS Baseline Quick Test | {dataset_type}/{instance}")
    print(f"max_iter={max_iter}")
    print(f"{'='*60}\n")

    result = run_single_alns_experiment(
        dataset_path=str(dataset_path),
        max_iter=max_iter,
        seed=42,
        verbose=True,
    )

    # 保存结果到与批量一致的目录
    out_path = Path(output_dir)
    results_dir = out_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    result_file = results_dir / f"{dataset_type}_{instance}_run1.json"
    result_data = {
        'exp_name':         f"{dataset_type}_{instance}_run1",
        'dataset_type':     dataset_type,
        'instance':         instance,
        'run_id':           1,
        'seed':             42,
        'algorithm':        'ALNS_Baseline',
        'max_iter':         max_iter,
        'timestamp':        datetime.now().isoformat(),
        'success':          result['success'],
        'best_cost':        result['best_cost'],
        'num_routes':       result['num_routes'],
        'elapsed_time':     result['elapsed_time'],
        'n_customers':      result['n_customers'],
        'total_iterations': result['total_iterations'],
    }
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'-'*60}")
    print(f"BEST_COST:  {result['best_cost']}")
    print(f"NUM_ROUTES: {result['num_routes']}")
    print(f"ELAPSED:    {result['elapsed_time']:.2f}s")
    print(f"{'-'*60}")
    print(result['output'])


# ──────────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ALNS Baseline Comparison Experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Quick single test
  python run_baseline_alns.py --instance c101 --type c1

  # Harder instance
  python run_baseline_alns.py --instance r201 --type r2 --maxiter 2000

  # Full batch experiment
  python run_baseline_alns.py --all

  # Full batch + custom iterations
  python run_baseline_alns.py --all --maxiter 1500
        """
    )

    parser.add_argument('--all',      action='store_true', help='Run full batch experiments')
    parser.add_argument('--type',     default='c1',  help='Dataset type (c1/c2/r1/r2/rc1/rc2)')
    parser.add_argument('--instance', default='c101', help='Instance name (without .txt)')
    parser.add_argument('--maxiter',  type=int, default=300, help='Max ALNS iterations (default 300)')
    parser.add_argument('--outdir',   default='experiments_alns_baseline', help='Batch output directory')

    args = parser.parse_args()

    if args.all:
        run_batch_alns_experiments(
            max_iter=args.maxiter,
            output_dir=args.outdir,
        )
    else:
        quick_test(
            dataset_type=args.type,
            instance=args.instance,
            max_iter=args.maxiter,
            output_dir=args.outdir,
        )
