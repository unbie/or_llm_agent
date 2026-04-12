# -*- coding: utf-8 -*-
"""
run_baseline_tabu.py - 禁忌搜索对比实验启动器
=============================================
功能:
  1. 对 Solomon Benchmark 数据集批量运行 Tabu Search 基线
  2. 输出与 run_batch_no_llm.py / run_baseline_ga.py 格式完全对齐
  3. 实验结果自动保存到 experiments_tabu_baseline/ 目录

用法:
  # 快速单个算例测试:
  python run_baseline_tabu.py --instance c101 --type c1

  # 全套对比实验:
  python run_baseline_tabu.py --all

  # 调整禁忌搜索参数:
  python run_baseline_tabu.py --all --maxiter 2000 --tenure 20
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

from baseline_tabu import TabuVRPSolver


# ──────────────────────────────────────────────────
# 数据加载 (与 run_batch_no_llm.py / run_baseline_ga.py 完全一致)
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

    # 补充 E_i / L_i 字段 (与项目其他脚本保持一致)
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

def run_single_tabu_experiment(
    dataset_path: str,
    max_iter: int = 1000,
    tabu_tenure: int = 15,
    neighborhood_size: int = 50,
    no_improve_restart: int = 150,
    seed: int = None,
    verbose: bool = True,
) -> dict:
    """
    对单个数据集运行 Tabu Search 实验。
    返回 dict 格式与 run_batch_no_llm 和 run_baseline_ga 完全兼容。
    """
    try:
        import random
        if seed is not None:
            random.seed(seed)

        data = load_solomon_data(dataset_path)
        n_cust = len([c for c in data['customers'] if c['id'] != 0])

        solver = TabuVRPSolver(
            data=data,
            max_iter=max_iter,
            tabu_tenure=tabu_tenure,
            neighborhood_size=neighborhood_size,
            no_improve_restart=no_improve_restart,
            seed=seed,
            verbose=verbose,
        )

        t0 = time.time()
        best_solution, best_cost = solver.solve()
        elapsed = time.time() - t0

        # ── 对齐输出格式 ──
        output_lines = [
            f"BEST_COST: {best_cost:.4f}",
            f"NUM_ROUTES: {len(best_solution)}",
            f"CUSTOMERS: {n_cust}",
            f"ELAPSED: {elapsed:.2f}s",
        ]
        for i, route in enumerate(best_solution):
            output_lines.append(f"Route {i+1}: {route}")

        return {
            'success':              True,
            'best_cost':            best_cost,
            'num_routes':           len(best_solution),
            'elapsed_time':         elapsed,
            'n_customers':          n_cust,
            'total_iterations':     len(solver.best_cost_history),
            'best_cost_history':    solver.best_cost_history,
            'output':               '\n'.join(output_lines),
        }

    except Exception as e:
        import traceback
        return {
            'success':              False,
            'best_cost':            None,
            'num_routes':           0,
            'elapsed_time':         0,
            'n_customers':          0,
            'total_iterations':     0,
            'best_cost_history':    [],
            'output':               f"ERROR: {e}\n{traceback.format_exc()}",
        }


# ──────────────────────────────────────────────────
# 批量实验 (与 ALNS 和 GA 使用相同算例集)
# ──────────────────────────────────────────────────

EXPERIMENT_SUITE = [
    ('c1',  'c101.txt', 3),
    ('c1',  'c105.txt', 3),
    ('c2',  'c201.txt', 3),
    ('c2',  'c205.txt', 3),
    ('r1',  'r101.txt', 3),
    ('r1',  'r105.txt', 3),
    ('r2',  'r201.txt', 3),
    ('r2',  'r205.txt', 3),
    ('rc1', 'rc101.txt', 3),
    ('rc1', 'rc105.txt', 3),
    ('rc2', 'rc201.txt', 3),
    ('rc2', 'rc205.txt', 3),
]


def run_batch_tabu_experiments(
    max_iter: int = 1000,
    tabu_tenure: int = 15,
    neighborhood_size: int = 50,
    output_dir: str = "experiments_tabu_baseline",
):
    """
    批量运行所有 Tabu Search 对比实验，结果存储到指定目录。
    """
    print("=" * 70)
    print("Tabu Search 基线对比实验")
    print(f"最大迭代: {max_iter} | 禁忌期限: {tabu_tenure} | 邻域大小: {neighborhood_size}")
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
    exp_idx = 0

    for dataset_type, instance, num_runs in EXPERIMENT_SUITE:
        dataset_path = dataset_base / dataset_type / instance
        if not dataset_path.exists():
            print(f"[跳过] 数据集不存在: {dataset_path}")
            continue

        instance_name = instance.replace('.txt', '')

        for run_id in range(1, num_runs + 1):
            exp_idx += 1
            exp_name = f"{dataset_type}_{instance_name}_run{run_id}"
            seed = run_id * 42 + hash(instance_name) % 1000

            print(f"[{exp_idx:2d}/{total_runs}] {exp_name}")

            result = run_single_tabu_experiment(
                dataset_path=str(dataset_path),
                max_iter=max_iter,
                tabu_tenure=tabu_tenure,
                neighborhood_size=neighborhood_size,
                seed=seed,
                verbose=False,  # 批量模式关闭详细日志
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
                'algorithm':        'TabuSearch',
                'max_iter':         max_iter,
                'tabu_tenure':      tabu_tenure,
                'neighborhood_size': neighborhood_size,
                'timestamp':        datetime.now().isoformat(),
                'success':          result['success'],
                'best_cost':        result['best_cost'],
                'num_routes':       result['num_routes'],
                'elapsed_time':     result['elapsed_time'],
                'n_customers':      result['n_customers'],
                'total_iterations': result['total_iterations'],
            }

            all_results.append(result_data)

            result_file = results_dir / f"{exp_name}.json"
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
    print("Tabu Search 批量实验完成!")
    print(f"  成功/总计: {success_count}/{len(all_results)}")
    if valid_costs:
        print(f"  平均成本: {sum(valid_costs)/len(valid_costs):.2f}")
        print(f"  最优成本: {min(valid_costs):.2f}")
        print(f"  最差成本: {max(valid_costs):.2f}")
    print(f"  结果保存: {results_dir}")
    print()
    print("三算法对比:")
    print("  ALNS(LLM)  -> experiments_batch/all_results.json")
    print("  GA         -> experiments_ga_baseline/all_results.json")
    print("  TabuSearch -> experiments_tabu_baseline/all_results.json")
    print("=" * 70)

    return all_results


# ──────────────────────────────────────────────────
# 快速单算例测试入口
# ──────────────────────────────────────────────────

def quick_test(
    dataset_type: str = 'c1',
    instance: str = 'c101',
    max_iter: int = 1000,
    tabu_tenure: int = 15,
    neighborhood_size: int = 50,
):
    """快速测试单个算例，打印详细日志。"""
    dataset_path = Path(f"data/1 Solomon Benchmark/{dataset_type}/{instance}.txt")
    if not dataset_path.exists():
        print(f"[错误] 数据集不存在: {dataset_path}")
        return

    print(f"\n{'='*60}")
    print(f"Tabu Search 快速测试 | {dataset_type}/{instance}")
    print(f"max_iter={max_iter} | tenure={tabu_tenure} | neighborhood={neighborhood_size}")
    print(f"{'='*60}\n")

    result = run_single_tabu_experiment(
        dataset_path=str(dataset_path),
        max_iter=max_iter,
        tabu_tenure=tabu_tenure,
        neighborhood_size=neighborhood_size,
        seed=42,
        verbose=True,
    )

    print(f"\n{'─'*60}")
    print(f"BEST_COST:  {result['best_cost']}")
    print(f"NUM_ROUTES: {result['num_routes']}")
    print(f"ELAPSED:    {result['elapsed_time']:.2f}s")
    print(f"{'─'*60}")
    print(result['output'])


# ──────────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Tabu Search 基线对比实验',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单算例快速测试
  python run_baseline_tabu.py --instance c101 --type c1

  # r 类困难算例
  python run_baseline_tabu.py --instance r201 --type r2 --maxiter 2000

  # 全套批量对比实验
  python run_baseline_tabu.py --all

  # 全套实验 + 自定义参数
  python run_baseline_tabu.py --all --maxiter 1500 --tenure 20 --neighborhood 80
        """
    )

    parser.add_argument('--all',          action='store_true', help='运行全套批量对比实验')
    parser.add_argument('--type',         default='c1',        help='数据集类型 (c1/c2/r1/r2/rc1/rc2)')
    parser.add_argument('--instance',     default='c101',      help='算例名称 (不含.txt)')
    parser.add_argument('--maxiter',      type=int, default=1000, help='最大迭代次数 (默认1000)')
    parser.add_argument('--tenure',       type=int, default=15,   help='禁忌期限 (默认15)')
    parser.add_argument('--neighborhood', type=int, default=50,   help='邻域评估上限 (默认50)')
    parser.add_argument('--outdir',       default='experiments_tabu_baseline', help='批量实验结果目录')

    args = parser.parse_args()

    if args.all:
        run_batch_tabu_experiments(
            max_iter=args.maxiter,
            tabu_tenure=args.tenure,
            neighborhood_size=args.neighborhood,
            output_dir=args.outdir,
        )
    else:
        quick_test(
            dataset_type=args.type,
            instance=args.instance,
            max_iter=args.maxiter,
            tabu_tenure=args.tenure,
            neighborhood_size=args.neighborhood,
        )
# python run_baseline_tabu.py --type r1 --instance r101 --maxiter 2000 --tenure 20