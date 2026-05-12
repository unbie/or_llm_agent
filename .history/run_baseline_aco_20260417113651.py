"""
run_baseline_aco.py - 蚁群算法对比实验启动器
============================================
功能:
  1. 对 Solomon Benchmark 指定算例运行 ACO 基线
  2. 默认覆盖 C101/C105/C201/C205 (与当前重点对比集一致)
  3. 输出格式与 ALNS / GA runner 对齐

用法:
  # 单算例快速测试
  python run_baseline_aco.py --instance c101 --type c1

  # 批量跑四个重点算例
  python run_baseline_aco.py --all
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline_aco import ACOVRPSolver


if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


def load_solomon_data(file_path: str) -> dict:
    data = {}
    customers = []
    vehicle_capacity = None

    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line == "" or any(line.startswith(kw) for kw in ["C", "VEHICLE", "NUMBER", "CUSTOMER"]):
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

    for line in lines:
        line = line.strip()
        if line == "" or any(line.startswith(kw) for kw in ["C", "VEHICLE", "NUMBER", "CUSTOMER", "CUST"]):
            continue
        parts = line.split()
        if parts[0].isdigit() and len(parts) >= 7:
            cid = int(parts[0])
            customers.append({
                "id": cid,
                "demand": int(parts[3]),
                "x": float(parts[1]),
                "y": float(parts[2]),
                "ready_time": float(parts[4]),
                "due_date": float(parts[5]),
                "service_time": float(parts[6]),
            })

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


def run_single_aco_experiment(
    dataset_path: str,
    n_ants: int = 40,
    max_iter: int = 300,
    alpha: float = 1.0,
    beta: float = 3.0,
    rho: float = 0.15,
    q0: float = 0.15,
    ls_enabled: bool = True,
    ls_steps: int = 25,
    seed: int | None = None,
    verbose: bool = True,
) -> dict:
    try:
        data = load_solomon_data(dataset_path)
        n_cust = len([c for c in data['customers'] if c['id'] != 0])

        solver = ACOVRPSolver(
            data=data,
            n_ants=n_ants,
            max_iter=max_iter,
            alpha=alpha,
            beta=beta,
            rho=rho,
            q0=q0,
            ls_enabled=ls_enabled,
            ls_steps=ls_steps,
            seed=seed,
            verbose=verbose,
        )

        t0 = time.time()
        best_solution, best_cost = solver.solve()
        elapsed = time.time() - t0

        output_lines = [
            f"BEST_COST: {best_cost:.4f}",
            f"NUM_ROUTES: {len(best_solution)}",
            f"CUSTOMERS: {n_cust}",
            f"ELAPSED: {elapsed:.2f}s",
        ]
        for i, route in enumerate(best_solution):
            output_lines.append(f"Route {i+1}: {route}")

        return {
            'success': True,
            'best_cost': best_cost,
            'num_routes': len(best_solution),
            'elapsed_time': elapsed,
            'n_customers': n_cust,
            'total_iterations': len(solver.best_cost_history),
            'best_cost_history': solver.best_cost_history,
            'output': '\n'.join(output_lines),
        }

    except Exception as e:
        import traceback
        return {
            'success': False,
            'best_cost': None,
            'num_routes': 0,
            'elapsed_time': 0,
            'n_customers': 0,
            'total_iterations': 0,
            'best_cost_history': [],
            'output': f"ERROR: {e}\n{traceback.format_exc()}",
        }


# 仅按你的要求跑四个重点算例
EXPERIMENT_SUITE = [
    ('c1', 'c101.txt', 3),
    ('c1', 'c105.txt', 3),
    ('c2', 'c201.txt', 3),
    ('c2', 'c205.txt', 3),
]


def run_batch_aco_experiments(
    n_ants: int = 40,
    max_iter: int = 300,
    alpha: float = 1.0,
    beta: float = 3.0,
    rho: float = 0.15,
    q0: float = 0.15,
    ls_enabled: bool = True,
    ls_steps: int = 25,
    only_c2: bool = False,
    output_dir: str = "experiments_aco_baseline",
):
    print("=" * 70)
    print("ACO 基线对比实验")
    print(f"蚂蚁数={n_ants} | 迭代={max_iter} | alpha={alpha} | beta={beta} | rho={rho} | q0={q0}")
    print(f"局部搜索={ls_enabled} | ls_steps={ls_steps}")
    print("=" * 70)
    print()

    out_path = Path(output_dir)
    results_dir = out_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset_base = Path("data/1 Solomon Benchmark")

    suite = EXPERIMENT_SUITE
    if only_c2:
        suite = [
            ('c2', 'c201.txt', 3),
            ('c2', 'c205.txt', 3),
        ]

    total_runs = sum(runs for _, _, runs in suite)
    print(f"总计: {len(suite)} 个数据集, {total_runs} 次运行")
    print()

    all_results = []
    exp_idx = 0

    for dataset_type, instance, num_runs in suite:
        dataset_path = dataset_base / dataset_type / instance
        if not dataset_path.exists():
            print(f"[SKIP] 数据集不存在: {dataset_path}")
            continue

        instance_name = instance.replace('.txt', '')

        for run_id in range(1, num_runs + 1):
            exp_idx += 1
            exp_name = f"{dataset_type}_{instance_name}_run{run_id}"
            seed = run_id * 42 + hash(instance_name) % 1000

            print(f"[{exp_idx:2d}/{total_runs}] {exp_name}")

            result = run_single_aco_experiment(
                dataset_path=str(dataset_path),
                n_ants=n_ants,
                max_iter=max_iter,
                alpha=alpha,
                beta=beta,
                rho=rho,
                q0=q0,
                ls_enabled=ls_enabled,
                ls_steps=ls_steps,
                seed=seed,
                verbose=False,
            )

            if result['success']:
                print(
                    f"  ✓ Cost={result['best_cost']:.2f} | "
                    f"Routes={result['num_routes']} | "
                    f"Time={result['elapsed_time']:.1f}s | "
                    f"Iters={result['total_iterations']}"
                )
            else:
                print(f"  ✗ 失败: {result['output'][:200]}")

            result_data = {
                'exp_name': exp_name,
                'dataset_type': dataset_type,
                'instance': instance_name,
                'run_id': run_id,
                'seed': seed,
                'algorithm': 'ACO',
                'n_ants': n_ants,
                'max_iter': max_iter,
                'alpha': alpha,
                'beta': beta,
                'rho': rho,
                'q0': q0,
                'ls_enabled': ls_enabled,
                'ls_steps': ls_steps,
                'timestamp': datetime.now().isoformat(),
                'success': result['success'],
                'best_cost': result['best_cost'],
                'num_routes': result['num_routes'],
                'elapsed_time': result['elapsed_time'],
                'n_customers': result['n_customers'],
                'total_iterations': result['total_iterations'],
            }

            all_results.append(result_data)

            result_file = results_dir / f"{exp_name}.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)

    summary_file = out_path / "all_results.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    success_count = sum(1 for r in all_results if r['success'])
    valid_costs = [r['best_cost'] for r in all_results if r['success'] and r['best_cost']]

    print()
    print("=" * 70)
    print("ACO 批量实验完成!")
    print(f"  成功/总计: {success_count}/{len(all_results)}")
    if valid_costs:
        print(f"  平均成本: {sum(valid_costs)/len(valid_costs):.2f}")
        print(f"  最优成本: {min(valid_costs):.2f}")
        print(f"  最差成本: {max(valid_costs):.2f}")
    print(f"  结果保存: {results_dir}")
    print("=" * 70)

    return all_results


def quick_test(
    dataset_type: str = 'c1',
    instance: str = 'c101',
    n_ants: int = 40,
    max_iter: int = 300,
    ls_enabled: bool = True,
    ls_steps: int = 25,
):
    dataset_path = Path(f"data/1 Solomon Benchmark/{dataset_type}/{instance}.txt")
    if not dataset_path.exists():
        print(f"[ERROR] 数据集不存在: {dataset_path}")
        return

    print(f"\n{'='*60}")
    print(f"ACO 快速测试 | {dataset_type}/{instance} | ants={n_ants} | iter={max_iter}")
    print(f"{'='*60}\n")

    result = run_single_aco_experiment(
        dataset_path=str(dataset_path),
        n_ants=n_ants,
        max_iter=max_iter,
        ls_enabled=ls_enabled,
        ls_steps=ls_steps,
        seed=42,
        verbose=True,
    )

    print(f"\n{'─'*60}")
    print(f"BEST_COST:  {result['best_cost']}")
    print(f"NUM_ROUTES: {result['num_routes']}")
    print(f"ELAPSED:    {result['elapsed_time']:.2f}s")
    print(f"{'─'*60}")
    print(result['output'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ACO 基线对比实验',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单算例
  python run_baseline_aco.py --instance c101 --type c1

  # 四个重点算例批量
  python run_baseline_aco.py --all

  # 调整参数
  python run_baseline_aco.py --all --ants 60 --iter 500 --beta 4.0
        """
    )

    parser.add_argument('--all', action='store_true', help='运行四个重点算例批量实验')
    parser.add_argument('--type', default='c1', help='数据集类型 (c1/c2)')
    parser.add_argument('--instance', default='c101', help='算例名称 (不含.txt)')
    parser.add_argument('--ants', type=int, default=40, help='蚂蚁数 (默认40)')
    parser.add_argument('--iter', type=int, default=300, help='最大迭代 (默认300)')
    parser.add_argument('--alpha', type=float, default=1.0, help='信息素权重 α')
    parser.add_argument('--beta', type=float, default=3.0, help='启发式权重 β')
    parser.add_argument('--rho', type=float, default=0.15, help='挥发率 ρ')
    parser.add_argument('--q0', type=float, default=0.15, help='贪心选择概率 q0')
    parser.add_argument('--no-ls', action='store_true', help='禁用局部搜索')
    parser.add_argument('--ls-steps', type=int, default=25, help='每轮局部搜索邻域尝试次数')
    parser.add_argument('--only-c2', action='store_true', help='批量模式仅跑 c2/c201 和 c2/c205')
    parser.add_argument('--outdir', default='experiments_aco_baseline', help='输出目录')

    args = parser.parse_args()

    if args.all:
        run_batch_aco_experiments(
            n_ants=args.ants,
            max_iter=args.iter,
            alpha=args.alpha,
            beta=args.beta,
            rho=args.rho,
            q0=args.q0,
            ls_enabled=not args.no_ls,
            ls_steps=args.ls_steps,
            only_c2=args.only_c2,
            output_dir=args.outdir,
        )
    else:
        quick_test(
            dataset_type=args.type,
            instance=args.instance,
            n_ants=args.ants,
            max_iter=args.iter,
            ls_enabled=not args.no_ls,
            ls_steps=args.ls_steps,
        )
