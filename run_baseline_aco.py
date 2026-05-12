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
import hashlib
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

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


def build_stable_seed(instance_name: str, run_id: int, base: int = 42) -> int:
    """构建跨进程稳定的随机种子，避免使用内置 hash 的随机化行为。"""
    digest = hashlib.md5(instance_name.encode('utf-8')).hexdigest()
    bucket = int(digest[:8], 16) % 1000
    return run_id * base + bucket


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


# 四个重点算例
CORE4_SUITE = [
    ('c1', 'c101.txt', 3),
    ('c1', 'c105.txt', 3),
    ('c2', 'c201.txt', 3),
    ('c2', 'c205.txt', 3),
]


# 跨类别核心 30 算例
CORE30_SUITE = [
    ('c1',  'c101.txt', 3),
    ('c1',  'c102.txt', 3),
    ('c1',  'c103.txt', 3),
    ('c1',  'c104.txt', 3),
    ('c1',  'c105.txt', 3),
    ('c2',  'c201.txt', 3),
    ('c2',  'c202.txt', 3),
    ('c2',  'c203.txt', 3),
    ('c2',  'c204.txt', 3),
    ('c2',  'c205.txt', 3),
    ('r1',  'r101.txt', 3),
    ('r1',  'r102.txt', 3),
    ('r1',  'r103.txt', 3),
    ('r1',  'r104.txt', 3),
    ('r1',  'r105.txt', 3),
    ('r2',  'r201.txt', 3),
    ('r2',  'r202.txt', 3),
    ('r2',  'r203.txt', 3),
    ('r2',  'r204.txt', 3),
    ('r2',  'r205.txt', 3),
    ('rc1', 'rc101.txt', 3),
    ('rc1', 'rc102.txt', 3),
    ('rc1', 'rc103.txt', 3),
    ('rc1', 'rc104.txt', 3),
    ('rc1', 'rc105.txt', 3),
    ('rc2', 'rc201.txt', 3),
    ('rc2', 'rc202.txt', 3),
    ('rc2', 'rc203.txt', 3),
    ('rc2', 'rc204.txt', 3),
    ('rc2', 'rc205.txt', 3),
]


FULL56_GROUPS = {
    'c1': [f'c10{i}.txt' for i in range(1, 10)],
    'c2': [f'c20{i}.txt' for i in range(1, 9)],
    'r1': [f'r10{i}.txt' for i in range(1, 10)] + ['r110.txt', 'r111.txt', 'r112.txt'],
    'r2': [f'r20{i}.txt' for i in range(1, 10)] + ['r210.txt', 'r211.txt'],
    'rc1': [f'rc10{i}.txt' for i in range(1, 9)],
    'rc2': [f'rc20{i}.txt' for i in range(1, 9)],
}


def build_experiment_suite(suite_name: str = 'core30', runs_per_instance: int = 3):
    suite_name = (suite_name or 'core30').lower()

    if suite_name == 'core4':
        return [(t, inst, runs_per_instance) for t, inst, _ in CORE4_SUITE]

    if suite_name == 'core30':
        return [(t, inst, runs_per_instance) for t, inst, _ in CORE30_SUITE]

    if suite_name == 'full56':
        suite = []
        for dataset_type, instances in FULL56_GROUPS.items():
            for inst in instances:
                suite.append((dataset_type, inst, runs_per_instance))
        return suite

    raise ValueError(f"不支持的 suite: {suite_name}")


def run_batch_aco_experiments(
    n_ants: int = 40,
    max_iter: int = 300,
    alpha: float = 1.0,
    beta: float = 3.0,
    rho: float = 0.15,
    q0: float = 0.15,
    ls_enabled: bool = True,
    ls_steps: int = 25,
    suite_name: str = 'core30',
    runs_per_instance: int = 1,
    only_c2: bool = False,
    save_history: bool = True,
    make_plots: bool = True,
    inner_progress: bool = False,
    output_dir: str = "experiments_aco_baseline",
):
    print("=" * 70)
    print("ACO 基线对比实验")
    print(f"蚂蚁数={n_ants} | 迭代={max_iter} | alpha={alpha} | beta={beta} | rho={rho} | q0={q0}")
    print(f"局部搜索={ls_enabled} | ls_steps={ls_steps}")
    print(f"内层迭代进度条={inner_progress}")
    print("=" * 70)
    print()

    out_path = Path(output_dir)
    results_dir = out_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset_base = Path("data/1 Solomon Benchmark")

    suite = build_experiment_suite(suite_name, runs_per_instance)
    if only_c2:
        suite = [(t, inst, runs) for t, inst, runs in suite if (t == 'c2' and inst in {'c201.txt', 'c205.txt'})]

    total_runs = sum(runs for _, _, runs in suite)
    print(f"实例集: {suite_name} | 每实例运行次数: {runs_per_instance}")
    print(f"总计: {len(suite)} 个数据集, {total_runs} 次运行")
    print()

    all_results = []

    run_tasks = []
    for dataset_type, instance, num_runs in suite:
        instance_name = instance.replace('.txt', '')
        for run_id in range(1, num_runs + 1):
            run_tasks.append((dataset_type, instance, instance_name, run_id))

    if tqdm is not None:
        task_iter = tqdm(run_tasks, total=len(run_tasks), desc="ACO Batch", ncols=100)
    else:
        task_iter = run_tasks

    exp_idx = 0

    for dataset_type, instance, instance_name, run_id in task_iter:
        dataset_path = dataset_base / dataset_type / instance
        if not dataset_path.exists():
            print(f"[SKIP] 数据集不存在: {dataset_path}")
            continue

        exp_idx += 1
        exp_name = f"{dataset_type}_{instance_name}_run{run_id}"
        seed = build_stable_seed(instance_name, run_id)
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
            verbose=True,
        )

        if result['success']:
            print(
                f"  ✓ {exp_name} | Cost={result['best_cost']:.2f} | "
                f"Routes={result['num_routes']} | Time={result['elapsed_time']:.1f}s | "
                f"Iters={result['total_iterations']}"
            )
        else:
            print(f"  ✗ {exp_name} 失败: {result['output'][:200]}")

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

        if save_history:
            result_data['best_cost_history'] = result.get('best_cost_history', [])

        all_results.append(result_data)

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

    summary_file = out_path / "all_results.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    if make_plots:
        try:
            plot_aco_convergence(all_results, out_path)
        except Exception as e:
            print(f"[WARN] 收敛图生成失败: {e}")

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


def plot_aco_convergence(all_results: list[dict], out_path: Path):
    """基于 best_cost_history 绘制收敛曲线图。"""
    grouped: dict[str, list[list[float]]] = {}
    for r in all_results:
        if not r.get('success'):
            continue
        hist = r.get('best_cost_history', [])
        if not hist:
            continue
        inst = f"{r.get('dataset_type', '')}_{r.get('instance', '')}"
        grouped.setdefault(inst, []).append([float(x) for x in hist])

    if not grouped:
        print("[WARN] 无可用 best_cost_history，跳过收敛图")
        return

    figures_dir = out_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 图1：各实例分面图（每次run细线 + 均值粗线）
    instances = sorted(grouped.keys())
    fig, axes = plt.subplots(len(instances), 1, figsize=(10, 4.2 * len(instances)), squeeze=False)

    for i, inst in enumerate(instances):
        ax = axes[i][0]
        hists = grouped[inst]
        min_len = min(len(h) for h in hists)
        arr = np.array([h[:min_len] for h in hists], dtype=float)
        x = np.arange(1, min_len + 1)

        for row in arr:
            ax.plot(x, row, linewidth=1.0, alpha=0.35, color="#4C78A8")

        mean_curve = arr.mean(axis=0)
        ax.plot(x, mean_curve, linewidth=2.2, color="#0B3C6D", label="Mean best cost")
        ax.set_title(f"ACO+LS Convergence - {inst}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best cost")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)

    fig.tight_layout()
    fig1 = figures_dir / "aco_ls_convergence_by_instance.png"
    fig.savefig(fig1, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 图2：均值±标准差
    fig, ax = plt.subplots(figsize=(10, 5.6))
    color_pool = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, inst in enumerate(instances):
        hists = grouped[inst]
        min_len = min(len(h) for h in hists)
        arr = np.array([h[:min_len] for h in hists], dtype=float)
        x = np.arange(1, min_len + 1)
        m = arr.mean(axis=0)
        s = arr.std(axis=0)
        c = color_pool[idx % len(color_pool)]
        ax.plot(x, m, linewidth=2.0, color=c, label=f"{inst} mean")
        ax.fill_between(x, m - s, m + s, color=c, alpha=0.15)

    ax.set_title("ACO+LS Convergence (Mean ± Std)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best cost")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig2 = figures_dir / "aco_ls_convergence_mean_std.png"
    fig.savefig(fig2, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"✓ Saved figure: {fig1}")
    print(f"✓ Saved figure: {fig2}")


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

    if result['success']:
        out_dir = Path("experiments_aco_baseline/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        exp_name = f"{dataset_type}_{instance}_run1"
        seed = build_stable_seed(instance, 1)
        result_data = {
            'exp_name': exp_name,
            'dataset_type': dataset_type,
            'instance': instance,
            'run_id': 1,
            'seed': seed,
            'algorithm': 'ACO',
            'n_ants': n_ants,
            'max_iter': max_iter,
            'alpha': 1.0,
            'beta': 3.0,
            'rho': 0.15,
            'q0': 0.15,
            'ls_enabled': ls_enabled,
            'ls_steps': ls_steps,
            'timestamp': datetime.now().isoformat(),
            'success': True,
            'best_cost': result['best_cost'],
            'num_routes': result['num_routes'],
            'elapsed_time': result['elapsed_time'],
            'n_customers': result['n_customers'],
            'total_iterations': result['total_iterations'],
            'best_cost_history': result.get('best_cost_history', [])
        }
        out_file = out_dir / f"{exp_name}.json"
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        print(f"\n[INFO] 运行结果已成功保存至: {out_file}")


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

    parser.add_argument('--all', action='store_true', help='运行批量实验')
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
    parser.add_argument('--suite', default='core30', choices=['core4', 'core30', 'full56'],
                        help='批量实例集: core30(默认) / core4 / full56')
    parser.add_argument('--runs', type=int, default=1, help='每个实例运行次数 (默认1)')
    parser.add_argument('--only-c2', action='store_true', help='批量模式仅跑 c2/c201 和 c2/c205')
    parser.add_argument('--no-save-history', action='store_true', help='不保存收敛历史')
    parser.add_argument('--no-plot', action='store_true', help='不生成收敛可视化图')
    parser.add_argument('--inner-progress', action='store_true', help='显示每个run的迭代进度条')
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
            suite_name=args.suite,
            runs_per_instance=args.runs,
            only_c2=args.only_c2,
            save_history=not args.no_save_history,
            make_plots=not args.no_plot,
            inner_progress=args.inner_progress,
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
