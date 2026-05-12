"""
run_baseline_ga.py - 遗传算法对比实验启动器
=============================================
功能:
  1. 对 Solomon Benchmark 数据集批量运行 GA 基线
  2. 输出与 run_batch_no_llm.py 格式对齐的实验结果
  3. 支持指定算例或运行全套对比实验

用法:
  # 快速单个算例测试:
  python run_baseline_ga.py --instance c101 --type c1

  # 全套对比实验 (与 ALNS 实验保持相同算例集):
  python run_baseline_ga.py --all

  # 调整 GA 参数:
  python run_baseline_ga.py --all --pop 100 --gen 800
"""

import os
import sys
import json
import time
import argparse
import math
import random
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline_ga import GAVRPSolver

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# ──────────────────────────────────────────────────
# 数据加载 (与 run_batch_no_llm.py 完全一致)
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

    # 解析客户节点
    for line in lines:
        line = line.strip()
        if line == "" or any(line.startswith(kw) for kw in ["C", "VEHICLE", "NUMBER", "CUSTOMER", "CUST"]):
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

    # 补充 E_i / L_i 字段 (与 run_batch_no_llm 保持一致)
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

def run_single_ga_experiment(
    dataset_path: str,
    pop_size: int = 80,
    max_gen: int = 500,
    cx_prob: float = 0.85,
    mut_prob: float = 0.15,
    seed: int = None,
    verbose: bool = True,
) -> dict:
    """
    对单个数据集运行 GA 实验。

    返回 dict 格式与 run_batch_no_llm.run_single_experiment 兼容:
      success, best_cost, num_routes, elapsed_time, output
    """
    try:
        data = load_solomon_data(dataset_path)
        n_cust = len([c for c in data['customers'] if c['id'] != 0])

        solver = GAVRPSolver(
            data=data,
            pop_size=pop_size,
            max_gen=max_gen,
            cx_prob=cx_prob,
            mut_prob=mut_prob,
            seed=seed,
            verbose=verbose,
        )

        t0 = time.time()
        best_solution, best_cost = solver.solve()
        elapsed = time.time() - t0

        # ── 输出对齐格式 (与 ALNS 脚本格式相同) ──
        output_lines = [
            f"BEST_COST: {best_cost:.4f}",
            f"NUM_ROUTES: {len(best_solution)}",
            f"CUSTOMERS: {n_cust}",
            f"ELAPSED: {elapsed:.2f}s",
        ]
        for i, route in enumerate(best_solution):
            output_lines.append(f"Route {i+1}: {route}")

        return {
            'success':      True,
            'best_cost':    best_cost,
            'num_routes':   len(best_solution),
            'elapsed_time': elapsed,
            'n_customers':  n_cust,
            'best_gen':     len(solver.best_cost_history),
            'best_cost_history': solver.best_cost_history,
            'output':       '\n'.join(output_lines),
        }

    except Exception as e:
        import traceback
        return {
            'success':      False,
            'best_cost':    None,
            'num_routes':   0,
            'elapsed_time': 0,
            'n_customers':  0,
            'best_gen':     0,
            'best_cost_history': [],
            'output':       f"ERROR: {e}\n{traceback.format_exc()}",
        }


# ──────────────────────────────────────────────────
# 批量实验 (与 ALNS 使用相同算例集)
# ──────────────────────────────────────────────────

# 与 run_batch_no_llm.py 保持一致的核心 30 实验配置
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
    """根据套件名构造批量实验列表。"""
    suite_name = (suite_name or 'core30').lower()

    if suite_name == 'core30':
        return [(t, inst, runs_per_instance) for t, inst, _ in CORE30_SUITE]

    if suite_name == 'full56':
        suite = []
        for dataset_type, instances in FULL56_GROUPS.items():
            for inst in instances:
                suite.append((dataset_type, inst, runs_per_instance))
        return suite

    raise ValueError(f"不支持的 suite: {suite_name}")


def run_batch_ga_experiments(
    pop_size: int = 80,
    max_gen: int = 500,
    suite_name: str = 'core30',
    runs_per_instance: int = 1,
    output_dir: str = "experiments_ga_baseline",
):
    """
    批量运行所有对比实验算例，结果存储到 experiments_ga_baseline/ 目录。
    目录结构与 experiments_batch/ 完全对齐，方便 result_analyzer_batch.py 直接读取。
    """
    print("=" * 70)
    print("GA 基线对比实验 - Genetic Algorithm Baseline")
    print(f"种群规模: {pop_size} | 最大代数: {max_gen}")
    print("=" * 70)
    print()

    out_path = Path(output_dir)
    results_dir = out_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset_base = Path("data/1 Solomon Benchmark")
    experiment_suite = build_experiment_suite(suite_name, runs_per_instance)

    total_runs = sum(runs for _, _, runs in experiment_suite)
    print(f"实例集: {suite_name} | 每实例运行次数: {runs_per_instance}")
    print(f"总计: {len(experiment_suite)} 个数据集, {total_runs} 次运行")
    print()

    all_results = []

    run_tasks = []
    for dataset_type, instance, num_runs in experiment_suite:
        instance_name = instance.replace('.txt', '')
        for run_id in range(1, num_runs + 1):
            run_tasks.append((dataset_type, instance, instance_name, run_id))

    if tqdm is not None:
        task_iter = tqdm(run_tasks, total=len(run_tasks), desc="GA Batch", ncols=100)
    else:
        task_iter = run_tasks

    exp_idx = 0

    for dataset_type, instance, instance_name, run_id in task_iter:
        dataset_path = dataset_base / dataset_type / instance
        if not dataset_path.exists():
            print(f"[跳过] 数据集不存在: {dataset_path}")
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

        result = run_single_ga_experiment(
            dataset_path=str(dataset_path),
            pop_size=pop_size,
            max_gen=max_gen,
            seed=seed,
            verbose=True,
        )

        if result['success']:
            print(f"  ✓ Cost={result['best_cost']:.2f} | "
                  f"Routes={result['num_routes']} | "
                  f"Time={result['elapsed_time']:.1f}s | "
                  f"Gen={result['best_gen']}")
        else:
            print(f"  ✗ 失败: {result['output'][:200]}")

        result_data = {
            'exp_name':     exp_name,
            'dataset_type': dataset_type,
            'instance':     instance_name,
            'run_id':       run_id,
            'seed':         seed,
            'algorithm':    'GA',
            'pop_size':     pop_size,
            'max_gen':      max_gen,
            'timestamp':    datetime.now().isoformat(),
            'success':      result['success'],
            'best_cost':    result['best_cost'],
            'num_routes':   result['num_routes'],
            'elapsed_time': result['elapsed_time'],
            'n_customers':  result['n_customers'],
            'best_gen':     result['best_gen'],
            # 不保存完整历史曲线以节省空间 (太长)
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
    print("GA 批量实验完成!")
    print(f"  成功/总计: {success_count}/{len(all_results)}")
    if valid_costs:
        print(f"  平均成本: {sum(valid_costs)/len(valid_costs):.2f}")
        print(f"  最优成本: {min(valid_costs):.2f}")
        print(f"  最差成本: {max(valid_costs):.2f}")
    print(f"  结果保存: {results_dir}")
    print()
    print("对比分析示例:")
    print("  python result_analyzer_batch.py  # 分析 ALNS 结果")
    print("  python compare_algorithms.py     # GA vs ALNS 对比图表 (待实现)")
    print("=" * 70)

    return all_results


# ──────────────────────────────────────────────────
# 快速单算例测试入口
# ──────────────────────────────────────────────────

def quick_test(dataset_type: str = 'c1', instance: str = 'c101',
               pop_size: int = 80, max_gen: int = 500, outdir: str = 'experiments_ga_baseline'):
    """快速测试单个算例，打印详细日志，并保存 json 结果到文件夹。"""
    dataset_path = Path(f"data/1 Solomon Benchmark/{dataset_type}/{instance}.txt")
    if not dataset_path.exists():
        print(f"[错误] 数据集不存在: {dataset_path}")
        return

    print(f"\n{'='*60}")
    print(f"GA 快速测试 | {dataset_type}/{instance} | pop={pop_size} | gen={max_gen}")
    print(f"{'='*60}\n")

    seed = 1 * 42 + hash(instance) % 1000
    result = run_single_ga_experiment(
        dataset_path=str(dataset_path),
        pop_size=pop_size,
        max_gen=max_gen,
        seed=seed,
        verbose=True,
    )

    print(f"\n{'─'*60}")
    print(f"BEST_COST: {result['best_cost']}")
    print(f"NUM_ROUTES: {result['num_routes']}")
    print(f"ELAPSED: {result['elapsed_time']:.2f}s")
    print(f"{'─'*60}")
    print(result['output'])

    if result['success']:
        # 写入 JSON, 参考 c2_c203_run1.json 的完整格式
        out_path = Path(outdir) / "results"
        out_path.mkdir(parents=True, exist_ok=True)
        
        json_data = {
            "exp_name": f"{dataset_type}_{instance}_run1",
            "dataset_type": dataset_type,
            "instance": instance,
            "run_id": 1,
            "seed": seed,
            "algorithm": "GA",
            "pop_size": pop_size,
            "max_gen": max_gen,
            "timestamp": datetime.now().isoformat(),
            "success": True,
            "best_cost": result["best_cost"],
            "num_routes": result["num_routes"],
            "elapsed_time": result["elapsed_time"],
            "n_customers": result["n_customers"],
            "best_gen": result["best_gen"]
        }
        
        json_file = out_path / f"{dataset_type}_{instance}_run1.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"\n[已写入] 单算例结果已保存至: {json_file}")


# ──────────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='GA 基线对比实验',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单算例快速测试 (c1类, c101实例)
  python run_baseline_ga.py --instance c101 --type c1

  # 单算例 r2 类
  python run_baseline_ga.py --instance r201 --type r2 --pop 80 --gen 800

  # 全套批量对比实验
  python run_baseline_ga.py --all

  # 全套实验 + 自定义 GA 参数
  python run_baseline_ga.py --all --pop 100 --gen 1000
        """
    )

    parser.add_argument('--all',      action='store_true', help='运行全套批量对比实验')
    parser.add_argument('--type',     default='c1',        help='数据集类型 (c1/c2/r1/r2/rc1/rc2)')
    parser.add_argument('--instance', default='c101',      help='算例名称 (不含.txt, 如 c101)')
    parser.add_argument('--pop',      type=int, default=80,  help='种群规模 (默认80)')
    parser.add_argument('--gen',      type=int, default=500, help='最大代数 (默认500)')
    parser.add_argument('--suite',    default='core30', choices=['core30', 'full56'],
                        help='批量实例集: core30(默认) 或 full56(全56算例)')
    parser.add_argument('--runs',     type=int, default=1, help='每个实例运行次数 (默认1)')
    parser.add_argument('--outdir',   default='experiments_ga_baseline', help='批量实验结果目录')

    args = parser.parse_args()

    if args.all:
        run_batch_ga_experiments(
            pop_size=args.pop,
            max_gen=args.gen,
            suite_name=args.suite,
            runs_per_instance=args.runs,
            output_dir=args.outdir,
        )
    else:
        quick_test(
            dataset_type=args.type,
            instance=args.instance,
            pop_size=args.pop,
            max_gen=args.gen,
            outdir=args.outdir,
        )
