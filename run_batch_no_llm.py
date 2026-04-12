"""
省Token批量实验脚本 - 使用固定算子代码运行多数据集实验
Zero Token Batch Experiment Runner

原理：
1. 使用预定义的HeuristicPlugin代码（不调用LLM）
2. 在多个Solomon数据集上运行
3. 每个数据集多次运行以测试稳定性

这样可以生成大量实验数据，但不消耗任何LLM token！
"""
import os
import sys
import json
import time
import random
import copy
import math
import re
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from heuristic_skeleton import HEURISTIC_SKELETON
from utils import FreshnessAndPenaltyCalculator

# ============================================================
# 预定义的HeuristicPlugin代码（基于LLM生成的优质版本）
# 这是一套经过验证的ALNS算子，不需要每次调用LLM生成
# ============================================================

FIXED_PLUGIN_CODE = '''
class HeuristicPlugin:
    def __init__(self, *args, **kwargs):
        _data = kwargs.get('data', {})
        self.capacity = _data.get('vehicle_capacity', 200)
        self.customers = _data.get('customers', [])
        self.dist_matrix = None
        self.solver = None
        self.customer_lookup = {c['id']: c for c in self.customers}

    def random_removal(self, solution, ratio):
        """随机移除算子：随机选择节点移除"""
        if not solution:
            return solution, []
        
        all_nodes = []
        for ri, route in enumerate(solution):
            for pi, node in enumerate(route):
                if node != 0:
                    all_nodes.append((ri, pi, node))
        
        if not all_nodes:
            return solution, []
        
        total = len(all_nodes)
        k = max(1, int(total * ratio)) if ratio <= 1 else min(total, int(ratio))
        k = min(k, total)
        
        selected = random.sample(all_nodes, k)
        selected.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        new_solution = [r[:] for r in solution]
        removed = []
        
        for ri, pi, node in selected:
            del new_solution[ri][pi]
            removed.append(node)
        
        new_solution = [r for r in new_solution if len(r) > 2]
        return new_solution, removed

    def route_removal(self, solution, ratio):
        """路径移除算子：移除整条路径"""
        if not solution:
            return solution, []
        
        non_empty = [(i, [n for n in r if n != 0]) for i, r in enumerate(solution) if any(n != 0 for n in r)]
        if not non_empty:
            return solution, []
        
        idx, customers = random.choice(non_empty)
        new_solution = [r[:] for i, r in enumerate(solution) if i != idx]
        return new_solution, customers

    def string_removal(self, solution, ratio):
        """连续节点移除算子：移除路径中连续的节点"""
        if not solution:
            return solution, []
        
        valid_routes = [(i, r) for i, r in enumerate(solution) if len([n for n in r if n != 0]) >= 2]
        if not valid_routes:
            return self.random_removal(solution, ratio)
        
        route_idx, route = random.choice(valid_routes)
        customers = [n for n in route if n != 0]
        
        string_len = max(2, int(len(customers) * ratio))
        string_len = min(string_len, len(customers))
        
        start = random.randint(0, len(customers) - string_len)
        removed = customers[start:start + string_len]
        
        new_solution = [r[:] for r in solution]
        new_route = [0] + [n for n in customers if n not in removed] + [0]
        
        if len(new_route) <= 2:
            new_solution = [r for i, r in enumerate(new_solution) if i != route_idx]
        else:
            new_solution[route_idx] = new_route
        
        return new_solution, removed

    def greedy_insert(self, solution, removed_nodes):
        """贪心插入算子：使用完整成本计算"""
        if not removed_nodes:
            return solution
        if not solution:
            return [[0, node, 0] for node in removed_nodes]
        
        remaining = list(removed_nodes)
        
        for node in removed_nodes:
            node_demand = self.customer_lookup.get(node, {}).get('demand', 0)
            best_cost_inc = float('inf')
            best_route_idx = None
            best_pos = None
            
            for ri, route in enumerate(solution):
                current_load = sum(self.customer_lookup.get(n, {}).get('demand', 0) for n in route if n != 0)
                if current_load + node_demand > self.capacity:
                    continue
                
                route_nodes = [self.solver.id_to_customer[n] for n in route]
                cost_before = self.solver.calculator.calculate_route_cost(route_nodes, self.dist_matrix)['variable_cost']
                
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [node] + route[pos:]
                    new_route_nodes = [self.solver.id_to_customer[n] for n in new_route]
                    cost_after = self.solver.calculator.calculate_route_cost(new_route_nodes, self.dist_matrix)['variable_cost']
                    cost_inc = cost_after - cost_before
                    
                    if cost_inc < best_cost_inc:
                        best_cost_inc = cost_inc
                        best_route_idx = ri
                        best_pos = pos
            
            if best_route_idx is not None:
                solution[best_route_idx] = solution[best_route_idx][:best_pos] + [node] + solution[best_route_idx][best_pos:]
                remaining.remove(node)
            else:
                solution.append([0, node, 0])
                remaining.remove(node)
        
        return solution

    def regret_insert(self, solution, removed_nodes):
        """后悔插入算子：基于后悔值选择插入"""
        if not removed_nodes:
            return solution
        if not solution:
            return [[0, node, 0] for node in removed_nodes]
        
        remaining = list(removed_nodes)
        
        while remaining:
            best_regret = -float('inf')
            best_node = None
            best_route_idx = None
            best_pos = None
            
            for node in remaining:
                node_demand = self.customer_lookup.get(node, {}).get('demand', 0)
                insert_costs = []
                
                for ri, route in enumerate(solution):
                    current_load = sum(self.customer_lookup.get(n, {}).get('demand', 0) for n in route if n != 0)
                    if current_load + node_demand > self.capacity:
                        continue
                    
                    route_nodes = [self.solver.id_to_customer[n] for n in route]
                    cost_before = self.solver.calculator.calculate_route_cost(route_nodes, self.dist_matrix)['variable_cost']
                    
                    best_pos_cost = float('inf')
                    best_pos_in_route = 1
                    
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [node] + route[pos:]
                        new_route_nodes = [self.solver.id_to_customer[n] for n in new_route]
                        cost_after = self.solver.calculator.calculate_route_cost(new_route_nodes, self.dist_matrix)['variable_cost']
                        cost_inc = cost_after - cost_before
                        
                        if cost_inc < best_pos_cost:
                            best_pos_cost = cost_inc
                            best_pos_in_route = pos
                    
                    insert_costs.append((best_pos_cost, ri, best_pos_in_route))
                
                if not insert_costs:
                    new_route_cost = self.dist_matrix[0][node] * 2 if self.dist_matrix else 100
                    insert_costs.append((new_route_cost, -1, 1))
                
                insert_costs.sort(key=lambda x: x[0])
                
                if len(insert_costs) >= 2:
                    regret = insert_costs[1][0] - insert_costs[0][0]
                else:
                    regret = 0
                
                if regret > best_regret:
                    best_regret = regret
                    best_node = node
                    best_route_idx = insert_costs[0][1]
                    best_pos = insert_costs[0][2]
            
            if best_node is None:
                break
            
            if best_route_idx == -1:
                solution.append([0, best_node, 0])
            else:
                solution[best_route_idx] = solution[best_route_idx][:best_pos] + [best_node] + solution[best_route_idx][best_pos:]
            
            remaining.remove(best_node)
        
        return solution
'''


def load_solomon_data(file_path):
    """加载Solomon数据集"""
    data = {}
    customers = []
    vehicle_capacity = None
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if line == "" or line.startswith("C") or line.startswith("VEHICLE") or line.startswith("NUMBER") or line.startswith("CUSTOMER"):
            continue
        parts = line.split()
        if len(parts) == 2 and vehicle_capacity is None:
            vehicle_capacity = int(parts[1])
            break
    
    if vehicle_capacity is None:
        vehicle_capacity = 200
    
    for line in lines:
        line = line.strip()
        if line == "" or line.startswith("C") or line.startswith("VEHICLE") or line.startswith("NUMBER") or line.startswith("CUSTOMER") or line.startswith("CUST"):
            continue
        parts = line.split()
        if parts[0].isdigit() and len(parts) >= 7:
            cust_id = int(parts[0])
            customers.append({
                "id": cust_id,
                "demand": int(parts[3]),
                "x": float(parts[1]),
                "y": float(parts[2]),
                "ready_time": float(parts[4]),
                "due_date": float(parts[5]),
                "service_time": float(parts[6])
            })
    
    data["vehicle_capacity"] = vehicle_capacity
    data["customers"] = customers
    return data


def run_single_experiment(dataset_path, max_iters=1000, seed=None):
    """运行单个实验（不调用LLM）"""
    if seed is not None:
        random.seed(seed)
    
    # 加载数据
    dataset = load_solomon_data(dataset_path)
    
    if dataset['customers']:
        depot = dataset['customers'][0]
        for cust in dataset['customers']:
            cust['E_i'] = max(0, cust['ready_time']) if cust['id'] != 0 else 0
            cust['L_i'] = min(depot['due_date'], cust['due_date']) if cust['id'] != 0 else depot['due_date']
    
    # 构建完整代码 - 添加必需的 FreshnessAndPenaltyCalculator 类
    utils_code = '''
import math

class FreshnessAndPenaltyCalculator:
    def __init__(self, config):
        def safe_get(cfg, key, default):
            if isinstance(cfg, dict):
                return cfg.get(key, default)
            elif hasattr(cfg, key):
                return getattr(cfg, key, default)
            else:
                return default
        
        self.f = safe_get(config, "vehicle_fixed_cost", 240)
        self.c = safe_get(config, "vehicle_distance_cost_per_km", 3)
        self.ct = safe_get(config, "cooling_cost_per_hour", 15)
        self.v = safe_get(config, "vehicle_speed_kmph", 40)
        self.p = safe_get(config, "product_price_per_ton", 5000)
        self.theta1 = safe_get(config, "theta_transport", 0.002)
        self.theta2 = safe_get(config, "theta_service", 0.005)
        self.delta1 = safe_get(config, "customer_loss_threshold", 0.02)
        self.z1 = safe_get(config, "early_penalty_per_hour", 20)
        self.z2 = safe_get(config, "late_penalty_per_hour", 40)

    def calculate_route_cost(self, route_nodes, dist_matrix):
        route_dist = 0.0
        c2_freshness = 0.0
        c3_penalty = 0.0
        curr_time = 0.0
        cum_service_h = 0.0

        for i in range(1, len(route_nodes)):
            prev = route_nodes[i - 1]
            curr = route_nodes[i]
            d = dist_matrix[prev['id']][curr['id']]
            route_dist += d
            drive_min = d * (60.0 / self.v)
            curr_time += drive_min

            if curr['id'] != 0:
                tik_h = curr_time / 60.0
                ri = 1 - math.exp(-self.theta1 * (tik_h - cum_service_h) - self.theta2 * cum_service_h)
                c2_freshness += self.p * curr['demand'] * max(ri - self.delta1, 0)

            if curr['id'] != 0:
                fi_t = 0.0
                ei, li = curr['ready_time'], curr['due_date']
                Ei, Li = curr['E_i'], curr['L_i']
                if Ei <= curr_time < ei:
                    fi_t = self.z1 * (ei - curr_time) / (ei - Ei + 1e-6)
                elif li < curr_time <= Li:
                    fi_t = self.z2 * (curr_time - li) / (Li - li + 1e-6)
                elif curr_time < Ei or curr_time > Li:
                    fi_t = 300.0
                c3_penalty += fi_t

            curr_time = max(curr_time, curr['ready_time']) + curr['service_time']
            cum_service_h += curr['service_time'] / 60.0

        c12 = route_dist * self.c
        total_drive_h = (curr_time / 60.0) - cum_service_h
        c13 = total_drive_h * self.ct

        return {
            "variable_cost": c12 + c13 + c2_freshness + c3_penalty,
            "c2": c2_freshness,
            "c3": c3_penalty,
            "dist": route_dist
        }
'''
    
    full_code = (
        "# -*- coding: utf-8 -*-\n"
        "import sys\n"
        "import io\n"
        "sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')\n"
        "sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')\n\n"
    ) + utils_code + "\n\n" + HEURISTIC_SKELETON.replace("from utils import FreshnessAndPenaltyCalculator", "# FreshnessAndPenaltyCalculator already defined above") + "\n\n" + FIXED_PLUGIN_CODE + "\n\n"
    full_code += (
        "if __name__ == '__main__':\n"
        "    import sys, traceback\n"
        f"    data = {repr(dataset)}\n"
        "    try:\n"
        "        plugin = HeuristicPlugin(data=data)\n"
        "        solver = HeuristicSolver(data, plugin)\n"
        f"        best_sol, best_cost = solver.solve(max_iters={max_iters})\n"
        "        print(f'BEST_COST: {best_cost}')\n"
        "        print(f'NUM_ROUTES: {len(best_sol)}')\n"
        "        for i, route in enumerate(best_sol):\n"
        "            print(f'Route {i+1}: {route}')\n"
        "    except Exception as e:\n"
        "        print(f'ERROR: {e}')\n"
        "        traceback.print_exc()\n"
    )
    
    # 执行代码
    import subprocess
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(full_code)
        temp_file = f.name
    
    try:
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=1800,  # 增加到30分钟（r类数据集更难）
            encoding='utf-8',
            errors='replace',
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        elapsed_time = time.time() - start_time
        
        output = result.stdout + result.stderr
        
        # 提取结果
        best_cost = None
        num_routes = 0
        
        cost_match = re.search(r'BEST_COST:\s*([\d.]+)', output)
        if cost_match:
            best_cost = float(cost_match.group(1))
        
        route_match = re.search(r'NUM_ROUTES:\s*(\d+)', output)
        if route_match:
            num_routes = int(route_match.group(1))
        
        success = best_cost is not None
        
        return {
            'success': success,
            'best_cost': best_cost,
            'num_routes': num_routes,
            'elapsed_time': elapsed_time,
            'output': output[:2000]  # 截断
        }
        
    except subprocess.TimeoutExpired:
        return {'success': False, 'best_cost': None, 'num_routes': 0, 'elapsed_time': 600, 'output': 'TIMEOUT'}
    except Exception as e:
        return {'success': False, 'best_cost': None, 'num_routes': 0, 'elapsed_time': 0, 'output': str(e)}
    finally:
        os.unlink(temp_file)


def run_batch_experiments():
    """运行批量实验"""
    print("=" * 70)
    print("省Token批量实验 - Zero Token Batch Experiments")
    print("使用固定算子代码，不调用LLM")
    print("=" * 70)
    print()
    
    # 创建输出目录
    output_dir = Path("experiments_batch")
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 实验配置
    dataset_base = Path("data/1 Solomon Benchmark")
    
    # Solomon数据集类型和实例
    experiments = [
        # (数据集类型, 实例名, 运行次数)
        ('c1', 'c101.txt', 3),
        ('c1', 'c105.txt', 3),
        ('c2', 'c201.txt', 3),
        ('c2', 'c205.txt', 3),
        ('r1', 'r101.txt', 3),
        ('r1', 'r105.txt', 3),
        ('r2', 'r201.txt', 3),
        ('r2', 'r205.txt', 3),
        ('rc1', 'rc101.txt', 3),
        ('rc1', 'rc105.txt', 3),
        ('rc2', 'rc201.txt', 3),
        ('rc2', 'rc205.txt', 3),
    ]
    
    # 计算总实验数
    total_runs = sum(runs for _, _, runs in experiments)
    print(f"总计: {len(experiments)} 个数据集, {total_runs} 次运行")
    print(f"预计时间: {total_runs * 2}-{total_runs * 5} 分钟")
    print()
    print("开始实验...")
    print()
    
    all_results = []
    exp_idx = 0
    
    for dataset_type, instance, num_runs in experiments:
        dataset_path = dataset_base / dataset_type / instance
        
        if not dataset_path.exists():
            print(f"[跳过] 数据集不存在: {dataset_path}")
            continue
        
        instance_name = instance.replace('.txt', '')
        
        for run_id in range(1, num_runs + 1):
            exp_idx += 1
            exp_name = f"{dataset_type}_{instance_name}_run{run_id}"
            
            print(f"[{exp_idx}/{total_runs}] {exp_name}")
            print(f"  数据集: {dataset_type}/{instance}")
            
            # 使用不同的随机种子
            seed = run_id * 42 + hash(instance_name) % 1000
            
            # r1/r2/rc数据集更难，需要更多迭代
            iters = 2000 if dataset_type in ['r1', 'r2', 'rc1', 'rc2'] else 1000
            result = run_single_experiment(str(dataset_path), max_iters=iters, seed=seed)
            
            if result['success']:
                print(f"  ✓ Cost={result['best_cost']:.2f}, Routes={result['num_routes']}, Time={result['elapsed_time']:.1f}s")
            else:
                print(f"  ✗ 失败")
            
            # 保存结果
            result_data = {
                'exp_name': exp_name,
                'dataset_type': dataset_type,
                'instance': instance_name,
                'run_id': run_id,
                'seed': seed,
                'timestamp': datetime.now().isoformat(),
                **result
            }
            
            all_results.append(result_data)
            
            # 单独保存每个实验结果
            result_file = results_dir / f"{exp_name}.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    # 保存汇总结果
    summary_file = output_dir / "all_results.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print()
    print("=" * 70)
    print("实验完成!")
    print(f"成功: {sum(1 for r in all_results if r['success'])} / {len(all_results)}")
    print(f"结果保存到: {results_dir}")
    print()
    print("下一步: 运行分析脚本")
    print("  python result_analyzer_batch.py")
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    run_batch_experiments()
