# -*- coding: utf-8 -*-
"""
Customer Scale Parameter Tuning Experiment

1. Uses pre-defined HeuristicPlugin code (no LLM calls)
2. Subsets Solomon Benchmark to different customer counts
3. Runs on 3 typical datasets (c101, r101, rc101)
4. Each config runs 3 times for stability

Zero LLM Token cost!
"""
import sys
import io
# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass
import os
import sys
import json
import time
import random
import copy
import math
import re
import hashlib
from pathlib import Path
from datetime import datetime
import subprocess
import tempfile

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from heuristic_skeleton import HEURISTIC_SKELETON

# ============================================================
# 预定义的HeuristicPlugin代码（与 run_batch_no_llm.py 保持一致）
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
        """随机移除算子"""
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
        """路径移除算子"""
        if not solution:
            return solution, []
        
        non_empty = [(i, [n for n in r if n != 0]) for i, r in enumerate(solution) if any(n != 0 for n in r)]
        if not non_empty:
            return solution, []
        
        idx, customers = random.choice(non_empty)
        new_solution = [r[:] for i, r in enumerate(solution) if i != idx]
        return new_solution, customers

    def string_removal(self, solution, ratio):
        """连续节点移除算子"""
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
        """后悔插入算子"""
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

# ============================================================
# FreshnessAndPenaltyCalculator 内联代码（避免import依赖）
# ============================================================

UTILS_CODE = '''
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


def subset_solomon_data(data, n_customers):
    """
    从完整Solomon数据中截取前 n_customers 个客户
    
    Args:
        data: 完整数据集（包含depot + 100个客户）
        n_customers: 截取的客户数量（不含depot）
    
    Returns:
        截取后的数据集，客户ID重新映射为 0~n_customers
    """
    depot = data['customers'][0]  # depot (id=0) 始终保留
    non_depot = data['customers'][1:]  # 排除 depot
    
    # 截取前 n_customers 个客户
    selected = non_depot[:n_customers]
    
    # 重新映射客户ID（确保连续: 0, 1, 2, ..., n_customers）
    id_mapping = {depot['id']: 0}
    new_customers = [dict(depot, id=0)]  # depot 保持 id=0
    
    for new_id, cust in enumerate(selected, start=1):
        id_mapping[cust['id']] = new_id
        new_cust = dict(cust, id=new_id)
        new_customers.append(new_cust)
    
    new_data = {
        'vehicle_capacity': data['vehicle_capacity'],
        'customers': new_customers
    }
    return new_data


def patch_time_windows(data):
    """为数据添加 E_i / L_i 时间窗字段"""
    if data['customers']:
        depot = data['customers'][0]
        for cust in data['customers']:
            cust['E_i'] = max(0, cust['ready_time']) if cust['id'] != 0 else 0
            cust['L_i'] = min(depot['due_date'], cust['due_date']) if cust['id'] != 0 else depot['due_date']


def run_single_experiment(dataset_path, n_customers, max_iters=1000, seed=None):
    """
    运行单个客户数调参实验（不调用LLM）
    
    Args:
        dataset_path: Solomon数据集路径
        n_customers: 使用的客户数量（截取子集）
        max_iters: ALNS最大迭代次数
        seed: 随机种子
    """
    if seed is not None:
        random.seed(seed)
    
    # 加载并截取数据
    full_data = load_solomon_data(dataset_path)
    
    if n_customers >= len(full_data['customers']) - 1:
        # 使用全部客户
        dataset = full_data
        actual_n = len(full_data['customers']) - 1
    else:
        dataset = subset_solomon_data(full_data, n_customers)
        actual_n = n_customers
    
    # 添加时间窗字段
    patch_time_windows(dataset)
    
    # 构建完整代码
    full_code = (
        "# -*- coding: utf-8 -*-\n"
        "import sys\n"
        "import io\n"
        "sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')\n"
        "sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')\n\n"
    )
    full_code += UTILS_CODE + "\n\n"
    full_code += HEURISTIC_SKELETON.replace(
        "from utils import FreshnessAndPenaltyCalculator",
        "# FreshnessAndPenaltyCalculator already defined above"
    ) + "\n\n"
    full_code += FIXED_PLUGIN_CODE + "\n\n"
    full_code += (
        "if __name__ == '__main__':\n"
        "    import sys, traceback, time\n"
        f"    data = {repr(dataset)}\n"
        "    try:\n"
        "        plugin = HeuristicPlugin(data=data)\n"
        "        solver = HeuristicSolver(data, plugin)\n"
        f"        best_sol, best_cost = solver.solve(max_iters={max_iters})\n"
        "        print(f'BEST_COST: {best_cost}')\n"
        "        print(f'NUM_ROUTES: {len(best_sol)}')\n"
        f"        print(f'NUM_CUSTOMERS: {actual_n}')\n"
        "        for i, route in enumerate(best_sol):\n"
        "            print(f'Route {i+1}: {route}')\n"
        "    except Exception as e:\n"
        "        print(f'ERROR: {e}')\n"
        "        traceback.print_exc()\n"
    )
    
    # 执行代码
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(full_code)
        temp_file = f.name
    
    try:
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=1800,  # 30分钟超时
            encoding='utf-8',
            errors='replace',
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        elapsed_time = time.time() - start_time
        
        output = result.stdout + result.stderr
        
        # 提取结果
        best_cost = None
        num_routes = 0
        last_improve_iter = 0
        
        cost_match = re.search(r'BEST_COST:\s*([\d.]+)', output)
        if cost_match:
            best_cost = float(cost_match.group(1))
        
        route_match = re.search(r'NUM_ROUTES:\s*(\d+)', output)
        if route_match:
            num_routes = int(route_match.group(1))
        
        # 提取最后改进迭代次数
        improve_match = re.search(r'最后改进:\s*第\s*(\d+)\s*次迭代', output)
        if improve_match:
            last_improve_iter = int(improve_match.group(1))
        
        # 提取改进次数
        improve_count = 0
        improve_count_match = re.search(r'改进次数:\s*(\d+)', output)
        if improve_count_match:
            improve_count = int(improve_count_match.group(1))
        
        success = best_cost is not None
        
        return {
            'success': success,
            'best_cost': best_cost,
            'num_routes': num_routes,
            'elapsed_time': round(elapsed_time, 2),
            'last_improve_iter': last_improve_iter,
            'improve_count': improve_count,
            'n_customers': actual_n,
            'cost_per_customer': round(best_cost / actual_n, 2) if best_cost and actual_n > 0 else None,
            'output': output[:3000]  # 截断保存
        }
        
    except subprocess.TimeoutExpired:
        return {
            'success': False, 'best_cost': None, 'num_routes': 0,
            'elapsed_time': 1800, 'last_improve_iter': 0, 'improve_count': 0,
            'n_customers': n_customers, 'cost_per_customer': None,
            'output': 'TIMEOUT'
        }
    except Exception as e:
        return {
            'success': False, 'best_cost': None, 'num_routes': 0,
            'elapsed_time': 0, 'last_improve_iter': 0, 'improve_count': 0,
            'n_customers': n_customers, 'cost_per_customer': None,
            'output': str(e)
        }
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def run_customer_scale_experiments():
    """运行客户数调参批量实验"""
    print("=" * 70)
    print("客户数/订单数调参实验 - Customer Scale Parameter Tuning")
    print("使用固定算子代码，不调用LLM，零Token消耗")
    print("=" * 70)
    print()
    
    # 创建输出目录
    output_dir = Path("experiments_customer_scale")
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # ============================================================
    # 实验配置
    # ============================================================
    
    # 客户数梯度
    customer_counts = [15, 25, 50, 75, 100]
    
    # 测试数据集（3种典型分布）
    dataset_base = Path("data/1 Solomon Benchmark")
    datasets = [
        ('c1', 'c101.txt', 'Clustered+Narrow TW'),
        ('r1', 'r101.txt', 'Random+Narrow TW'),
        ('rc1', 'rc101.txt', 'Mixed+Narrow TW'),
    ]

    def build_stable_seed(instance_name: str, n_customers: int, run_id: int, base: int = 42) -> int:
        """构建跨进程稳定种子，避免使用内置 hash 导致重复实验不可复现。"""
        key = f"{instance_name}_{n_customers}"
        digest = hashlib.md5(key.encode('utf-8')).hexdigest()
        bucket = int(digest[:8], 16) % 10000
        return run_id * base + bucket
    
    # 每个配置运行次数
    num_runs = 3
    
    # ALNS 迭代次数（根据客户数自适应调整）
    def get_max_iters(n_customers):
        """根据客户数调整迭代次数"""
        if n_customers <= 25:
            return 500   # 小规模不需要太多迭代
        elif n_customers <= 50:
            return 800
        elif n_customers <= 75:
            return 1200  # 75规模更容易早熟，适当增加迭代预算
        else:
            return 1400  # 100规模需要更多迭代

    def is_premature_convergence(result: dict, n_customers: int, max_iters: int) -> bool:
        """判断是否出现早熟收敛（用于n=75波动修复）。"""
        if not result.get('success', False):
            return False
        if n_customers < 75:
            return False
        last_improve = int(result.get('last_improve_iter', 0) or 0)
        improve_count = int(result.get('improve_count', 0) or 0)
        # 最后改进发生得过早，且改进次数偏少，视为可疑早熟
        return last_improve > 0 and last_improve < max(40, int(0.08 * max_iters)) and improve_count <= 20
    
    # 计算总实验数
    total_experiments = len(customer_counts) * len(datasets) * num_runs
    print(f"实验配置:")
    print(f"  客户数梯度: {customer_counts}")
    print(f"  数据集: {[d[0]+'/'+d[1] for d in datasets]}")
    print(f"  每配置运行次数: {num_runs}")
    print(f"  总实验数: {total_experiments}")
    print(f"  预计时间: {total_experiments * 1.5:.0f} - {total_experiments * 8:.0f} 分钟")
    print()
    
    # ============================================================
    # 运行实验
    # ============================================================
    
    all_results = []
    exp_idx = 0
    failed_count = 0
    
    for dataset_type, instance, desc in datasets:
        dataset_path = dataset_base / dataset_type / instance
        
        if not dataset_path.exists():
            print(f"[跳过] 数据集不存在: {dataset_path}")
            continue
        
        instance_name = instance.replace('.txt', '')
        
        print(f"\n{'─' * 60}")
        print(f"数据集: {dataset_type}/{instance} ({desc})")
        print(f"{'─' * 60}")
        
        for n_customers in customer_counts:
            for run_id in range(1, num_runs + 1):
                exp_idx += 1
                exp_name = f"scale_{dataset_type}_{instance_name}_n{n_customers}_run{run_id}"
                
                max_iters = get_max_iters(n_customers)
                seed = build_stable_seed(instance_name, n_customers, run_id)
                
                print(f"\n  [{exp_idx}/{total_experiments}] {exp_name}")
                print(f"    客户数: {n_customers}, 迭代: {max_iters}, 种子: {seed}")
                
                result = run_single_experiment(
                    str(dataset_path),
                    n_customers=n_customers,
                    max_iters=max_iters,
                    seed=seed
                )

                # 针对 n=75/100 的异常波动：检测早熟收敛后进行最多2次重启，取最优
                rerun_count = 0
                selected_seed = seed
                best_result = result
                while is_premature_convergence(best_result, n_customers, max_iters) and rerun_count < 2:
                    rerun_count += 1
                    retry_seed = seed + 9973 * rerun_count
                    print(f"    [RETRY] 检测到早熟收敛，重启第{rerun_count}次，seed={retry_seed}")
                    retry_result = run_single_experiment(
                        str(dataset_path),
                        n_customers=n_customers,
                        max_iters=max_iters,
                        seed=retry_seed
                    )
                    if retry_result.get('success') and (
                        (not best_result.get('success')) or
                        (retry_result.get('best_cost') is not None and best_result.get('best_cost') is not None and retry_result['best_cost'] < best_result['best_cost'])
                    ):
                        best_result = retry_result
                        selected_seed = retry_seed

                result = best_result
                
                if result['success']:
                    print(f"    [OK] Cost={result['best_cost']:.2f}, "
                          f"Routes={result['num_routes']}, "
                          f"Cost/Cust={result['cost_per_customer']:.2f}, "
                          f"Time={result['elapsed_time']:.1f}s, "
                          f"LastImprove=Iter{result['last_improve_iter']}, "
                          f"Retries={rerun_count}")
                else:
                    failed_count += 1
                    print(f"    [FAIL]")
                
                # 保存结果
                result_data = {
                    'exp_name': exp_name,
                    'dataset_type': dataset_type,
                    'instance': instance_name,
                    'n_customers': n_customers,
                    'run_id': run_id,
                    'seed': selected_seed,
                    'max_iters': max_iters,
                    'retry_count': rerun_count,
                    'timestamp': datetime.now().isoformat(),
                    **result
                }
                
                all_results.append(result_data)
                
                # 单独保存每个实验结果
                result_file = results_dir / f"{exp_name}.json"
                with open(result_file, 'w', encoding='utf-8') as f:
                    # 不保存冗长的 output 到单独文件
                    save_data = {k: v for k, v in result_data.items() if k != 'output'}
                    json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    # ============================================================
    # 保存汇总结果
    # ============================================================
    
    summary_file = output_dir / "all_results.json"
    
    # 汇总文件中也不保存冗长的 output
    save_results = []
    for r in all_results:
        save_r = {k: v for k, v in r.items() if k != 'output'}
        save_results.append(save_r)
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    
    # ============================================================
    # 打印汇总统计
    # ============================================================
    
    print("\n" + "=" * 70)
    print("实验完成!")
    print("=" * 70)
    print(f"  成功: {sum(1 for r in all_results if r['success'])} / {len(all_results)}")
    print(f"  失败: {failed_count}")
    print(f"  结果目录: {results_dir}")
    print(f"  汇总文件: {summary_file}")
    print()
    
    # 按客户数分组统计
    print("按客户数分组统计:")
    print(f"  {'客户数':>6}  {'平均成本':>12}  {'平均车辆':>8}  {'平均时间':>10}  {'平均成本/客户':>14}")
    print(f"  {'─'*6}  {'─'*12}  {'─'*8}  {'─'*10}  {'─'*14}")
    
    for n_c in customer_counts:
        group = [r for r in all_results if r['n_customers'] == n_c and r['success']]
        if group:
            avg_cost = sum(r['best_cost'] for r in group) / len(group)
            avg_routes = sum(r['num_routes'] for r in group) / len(group)
            avg_time = sum(r['elapsed_time'] for r in group) / len(group)
            avg_cpc = sum(r['cost_per_customer'] for r in group if r['cost_per_customer']) / len(group)
            print(f"  {n_c:>6}  {avg_cost:>12.2f}  {avg_routes:>8.1f}  {avg_time:>8.1f}s  {avg_cpc:>14.2f}")
    
    print()
    print("下一步: 运行分析脚本生成图表")
    print("  python analyze_customer_scale.py")
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    run_customer_scale_experiments()
