# -*- coding: utf-8 -*-
"""
run_llm_optimize.py — LLM 迭代优化 ALNS 算子
==============================================
核心思路：运行已有算子 → 收集性能数据 → LLM 分析瓶颈并修改代码 → 再运行 → 迭代

用法:
    python run_llm_optimize.py
    python run_llm_optimize.py --dataset c105 --rounds 5
    python run_llm_optimize.py --dataset c101 --rounds 3 --iters 500
"""

import io
import sys

# Windows 编码修复：避免 GBK 编码错误
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', write_through=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', write_through=True)
    except Exception:
        pass

import openai
from dotenv import load_dotenv
import os
import re
import json

# 加载可能存在的 .env 变量
load_dotenv()
import time
import copy
import math
import random
import types
import argparse
import traceback
import difflib
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from baseline_alns import ALNSVRPSolver
from optimize_prompts import (
    OPTIMIZE_PROMPT_TEMPLATE,
    COST_MODEL_DESCRIPTION,
    SOLVER_API_DESCRIPTION,
    DATASET_CHARACTERISTICS,
    HISTORY_SECTION_TEMPLATE,
    HISTORY_ENTRY_TEMPLATE,
)

# ============================================================
# API 配置
# ============================================================
API_KEY = "	fba8fd46-2ec0-4590-908d-75891f90c981"
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
MODEL_NAME = "deepseek-v3-2-251201"

# 增加超时时间，DeepSeek推理长上下文（10k+ token）由于 prefill 计算，可能需要几十秒才输出第一个 token
client = openai.OpenAI(
    api_key=API_KEY, 
    base_url=BASE_URL,
    timeout=300.0
)


# ============================================================
# 基线算子代码（从 baseline_alns.py 提取的 5 个算子方法）
# ============================================================
BASELINE_OPERATOR_CODE = '''
def _random_removal(self, solution, ratio):
    """随机移除: 分散选择节点"""
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
    new_sol = [r[:] for r in solution]
    removed = []
    for ri, pi, node in selected:
        del new_sol[ri][pi]
        removed.append(node)
    new_sol = [r for r in new_sol if len(r) > 2]
    return new_sol, removed


def _route_removal(self, solution, ratio):
    """路径移除: 随机移除整条路径"""
    routes_info = []
    for ri, route in enumerate(solution):
        custs = [n for n in route if n != 0]
        if custs:
            routes_info.append((ri, custs))
    if not routes_info:
        return solution, []

    total_customers = sum(len(c) for _, c in routes_info)
    target = max(1, int(total_customers * ratio)) if ratio <= 1 else int(ratio)

    random.shuffle(routes_info)
    removed = []
    to_remove_idx = []
    for ri, custs in routes_info:
        removed.extend(custs)
        to_remove_idx.append(ri)
        if len(removed) >= target:
            break

    to_remove_idx.sort(reverse=True)
    new_sol = [r[:] for r in solution]
    for ri in to_remove_idx:
        del new_sol[ri]
    return new_sol, removed[:target]


def _string_removal(self, solution, ratio):
    """连续节点移除: 移除路径上的一段连续节点"""
    all_custs = [n for route in solution for n in route if n != 0]
    if not all_custs:
        return solution, []
    total = len(all_custs)
    n_remove = max(1, int(total * ratio)) if ratio <= 1 else min(total, int(ratio))
    n_remove = min(n_remove, total)

    removed = []
    new_sol = [r[:] for r in solution]
    while len(removed) < n_remove:
        valid = [(i, r) for i, r in enumerate(new_sol) if len(r) > 2]
        if not valid:
            break
        ri, route = random.choice(valid)
        start = random.randint(1, len(route) - 2)
        slen = min(random.randint(1, 3),
                   len(route) - start - 1,
                   n_remove - len(removed))
        for _ in range(slen):
            if start < len(new_sol[ri]) - 1:
                node = new_sol[ri][start]
                if node != 0:
                    removed.append(node)
                    del new_sol[ri][start]
    new_sol = [r for r in new_sol if len(r) > 2]
    return new_sol, removed


def _greedy_insert(self, solution, removed_nodes):
    """贪心插入: 每个节点找全局最优位置 (完整成本计算)"""
    if not removed_nodes:
        return solution
    if not solution:
        return [[0, n, 0] for n in removed_nodes]

    new_sol = [r[:] for r in solution]
    for node in removed_nodes:
        node_demand = self.customer_lookup[node].get('demand', 0)
        best_inc = float('inf')
        best_ri = None
        best_pos = None

        for ri, route in enumerate(new_sol):
            ld = sum(self.customer_lookup[n].get('demand', 0) for n in route if n != 0)
            if ld + node_demand > self.capacity:
                continue
            route_before = [self.id_to_customer[n] for n in route]
            c_before = self.calculator.calculate_route_cost(
                route_before, self.dist_matrix)['variable_cost']
            for pos in range(1, len(route)):
                route_after = route[:pos] + [node] + route[pos:]
                ra_nodes = [self.id_to_customer[n] for n in route_after]
                c_after = self.calculator.calculate_route_cost(
                    ra_nodes, self.dist_matrix)['variable_cost']
                inc = c_after - c_before
                if inc < best_inc:
                    best_inc = inc
                    best_ri = ri
                    best_pos = pos

        # 新路径成本
        nr = [0, node, 0]
        nr_nodes = [self.id_to_customer[n] for n in nr]
        nr_cost = self.calculator.calculate_route_cost(
            nr_nodes, self.dist_matrix)['variable_cost']
        nr_cost += self.calculator.f

        if best_ri is not None and best_inc <= nr_cost:
            new_sol[best_ri].insert(best_pos, node)
        else:
            new_sol.append([0, node, 0])
    return new_sol


def _regret_insert(self, solution, removed_nodes):
    """后悔插入: 优先插入后悔值最大的节点 (完整成本计算)"""
    if not removed_nodes:
        return solution
    if not solution:
        return [[0, n, 0] for n in removed_nodes]

    new_sol = [r[:] for r in solution]
    remaining = list(removed_nodes)

    while remaining:
        best_regret = -float('inf')
        best_node = None
        best_ri = None
        best_pos = None

        for node in remaining:
            nd = self.customer_lookup[node].get('demand', 0)
            route_bests = []

            for ri, route in enumerate(new_sol):
                ld = sum(self.customer_lookup[n].get('demand', 0)
                         for n in route if n != 0)
                if ld + nd > self.capacity:
                    continue
                route_before = [self.id_to_customer[n] for n in route]
                c_before = self.calculator.calculate_route_cost(
                    route_before, self.dist_matrix)['variable_cost']
                best_inc_r = float('inf')
                best_pos_r = 1
                for pos in range(1, len(route)):
                    ra = route[:pos] + [node] + route[pos:]
                    ra_n = [self.id_to_customer[n] for n in ra]
                    c_after = self.calculator.calculate_route_cost(
                        ra_n, self.dist_matrix)['variable_cost']
                    inc = c_after - c_before
                    if inc < best_inc_r:
                        best_inc_r = inc
                        best_pos_r = pos
                route_bests.append((best_inc_r, ri, best_pos_r))

            # 新路径选项
            nr = [0, node, 0]
            nr_nodes = [self.id_to_customer[n] for n in nr]
            nr_cost = self.calculator.calculate_route_cost(
                nr_nodes, self.dist_matrix)['variable_cost']
            nr_cost += self.calculator.f
            route_bests.append((nr_cost, None, None))

            route_bests.sort(key=lambda x: x[0])
            regret = route_bests[1][0] - route_bests[0][0] if len(route_bests) >= 2 else 0

            if regret > best_regret:
                best_regret = regret
                best_node = node
                best_ri = route_bests[0][1]
                best_pos = route_bests[0][2]

        if best_node is None:
            for n in remaining:
                new_sol.append([0, n, 0])
            break

        if best_ri is not None:
            new_sol[best_ri].insert(best_pos, best_node)
        else:
            new_sol.append([0, best_node, 0])
        remaining.remove(best_node)

    return new_sol
'''


# ============================================================
# 数据加载
# ============================================================
def load_solomon_data(file_path):
    """加载 Solomon 数据集"""
    data = {}
    customers = []
    vehicle_capacity = None

    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line == "" or line.startswith("C") or line.startswith("VEHICLE") \
                or line.startswith("NUMBER") or line.startswith("CUSTOMER"):
            continue
        parts = line.split()
        if len(parts) == 2 and vehicle_capacity is None:
            vehicle_capacity = int(parts[1])
            break

    if vehicle_capacity is None:
        vehicle_capacity = 200

    for line in lines:
        line = line.strip()
        if line == "" or line.startswith("C") or line.startswith("VEHICLE") \
                or line.startswith("NUMBER") or line.startswith("CUSTOMER") \
                or line.startswith("CUST"):
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
                "service_time": float(parts[6]),
            })

    data["vehicle_capacity"] = vehicle_capacity
    data["customers"] = customers

    # 添加 E_i, L_i 字段
    if customers:
        depot = customers[0]
        for cust in customers:
            cust['E_i'] = max(0, cust['ready_time']) if cust['id'] != 0 else 0
            cust['L_i'] = min(depot['due_date'], cust['due_date']) \
                if cust['id'] != 0 else depot['due_date']

    return data


# ============================================================
# LLM 接口（流式输出）
# ============================================================
def query_llm(messages, model_name=MODEL_NAME, temperature=0.6, max_retries=3):
    """调用 LLM 获取响应，使用流式输出，带重试机制。"""
    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                wait = 5 * attempt
                print(f"[LLM] 第 {attempt}/{max_retries} 次重试 (等待 {wait}s)...")
                time.sleep(wait)

            print("[LLM] 正在向大模型发送请求并等待推理（提示词较长，通常需要等待20-60秒出现第一个字符）...")
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                stream=True,
                max_tokens=8192,
            )

            full_response = ""
            print("[LLM] ", end="", flush=True)

            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                # 推理模型的思考过程
                reasoning = getattr(delta, 'reasoning_content', None)
                if reasoning:
                    print(reasoning, end="", flush=True)

                content = getattr(delta, 'content', None)
                if content:
                    print(content, end="", flush=True)
                    full_response += content

            print()
            return full_response

        except KeyboardInterrupt:
            print(f"\n[LLM API] 被用户终止 (KeyboardInterrupt)。保留当前已生成的部分({len(full_response)}字符)。")
            return full_response
        except Exception as e:
            print(f"\n[LLM API Error] attempt {attempt}/{max_retries}: {e}")
            if attempt == max_retries:
                print("[LLM] 所有重试均失败")
                return ""

    return ""


# ============================================================
# 性能指标收集
# ============================================================
def collect_metrics(solver, best_solution, best_cost, elapsed_time):
    """求解完成后收集完整性能指标"""
    metrics = {
        'best_cost': best_cost,
        'num_routes': len(best_solution),
        'elapsed_time': elapsed_time,
    }

    # --- 初始成本 ---
    if solver.cost_history:
        metrics['initial_cost'] = solver.cost_history[0]
    else:
        metrics['initial_cost'] = best_cost

    improvement = metrics['initial_cost'] - best_cost
    metrics['improvement_pct'] = (improvement / metrics['initial_cost'] * 100) \
        if metrics['initial_cost'] > 0 else 0

    # --- 算子统计 ---
    d_names = ['_random_removal', '_route_removal', '_string_removal']
    i_names = ['_greedy_insert', '_regret_insert']

    op_stats_str = ""
    for idx, name in enumerate(d_names):
        s = solver.op_stats['destroy'].get(idx, {'uses': 0, 'successes': 0})
        rate = (s['successes'] / s['uses'] * 100) if s['uses'] > 0 else 0
        op_stats_str += f"  {name}: {s['uses']} 次使用, {s['successes']} 次成功 ({rate:.1f}%)\n"
    for idx, name in enumerate(i_names):
        s = solver.op_stats['insert'].get(idx, {'uses': 0, 'successes': 0})
        rate = (s['successes'] / s['uses'] * 100) if s['uses'] > 0 else 0
        op_stats_str += f"  {name}: {s['uses']} 次使用, {s['successes']} 次成功 ({rate:.1f}%)\n"
    metrics['operator_stats'] = op_stats_str.rstrip()
    metrics['op_stats_raw'] = copy.deepcopy(solver.op_stats)

    # --- 收敛分析 ---
    bch = solver.best_cost_history
    total_iters = len(bch)
    metrics['total_iters'] = total_iters

    # 最后改善迭代
    last_improve = 0
    if len(bch) > 1:
        for i in range(1, len(bch)):
            if bch[i] < bch[i - 1] - 1e-6:
                last_improve = i
    metrics['last_improve_iter'] = last_improve
    metrics['stagnation_iters'] = total_iters - last_improve

    # 前 25% 迭代的改善占比
    total_improve = bch[0] - bch[-1] if bch else 0
    quarter = max(1, total_iters // 4)
    early_improve = bch[0] - bch[min(quarter, len(bch) - 1)] if bch else 0
    metrics['early_improvement_pct'] = (early_improve / total_improve * 100) \
        if total_improve > 1e-6 else 100.0

    # --- 成本分解 ---
    total_c12 = 0
    total_c13 = 0
    total_c2 = 0
    total_c3 = 0
    for route_ids in best_solution:
        route_nodes = [solver.id_to_customer[n] for n in route_ids]
        info = solver.calculator.calculate_route_cost(route_nodes, solver.dist_matrix)
        route_dist = info['dist']
        total_c12 += route_dist * solver.calculator.c
        total_c2 += info['c2']
        total_c3 += info['c3']
        # C13 = variable_cost - C12 - C2 - C3
        total_c13 += info['variable_cost'] - (route_dist * solver.calculator.c) \
            - info['c2'] - info['c3']

    c11 = len(best_solution) * solver.calculator.f
    total_all = c11 + total_c12 + total_c13 + total_c2 + total_c3
    if total_all < 1e-6:
        total_all = 1.0

    metrics['c11'] = c11
    metrics['c12'] = total_c12
    metrics['c13'] = total_c13
    metrics['c2'] = total_c2
    metrics['c3'] = total_c3
    metrics['c11_pct'] = c11 / total_all * 100
    metrics['c12_pct'] = total_c12 / total_all * 100
    metrics['c13_pct'] = total_c13 / total_all * 100
    metrics['c2_pct'] = total_c2 / total_all * 100
    metrics['c3_pct'] = total_c3 / total_all * 100

    # --- 路径统计 ---
    customers_per_route = []
    load_utilizations = []
    for route_ids in best_solution:
        custs = [n for n in route_ids if n != 0]
        customers_per_route.append(len(custs))
        total_demand = sum(solver.customer_lookup[n].get('demand', 0) for n in custs)
        load_utilizations.append(total_demand / solver.capacity * 100
                                 if solver.capacity > 0 else 0)

    metrics['avg_customers_per_route'] = sum(customers_per_route) / len(customers_per_route) \
        if customers_per_route else 0
    metrics['avg_load_utilization'] = sum(load_utilizations) / len(load_utilizations) \
        if load_utilizations else 0

    return metrics


# ============================================================
# 算子动态注入
# ============================================================
def inject_operators(solver, operator_code_str):
    """将 LLM 优化后的算子代码动态注入到 solver 实例中"""
    namespace = {
        'math': math,
        'random': random,
        'copy': copy,
    }

    try:
        exec(operator_code_str, namespace)
    except Exception as e:
        print(f"[Error] 算子代码编译失败: {e}")
        return False

    method_names = [
        '_random_removal', '_route_removal', '_string_removal',
        '_greedy_insert', '_regret_insert',
    ]

    injected = 0
    for name in method_names:
        if name in namespace and callable(namespace[name]):
            bound_method = types.MethodType(namespace[name], solver)
            setattr(solver, name, bound_method)
            injected += 1
        else:
            print(f"[Warning] 方法 {name} 未找到或不可调用")

    if injected < 5:
        print(f"[Error] 仅注入 {injected}/5 个方法")
        return False

    # 重新注册算子列表（关键！）
    solver.destroy_ops = [
        solver._random_removal,
        solver._route_removal,
        solver._string_removal,
    ]
    solver.insert_ops = [
        solver._greedy_insert,
        solver._regret_insert,
    ]

    # 重置权重
    solver.d_weights = [1.0] * len(solver.destroy_ops)
    solver.i_weights = [1.0] * len(solver.insert_ops)

    # 重置统计
    solver.op_stats = {
        'destroy': {i: {'uses': 0, 'successes': 0, 'score': 0}
                    for i in range(len(solver.destroy_ops))},
        'insert': {i: {'uses': 0, 'successes': 0, 'score': 0}
                   for i in range(len(solver.insert_ops))},
    }
    solver.cost_history = []
    solver.best_cost_history = []

    print(f"[Inject] 成功注入 {injected} 个优化算子")
    return True


# ============================================================
# 从 LLM 响应中提取算子代码
# ============================================================
def extract_operator_code(llm_response):
    """从 LLM 响应中提取算子代码，同时提取分析说明"""

    # 提取分析说明（代码块之前的文字）
    analysis = ""
    code_block_start = llm_response.find("```python")
    if code_block_start > 0:
        analysis = llm_response[:code_block_start].strip()
    else:
        code_block_start = llm_response.find("```")
        if code_block_start > 0:
            analysis = llm_response[:code_block_start].strip()
        else:
            analysis = llm_response

    # 提取 Python 代码块
    code_match = re.search(r'```python\s*\n(.*?)```', llm_response, re.DOTALL)
    if not code_match:
        # 兜底: 尝试找不带语言标记的代码块
        code_match = re.search(r'```\s*\n(.*?)```', llm_response, re.DOTALL)
        
    if not code_match:
        # 再兜底: 可能是由于被截断，没有闭合的 ```
        code_match = re.search(r'```python\s*\n(.*)', llm_response, re.DOTALL)
        
    if not code_match:
        code_match = re.search(r'```\s*\n(.*)', llm_response, re.DOTALL)

    if not code_match:
        # 终极兜底: 如果连 ``` 都没有，直接尝试寻找 def 关键字
        if "def _random_removal" in llm_response:
            code = llm_response[llm_response.find("def _random_removal"):]
            return code.strip(), analysis
            
        print("[Error] 无法从 LLM 响应中提取代码块")
        return None, analysis

    code = code_match.group(1).strip()

    # 如果 LLM 把代码放在 class 里面，提取方法部分
    if 'class ' in code:
        lines = code.split('\n')
        method_lines = []
        in_class = False
        skip_init = False

        for line in lines:
            if line.strip().startswith('class '):
                in_class = True
                continue
            if in_class:
                if line.strip().startswith('def __init__'):
                    skip_init = True
                    continue
                if skip_init:
                    if line.strip().startswith('def '):
                        skip_init = False
                    else:
                        continue
                if line.strip().startswith('def _'):
                    # 去掉一级缩进（class 内部→顶层函数）
                    method_lines.append(line[4:] if line.startswith('    ') else line)
                elif in_class and method_lines:
                    method_lines.append(line[4:] if line.startswith('    ') else line)

        if method_lines:
            code = '\n'.join(method_lines)

    return code, analysis


def validate_operator_code(code):
    """验证算子代码的合法性"""
    # 1. 语法检查
    try:
        compile(code, '<operator_code>', 'exec')
    except SyntaxError as e:
        print(f"[Validate] 语法错误: {e}")
        return False

    # 2. 必要方法检查
    required = ['_random_removal', '_route_removal', '_string_removal',
                '_greedy_insert', '_regret_insert']
    for method in required:
        if f'def {method}' not in code:
            print(f"[Validate] 缺少方法: {method}")
            return False

    # 3. 检查是否使用了完整成本计算（修复算子中）
    insert_section = code[code.find('def _greedy_insert'):]
    if 'calculate_route_cost' not in insert_section:
        print("[Validate] 警告: 修复算子未使用 calculate_route_cost")
        # 不强制失败，但给出警告

    return True


# ============================================================
# 运行求解器
# ============================================================
def run_solver(data, operator_code, max_iter=500, seed=42, verbose=True):
    """用指定的算子代码运行 ALNS 求解器，返回 (solution, cost, metrics)"""

    random.seed(seed)

    solver = ALNSVRPSolver(
        data, max_iter=max_iter, seed=seed, verbose=verbose
    )

    # 强制打开迭代进度输出
    solver.verbose = True

    # 注入算子
    if not inject_operators(solver, operator_code):
        return None, float('inf'), None

    # 求解
    start_time = time.time()
    try:
        best_solution, best_cost = solver.solve()
        elapsed = time.time() - start_time
    except Exception as e:
        print(f"[Error] 求解失败: {e}")
        traceback.print_exc()
        return None, float('inf'), None

    # 收集指标
    metrics = collect_metrics(solver, best_solution, best_cost, elapsed)

    return best_solution, best_cost, metrics


def evaluate_operator(data, operator_code, max_iter=500, seed=42,
                      eval_runs=1, seed_step=97, verbose=False):
    """多随机种子评估算子，返回稳健统计结果。

    返回字段:
      - mean_cost / std_cost: 多次运行均值与标准差
      - best_cost: 多次中的最优单次成本
      - best_solution / best_metrics: 最优单次对应结果
      - costs / seeds: 每次运行明细
    """
    eval_runs = max(1, int(eval_runs))
    costs = []
    seeds = []
    best_solution = None
    best_metrics = None
    best_cost = float('inf')

    for i in range(eval_runs):
        run_seed = seed + i * seed_step
        sol, cost, metrics = run_solver(
            data,
            operator_code,
            max_iter=max_iter,
            seed=run_seed,
            verbose=verbose,
        )
        if metrics is None:
            return None

        costs.append(float(cost))
        seeds.append(run_seed)

        if cost < best_cost:
            best_cost = float(cost)
            best_solution = sol
            best_metrics = metrics

    mean_cost = sum(costs) / len(costs)
    if len(costs) > 1:
        var = sum((c - mean_cost) ** 2 for c in costs) / (len(costs) - 1)
        std_cost = math.sqrt(var)
    else:
        std_cost = 0.0

    # 在 metrics 中附加稳健评估字段，供 Prompt 与日志复用
    if best_metrics is None:
        return None
    best_metrics = dict(best_metrics)
    best_metrics['eval_runs'] = eval_runs
    best_metrics['eval_mean_cost'] = mean_cost
    best_metrics['eval_std_cost'] = std_cost
    best_metrics['eval_best_cost'] = best_cost

    return {
        'mean_cost': mean_cost,
        'std_cost': std_cost,
        'best_cost': best_cost,
        'best_solution': best_solution,
        'best_metrics': best_metrics,
        'costs': costs,
        'seeds': seeds,
    }


# ============================================================
# 构建优化 Prompt
# ============================================================
def build_prompt(operator_code, metrics, round_num, dataset_name, dataset_type,
                 data, history=None):
    """构建发送给 LLM 的优化 Prompt"""

    n_customers = len([c for c in data['customers'] if c['id'] != 0])
    dataset_desc = DATASET_CHARACTERISTICS.get(dataset_type,
                                                f"数据集类型: {dataset_type}")

    # 额外分析
    extra_analysis = ""
    if history:
        entries = []
        for h in history:
            direction = "↓" if h['change'] < 0 else "↑"
            entries.append(HISTORY_ENTRY_TEMPLATE.format(
                round_num=h['round'],
                cost=h['cost'],
                prev_cost=h['prev_cost'],
                change_direction=direction,
                change_pct=abs(h['change']),
                modification_summary=h.get('summary', '(无)'),
            ))
        extra_analysis = HISTORY_SECTION_TEMPLATE.format(
            history_entries='\n'.join(entries)
        )

    best_cost_for_prompt = metrics.get('eval_mean_cost', metrics['best_cost'])

    prompt = OPTIMIZE_PROMPT_TEMPLATE.format(
        cost_model=COST_MODEL_DESCRIPTION,
        solver_api=SOLVER_API_DESCRIPTION,
        round_num=round_num,
        current_operator_code=operator_code.strip(),
        dataset_name=dataset_name,
        dataset_desc=dataset_desc,
        n_customers=n_customers,
        vehicle_capacity=data['vehicle_capacity'],
        best_cost=best_cost_for_prompt,
        initial_cost=metrics['initial_cost'],
        improvement_pct=metrics['improvement_pct'],
        num_routes=metrics['num_routes'],
        elapsed_time=metrics['elapsed_time'],
        operator_stats=metrics['operator_stats'],
        last_improve_iter=metrics['last_improve_iter'],
        total_iters=metrics['total_iters'],
        early_improvement_pct=metrics['early_improvement_pct'],
        stagnation_iters=metrics['stagnation_iters'],
        c11=metrics['c11'], c11_pct=metrics['c11_pct'],
        c12=metrics['c12'], c12_pct=metrics['c12_pct'],
        c13=metrics['c13'], c13_pct=metrics['c13_pct'],
        c2=metrics['c2'], c2_pct=metrics['c2_pct'],
        c3=metrics['c3'], c3_pct=metrics['c3_pct'],
        avg_customers_per_route=metrics['avg_customers_per_route'],
        avg_load_utilization=metrics['avg_load_utilization'],
        extra_analysis=extra_analysis,
    )

    return prompt


# ============================================================
# 主优化循环
# ============================================================
def run_optimization(dataset_path, dataset_type="c1", max_rounds=5,
                     iters_per_round=500, final_iters=1000, seed=42,
                     eval_runs=1, seed_step=97):
    """主优化循环"""

    dataset_name = Path(dataset_path).stem
    output_dir = Path("experiments_llm_optimize") / f"{dataset_type}_{dataset_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Start] 当前实例: {dataset_type}/{dataset_name}")

    # 若已有 summary.json 或已有 round_* 结果，则跳过
    has_summary = (output_dir / "summary.json").exists()
    has_rounds = any(
        (output_dir / name).is_dir() and name.startswith("round_")
        for name in os.listdir(output_dir)
    ) if output_dir.exists() else False

    if has_summary or has_rounds:
        print(f"[Skip] 已有结果，跳过: {output_dir}")
        return None, None

    print("=" * 70)
    print("LLM 迭代优化 ALNS 算子")
    print("=" * 70)
    print(f"数据集: {dataset_type}/{dataset_name}")
    print(f"优化轮数: {max_rounds}")
    print(f"每轮迭代: {iters_per_round}")
    print(f"最终验证: {final_iters}")
    print(f"评估重复: {eval_runs} 次 (seed_step={seed_step})")
    print(f"输出目录: {output_dir}")
    print("=" * 70)
    print()

    # 加载数据
    data = load_solomon_data(dataset_path)
    n_customers = len([c for c in data['customers'] if c['id'] != 0])
    print(f"[Data] {n_customers} 客户, 容量 {data['vehicle_capacity']}")
    print()

    # ===== Round 0: 基线运行 =====
    print("=" * 60)
    print("Round 0: 基线算子 (Baseline)")
    print("=" * 60)

    current_code = BASELINE_OPERATOR_CODE
    baseline_eval = evaluate_operator(
        data,
        current_code,
        max_iter=iters_per_round,
        seed=seed,
        eval_runs=eval_runs,
        seed_step=seed_step,
        verbose=False,
    )

    if baseline_eval is None:
        print("[Fatal] 基线算子运行失败")
        return

    best_solution = baseline_eval['best_solution']
    metrics = baseline_eval['best_metrics']
    best_cost = baseline_eval['mean_cost']

    baseline_cost = best_cost
    best_ever_cost = best_cost
    best_ever_code = current_code

    # 保存 Round 0
    round_dir = output_dir / "round_0_baseline"
    round_dir.mkdir(exist_ok=True)
    (round_dir / "code.py").write_text(current_code, encoding='utf-8')
    with open(round_dir / "result.json", 'w', encoding='utf-8') as f:
        json.dump({
            'round': 0,
            'type': 'baseline',
            'best_cost': baseline_eval['best_cost'],
            'mean_cost': baseline_eval['mean_cost'],
            'std_cost': baseline_eval['std_cost'],
            'eval_runs': eval_runs,
            'seeds': baseline_eval['seeds'],
            'num_routes': metrics['num_routes'],
            'elapsed_time': metrics['elapsed_time'],
            'metrics': {k: v for k, v in metrics.items()
                       if k not in ('op_stats_raw',)},
        }, f, indent=2, ensure_ascii=False, default=str)

    if eval_runs > 1:
        print(f"\n[Round 0] 基线均值成本: {baseline_eval['mean_cost']:.2f} ± {baseline_eval['std_cost']:.2f}, 最优: {baseline_eval['best_cost']:.2f}")
    else:
        print(f"\n[Round 0] 基线成本: {best_cost:.2f}, 车辆: {metrics['num_routes']}")
    print()

    # 优化历史
    history = []
    round_results = [{
        'round': 0,
        'type': 'baseline',
        'cost': best_cost,
        'best_cost': baseline_eval['best_cost'],
        'std_cost': baseline_eval['std_cost'],
        'num_routes': metrics['num_routes'],
    }]

    # ===== Round 1 ~ N: LLM 迭代优化 =====
    accepted_once = False
    for round_num in range(1, max_rounds + 1):
        print("=" * 60)
        print(f"Round {round_num}/{max_rounds}: LLM 优化")
        print("=" * 60)

        # 1. 构建 Prompt
        prompt = build_prompt(
            current_code, metrics, round_num,
            dataset_name, dataset_type, data, history
        )

        # 2-4: 调用 LLM 并提取验证代码（带有重试机制）
        max_code_retries = 3
        new_code = None
        analysis = ""
        for code_attempt in range(1, max_code_retries + 1):
            if code_attempt > 1:
                print(f"[Round {round_num}] 第 {code_attempt}/{max_code_retries} 次尝试生成有效的代码...")
            
            print(f"\n[LLM] 分析算子性能并生成优化代码...")
            messages = [{"role": "user", "content": prompt}]
            llm_response = query_llm(messages)

            if not llm_response:
                print(f"[Round {round_num}] LLM 无响应")
                continue

            # 3. 提取代码
            extracted_code, extracted_analysis = extract_operator_code(llm_response)
            if not extracted_code:
                print(f"[Round {round_num}] 无法从回复中提取 Python 代码段")
                continue

            # 4. 验证代码
            if not validate_operator_code(extracted_code):
                print(f"[Round {round_num}] 代码段不完整或有语法错误（可能被截断）。")
                # 把提示词更新一下，提醒 LLM 注意输出完整
                prompt += "\n\n【系统提醒】上一轮输出的代码不完整或有语法错误（缺少部分算子方法或被截断）。请务必**确保输出的代码完整包含所有 5 个方法**，不要省略内容！"
                continue

            # 成功则跳出重试循环
            new_code = extracted_code
            analysis = extracted_analysis
            break

        if not new_code:
            print(f"[Round {round_num}] 连续 {max_code_retries} 次代码生成均失败，跳过本轮")
            round_results.append({
                'round': round_num, 'type': 'skipped', 'cost': best_cost,
                'reason': 'code extraction or validation failed after retries'
            })
            continue

        # 5. 运行优化后的算子
        print(f"\n[Run] 运行优化后的算子 (iter={iters_per_round})...")
        new_eval = evaluate_operator(
            data,
            new_code,
            max_iter=iters_per_round,
            seed=seed,
            eval_runs=eval_runs,
            seed_step=seed_step,
            verbose=False,
        )

        if new_eval is None:
            print(f"[Round {round_num}] 运行失败，回退到上一版")
            round_results.append({
                'round': round_num, 'type': 'failed', 'cost': best_cost,
                'reason': 'runtime error'
            })
            continue

        new_solution = new_eval['best_solution']
        new_cost = new_eval['mean_cost']
        new_metrics = new_eval['best_metrics']

        # 6. 对比结果
        prev_cost = best_cost
        cost_change = (new_cost - prev_cost) / prev_cost * 100

        if new_cost < prev_cost:
            status = "✓ 改善"
            decision = "accepted"
            current_code = new_code
            best_cost = new_cost
            metrics = new_metrics
            if new_cost < best_ever_cost:
                best_ever_cost = new_cost
                best_ever_code = new_code
        else:
            status = "✗ 退步"
            decision = "rejected"
            # 不更新 current_code 和 metrics，保持上一版

        print(f"\n[Round {round_num}] {status}")
        print(f"  上一版(评估均值): {prev_cost:.2f}")
        if eval_runs > 1:
            print(f"  本轮评估均值:     {new_eval['mean_cost']:.2f} ± {new_eval['std_cost']:.2f} ({cost_change:+.2f}%)")
            print(f"  本轮单次最优:     {new_eval['best_cost']:.2f}")
        else:
            print(f"  本  轮: {new_cost:.2f} ({cost_change:+.2f}%)")
        print(f"  决  定: {decision}")
        print(f"  历史最优(评估均值): {best_ever_cost:.2f}")

        # 记录历史（保存完整分析）
        history.append({
            'round': round_num,
            'cost': new_cost,
            'prev_cost': prev_cost,
            'change': cost_change,
            'summary': analysis if analysis else '(无分析)',
        })

        # 保存本轮结果
        round_dir = output_dir / f"round_{round_num}_{decision}"
        round_dir.mkdir(exist_ok=True)
        (round_dir / "code.py").write_text(new_code, encoding='utf-8')
        (round_dir / "llm_analysis.md").write_text(
            f"# Round {round_num} - LLM 分析\n\n{analysis}\n\n"
            f"## 性能对比\n- 优化前: {prev_cost:.2f}\n- 优化后: {new_cost:.2f}\n"
            f"- 变化: {cost_change:+.2f}%\n- 决定: {decision}\n",
            encoding='utf-8'
        )
        (round_dir / "llm_full_response.md").write_text(
            llm_response, encoding='utf-8'
        )
        # 保存完整改进摘要（原始分析 + 完整回复）
        (round_dir / "llm_improvements.txt").write_text(
            f"[LLM Analysis]\n{analysis}\n\n"
            f"[LLM Full Response]\n{llm_response}\n",
            encoding='utf-8'
        )
        # 保存结构化改进说明（代码 + 思路）
        with open(round_dir / "llm_improvements.json", 'w', encoding='utf-8') as f:
            json.dump({
                'analysis': analysis,
                'full_response': llm_response,
                'code': new_code,
                'decision': decision,
                'prev_cost': prev_cost,
                'new_cost': new_cost,
                'change_pct': cost_change,
            }, f, indent=2, ensure_ascii=False, default=str)
        with open(round_dir / "result.json", 'w', encoding='utf-8') as f:
            json.dump({
                'round': round_num,
                'type': decision,
                'cost': new_cost,
                'best_cost': new_eval['best_cost'],
                'mean_cost': new_eval['mean_cost'],
                'std_cost': new_eval['std_cost'],
                'eval_runs': eval_runs,
                'seeds': new_eval['seeds'],
                'prev_cost': prev_cost,
                'change_pct': cost_change,
                'num_routes': new_metrics['num_routes'],
                'elapsed_time': new_metrics['elapsed_time'],
                'metrics': {k: v for k, v in new_metrics.items()
                           if k not in ('op_stats_raw',)},
            }, f, indent=2, ensure_ascii=False, default=str)

        round_results.append({
            'round': round_num,
            'type': decision,
            'cost': new_cost,
            'best_cost': new_eval['best_cost'],
            'std_cost': new_eval['std_cost'],
            'prev_cost': prev_cost,
            'change_pct': cost_change,
            'num_routes': new_metrics['num_routes'],
        })

        if decision == "accepted":
            print("\n[Stop] 已接受改进结果，提前结束优化流程")
            accepted_once = True
            break

        print()

    # 若仍无提升，记录未改进
    if not accepted_once:
        (output_dir / "no_improvement.txt").write_text(
            "未改进：多轮优化未优于基线，保留基线结果。\n",
            encoding='utf-8'
        )

    # ===== 最终验证（用更多迭代） =====
    print("=" * 60)
    print(f"最终验证: 用最优算子运行 {final_iters} 次迭代")
    print("=" * 60)

    final_eval = evaluate_operator(
        data,
        best_ever_code,
        max_iter=final_iters,
        seed=seed,
        eval_runs=eval_runs,
        seed_step=seed_step,
        verbose=False,
    )

    if final_eval:
        final_solution = final_eval['best_solution']
        final_cost = final_eval['mean_cost']
        final_metrics = final_eval['best_metrics']
        print(f"\n[最终结果]")
        print(f"  基线评估均值: {baseline_cost:.2f}")
        if eval_runs > 1:
            print(f"  最终评估均值: {final_eval['mean_cost']:.2f} ± {final_eval['std_cost']:.2f}")
            print(f"  最终单次最优: {final_eval['best_cost']:.2f}")
        else:
            print(f"  最终成本:     {final_cost:.2f}")
        total_improve = (baseline_cost - final_cost) / baseline_cost * 100
        print(f"  总改善率:   {total_improve:.2f}%")
        print(f"  车辆数:     {final_metrics['num_routes']}")

        # 保存最终结果
        final_dir = output_dir / "final_best"
        final_dir.mkdir(exist_ok=True)
        (final_dir / "code.py").write_text(best_ever_code, encoding='utf-8')
        with open(final_dir / "result.json", 'w', encoding='utf-8') as f:
            json.dump({
                'type': 'final',
                'baseline_cost': baseline_cost,
                'final_cost': final_cost,
                'final_best_cost': final_eval['best_cost'],
                'final_std_cost': final_eval['std_cost'],
                'eval_runs': eval_runs,
                'seeds': final_eval['seeds'],
                'total_improvement_pct': total_improve,
                'num_routes': final_metrics['num_routes'],
                'elapsed_time': final_metrics['elapsed_time'],
            }, f, indent=2, ensure_ascii=False, default=str)

    # ===== 汇总 =====
    print("\n" + "=" * 60)
    print("优化汇总")
    print("=" * 60)
    for r in round_results:
        status_icon = {"baseline": "[B]", "accepted": "[+]", "rejected": "[-]",
                       "skipped": "[S]", "failed": "[X]"}.get(r['type'], '[?]')
        cost_str = f"{r['cost']:.2f}"
        change_str = ""
        if 'change_pct' in r:
            change_str = f" ({r['change_pct']:+.2f}%)"
        print(f"  {status_icon} Round {r['round']}: {cost_str}{change_str} [{r['type']}]")

    print(f"\n  Baseline: {baseline_cost:.2f} -> Best: {best_ever_cost:.2f}")
    total_improve = (baseline_cost - best_ever_cost) / baseline_cost * 100
    print(f"  Total improvement: {total_improve:.2f}%")

    # 保存汇总
    with open(output_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump({
            'dataset': f"{dataset_type}/{dataset_name}",
            'n_customers': n_customers,
            'max_rounds': max_rounds,
            'iters_per_round': iters_per_round,
            'eval_runs': eval_runs,
            'seed_step': seed_step,
            'baseline_cost': baseline_cost,
            'best_cost': best_ever_cost,
            'total_improvement_pct': total_improve,
            'rounds': round_results,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  结果已保存到: {output_dir}")
    print("=" * 60)

    return best_ever_code, best_ever_cost


def build_core30_suite():
    return [
        ('c1', 'c101'), ('c1', 'c102'), ('c1', 'c103'), ('c1', 'c104'), ('c1', 'c105'),
        ('c2', 'c201'), ('c2', 'c202'), ('c2', 'c203'), ('c2', 'c204'), ('c2', 'c205'),
        ('r1', 'r101'), ('r1', 'r102'), ('r1', 'r103'), ('r1', 'r104'), ('r1', 'r105'),
        ('r2', 'r201'), ('r2', 'r202'), ('r2', 'r203'), ('r2', 'r204'), ('r2', 'r205'),
        ('rc1', 'rc101'), ('rc1', 'rc102'), ('rc1', 'rc103'), ('rc1', 'rc104'), ('rc1', 'rc105'),
        ('rc2', 'rc201'), ('rc2', 'rc202'), ('rc2', 'rc203'), ('rc2', 'rc204'), ('rc2', 'rc205'),
    ]


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM 迭代优化 ALNS 算子")
    parser.add_argument('--dataset', type=str, default='c101',
                        help='数据集名称 (默认 c101)')
    parser.add_argument('--type', type=str, default='c1',
                        help='数据集类型 (默认 c1)')
    parser.add_argument('--suite', type=str, default=None,
                        choices=['core30'], help='批量实例集 (core30)')
    parser.add_argument('--rounds', type=int, default=5,
                        help='优化轮数 (默认 5)')
    parser.add_argument('--iters', type=int, default=500,
                        help='每轮求解迭代次数 (默认 500)')
    parser.add_argument('--final-iters', type=int, default=1000,
                        help='最终验证迭代次数 (默认 1000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认 42)')
    parser.add_argument('--eval-runs', type=int, default=1,
                        help='每次算子评估重复次数（多seed均值，默认 1）')
    parser.add_argument('--seed-step', type=int, default=97,
                        help='多seed评估的步长（默认 97）')
    args = parser.parse_args()

    dataset_base = Path("data/1 Solomon Benchmark")

    if args.suite == 'core30':
        suite = build_core30_suite()
        for dataset_type, dataset_name in suite:
            dataset_file = f"{dataset_name}.txt"
            dataset_path = dataset_base / dataset_type / dataset_file
            if not dataset_path.exists():
                print(f"[Skip] 数据集不存在: {dataset_path}")
                continue
            run_optimization(
                dataset_path=str(dataset_path),
                dataset_type=dataset_type,
                max_rounds=args.rounds,
                iters_per_round=args.iters,
                final_iters=args.final_iters,
                seed=args.seed,
                eval_runs=args.eval_runs,
                seed_step=args.seed_step,
            )
    else:
        dataset_file = f"{args.dataset}.txt"
        dataset_path = dataset_base / args.type / dataset_file

        if not dataset_path.exists():
            print(f"[Error] 数据集不存在: {dataset_path}")
            print(f"  可用的路径格式: data/1 Solomon Benchmark/<type>/<name>.txt")
            sys.exit(1)

        run_optimization(
            dataset_path=str(dataset_path),
            dataset_type=args.type,
            max_rounds=args.rounds,
            iters_per_round=args.iters,
            final_iters=args.final_iters,
            seed=args.seed,
            eval_runs=args.eval_runs,
            seed_step=args.seed_step,
        )
