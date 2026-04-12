"""
baseline_ga.py - 遗传算法 (GA) 基线求解器
=========================================
用途: 作为对比实验的"地板基线"，与 ALNS+LLM 方案比较性能差距。

算法设计:
  - 染色体编码: Giant Tour (客户ID的一维全排列)
  - 解码方式: 贪心拆分算法 (Greedy Split) → 二维路线列表
  - 适应度:   调用 FreshnessAndPenaltyCalculator，与 ALNS 使用完全相同的成本函数
  - 选择:     锦标赛选择 (Tournament Selection)
  - 交叉:     顺序交叉 OX1 (Order Crossover)
  - 变异:     Swap / Insertion / Inversion 三种变异随机选择
  - 精英保留: 保留每代最优个体 (Elitism)

兼容性:
  - data 格式与 run_batch_no_llm.py 完全一致
  - 成本函数使用 utils.FreshnessAndPenaltyCalculator (含 C2 新鲜度 + C3 时间惩罚)
  - 输出格式与 ALNS 对齐: BEST_COST / NUM_ROUTES / Route 1: [...]
"""

import sys
import random
import copy
import math
import time
from typing import List, Tuple, Optional

# Windows GBK 终端 UTF-8 输出修复
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ──────────────────────────────────────────────────
# 工具函数: 成本计算器 (解耦导入，方便独立运行测试)
# ──────────────────────────────────────────────────

def _build_calculator(data: dict):
    """从 data 字典构建 FreshnessAndPenaltyCalculator。"""
    try:
        from utils import FreshnessAndPenaltyCalculator
    except ImportError:
        # 若 utils 不在 path 中，提供内联版本
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from utils import FreshnessAndPenaltyCalculator
    return FreshnessAndPenaltyCalculator(data)


def _build_dist_matrix(customers: list) -> list:
    """构建欧氏距离矩阵 (与 HeuristicSolver 逻辑一致)。"""
    n = len(customers)
    dm = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                xi, yi = customers[i]['x'], customers[i]['y']
                xj, yj = customers[j]['x'], customers[j]['y']
                dm[i][j] = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
    return dm


# ──────────────────────────────────────────────────
# 核心类: GAVRPSolver
# ──────────────────────────────────────────────────

class GAVRPSolver:
    """
    基于遗传算法的 VRP-TW 求解器。

    参数:
        data         : dict, 与 run_batch_no_llm.py 一致的数据字典
        pop_size     : int, 种群规模 (默认80)
        max_gen      : int, 最大代数 (默认500)
        cx_prob      : float, 交叉概率 (默认0.85)
        mut_prob     : float, 变异概率 (默认0.15)
        tournament_k : int, 锦标赛规模 (默认5)
        elite_size   : int, 精英个体数量 (默认2)
        seed         : int | None, 随机种子
        verbose      : bool, 是否打印详细日志
    """

    def __init__(
        self,
        data: dict,
        pop_size: int = 80,
        max_gen: int = 500,
        cx_prob: float = 0.85,
        mut_prob: float = 0.15,
        tournament_k: int = 5,
        elite_size: int = 2,
        seed: Optional[int] = None,
        verbose: bool = True,
    ):
        self.data = data
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.tournament_k = tournament_k
        self.elite_size = elite_size
        self.verbose = verbose

        if seed is not None:
            random.seed(seed)

        # ── 从 data 提取基础信息 ──
        self.customers = data['customers']
        self.n = len(self.customers)
        self.capacity = data.get('vehicle_capacity', 200)
        self.vehicle_speed = data.get('vehicle_speed_kmph', 40)

        # 仓库 (id=0), 客户节点 (id!=0)
        self.depot = next(c for c in self.customers if c['id'] == 0)
        self.non_depot = [c for c in self.customers if c['id'] != 0]
        self.cust_ids = [c['id'] for c in self.non_depot]  # 染色体基因池

        # 索引映射 (id → customer dict)
        self.id_to_customer = {c['id']: c for c in self.customers}

        # 距离矩阵 & 成本计算器
        self.dist_matrix = _build_dist_matrix(self.customers)
        self.calculator = _build_calculator(data)

        # 记录历史 (用于绘优化曲线)
        self.best_cost_history: List[float] = []
        self.avg_cost_history: List[float] = []

        if self.verbose:
            print(f"[GA] 初始化完成 | 客户数: {len(self.non_depot)} | "
                  f"种群: {pop_size} | 代数: {max_gen} | 容量: {self.capacity}")

    # ═══════════════════════════════════════════════
    # 1. 解码: Giant Tour → 路线列表 (贪心拆分)
    # ═══════════════════════════════════════════════

    def decode(self, tour: List[int]) -> List[List[int]]:
        """
        贪心拆分算法 (Greedy Split / Feasibility Check Decoder).

        按 tour 顺序依次将客户加入当前路线；当出现以下情况时，
        强制当前车辆回场，开启新路线:
          - 超出车辆容量
          - 到达时间超过客户软硬时间窗上界 L_i

        返回: 二维路线列表, 每条路线形如 [0, c1, c2, ..., 0]
        """
        routes = []
        current_route = [0]          # 从仓库出发
        current_load = 0
        current_time = 0.0           # 离开仓库时刻 (分钟)
        current_node_id = 0          # 当前位置 (仓库id=0)

        for cust_id in tour:
            cust = self.id_to_customer[cust_id]
            demand = cust.get('demand', 0)
            ready  = cust.get('ready_time', 0)
            due    = cust.get('due_date', float('inf'))
            L_i    = cust.get('L_i', due)   # 软时间窗上界 (如无则等于 due_date)
            svc    = cust.get('service_time', 0)

            # 行驶时间
            travel = self.dist_matrix[current_node_id][cust_id] * (60.0 / self.vehicle_speed)
            arrival = current_time + travel

            # ── 可行性判断: 超容量 或 到达时间超出 L_i/due_date ──
            capacity_ok = (current_load + demand <= self.capacity)
            time_ok = (arrival <= max(due, L_i))

            if not capacity_ok or not time_ok:
                # 当前路线已满，回仓库并开启新路线
                current_route.append(0)
                if len(current_route) > 2:
                    routes.append(current_route)
                # 重置状态
                current_route = [0]
                current_load = 0
                current_time = 0.0
                current_node_id = 0
                # 重新计算从仓库出发
                travel = self.dist_matrix[0][cust_id] * (60.0 / self.vehicle_speed)
                arrival = travel

            # 更新状态
            current_load += demand
            current_time = max(arrival, ready) + svc   # 等候 + 服务
            current_node_id = cust_id
            current_route.append(cust_id)

        # 收尾: 最后一条路线
        current_route.append(0)
        if len(current_route) > 2:
            routes.append(current_route)

        return routes

    # ═══════════════════════════════════════════════
    # 2. 适应度计算
    # ═══════════════════════════════════════════════

    def calc_cost(self, routes: List[List[int]]) -> float:
        """
        计算路线总成本 (与 HeuristicSolver.cost 逻辑完全一致):
          固定成本 (C11) + 变动成本 (C12 + C13 + C2 + C3)
        """
        if not routes:
            return float('inf')

        fixed_cost = len(routes) * self.calculator.f
        variable_cost = 0.0
        for route_ids in routes:
            route_nodes = [self.id_to_customer[nid] for nid in route_ids]
            result = self.calculator.calculate_route_cost(route_nodes, self.dist_matrix)
            variable_cost += result['variable_cost']

        return fixed_cost + variable_cost

    def fitness(self, tour: List[int]) -> Tuple[float, List[List[int]]]:
        """返回 (cost, routes). cost 越小越优。"""
        routes = self.decode(tour)
        cost = self.calc_cost(routes)
        return cost, routes

    # ═══════════════════════════════════════════════
    # 3. 种群初始化
    # ═══════════════════════════════════════════════

    def _init_population(self) -> List[List[int]]:
        """
        生成初始种群。混合策略提高多样性:
          - 70% 随机排列
          - 15% 按时间窗紧迫度 (due_date 升序) + 轻度扰动
          - 15% 按需求降序 + 轻度扰动
        """
        population = []
        n_random = int(self.pop_size * 0.70)
        n_tw = int(self.pop_size * 0.15)
        n_demand = self.pop_size - n_random - n_tw

        # 随机排列
        for _ in range(n_random):
            t = self.cust_ids[:]
            random.shuffle(t)
            population.append(t)

        # 时间窗排序 + 扰动
        tw_base = sorted(self.cust_ids, key=lambda cid: self.id_to_customer[cid].get('due_date', float('inf')))
        for _ in range(n_tw):
            t = tw_base[:]
            self._mutate_inplace(t, prob=0.3)
            population.append(t)

        # 需求降序 + 扰动
        demand_base = sorted(self.cust_ids, key=lambda cid: self.id_to_customer[cid].get('demand', 0), reverse=True)
        for _ in range(n_demand):
            t = demand_base[:]
            self._mutate_inplace(t, prob=0.3)
            population.append(t)

        return population

    # ═══════════════════════════════════════════════
    # 4. 遗传算子
    # ═══════════════════════════════════════════════

    def _tournament_select(self, population: List[List[int]], costs: List[float]) -> List[int]:
        """锦标赛选择: 随机抽 tournament_k 个个体，返回成本最小者的副本。"""
        k = min(self.tournament_k, len(population))
        candidates_idx = random.sample(range(len(population)), k)
        best_idx = min(candidates_idx, key=lambda i: costs[i])
        return population[best_idx][:]

    def _ox1_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Order Crossover (OX1): 经典排列交叉算子。
        随机选一个子片段, 从 p1 直接继承; 其余位置按 p2 中的顺序填充。
        对称地生成两个子代。
        """
        size = len(parent1)
        if size < 3:
            return parent1[:], parent2[:]

        a, b = sorted(random.sample(range(size), 2))

        def _ox1_single(p1, p2):
            child = [None] * size
            child[a:b+1] = p1[a:b+1]
            segment_set = set(child[a:b+1])
            fill_vals = [x for x in p2 if x not in segment_set]
            fill_idx = 0
            for i in range(size):
                if child[i] is None:
                    child[i] = fill_vals[fill_idx]
                    fill_idx += 1
            return child

        child1 = _ox1_single(parent1, parent2)
        child2 = _ox1_single(parent2, parent1)
        return child1, child2

    def _mutate_inplace(self, tour: List[int], prob: float = None) -> None:
        """
        原地变异 (随机选三种策略之一):
          - Swap:      随机交换两个位置
          - Insertion: 随机把一个基因抽出并插入别处
          - Inversion: 随机翻转一个子段
        """
        if prob is None:
            prob = self.mut_prob
        if random.random() > prob or len(tour) < 2:
            return

        strategy = random.choice(['swap', 'insert', 'invert'])

        if strategy == 'swap':
            i, j = random.sample(range(len(tour)), 2)
            tour[i], tour[j] = tour[j], tour[i]

        elif strategy == 'insert':
            i = random.randrange(len(tour))
            gene = tour.pop(i)
            j = random.randrange(len(tour))
            tour.insert(j, gene)

        else:  # inversion
            i, j = sorted(random.sample(range(len(tour)), 2))
            tour[i:j+1] = tour[i:j+1][::-1]

    # ═══════════════════════════════════════════════
    # 5. 主进化循环
    # ═══════════════════════════════════════════════

    def solve(self) -> Tuple[List[List[int]], float]:
        """
        执行遗传算法主循环。

        返回:
            best_solution : 最优路线列表 [[0,c1,...,0], ...]
            best_cost     : 对应总成本
        """
        start_time = time.time()

        # ── 初始化种群 ──
        population = self._init_population()

        # ── 首次评估 ──
        costs_routes = [self.fitness(ind) for ind in population]
        costs = [cr[0] for cr in costs_routes]
        routes_cache = [cr[1] for cr in costs_routes]

        best_idx = min(range(len(costs)), key=lambda i: costs[i])
        best_cost = costs[best_idx]
        best_solution = copy.deepcopy(routes_cache[best_idx])
        best_tour = population[best_idx][:]

        if self.verbose:
            print(f"[GA] 初始最优成本: {best_cost:.2f} | "
                  f"路径数: {len(best_solution)} | "
                  f"代数: {self.max_gen} | 迭代次数: {self.max_gen}")

        no_improve_gen = 0

        for gen in range(self.max_gen):
            new_population = []

            # ── 精英保留 ──
            sorted_idx = sorted(range(len(costs)), key=lambda i: costs[i])
            for ei in sorted_idx[:self.elite_size]:
                new_population.append(population[ei][:])

            # ── 繁殖 ──
            while len(new_population) < self.pop_size:
                p1 = self._tournament_select(population, costs)
                p2 = self._tournament_select(population, costs)

                if random.random() < self.cx_prob:
                    c1, c2 = self._ox1_crossover(p1, p2)
                else:
                    c1, c2 = p1[:], p2[:]

                self._mutate_inplace(c1)
                self._mutate_inplace(c2)

                new_population.append(c1)
                if len(new_population) < self.pop_size:
                    new_population.append(c2)

            population = new_population

            # ── 评估 ──
            costs_routes = [self.fitness(ind) for ind in population]
            costs = [cr[0] for cr in costs_routes]
            routes_cache = [cr[1] for cr in costs_routes]

            # ── 更新最优 ──
            gen_best_idx = min(range(len(costs)), key=lambda i: costs[i])
            gen_best_cost = costs[gen_best_idx]

            improved = False
            if gen_best_cost < best_cost:
                best_cost = gen_best_cost
                best_solution = copy.deepcopy(routes_cache[gen_best_idx])
                best_tour = population[gen_best_idx][:]
                no_improve_gen = 0
                improved = True
            else:
                no_improve_gen += 1

            avg_cost = sum(c for c in costs if c < float('inf')) / max(1, len(costs))
            self.best_cost_history.append(best_cost)
            self.avg_cost_history.append(avg_cost)

            # ── 日志 ──
            if self.verbose and (improved or (gen + 1) % 50 == 0):
                elapsed = time.time() - start_time
                flag = "[NEW BEST]" if improved else ""
                print(f"Iter {gen+1:4d}: Best={best_cost:.2f} | Avg={avg_cost:.2f} | "
                      f"Routes={len(best_solution)} | {flag} [{elapsed:.1f}s]")

            # ── 早停: 连续 200 代无改进 ──
            if no_improve_gen >= 200:
                if self.verbose:
                    print(f"[GA] 连续 200 代无改进，提前终止 (第 {gen+1} 代)")
                break

        # ── 最终汇报 ──
        elapsed = time.time() - start_time
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"【GA 最终结果】最优成本: {best_cost:.2f}")
            print(f"【GA 最终结果】路径数量: {len(best_solution)}")
            print(f"【GA 最终结果】总耗时: {elapsed:.2f}s")
            print(f"【GA 最终结果】运行代数: {len(self.best_cost_history)}")
            print(f"{'='*60}")

        return best_solution, best_cost
