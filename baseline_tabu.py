# -*- coding: utf-8 -*-
"""
baseline_tabu.py - 禁忌搜索 (Tabu Search) 基线求解器
====================================================
用途: 作为对比实验的第二个基线算法，与 ALNS+LLM 和 GA 方案共同比较。

算法设计:
  - 初始解:      贪心构造 (最近邻 + 时间窗感知)
  - 邻域结构:    三种移动算子交替探索
      * Or-Opt-1  (Relocate): 将单个客户从原路线取出，插入其他路线的最优位置
      * Or-Opt-2  (Move2):    将两个连续客户作为一组整体迁移
      * 2-opt*    (CrossSwap): 跨路线交换两段尾部 (Lin-Kernighan风格)
  - 禁忌表:      记录最近 tabu_tenure 次移动，阻止反向恢复
  - 破禁准则:    若候选解优于历史全局最优，无论是否禁忌均强制接受 (Aspiration)
  - 多样化:      连续一定迭代无改进时，随机扰动当前解 (Random Restart from best)

兼容性:
  - data 格式与整个项目完全一致
  - 成本函数使用 utils.FreshnessAndPenaltyCalculator (含 C2 + C3)
  - 输出格式: BEST_COST / NUM_ROUTES / Route i: [...]
"""

import sys
import random
import copy
import math
import time
from typing import List, Tuple, Dict, Optional, Set

# Windows GBK 终端 UTF-8 输出修复
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


# ──────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────

def _build_calculator(data: dict):
    """构建 FreshnessAndPenaltyCalculator (解耦导入)。"""
    try:
        from utils import FreshnessAndPenaltyCalculator
    except ImportError:
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from utils import FreshnessAndPenaltyCalculator
    return FreshnessAndPenaltyCalculator(data)


def _build_dist_matrix(customers: list) -> list:
    """构建欧氏距离矩阵。"""
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
# 核心类: TabuVRPSolver
# ──────────────────────────────────────────────────

class TabuVRPSolver:
    """
    基于禁忌搜索的 VRP-TW 求解器。

    参数:
        data          : dict, 标准数据字典
        max_iter      : int, 最大迭代次数 (默认 1000)
        tabu_tenure   : int, 禁忌期限，即每个动作被禁忌的迭代数 (默认 15)
        neighborhood_size: int, 每轮评估的邻域候选数量上限 (默认 50)
        no_improve_restart: int, 无改进多少轮后从最优解重启 (默认 150)
        seed          : int | None, 随机种子
        verbose       : bool, 是否打印详细日志
    """

    def __init__(
        self,
        data: dict,
        max_iter: int = 1000,
        tabu_tenure: int = 15,
        neighborhood_size: int = 50,
        no_improve_restart: int = 150,
        seed: Optional[int] = None,
        verbose: bool = True,
    ):
        self.data = data
        self.max_iter = max_iter
        self.tabu_tenure = tabu_tenure
        self.neighborhood_size = neighborhood_size
        self.no_improve_restart = no_improve_restart
        self.verbose = verbose

        if seed is not None:
            random.seed(seed)

        # ── 客户信息 ──
        self.customers = data['customers']
        self.n = len(self.customers)
        self.capacity = data.get('vehicle_capacity', 200)
        self.vehicle_speed = data.get('vehicle_speed_kmph', 40)

        self.depot = next(c for c in self.customers if c['id'] == 0)
        self.non_depot = [c for c in self.customers if c['id'] != 0]
        self.id_to_customer = {c['id']: c for c in self.customers}

        # ── 距离矩阵 & 成本计算 ──
        self.dist_matrix = _build_dist_matrix(self.customers)
        self.calculator = _build_calculator(data)

        # ── 禁忌表: {move_key: expire_iter} ──
        # move_key = (node_id, from_route_idx_hash, to_route_idx_hash)
        self.tabu_list: Dict[tuple, int] = {}

        # ── 历史记录 ──
        self.best_cost_history: List[float] = []
        self.current_cost_history: List[float] = []

        if self.verbose:
            print(f"[Tabu] 初始化完成 | 客户数: {len(self.non_depot)} | "
                  f"迭代: {max_iter} | 禁忌期限: {tabu_tenure} | "
                  f"邻域大小: {neighborhood_size}")

    # ═══════════════════════════════════════════════
    # 1. 成本计算
    # ═══════════════════════════════════════════════

    def calc_cost(self, routes: List[List[int]]) -> float:
        """
        总成本 = C11(固定) + C12(距离) + C13(制冷) + C2(鲜度) + C3(时间惩罚)
        与 HeuristicSolver.cost 完全一致。
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

    def route_cost(self, route: List[int]) -> float:
        """计算单条路线的变动成本 (不含固定成本)。"""
        if len(route) <= 2:
            return 0.0
        route_nodes = [self.id_to_customer[nid] for nid in route]
        return self.calculator.calculate_route_cost(route_nodes, self.dist_matrix)['variable_cost']

    # ═══════════════════════════════════════════════
    # 2. 可行性检查
    # ═══════════════════════════════════════════════

    def _route_feasible(self, route: List[int]) -> bool:
        """
        检查单条路线是否同时满足容量约束 + 软硬时间窗约束。
        使用 due_date 作为硬约束上界（到达不能超过 due_date）。
        """
        load = 0
        curr_time = 0.0
        prev_id = 0  # 从仓库出发

        for nid in route:
            if nid == 0:
                continue
            cust = self.id_to_customer[nid]
            load += cust.get('demand', 0)
            if load > self.capacity:
                return False

            travel = self.dist_matrix[prev_id][nid] * (60.0 / self.vehicle_speed)
            curr_time += travel

            due = cust.get('due_date', float('inf'))
            if curr_time > due:
                return False

            curr_time = max(curr_time, cust.get('ready_time', 0)) + cust.get('service_time', 0)
            prev_id = nid

        return True

    # ═══════════════════════════════════════════════
    # 3. 初始解构造 (贪心时间窗感知)
    # ═══════════════════════════════════════════════

    def _build_initial_solution(self) -> List[List[int]]:
        """
        贪心最近邻算法构造初始解，感知时间窗和容量约束。
        尝试三种排列策略，选最优。
        """
        best_sol = None
        best_cost = float('inf')

        strategies = [
            sorted(self.non_depot, key=lambda c: c.get('due_date', float('inf'))),   # 时间窗紧迫
            sorted(self.non_depot, key=lambda c: c.get('demand', 0), reverse=True),   # 大需求优先
            sorted(self.non_depot, key=lambda c: (                                    # 极角排序
                math.atan2(c.get('y', 0) - self.depot.get('y', 0),
                           c.get('x', 0) - self.depot.get('x', 0))
            )),
        ]

        for candidates in strategies:
            sol = self._greedy_construct(candidates)
            cost = self.calc_cost(sol)
            if cost < best_cost:
                best_cost = cost
                best_sol = sol

        return best_sol

    def _greedy_construct(self, ordered_candidates: list) -> List[List[int]]:
        """按给定顺序的客户列表进行贪心插入构造。"""
        routes = []
        remaining = list(ordered_candidates)

        while remaining:
            route = [0]
            load = 0
            curr_time = 0.0
            prev_id = 0

            inserted = True
            while inserted and remaining:
                inserted = False
                best_cust = None
                best_score = float('inf')

                for c in remaining:
                    cid = c['id']
                    demand = c.get('demand', 0)
                    if load + demand > self.capacity:
                        continue

                    travel = self.dist_matrix[prev_id][cid] * (60.0 / self.vehicle_speed)
                    arrival = curr_time + travel
                    due = c.get('due_date', float('inf'))
                    L_i = c.get('L_i', due)

                    if arrival > max(due, L_i):
                        continue

                    # 打分：距离 + 时间紧迫惩罚
                    urgency = max(0, arrival - due) * 0.5
                    score = self.dist_matrix[prev_id][cid] + urgency

                    if score < best_score:
                        best_score = score
                        best_cust = c

                if best_cust:
                    inserted = True
                    route.append(best_cust['id'])
                    load += best_cust.get('demand', 0)
                    travel = self.dist_matrix[prev_id][best_cust['id']] * (60.0 / self.vehicle_speed)
                    curr_time = max(curr_time + travel, best_cust.get('ready_time', 0))
                    curr_time += best_cust.get('service_time', 0)
                    prev_id = best_cust['id']
                    remaining.remove(best_cust)

            route.append(0)
            if len(route) > 2:
                routes.append(route)

        # 强制分配剩余无法合并的客户
        for c in remaining:
            routes.append([0, c['id'], 0])

        return routes

    # ═══════════════════════════════════════════════
    # 4. 邻域算子
    # ═══════════════════════════════════════════════

    def _get_neighbors_relocate(
        self,
        routes: List[List[int]],
        current_cost: float,
        iteration: int,
    ) -> List[tuple]:
        """
        Or-Opt-1 (Relocate): 将单个客户从原路线取出，插入其他路线的最优位置。
        返回候选列表: [(delta_cost, new_routes, move_key)]
        delta_cost < 0 表示改进。
        """
        candidates = []
        n_routes = len(routes)

        for r_from in range(n_routes):
            customers_in_route = [nid for nid in routes[r_from] if nid != 0]
            if not customers_in_route:
                continue

            for pos_in, node in enumerate(customers_in_route):
                # 计算从原路线移除后的收益
                route_without = [0] + [n for n in customers_in_route if n != node] + [0]
                cost_from_before = self.route_cost(routes[r_from])
                cost_from_after = self.route_cost(route_without) if len(route_without) > 2 else 0.0
                # 若原路线变空，还少一辆车 (减固定成本)
                remove_fixed = self.calculator.f if len(route_without) <= 2 else 0.0
                saving_from = cost_from_before - cost_from_after + remove_fixed

                # 尝试插入其他路线
                for r_to in range(n_routes):
                    if r_to == r_from:
                        continue

                    # 容量检查
                    cur_load = sum(self.id_to_customer[nid].get('demand', 0)
                                   for nid in routes[r_to] if nid != 0)
                    node_demand = self.id_to_customer[node].get('demand', 0)
                    if cur_load + node_demand > self.capacity:
                        continue

                    cost_to_before = self.route_cost(routes[r_to])
                    customers_to = [n for n in routes[r_to] if n != 0]

                    # 找最优插入位置
                    best_pos_cost = float('inf')
                    best_new_route_to = None
                    for ins_pos in range(len(customers_to) + 1):
                        new_to_customers = customers_to[:ins_pos] + [node] + customers_to[ins_pos:]
                        new_route_to = [0] + new_to_customers + [0]
                        if not self._route_feasible(new_route_to):
                            continue
                        c_after = self.route_cost(new_route_to)
                        if c_after < best_pos_cost:
                            best_pos_cost = c_after
                            best_new_route_to = new_route_to

                    if best_new_route_to is None:
                        continue

                    cost_inc_to = best_pos_cost - cost_to_before
                    delta = cost_inc_to - saving_from

                    # 构造新解
                    new_routes = [r[:] for r in routes]
                    new_routes[r_from] = route_without if len(route_without) > 2 else None
                    new_routes[r_to] = best_new_route_to
                    new_routes = [r for r in new_routes if r is not None]

                    move_key = ('relocate', node, r_from, r_to)
                    candidates.append((delta, new_routes, move_key))

        return candidates

    def _get_neighbors_or_opt2(
        self,
        routes: List[List[int]],
        current_cost: float,
        iteration: int,
    ) -> List[tuple]:
        """
        Or-Opt-2 (Move2): 将两个连续客户作为整体从原路线迁移到其他路线。
        """
        candidates = []
        n_routes = len(routes)

        for r_from in range(n_routes):
            customers_in_route = [nid for nid in routes[r_from] if nid != 0]
            if len(customers_in_route) < 2:
                continue

            for i in range(len(customers_in_route) - 1):
                seg = [customers_in_route[i], customers_in_route[i + 1]]
                seg_demand = sum(self.id_to_customer[n].get('demand', 0) for n in seg)

                route_without = [0] + [n for n in customers_in_route if n not in seg] + [0]
                cost_from_before = self.route_cost(routes[r_from])
                cost_from_after = self.route_cost(route_without) if len(route_without) > 2 else 0.0
                remove_fixed = self.calculator.f if len(route_without) <= 2 else 0.0
                saving_from = cost_from_before - cost_from_after + remove_fixed

                for r_to in range(n_routes):
                    if r_to == r_from:
                        continue

                    cur_load = sum(self.id_to_customer[nid].get('demand', 0)
                                   for nid in routes[r_to] if nid != 0)
                    if cur_load + seg_demand > self.capacity:
                        continue

                    cost_to_before = self.route_cost(routes[r_to])
                    customers_to = [n for n in routes[r_to] if n != 0]

                    best_pos_cost = float('inf')
                    best_new_route_to = None
                    for ins_pos in range(len(customers_to) + 1):
                        new_to_customers = customers_to[:ins_pos] + seg + customers_to[ins_pos:]
                        new_route_to = [0] + new_to_customers + [0]
                        if not self._route_feasible(new_route_to):
                            continue
                        c_after = self.route_cost(new_route_to)
                        if c_after < best_pos_cost:
                            best_pos_cost = c_after
                            best_new_route_to = new_route_to

                    if best_new_route_to is None:
                        continue

                    cost_inc_to = best_pos_cost - cost_to_before
                    delta = cost_inc_to - saving_from

                    new_routes = [r[:] for r in routes]
                    new_routes[r_from] = route_without if len(route_without) > 2 else None
                    new_routes[r_to] = best_new_route_to
                    new_routes = [r for r in new_routes if r is not None]

                    move_key = ('or2', tuple(seg), r_from, r_to)
                    candidates.append((delta, new_routes, move_key))

        return candidates

    def _get_neighbors_swap(
        self,
        routes: List[List[int]],
        current_cost: float,
        iteration: int,
    ) -> List[tuple]:
        """
        跨路线单点交换 (Cross-Route Swap):
        从两条不同路线各取一个客户，相互交换位置。
        """
        candidates = []
        n_routes = len(routes)

        route_pairs = []
        for r1 in range(n_routes):
            for r2 in range(r1 + 1, n_routes):
                route_pairs.append((r1, r2))

        # 限制评估对的数量，避免 O(n^4) 爆炸
        if len(route_pairs) > 30:
            route_pairs = random.sample(route_pairs, 30)

        for r1, r2 in route_pairs:
            c1_list = [n for n in routes[r1] if n != 0]
            c2_list = [n for n in routes[r2] if n != 0]
            if not c1_list or not c2_list:
                continue

            # 同样限制每对路线抽样几个客户比较
            sample1 = random.sample(c1_list, min(5, len(c1_list)))
            sample2 = random.sample(c2_list, min(5, len(c2_list)))

            for n1 in sample1:
                for n2 in sample2:
                    d1 = self.id_to_customer[n1].get('demand', 0)
                    d2 = self.id_to_customer[n2].get('demand', 0)

                    load1 = sum(self.id_to_customer[n].get('demand', 0) for n in c1_list)
                    load2 = sum(self.id_to_customer[n].get('demand', 0) for n in c2_list)

                    # 交换后容量检查
                    if (load1 - d1 + d2) > self.capacity:
                        continue
                    if (load2 - d2 + d1) > self.capacity:
                        continue

                    new_c1 = [n2 if n == n1 else n for n in c1_list]
                    new_c2 = [n1 if n == n2 else n for n in c2_list]
                    new_r1 = [0] + new_c1 + [0]
                    new_r2 = [0] + new_c2 + [0]

                    if not self._route_feasible(new_r1) or not self._route_feasible(new_r2):
                        continue

                    delta = (self.route_cost(new_r1) + self.route_cost(new_r2)
                             - self.route_cost(routes[r1]) - self.route_cost(routes[r2]))

                    new_routes = [r[:] for r in routes]
                    new_routes[r1] = new_r1
                    new_routes[r2] = new_r2

                    move_key = ('swap', min(n1, n2), max(n1, n2))
                    candidates.append((delta, new_routes, move_key))

        return candidates

    # ═══════════════════════════════════════════════
    # 5. 禁忌表操作
    # ═══════════════════════════════════════════════

    def _is_tabu(self, move_key: tuple, iteration: int) -> bool:
        """判断一个移动是否在禁忌期内。"""
        return self.tabu_list.get(move_key, 0) > iteration

    def _add_tabu(self, move_key: tuple, iteration: int) -> None:
        """将移动加入禁忌表，禁忌到 iteration + tabu_tenure。"""
        self.tabu_list[move_key] = iteration + self.tabu_tenure
        # 清理过期记录，避免禁忌表无限增长
        if len(self.tabu_list) > 500:
            expired = [k for k, v in self.tabu_list.items() if v <= iteration]
            for k in expired:
                del self.tabu_list[k]

    # ═══════════════════════════════════════════════
    # 6. 随机扰动 (多样化机制)
    # ═══════════════════════════════════════════════

    def _perturb(self, routes: List[List[int]], strength: int = 3) -> List[List[int]]:
        """
        随机扰动: 执行 `strength` 次随机 Relocate 移动，产生多样化解。
        用于从局部最优中跳出。
        """
        new_routes = copy.deepcopy(routes)
        for _ in range(strength):
            non_empty = [r for r in new_routes if len(r) > 2]
            if len(non_empty) < 2:
                break

            r_from_route = random.choice(non_empty)
            r_from_idx = new_routes.index(r_from_route)
            cust_list = [n for n in r_from_route if n != 0]
            if not cust_list:
                continue
            node = random.choice(cust_list)
            node_demand = self.id_to_customer[node].get('demand', 0)

            # 可行的目标路线 (不含自身)
            feasible_to = []
            for idx, r in enumerate(new_routes):
                if idx == r_from_idx:
                    continue
                cur_load = sum(self.id_to_customer[n].get('demand', 0) for n in r if n != 0)
                if cur_load + node_demand <= self.capacity:
                    feasible_to.append(idx)

            if not feasible_to:
                continue

            r_to_idx = random.choice(feasible_to)
            cust_to = [n for n in new_routes[r_to_idx] if n != 0]
            ins_pos = random.randint(0, len(cust_to))

            # 执行移动
            new_from = [0] + [n for n in cust_list if n != node] + [0]
            new_to = [0] + cust_to[:ins_pos] + [node] + cust_to[ins_pos:] + [0]

            new_routes[r_from_idx] = new_from
            new_routes[r_to_idx] = new_to
            new_routes = [r for r in new_routes if len(r) > 2]

        return new_routes

    # ═══════════════════════════════════════════════
    # 7. 主搜索循环
    # ═══════════════════════════════════════════════

    def solve(self) -> Tuple[List[List[int]], float]:
        """
        执行禁忌搜索主循环。

        返回:
            best_solution : 最优路线列表
            best_cost     : 对应总成本
        """
        start_time = time.time()

        # ── 构建初始解 ──
        current_routes = self._build_initial_solution()
        current_cost = self.calc_cost(current_routes)
        best_routes = copy.deepcopy(current_routes)
        best_cost = current_cost

        if self.verbose:
            print(f"[Tabu] 初始解 | 成本: {current_cost:.2f} | 路径数: {len(current_routes)} | "
                  f"迭代次数: {self.max_iter}")

        no_improve_count = 0
        improvement_count = 0

        for iteration in range(self.max_iter):

            # ── 生成邻域候选 ──
            all_candidates = []

            # Or-Opt-1 (Relocate) — 最核心的邻域
            reloc = self._get_neighbors_relocate(current_routes, current_cost, iteration)
            all_candidates.extend(reloc)

            # Or-Opt-2 (Move2) — 每隔几轮尝试一次，避免每轮都耗时过久
            if iteration % 3 == 0:
                or2 = self._get_neighbors_or_opt2(current_routes, current_cost, iteration)
                all_candidates.extend(or2)

            # Cross-Route Swap
            swap = self._get_neighbors_swap(current_routes, current_cost, iteration)
            all_candidates.extend(swap)

            if not all_candidates:
                # 邻域为空，直接扰动
                current_routes = self._perturb(best_routes, strength=5)
                current_cost = self.calc_cost(current_routes)
                no_improve_count += 1
                continue

            # ── 按 delta 升序排列 (delta 越小越好) ──
            all_candidates.sort(key=lambda x: x[0])

            # ── 选择最佳可接受移动 (禁忌过滤 + 破禁准则) ──
            chosen_delta = None
            chosen_routes = None
            chosen_key = None

            for delta, new_routes, move_key in all_candidates[:self.neighborhood_size]:
                new_cost = current_cost + delta

                # 破禁准则: 若超越历史最优，无条件接受
                aspiration = (new_cost < best_cost)

                if aspiration or not self._is_tabu(move_key, iteration):
                    chosen_delta = delta
                    chosen_routes = new_routes
                    chosen_key = move_key
                    break

            # 若所有候选都被禁忌 (极端情况)，强制接受最佳
            if chosen_routes is None and all_candidates:
                chosen_delta, chosen_routes, chosen_key = all_candidates[0]

            if chosen_routes is None:
                continue

            # ── 更新当前解 ──
            current_routes = chosen_routes
            current_cost = current_cost + chosen_delta
            self._add_tabu(chosen_key, iteration)

            # ── 更新全局最优 ──
            improved = False
            if current_cost < best_cost:
                best_cost = current_cost
                best_routes = copy.deepcopy(current_routes)
                no_improve_count = 0
                improvement_count += 1
                improved = True
            else:
                no_improve_count += 1

            # ── 记录历史 ──
            self.best_cost_history.append(best_cost)
            self.current_cost_history.append(current_cost)

            # ── 日志 ──
            if self.verbose and (improved or (iteration + 1) % 100 == 0):
                elapsed = time.time() - start_time
                flag = "[NEW BEST]" if improved else ""
                print(f"Iter {iteration+1:4d}: Current={current_cost:.2f} | "
                      f"Best={best_cost:.2f} | Routes={len(current_routes)} | "
                      f"{flag} [{elapsed:.1f}s]")

            # ── 多样化: 长时间无改进时扰动 ──
            if no_improve_count >= self.no_improve_restart:
                if self.verbose:
                    print(f"[Tabu] Iter {iteration+1}: {self.no_improve_restart} 轮无改进, 从最优解扰动重启")
                current_routes = self._perturb(copy.deepcopy(best_routes), strength=5)
                current_cost = self.calc_cost(current_routes)
                no_improve_count = 0
                # 清理禁忌表 (重启时赦免所有)
                self.tabu_list.clear()

        # ── 最终汇报 ──
        elapsed = time.time() - start_time
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[Tabu 最终结果] 最优成本: {best_cost:.2f}")
            print(f"[Tabu 最终结果] 路径数量: {len(best_routes)}")
            print(f"[Tabu 最终结果] 改进次数: {improvement_count}")
            print(f"[Tabu 最终结果] 总耗时:   {elapsed:.2f}s")
            print(f"{'='*60}")

        return best_routes, best_cost
