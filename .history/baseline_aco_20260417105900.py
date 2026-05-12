"""
baseline_aco.py - 蚁群算法 (ACO) 基线求解器
=============================================
用途: 在与 ALNS / GA 相同成本模型下，提供可比的 ACO 基线。

设计要点:
  - 路径构造: 基于信息素 + 启发式距离的客户访问序列 (Giant Tour)
  - 解码方式: 贪心拆分 (容量 + 时间窗可行性)
  - 成本函数: utils.FreshnessAndPenaltyCalculator (与 ALNS / GA 对齐)
"""

from __future__ import annotations

import math
import random
import copy
import time
from typing import List, Tuple, Optional


def _build_calculator(data: dict):
    try:
        from utils import FreshnessAndPenaltyCalculator
    except ImportError:
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from utils import FreshnessAndPenaltyCalculator
    return FreshnessAndPenaltyCalculator(data)


def _build_dist_matrix(customers: list) -> list:
    n = len(customers)
    dm = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi = customers[i]['x'], customers[i]['y']
            xj, yj = customers[j]['x'], customers[j]['y']
            d = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            dm[i][j] = d
            dm[j][i] = d
    return dm


class ACOVRPSolver:
    """ACO VRP-TW 基线求解器。"""

    def __init__(
        self,
        data: dict,
        n_ants: int = 40,
        max_iter: int = 300,
        alpha: float = 1.0,
        beta: float = 3.0,
        rho: float = 0.15,
        q0: float = 0.15,
        seed: Optional[int] = None,
        verbose: bool = True,
    ):
        self.data = data
        self.n_ants = n_ants
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        self.verbose = verbose

        if seed is not None:
            random.seed(seed)

        self.customers = data['customers']
        self.capacity = data.get('vehicle_capacity', 200)
        self.id_to_customer = {c['id']: c for c in self.customers}

        self.dist_matrix = _build_dist_matrix(self.customers)
        self.calculator = _build_calculator(data)

        self.customer_ids = [c['id'] for c in self.customers if c['id'] != 0]
        self.n_nodes = len(self.customers)

        # 假设 Solomon 的 id 与索引一致；若不一致则使用映射兜底
        self.id_to_idx = {c['id']: i for i, c in enumerate(self.customers)}

        self.pheromone = [[1.0] * self.n_nodes for _ in range(self.n_nodes)]
        self.best_cost_history: List[float] = []

    def _dist_by_id(self, id_i: int, id_j: int) -> float:
        return self.dist_matrix[self.id_to_idx[id_i]][self.id_to_idx[id_j]]

    def decode(self, tour: List[int]) -> List[List[int]]:
        """Giant Tour -> 多条可行路径。"""
        routes = []
        route = [0]
        current_load = 0
        current_time = 0.0
        current = 0

        for cid in tour:
            c = self.id_to_customer[cid]
            demand = c.get('demand', 0)
            ready = c.get('ready_time', 0)
            due = c.get('due_date', float('inf'))
            l_i = c.get('L_i', due)
            service = c.get('service_time', 0)

            travel = self._dist_by_id(current, cid) * (60.0 / self.calculator.v)
            arrival = current_time + travel

            capacity_ok = (current_load + demand <= self.capacity)
            time_ok = (arrival <= max(due, l_i))

            if not capacity_ok or not time_ok:
                route.append(0)
                if len(route) > 2:
                    routes.append(route)
                route = [0]
                current_load = 0
                current_time = 0.0
                current = 0
                travel = self._dist_by_id(0, cid) * (60.0 / self.calculator.v)
                arrival = travel

            current_load += demand
            current_time = max(arrival, ready) + service
            route.append(cid)
            current = cid

        route.append(0)
        if len(route) > 2:
            routes.append(route)
        return routes

    def calc_cost(self, routes: List[List[int]]) -> float:
        if not routes:
            return float('inf')
        fixed_cost = len(routes) * self.calculator.f
        variable_cost = 0.0
        for route_ids in routes:
            route_nodes = [self.id_to_customer[nid] for nid in route_ids]
            info = self.calculator.calculate_route_cost(route_nodes, self.dist_matrix)
            variable_cost += info['variable_cost']
        return fixed_cost + variable_cost

    def _construct_tour(self) -> List[int]:
        """构造一只蚂蚁的客户访问序列。"""
        unvisited = set(self.customer_ids)
        current = 0
        tour = []

        while unvisited:
            candidates = list(unvisited)
            desirability = []

            i = self.id_to_idx[current]
            for cid in candidates:
                j = self.id_to_idx[cid]
                tau = max(1e-12, self.pheromone[i][j])
                d = self.dist_matrix[i][j]
                eta = 1.0 / (d + 1e-6)
                desirability.append((tau ** self.alpha) * (eta ** self.beta))

            if random.random() < self.q0:
                # 贪心选择
                next_id = candidates[max(range(len(candidates)), key=lambda k: desirability[k])]
            else:
                total = sum(desirability)
                if total <= 1e-12:
                    next_id = random.choice(candidates)
                else:
                    r = random.random() * total
                    acc = 0.0
                    next_id = candidates[-1]
                    for cid, w in zip(candidates, desirability):
                        acc += w
                        if acc >= r:
                            next_id = cid
                            break

            tour.append(next_id)
            unvisited.remove(next_id)
            current = next_id

        return tour

    def _evaporate(self):
        keep = max(0.0, 1.0 - self.rho)
        for i in range(self.n_nodes):
            row = self.pheromone[i]
            for j in range(self.n_nodes):
                row[j] *= keep
                if row[j] < 1e-12:
                    row[j] = 1e-12

    def _deposit(self, tour: List[int], cost: float):
        if cost <= 0:
            return
        delta = 1.0 / cost
        full = [0] + tour + [0]
        for k in range(len(full) - 1):
            i = self.id_to_idx[full[k]]
            j = self.id_to_idx[full[k + 1]]
            self.pheromone[i][j] += delta
            self.pheromone[j][i] += delta

    def solve(self) -> Tuple[List[List[int]], float]:
        start = time.time()
        global_best_cost = float('inf')
        global_best_routes: List[List[int]] = []
        global_best_tour: List[int] = []

        for it in range(1, self.max_iter + 1):
            iter_best_cost = float('inf')
            iter_best_routes = None
            iter_best_tour = None

            for _ in range(self.n_ants):
                tour = self._construct_tour()
                routes = self.decode(tour)
                cost = self.calc_cost(routes)

                if cost < iter_best_cost:
                    iter_best_cost = cost
                    iter_best_routes = routes
                    iter_best_tour = tour

                if cost < global_best_cost:
                    global_best_cost = cost
                    global_best_routes = copy.deepcopy(routes)
                    global_best_tour = tour[:]

            self._evaporate()
            if iter_best_tour is not None:
                self._deposit(iter_best_tour, iter_best_cost)
            if global_best_tour:
                self._deposit(global_best_tour, global_best_cost)

            self.best_cost_history.append(global_best_cost)

            if self.verbose and (it == 1 or it % 20 == 0):
                elapsed = time.time() - start
                print(
                    f"Iter {it:4d}: Best={global_best_cost:.2f} | "
                    f"Routes={len(global_best_routes)} | [{elapsed:.1f}s]"
                )

        return global_best_routes, global_best_cost
