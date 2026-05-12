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

try:
    from tqdm import trange, tqdm
except Exception:
    trange = None
    tqdm = None


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
        ls_enabled: bool = True,
        ls_steps: int = 25,
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
        self.ls_enabled = ls_enabled
        self.ls_steps = max(0, int(ls_steps))
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
        """使用 Optimal Split 算法将 Giant Tour 拆分为最优的路线列表。"""
        n = len(tour)
        # V[i] 存储拆分前 i 个节点的最优成本
        V = [float('inf')] * (n + 1)
        V[0] = 0.0
        # P[i] 存储到达 i 的最优前驱节点
        P = [0] * (n + 1)

        for i in range(n):
            if V[i] == float('inf'):
                continue
            
            route_ids = [0]
            current_load = 0
            current_time = 0.0
            current_node = 0
            
            for j in range(i + 1, n + 1):
                cid = tour[j - 1]
                c = self.id_to_customer[cid]
                demand = c.get('demand', 0)
                ready = c.get('ready_time', 0)
                due = c.get('due_date', float('inf'))
                l_i = c.get('L_i', due)
                svc = c.get('service_time', 0)
                
                travel = self._dist_by_id(current_node, cid) * (60.0 / self.calculator.v)
                arrival = current_time + travel
                
                if current_load + demand > self.capacity:
                    break
                    
                # 只有当包含多于1个客户时才因时间窗超限而断开，保证必定有解
                if j > i + 1 and arrival > max(due, l_i):
                    break
                    
                current_load += demand
                current_time = max(arrival, ready) + svc
                current_node = cid
                route_ids.append(cid)
                
                # 计算精确成本
                temp_route = [self.id_to_customer[nid] for nid in route_ids] + [self.id_to_customer[0]]
                info = self.calculator.calculate_route_cost(temp_route, self.dist_matrix)
                edge_cost = self.calculator.f + info['variable_cost']
                
                if V[i] + edge_cost < V[j]:
                    V[j] = V[i] + edge_cost
                    P[j] = i
                    
                # 提前剪枝: 如果成本过大(例如 C2 货损爆发或严重的硬时间窗惩罚)
                if info['variable_cost'] > 2e4:
                    break

        routes = []
        curr = n
        while curr > 0:
            prev = P[curr]
            route = [0] + tour[prev:curr] + [0]
            routes.append(route)
            curr = prev
            
        routes.reverse()
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
        """构造一只蚂蚁的客户访问序列。融合时间窗等待时间作为启发式因子。"""
        unvisited = set(self.customer_ids)
        current = 0
        tour = []
        
        current_time = 0.0
        current_load = 0

        while unvisited:
            candidates = list(unvisited)
            desirability = []

            i = self.id_to_idx[current]
            for cid in candidates:
                j = self.id_to_idx[cid]
                tau = max(1e-12, self.pheromone[i][j])
                d = self.dist_matrix[i][j]
                
                # 预估到达时间和等待时间
                c = self.id_to_customer[cid]
                travel = d * (60.0 / self.calculator.v)
                arrival = current_time + travel
                
                due = c.get('due_date', float('inf'))
                l_i = c.get('L_i', due)
                
                # 如果超载或超过最晚时间窗，概念上相当于从仓库出发新路线
                if current_load + c.get('demand', 0) > self.capacity or arrival > max(due, l_i):
                    travel_from_depot = self.dist_matrix[0][j] * (60.0 / self.calculator.v)
                    arrival = travel_from_depot
                    
                wait = max(0.0, c.get('ready_time', 0) - arrival)
                
                # 启发式因子：结合距离和等待时间
                eta = 1.0 / (d + 0.5 * wait + 1e-6)
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
            
            # 更新蚂蚁当前状态
            c = self.id_to_customer[next_id]
            demand = c.get('demand', 0)
            due = c.get('due_date', float('inf'))
            l_i = c.get('L_i', due)
            
            travel = self.dist_matrix[i][self.id_to_idx[next_id]] * (60.0 / self.calculator.v)
            arrival = current_time + travel
            
            if current_load + demand > self.capacity or arrival > max(due, l_i):
                current_load = demand
                arrival = self.dist_matrix[0][self.id_to_idx[next_id]] * (60.0 / self.calculator.v)
                current_time = max(arrival, c.get('ready_time', 0)) + c.get('service_time', 0)
            else:
                current_load += demand
                current_time = max(arrival, c.get('ready_time', 0)) + c.get('service_time', 0)
                
            current = next_id

        return tour

    def _tour_cost(self, tour: List[int]) -> Tuple[List[List[int]], float]:
        routes = self.decode(tour)
        return routes, self.calc_cost(routes)

    def _local_search_tour(
        self,
        tour: List[int],
        base_cost: Optional[float] = None,
    ) -> Tuple[List[int], List[List[int]], float]:
        """对单个 Giant Tour 做轻量局部搜索（swap/insert/inversion）。"""
        if not tour or self.ls_steps <= 0:
            routes, cost = self._tour_cost(tour)
            return tour[:], routes, cost

        best_tour = tour[:]
        if base_cost is None:
            best_routes, best_cost = self._tour_cost(best_tour)
        else:
            best_routes = self.decode(best_tour)
            best_cost = base_cost

        n = len(best_tour)
        if n < 2:
            return best_tour, best_routes, best_cost

        for _ in range(self.ls_steps):
            cand = best_tour[:]
            move = random.choice(['swap', 'insert', 'invert'])

            if move == 'swap':
                i, j = random.sample(range(n), 2)
                cand[i], cand[j] = cand[j], cand[i]
            elif move == 'insert':
                i = random.randrange(n)
                v = cand.pop(i)
                j = random.randrange(n)
                cand.insert(j, v)
            else:
                i, j = sorted(random.sample(range(n), 2))
                cand[i:j + 1] = reversed(cand[i:j + 1])

            cand_routes, cand_cost = self._tour_cost(cand)
            if cand_cost < best_cost:
                best_tour = cand
                best_routes = cand_routes
                best_cost = cand_cost

        return best_tour, best_routes, best_cost

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

        use_tqdm = self.verbose and trange is not None
        if use_tqdm:
            iter_range = trange(1, self.max_iter + 1, desc="ACO Iter", ncols=100, mininterval=0.5)
        else:
            iter_range = range(1, self.max_iter + 1)

        for it in iter_range:
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

            # 每轮只对迭代最优解做一次局部搜索，控制计算量
            if self.ls_enabled and iter_best_tour is not None and self.ls_steps > 0:
                ls_tour, ls_routes, ls_cost = self._local_search_tour(
                    iter_best_tour,
                    base_cost=iter_best_cost,
                )
                if ls_cost < iter_best_cost:
                    iter_best_tour = ls_tour
                    iter_best_routes = ls_routes
                    iter_best_cost = ls_cost
                if ls_cost < global_best_cost:
                    global_best_cost = ls_cost
                    global_best_routes = copy.deepcopy(ls_routes)
                    global_best_tour = ls_tour[:]

            self._evaporate()
            if iter_best_tour is not None:
                self._deposit(iter_best_tour, iter_best_cost)
            if global_best_tour:
                self._deposit(global_best_tour, global_best_cost)

            self.best_cost_history.append(global_best_cost)

            if self.verbose and (not use_tqdm) and (it == 1 or it % 20 == 0):
                elapsed = time.time() - start
                print(
                    f"Iter {it:4d}: Best={global_best_cost:.2f} | "
                    f"Routes={len(global_best_routes)} | [{elapsed:.1f}s]"
                )

            # tqdm 模式下也定期写出心跳日志，避免用户误判“卡住”
            if self.verbose and use_tqdm and (it == 1 or it % 50 == 0):
                elapsed = time.time() - start
                try:
                    iter_range.set_postfix(best=f"{global_best_cost:.2f}", routes=len(global_best_routes), elapsed=f"{elapsed:.1f}s")
                    if tqdm is not None:
                        tqdm.write(f"[ACO] iter={it}/{self.max_iter}, best={global_best_cost:.2f}, routes={len(global_best_routes)}, elapsed={elapsed:.1f}s")
                except Exception:
                    pass

        return global_best_routes, global_best_cost
