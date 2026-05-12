# -*- coding: utf-8 -*-
"""
baseline_alns.py - 标准 ALNS 求解器（手工算子，不使用 LLM）
==========================================================
提供与 LLM-ALNS 完全可比的基线:
  - 相同的成本评估函数 (FreshnessAndPenaltyCalculator)
  - 相同的 ALNS 框架 (模拟退火接受、轮盘赌调度、权重自适应)
  - 相同的初始解构建策略
  - 唯一区别：5 个算子是手工编写的，而非 LLM 生成

用法:
    from baseline_alns import ALNSVRPSolver
    solver = ALNSVRPSolver(data, max_iter=300, seed=42)
    best_solution, best_cost = solver.solve()
"""

import math
import copy
import random
from utils import FreshnessAndPenaltyCalculator


class ALNSVRPSolver:
    """
    标准自适应大邻域搜索 (ALNS) 求解器
    内置 3 个破坏算子 + 2 个修复算子
    """

    def __init__(
        self,
        data: dict,
        max_iter: int = 300,
        destruction_ratio: float = 0.3,
        seed: int = None,
        verbose: bool = True,
    ):
        self.data = data
        self.max_iter = max_iter
        self.destruction_ratio = destruction_ratio
        self.verbose = verbose

        if seed is not None:
            random.seed(seed)

        self.customers = data['customers']
        self.n = len(self.customers)
        self.capacity = data.get('vehicle_capacity', 200)
        self.id_to_customer = {c['id']: c for c in self.customers}
        self.customer_lookup = {c['id']: c for c in self.customers}

        # 距离矩阵
        self.dist_matrix = self._build_distance_matrix()

        # 成本计算器 (与 LLM-ALNS 共用)
        self.calculator = FreshnessAndPenaltyCalculator(data)

        # ALNS 算子权重
        self.destroy_ops = [
            self._random_removal,
            self._route_removal,
            self._string_removal,
        ]
        self.insert_ops = [
            self._greedy_insert,
            self._regret_insert,
        ]
        self.d_weights = [1.0] * len(self.destroy_ops)
        self.i_weights = [1.0] * len(self.insert_ops)
        self.last_d_idx = 0
        self.last_i_idx = 0

        # 自适应参数 (与 heuristic_skeleton 一致)
        self.rho = 0.1
        self.sigma1 = 33   # 全局最优
        self.sigma2 = 9    # 改善
        self.sigma3 = 3    # 接受但未改善
        self.sigma4 = 0    # 拒绝

        # 统计
        self.op_stats = {
            'destroy': {i: {'uses': 0, 'successes': 0, 'score': 0}
                        for i in range(len(self.destroy_ops))},
            'insert':  {i: {'uses': 0, 'successes': 0, 'score': 0}
                        for i in range(len(self.insert_ops))},
        }

        # 公共输出 (供 runner 读取)
        self.cost_history = []
        self.best_cost_history = []

        # 候选插入位置数量（加速 r/rc 等难实例）
        self.candidate_list_size = 10

    def _candidate_positions(self, route, node_id):
        """只评估少量候选插入位置，降低成本计算开销。"""
        if not route or len(route) <= 2:
            return range(1, len(route))

        k = self.candidate_list_size
        if k is None or k <= 0 or k >= len(route):
            return range(1, len(route))

        scored = []
        for pos in range(1, len(route)):
            prev_id = route[pos - 1]
            next_id = route[pos]
            d = self.dist_matrix[node_id][prev_id] + self.dist_matrix[node_id][next_id]
            scored.append((d, pos))

        scored.sort(key=lambda x: x[0])
        return [pos for _, pos in scored[:k]]

    # ──────────────────────────────────────────
    # 基础工具
    # ──────────────────────────────────────────

    def _build_distance_matrix(self):
        n = self.n
        dm = [[0.0] * n for _ in range(n)]
        for i in range(n):
            xi, yi = self.customers[i]['x'], self.customers[i]['y']
            for j in range(i + 1, n):
                xj, yj = self.customers[j]['x'], self.customers[j]['y']
                d = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                dm[i][j] = d
                dm[j][i] = d
        return dm

    def _route_distance(self, route):
        return sum(self.dist_matrix[route[k]][route[k + 1]]
                   for k in range(len(route) - 1))

    # ──────────────────────────────────────────
    # 成本 & 验证
    # ──────────────────────────────────────────

    def cost(self, solution):
        if not solution:
            return float('inf')
        num_vehicles = len(solution)
        fixed_cost = num_vehicles * self.calculator.f
        total_var = 0.0
        for route_ids in solution:
            route_nodes = [self.id_to_customer[nid] for nid in route_ids]
            info = self.calculator.calculate_route_cost(route_nodes, self.dist_matrix)
            total_var += info['variable_cost']
        return fixed_cost + total_var

    def validate(self, solution):
        if not solution:
            return False
        visited = set()
        for route_ids in solution:
            load = 0
            curr_time = 0.0
            for i, nid in enumerate(route_ids):
                c = self.id_to_customer[nid]
                if nid != 0:
                    visited.add(nid)
                    load += c.get('demand', 0)
                if i > 0:
                    prev_id = route_ids[i - 1]
                    d = self.dist_matrix[prev_id][nid]
                    curr_time += d * (60.0 / self.calculator.v)
                if nid != 0:
                    if curr_time > c.get('due_date', float('inf')):
                        return False
                    curr_time = max(curr_time, c.get('ready_time', 0))
                    curr_time += c.get('service_time', 0)
            if load > self.capacity:
                return False
        expected = {c['id'] for c in self.customers if c['id'] != 0}
        return visited == expected

    # ──────────────────────────────────────────
    # 破坏算子
    # ──────────────────────────────────────────

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

    # ──────────────────────────────────────────
    # 修复算子
    # ──────────────────────────────────────────

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
                for pos in self._candidate_positions(route, node):
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
                    for pos in self._candidate_positions(route, node):
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

    # ──────────────────────────────────────────
    # 权重自适应
    # ──────────────────────────────────────────

    def _update_weights(self, reward: str):
        score_map = {
            'global_best': self.sigma1,
            'improved':    self.sigma2,
            'accepted':    self.sigma3,
            'rejected':    self.sigma4,
        }
        score = score_map.get(reward, 0)
        self.d_weights[self.last_d_idx] = (
            self.d_weights[self.last_d_idx] * (1 - self.rho) + score * self.rho)
        self.i_weights[self.last_i_idx] = (
            self.i_weights[self.last_i_idx] * (1 - self.rho) + score * self.rho)
        self.d_weights = [max(w, 0.1) for w in self.d_weights]
        self.i_weights = [max(w, 0.1) for w in self.i_weights]
        self.op_stats['destroy'][self.last_d_idx]['uses'] += 1
        self.op_stats['destroy'][self.last_d_idx]['score'] += score
        self.op_stats['insert'][self.last_i_idx]['uses'] += 1
        self.op_stats['insert'][self.last_i_idx]['score'] += score
        if reward in ('global_best', 'improved'):
            self.op_stats['destroy'][self.last_d_idx]['successes'] += 1
            self.op_stats['insert'][self.last_i_idx]['successes'] += 1

    # ──────────────────────────────────────────
    # 局部搜索
    # ──────────────────────────────────────────

    def _two_opt(self, route):
        if len(route) < 4:
            return route
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    a, b = route[i - 1], route[i]
                    c, d = route[j], route[j + 1]
                    old = self.dist_matrix[a][b] + self.dist_matrix[c][d]
                    new = self.dist_matrix[a][c] + self.dist_matrix[b][d]
                    if new < old - 1e-3:
                        route[i:j + 1] = route[i:j + 1][::-1]
                        improved = True
        return route

    # ──────────────────────────────────────────
    # 初始解构建
    # ──────────────────────────────────────────

    def _build_initial_solution(self):
        non_depot = [c for c in self.customers if c['id'] != 0]
        id_to_idx = {c['id']: i for i, c in enumerate(self.customers)}

        strategies = [
            lambda cs: sorted(cs, key=lambda c: c.get('demand', 0), reverse=True),
            lambda cs: sorted(cs, key=lambda c: c.get('due_date', 1e9) - c.get('ready_time', 0)),
            lambda cs: random.sample(cs, len(cs)),
        ]

        # 大实例仅使用一种策略，避免初始解构建过慢
        if len(non_depot) >= 80:
            strategies = [strategies[0]]

        # 初始解候选集大小（加速）
        candidate_k = 30

        best_sol = None
        best_c = float('inf')

        for strat in strategies:
            remaining = strat(list(non_depot))
            solution = []
            while remaining:
                route = [0]
                load = 0
                pos_id = 0
                curr_time = 0.0
                while remaining:
                    best_cust = None
                    best_score = float('inf')
                    if len(remaining) > candidate_k:
                        candidates = random.sample(remaining, candidate_k)
                    else:
                        candidates = remaining
                    for c in candidates:
                        cid = c['id']
                        if load + c.get('demand', 0) > self.capacity:
                            continue
                        d = self.dist_matrix[id_to_idx[pos_id]][id_to_idx[cid]]
                        tt = d * (60.0 / self.calculator.v)
                        arr = curr_time + tt
                        li = c.get('L_i', c.get('due_date', float('inf')))
                        if arr > li:
                            continue
                        urg = max(0, arr - c.get('due_date', li))
                        sc = d + urg * 0.5
                        if sc < best_score:
                            best_score = sc
                            best_cust = c
                    if best_cust is None:
                        break
                    route.append(best_cust['id'])
                    load += best_cust.get('demand', 0)
                    d = self.dist_matrix[id_to_idx[pos_id]][id_to_idx[best_cust['id']]]
                    curr_time += d * (60.0 / self.calculator.v)
                    curr_time = max(curr_time, best_cust.get('ready_time', 0))
                    curr_time += best_cust.get('service_time', 0)
                    pos_id = best_cust['id']
                    remaining.remove(best_cust)
                route.append(0)
                if len(route) > 2:
                    solution.append(route)
            for c in remaining:
                solution.append([0, c['id'], 0])
            c_val = self.cost(solution)
            if c_val < best_c:
                best_c = c_val
                best_sol = copy.deepcopy(solution)

        return best_sol, best_c

    # ──────────────────────────────────────────
    # 主求解循环
    # ──────────────────────────────────────────

    def solve(self):
        non_depot = [c for c in self.customers if c['id'] != 0]

        if self.verbose:
            print("[ALNS-Baseline] Building initial solution ...")

        current_solution, current_cost = self._build_initial_solution()
        is_feasible = self.validate(current_solution)

        if self.verbose:
            print(f"[Initial] Cost: {current_cost:.2f} | Routes: {len(current_solution)} | Feasible: {is_feasible}")

        if not is_feasible or current_cost == float('inf'):
            current_solution = [[0, c['id'], 0] for c in non_depot]
            random.shuffle(current_solution)
            current_cost = self.cost(current_solution)

        best_solution = copy.deepcopy(current_solution)
        best_cost = current_cost

        # 温度参数 (与 heuristic_skeleton 一致)
        T = max(current_cost * 0.10, 100) if current_cost < float('inf') else 100
        target_ratio = 0.02
        alpha = target_ratio ** (1.0 / self.max_iter)

        segment_size = max(50, int(self.max_iter * 0.25))
        restart_threshold = max(60, int(self.max_iter * 0.6))

        improvement_count = 0
        no_improve = 0
        last_improve_iter = 0

        if self.verbose:
            print(f"[Params] T0={T:.2f}, alpha={alpha:.6f}, iters={self.max_iter}")
            print("[ALNS-Baseline] Starting iterations ...\n")

        for it in range(self.max_iter):
            # 自适应破坏程度
            thr_med = self.max_iter * 0.27
            thr_high = self.max_iter * 0.53
            if no_improve > thr_high:
                base_rm = max(3, int(len(non_depot) * 0.20))
                max_rm = max(5, int(len(non_depot) * 0.30))
            elif no_improve > thr_med:
                base_rm = max(2, int(len(non_depot) * 0.15))
                max_rm = max(4, int(len(non_depot) * 0.20))
            else:
                base_rm = max(2, int(len(non_depot) * 0.12))
                max_rm = max(3, int(len(non_depot) * 0.18))
            num_rm = random.randint(base_rm, max_rm)

            # 选择算子
            self.last_d_idx = random.choices(
                range(len(self.destroy_ops)), weights=self.d_weights)[0]
            self.last_i_idx = random.choices(
                range(len(self.insert_ops)), weights=self.i_weights)[0]
            d_op = self.destroy_ops[self.last_d_idx]
            i_op = self.insert_ops[self.last_i_idx]

            try:
                temp_sol = copy.deepcopy(current_solution)
                partial, removed = d_op(temp_sol, num_rm)
                if removed:
                    new_sol = i_op(partial, removed)
                else:
                    new_sol = partial
                if not self.validate(new_sol):
                    raise ValueError("infeasible")
                new_cost = self.cost(new_sol)
            except Exception:
                new_cost = float('inf')
                new_sol = current_solution

            delta = new_cost - current_cost

            if new_cost < current_cost:
                current_solution = new_sol
                current_cost = new_cost
                no_improve = 0
                if new_cost < best_cost:
                    best_solution = copy.deepcopy(new_sol)
                    best_cost = new_cost
                    improvement_count += 1
                    last_improve_iter = it
                    self._update_weights('global_best')
                    if self.verbose:
                        print(f"Iter {it+1:4d}: ** New best = {best_cost:.2f} | Routes={len(best_solution)}")
                else:
                    self._update_weights('improved')
            elif T > 0.01 and delta < float('inf'):
                limit = max(current_cost * 1.5, best_cost * 2.0)
                if new_cost <= limit:
                    if random.random() < math.exp(-delta / T):
                        current_solution = new_sol
                        current_cost = new_cost
                        self._update_weights('accepted')
                        no_improve += 1
                    else:
                        self._update_weights('rejected')
                        no_improve += 1
                else:
                    self._update_weights('rejected')
                    no_improve += 1
            else:
                self._update_weights('rejected')
                no_improve += 1

            T *= alpha

            self.cost_history.append(current_cost)
            self.best_cost_history.append(best_cost)

            # 重启
            if no_improve > restart_threshold:
                current_solution = copy.deepcopy(best_solution)
                current_cost = best_cost
                no_improve = 0
                T = max(best_cost * 0.10, 100) * 0.5
                if self.verbose:
                    print(f"Iter {it+1:4d}: Restart (reheat T={T:.1f})")

            # 权重归一化
            if (it + 1) % segment_size == 0:
                ds = sum(self.d_weights)
                if ds > 0:
                    self.d_weights = [w / ds * len(self.d_weights) for w in self.d_weights]
                iss = sum(self.i_weights)
                if iss > 0:
                    self.i_weights = [w / iss * len(self.i_weights) for w in self.i_weights]

            if self.verbose and (it + 1) % 50 == 0:
                print(f"Iter {it+1:4d}: Current={current_cost:.2f}, Best={best_cost:.2f}, T={T:.2f}")

        # 后处理 2-opt
        for idx in range(len(best_solution)):
            best_solution[idx] = self._two_opt(best_solution[idx])
        post_cost = self.cost(best_solution)
        if post_cost < best_cost:
            best_cost = post_cost
            if self.verbose:
                print(f"[Post] 2-opt improved to {best_cost:.2f}")

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[Result] Best cost:  {best_cost:.2f}")
            print(f"[Result] Vehicles:   {len(best_solution)}")
            print(f"[Result] Improvements: {improvement_count}")
            print(f"[Result] Last improve: iter {last_improve_iter}")
            print(f"{'='*60}")

            d_names = ['random_removal', 'route_removal', 'string_removal']
            i_names = ['greedy_insert', 'regret_insert']
            print("\n[Operator Stats]")
            for i, nm in enumerate(d_names):
                s = self.op_stats['destroy'][i]
                r = (s['successes'] / s['uses'] * 100) if s['uses'] > 0 else 0
                print(f"  {nm}: {s['uses']} uses, {s['successes']} success ({r:.1f}%)")
            for i, nm in enumerate(i_names):
                s = self.op_stats['insert'][i]
                r = (s['successes'] / s['uses'] * 100) if s['uses'] > 0 else 0
                print(f"  {nm}: {s['uses']} uses, {s['successes']} success ({r:.1f}%)")

        return best_solution, best_cost
