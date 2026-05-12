
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
    """贪心插入: 每个节点找全局最优位置 (完整成本计算 + 候选列表加速)"""
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
    """后悔插入: 优先插入后悔值最大的节点 (完整成本计算 + 候选列表加速)"""
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
