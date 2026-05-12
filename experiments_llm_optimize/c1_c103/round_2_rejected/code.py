def _random_removal(self, solution, ratio):
    """随机移除: 但优先移除时间窗紧张的节点（概率加权）"""
    all_nodes = []
    time_window_tightness = []
    
    # 计算每个节点的时间窗紧张度
    for ri, route in enumerate(solution):
        if len(route) <= 2:
            continue
            
        # 计算路径到达时间
        arrival_times = [0]
        for i in range(1, len(route)-1):
            prev_node = route[i-1]
            curr_node = route[i]
            travel_time = self.dist_matrix[prev_node][curr_node] / self.calculator.v
            service_time = self.customer_lookup[prev_node].get('service_time', 0) if prev_node != 0 else 0
            arrival = arrival_times[-1] + service_time + travel_time
            arrival_times.append(arrival)
        
        # 评估每个客户的时间窗紧张度
        for pi, node in enumerate(route):
            if node == 0:
                continue
            cust = self.customer_lookup[node]
            ready, due = cust['ready_time'], cust['due_date']
            arrival = arrival_times[pi-1] if pi > 0 else 0
            
            # 紧张度 = 距离时间窗边界的最近距离（越小越紧张）
            if arrival < ready:
                tightness = ready - arrival
            elif arrival > due:
                tightness = arrival - due
            else:
                # 在时间窗内，计算到边界的距离
                tightness = min(arrival - ready, due - arrival)
            
            all_nodes.append((ri, pi, node))
            time_window_tightness.append(max(0.1, 1.0 / (tightness + 1.0)))  # 紧张度越高，权重越大
    
    if not all_nodes:
        return solution, []
    
    total = len(all_nodes)
    k = max(1, int(total * ratio)) if ratio <= 1 else min(total, int(ratio))
    
    # 根据时间窗紧张度加权采样
    if len(time_window_tightness) == len(all_nodes):
        # 归一化权重
        weights = [w/sum(time_window_tightness) for w in time_window_tightness]
        selected_indices = random.choices(range(len(all_nodes)), weights=weights, k=k)
        selected = [all_nodes[i] for i in selected_indices]
    else:
        selected = random.sample(all_nodes, k)
    
    selected.sort(key=lambda x: (x[0], x[1]), reverse=True)
    new_sol = [r[:] for r in solution]
    removed = []
    for ri, pi, node in selected:
        del new_sol[ri][pi]
        removed.append(node)
    
    # 清理空路径
    new_sol = [r for r in new_sol if len(r) > 2]
    return new_sol, removed


def _route_removal(self, solution, ratio):
    """路径移除: 优先移除成本效益低的路径（单位客户成本高）"""
    routes_info = []
    route_costs = []
    
    for ri, route in enumerate(solution):
        custs = [n for n in route if n != 0]
        if not custs:
            continue
        
        # 计算路径成本
        route_nodes = [self.id_to_customer[n] for n in route]
        cost_info = self.calculator.calculate_route_cost(route_nodes, self.dist_matrix)
        total_cost = cost_info['variable_cost'] + self.calculator.f
        
        # 成本效益 = 总成本 / 客户数（越低越好）
        cost_per_customer = total_cost / len(custs)
        
        routes_info.append((ri, custs, cost_per_customer))
        route_costs.append(cost_per_customer)
    
    if not routes_info:
        return solution, []
    
    total_customers = sum(len(c) for _, c, _ in routes_info)
    target = max(1, int(total_customers * ratio)) if ratio <= 1 else int(ratio)
    
    # 按成本效益排序（成本效益高的先移除）
    routes_info.sort(key=lambda x: x[2], reverse=True)
    
    removed = []
    to_remove_idx = []
    
    for ri, custs, _ in routes_info:
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
    """连续节点移除: 优先移除地理位置连续的节点（基于距离矩阵）"""
    all_custs = [n for route in solution for n in route if n != 0]
    if not all_custs:
        return solution, []
    
    total = len(all_custs)
    n_remove = max(1, int(total * ratio)) if ratio <= 1 else min(total, int(ratio))
    n_remove = min(n_remove, total)
    
    removed = []
    new_sol = [r[:] for r in solution]
    
    while len(removed) < n_remove:
        # 选择非空路径
        valid_routes = [(i, r) for i, r in enumerate(new_sol) if len(r) > 2]
        if not valid_routes:
            break
        
        # 随机选择一条路径
        ri, route = random.choice(valid_routes)
        
        # 在路径中寻找连续节点（基于距离）
        if len(route) <= 3:
            start = 1
            slen = min(len(route) - 2, n_remove - len(removed))
        else:
            # 寻找路径中距离最近的两个连续节点作为起点
            min_dist = float('inf')
            best_start = 1
            
            for i in range(1, len(route) - 2):
                node1 = route[i]
                node2 = route[i + 1]
                if node1 == 0 or node2 == 0:
                    continue
                dist = self.dist_matrix[node1][node2]
                if dist < min_dist:
                    min_dist = dist
                    best_start = i
            
            start = best_start
            # 移除长度：1-3个节点，但不超过剩余需要移除的数量
            slen = min(random.randint(1, 3), 
                      len(route) - start - 1,
                      n_remove - len(removed))
        
        # 执行移除
        for _ in range(slen):
            if start < len(new_sol[ri]) - 1:
                node = new_sol[ri][start]
                if node != 0:
                    removed.append(node)
                    del new_sol[ri][start]
    
    # 清理空路径
    new_sol = [r for r in new_sol if len(r) > 2]
    return new_sol, removed


def _greedy_insert(self, solution, removed_nodes):
    """贪心插入: 使用候选位置大幅减少计算量"""
    if not removed_nodes:
        return solution
    if not solution:
        return [[0, n, 0] for n in removed_nodes]
    
    new_sol = [r[:] for r in solution]
    
    # 缓存路径成本
    route_cost_cache = {}
    for ri, route in enumerate(new_sol):
        route_nodes = [self.id_to_customer[n] for n in route]
        cost_info = self.calculator.calculate_route_cost(route_nodes, self.dist_matrix)
        route_cost_cache[ri] = cost_info['variable_cost']
    
    for node in removed_nodes:
        node_demand = self.customer_lookup[node].get('demand', 0)
        best_inc = float('inf')
        best_ri = None
        best_pos = None
        
        # 评估新路径成本
        nr = [0, node, 0]
        nr_nodes = [self.id_to_customer[n] for n in nr]
        nr_cost_info = self.calculator.calculate_route_cost(nr_nodes, self.dist_matrix)
        nr_cost = nr_cost_info['variable_cost'] + self.calculator.f
        
        for ri, route in enumerate(new_sol):
            # 容量检查
            ld = sum(self.customer_lookup[n].get('demand', 0) for n in route if n != 0)
            if ld + node_demand > self.capacity:
                continue
            
            # 使用候选位置而不是遍历所有位置
            candidate_positions = self._candidate_positions(route, node)
            if not candidate_positions:
                continue
            
            c_before = route_cost_cache.get(ri)
            if c_before is None:
                route_nodes = [self.id_to_customer[n] for n in route]
                cost_info = self.calculator.calculate_route_cost(route_nodes, self.dist_matrix)
                c_before = cost_info['variable_cost']
                route_cost_cache[ri] = c_before
            
            for pos in candidate_positions:
                # 构建插入后的路径
                route_after = route[:pos] + [node] + route[pos:]
                ra_nodes = [self.id_to_customer[n] for n in route_after]
                c_after = self.calculator.calculate_route_cost(ra_nodes, self.dist_matrix)['variable_cost']
                inc = c_after - c_before
                
                if inc < best_inc:
                    best_inc = inc
                    best_ri = ri
                    best_pos = pos
        
        # 决策：插入现有路径还是创建新路径
        if best_ri is not None and best_inc <= nr_cost:
            new_sol[best_ri].insert(best_pos, node)
            # 更新缓存
            route_after = new_sol[best_ri]
            ra_nodes = [self.id_to_customer[n] for n in route_after]
            new_cost = self.calculator.calculate_route_cost(ra_nodes, self.dist_matrix)['variable_cost']
            route_cost_cache[best_ri] = new_cost
        else:
            new_sol.append([0, node, 0])
            # 为新路径添加缓存
            ri = len(new_sol) - 1
            nr_nodes = [self.id_to_customer[n] for n in new_sol[ri]]
            new_cost = self.calculator.calculate_route_cost(nr_nodes, self.dist_matrix)['variable_cost']
            route_cost_cache[ri] = new_cost
    
    return new_sol


def _regret_insert(self, solution, removed_nodes):
    """后悔插入: 使用Regret-3和候选位置，大幅优化计算"""
    if not removed_nodes:
        return solution
    if not solution:
        return [[0, n, 0] for n in removed_nodes]
    
    new_sol = [r[:] for r in solution]
    remaining = list(removed_nodes)
    
    # 缓存路径成本
    route_cost_cache = {}
    for ri, route in enumerate(new_sol):
        route_nodes = [self.id_to_customer[n] for n in route]
        cost_info = self.calculator.calculate_route_cost(route_nodes, self.dist_matrix)
        route_cost_cache[ri] = cost_info['variable_cost']
    
    while remaining:
        best_regret = -float('inf')
        best_node = None
        best_ri = None
        best_pos = None
        
        # 计算新路径成本（用于所有节点）
        new_route_base_cost = {}
        for node in remaining:
            nr = [0, node, 0]
            nr_nodes = [self.id_to_customer[n] for n in nr]
            nr_cost_info = self.calculator.calculate_route_cost(nr_nodes, self.dist_matrix)
            new_route_base_cost[node] = nr_cost_info['variable_cost'] + self.calculator.f
        
        for node in remaining:
            nd = self.customer_lookup[node].get('demand', 0)
            insertion_costs = []  # (cost, ri, pos)
            
            # 评估现有路径的插入
            for ri, route in enumerate(new_sol):
                # 容量检查
                ld = sum(self.customer_lookup[n].get('demand', 0) for n in route if n != 0)
                if ld + nd > self.capacity:
                    continue
                
                # 使用候选位置
                candidate_positions = self._candidate_positions(route, node)
                if not candidate_positions:
                    continue
                
                c_before = route_cost_cache.get(ri)
                if c_before is None:
                    route_nodes = [self.id_to_customer[n] for n in route]
                    cost_info = self.calculator.calculate_route_cost(route_nodes, self.dist_matrix)
                    c_before = cost_info['variable_cost']
                    route_cost_cache[ri] = c_before
                
                # 评估每个候选位置
                for pos in candidate_positions:
                    route_after = route[:pos] + [node] + route[pos:]
                    ra_nodes = [self.id_to_customer[n] for n in route_after]
                    c_after = self.calculator.calculate_route_cost(ra_nodes, self.dist_matrix)['variable_cost']
                    inc = c_after - c_before
                    insertion_costs.append((inc, ri, pos))
            
            # 添加新路径选项
            insertion_costs.append((new_route_base_cost[node], None, None))
            
            # 按成本排序，取前3个最佳位置计算Regret-3
            insertion_costs.sort(key=lambda x: x[0])
            
            if len(insertion_costs) >= 3:
                # Regret-3: 计算插入第2、3佳位置与最佳位置的差异
                regret = (insertion_costs[1][0] - insertion_costs[0][0]) + \
                        (insertion_costs[2][0] - insertion_costs[0][0])
            elif len(insertion_costs) >= 2:
                # Regret-2
                regret = insertion_costs[1][0] - insertion_costs[0][0]
            else:
                regret = 0
            
            if regret > best_regret:
                best_regret = regret
                best_node = node
                best_ri = insertion_costs[0][1]
                best_pos = insertion_costs[0][2]
        
        if best_node is None:
            # 所有剩余节点都无法插入现有路径，创建新路径
            for n in remaining:
                new_sol.append([0, n, 0])
            break
        
        # 执行最佳插入
        if best_ri is not None:
            new_sol[best_ri].insert(best_pos, best_node)
            # 更新缓存
            route_after = new_sol[best_ri]
            ra_nodes = [self.id_to_customer[n] for n in route_after]
            new_cost = self.calculator.calculate_route_cost(ra_nodes, self.dist_matrix)['variable_cost']
            route_cost_cache[best_ri] = new_cost
        else:
            new_sol.append([0, best_node, 0])
            # 为新路径添加缓存
            ri = len(new_sol) - 1
            nr_nodes = [self.id_to_customer[n] for n in new_sol[ri]]
            new_cost = self.calculator.calculate_route_cost(nr_nodes, self.dist_matrix)['variable_cost']
            route_cost_cache[ri] = new_cost
        
        remaining.remove(best_node)
    
    return new_sol