def _random_removal(self, solution, ratio):
    """改进的随机移除：基于时间窗压力加权选择"""
    all_nodes = []
    for ri, route in enumerate(solution):
        for pi, node in enumerate(route):
            if node != 0:
                # 计算时间窗压力：时间窗越窄，压力越大
                cust = self.customer_lookup[node]
                time_window_width = cust['due_date'] - cust['ready_time']
                # 归一化压力值：时间窗越窄，权重越高
                pressure = 1.0 / (time_window_width + 1.0)  # +1避免除零
                all_nodes.append((ri, pi, node, pressure))
    
    if not all_nodes:
        return solution, []
    
    total = len(all_nodes)
    k = max(1, int(total * ratio)) if ratio <= 1 else min(total, int(ratio))
    k = min(k, total)
    
    # 基于压力值加权随机选择
    weights = [item[3] for item in all_nodes]
    selected_indices = random.choices(range(len(all_nodes)), weights=weights, k=k)
    selected = [all_nodes[i] for i in selected_indices]
    
    # 去重并排序
    selected = list(set(selected))
    selected.sort(key=lambda x: (x[0], x[1]), reverse=True)
    
    new_sol = [r[:] for r in solution]
    removed = []
    for ri, pi, node, _ in selected:
        if len(removed) >= k:
            break
        if pi < len(new_sol[ri]) and new_sol[ri][pi] == node:
            del new_sol[ri][pi]
            removed.append(node)
    
    new_sol = [r for r in new_sol if len(r) > 2]
    return new_sol, removed[:k]


def _route_removal(self, solution, ratio):
    """高货损路径移除：优先移除货损成本高的路径"""
    if not solution:
        return solution, []
    
    # 计算每条路径的货损成本
    route_costs = []
    for ri, route in enumerate(solution):
        custs = [n for n in route if n != 0]
        if not custs:
            continue
            
        route_nodes = [self.id_to_customer[n] for n in route]
        cost_info = self.calculator.calculate_route_cost(route_nodes, self.dist_matrix)
        # 重点关注货损成本C2
        c2_cost = cost_info.get('c2', 0)
        route_costs.append((ri, custs, c2_cost))
    
    if not route_costs:
        return solution, []
    
    # 按货损成本降序排序
    route_costs.sort(key=lambda x: x[2], reverse=True)
    
    total_customers = sum(len(c) for _, c, _ in route_costs)
    target = max(1, int(total_customers * ratio)) if ratio <= 1 else int(ratio)
    
    removed = []
    to_remove_idx = []
    for ri, custs, _ in route_costs:
        if len(removed) + len(custs) > target and len(removed) > 0:
            # 如果添加这条路径会超过目标，则跳过
            continue
        removed.extend(custs)
        to_remove_idx.append(ri)
        if len(removed) >= target:
            break
    
    # 如果一条路径都没移除，至少移除货损最高的路径
    if not to_remove_idx and route_costs:
        ri, custs, _ = route_costs[0]
        removed.extend(custs)
        to_remove_idx.append(ri)
    
    to_remove_idx.sort(reverse=True)
    new_sol = [r[:] for r in solution]
    for ri in to_remove_idx:
        del new_sol[ri]
    
    return new_sol, removed[:target]


def _string_removal(self, solution, ratio):
    """时空聚类移除：移除同一时间窗和地理区域的连续节点"""
    all_custs = [n for route in solution for n in route if n != 0]
    if not all_custs:
        return solution, []
    
    total = len(all_custs)
    n_remove = max(1, int(total * ratio)) if ratio <= 1 else min(total, int(ratio))
    n_remove = min(n_remove, total)
    
    removed = []
    new_sol = [r[:] for r in solution]
    
    # 找到所有非空路径
    valid_routes = [(i, r) for i, r in enumerate(new_sol) if len(r) > 2]
    if not valid_routes:
        return new_sol, removed
    
    # 选择一条随机路径
    ri, route = random.choice(valid_routes)
    
    # 找到路径中时间窗最密集的区域
    if len(route) > 3:
        # 计算每个位置的时间窗密度
        density = []
        for i in range(1, len(route) - 1):
            if route[i] == 0:
                continue
            cust = self.customer_lookup[route[i]]
            # 计算与前后客户的时间窗重叠度
            window_center = (cust['ready_time'] + cust['due_date']) / 2
            density.append((i, window_center))
        
        if density:
            # 按时间窗中心排序，找到最密集的连续段
            density.sort(key=lambda x: x[1])
            # 选择中间段作为起始点
            mid_idx = len(density) // 2
            start_pos = density[mid_idx][0]
            
            # 确定移除长度
            max_len = min(5, len(route) - start_pos - 1, n_remove)
            slen = random.randint(2, max_len)
            
            # 移除连续段
            for offset in range(slen):
                pos = start_pos + offset
                if pos < len(new_sol[ri]) and new_sol[ri][pos] != 0:
                    removed.append(new_sol[ri][pos])
                    del new_sol[ri][pos]
    
    # 如果时空聚类移除不够，补充随机连续移除
    while len(removed) < n_remove:
        valid = [(i, r) for i, r in enumerate(new_sol) if len(r) > 2]
        if not valid:
            break
        
        ri, route = random.choice(valid)
        if len(route) <= 2:
            continue
            
        start = random.randint(1, len(route) - 2)
        remaining = n_remove - len(removed)
        slen = min(random.randint(1, 3), len(route) - start - 1, remaining)
        
        for _ in range(slen):
            if start < len(new_sol[ri]) and new_sol[ri][start] != 0:
                removed.append(new_sol[ri][start])
                del new_sol[ri][start]
    
    new_sol = [r for r in new_sol if len(r) > 2]
    return new_sol, removed[:n_remove]


def _greedy_insert(self, solution, removed_nodes):
    """改进的贪心插入：使用候选位置+噪声扰动"""
    if not removed_nodes:
        return solution
    if not solution:
        return [[0, n, 0] for n in removed_nodes]
    
    new_sol = [r[:] for r in solution]
    
    # 打乱节点顺序以增加多样性
    nodes_to_insert = list(removed_nodes)
    random.shuffle(nodes_to_insert)
    
    for node in nodes_to_insert:
        node_demand = self.customer_lookup[node].get('demand', 0)
        node_info = self.id_to_customer[node]
        
        best_score = float('inf')
        best_ri = None
        best_pos = None
        
        # 评估现有路径
        for ri, route in enumerate(new_sol):
            # 容量检查
            current_load = sum(self.customer_lookup[n].get('demand', 0) for n in route if n != 0)
            if current_load + node_demand > self.capacity:
                continue
            
            # 获取候选位置
            candidate_positions = self._candidate_positions(route, node)
            if not candidate_positions:
                continue
            
            route_before = [self.id_to_customer[n] for n in route]
            c_before = self.calculator.calculate_route_cost(
                route_before, self.dist_matrix)['variable_cost']
            
            for pos in candidate_positions:
                # 构建插入后的路径
                route_after = route[:pos] + [node] + route[pos:]
                ra_nodes = [self.id_to_customer[n] for n in route_after]
                c_after = self.calculator.calculate_route_cost(
                    ra_nodes, self.dist_matrix)['variable_cost']
                
                inc = c_after - c_before
                
                # 添加噪声扰动（±10%）
                noise = random.uniform(0.9, 1.1)
                score = inc * noise
                
                if score < best_score:
                    best_score = score
                    best_ri = ri
                    best_pos = pos
        
        # 评估新路径
        new_route = [0, node, 0]
        new_route_nodes = [self.id_to_customer[n] for n in new_route]
        new_route_cost = self.calculator.calculate_route_cost(
            new_route_nodes, self.dist_matrix)['variable_cost']
        new_route_cost += self.calculator.f
        
        # 添加噪声扰动
        new_score = new_route_cost * random.uniform(0.9, 1.1)
        
        if best_ri is not None and best_score <= new_score:
            new_sol[best_ri].insert(best_pos, node)
        else:
            new_sol.append([0, node, 0])
    
    return new_sol


def _regret_insert(self, solution, removed_nodes):
    """Regret-3插入：考虑前3个最佳位置的机会成本"""
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
            node_demand = self.customer_lookup[node].get('demand', 0)
            node_info = self.id_to_customer[node]
            
            # 收集所有可行插入选项
            options = []
            
            # 现有路径选项
            for ri, route in enumerate(new_sol):
                current_load = sum(self.customer_lookup[n].get('demand', 0) for n in route if n != 0)
                if current_load + node_demand > self.capacity:
                    continue
                
                route_before = [self.id_to_customer[n] for n in route]
                c_before = self.calculator.calculate_route_cost(
                    route_before, self.dist_matrix)['variable_cost']
                
                # 使用候选位置而不是遍历所有位置
                candidate_positions = self._candidate_positions(route, node)
                for pos in candidate_positions:
                    route_after = route[:pos] + [node] + route[pos:]
                    ra_nodes = [self.id_to_customer[n] for n in route_after]
                    c_after = self.calculator.calculate_route_cost(
                        ra_nodes, self.dist_matrix)['variable_cost']
                    
                    inc = c_after - c_before
                    options.append((inc, ri, pos))
            
            # 新路径选项
            new_route = [0, node, 0]
            new_route_nodes = [self.id_to_customer[n] for n in new_route]
            new_route_cost = self.calculator.calculate_route_cost(
                new_route_nodes, self.dist_matrix)['variable_cost']
            new_route_cost += self.calculator.f
            options.append((new_route_cost, None, None))
            
            # 按成本排序
            options.sort(key=lambda x: x[0])
            
            # 计算Regret-3：前3个最佳位置的机会成本差异
            if len(options) >= 3:
                # regret = sum(option[i] - option[0]) for i in 1,2
                regret = (options[1][0] - options[0][0]) + (options[2][0] - options[0][0])
            elif len(options) >= 2:
                regret = options[1][0] - options[0][0]
            else:
                regret = 0
            
            # 加权调整：给新路径选项更高的权重（鼓励减少车辆数）
            if options[0][1] is None:  # 最佳选项是新路径
                regret *= 1.2  # 增加20%权重
            
            if regret > best_regret:
                best_regret = regret
                best_node = node
                best_ri = options[0][1]
                best_pos = options[0][2]
        
        if best_node is None:
            # 如果找不到合适位置，创建新路径
            for n in remaining:
                new_sol.append([0, n, 0])
            break
        
        # 执行插入
        if best_ri is not None:
            new_sol[best_ri].insert(best_pos, best_node)
        else:
            new_sol.append([0, best_node, 0])
        
        remaining.remove(best_node)
    
    return new_sol