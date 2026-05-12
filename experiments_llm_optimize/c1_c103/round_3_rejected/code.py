def _random_removal(self, solution, ratio):
    """随机移除: 分散选择节点，但优先移除高成本节点"""
    all_nodes = []
    for ri, route in enumerate(solution):
        if len(route) <= 2:
            continue
            
        # 计算路径中每个节点的成本贡献
        route_customers = [self.id_to_customer[n] for n in route if n != 0]
        if not route_customers:
            continue
            
        # 计算完整路径成本
        route_cost = self.calculator.calculate_route_cost(route_customers, self.dist_matrix)
        base_cost = route_cost['variable_cost']
        
        # 计算移除每个节点后的成本变化
        for pi, node in enumerate(route):
            if node == 0:
                continue
                
            # 创建移除该节点后的路径
            route_without = [n for n in route if n != node]
            if len(route_without) <= 2:  # 只剩下仓库
                cost_without = 0
            else:
                route_without_customers = [self.id_to_customer[n] for n in route_without if n != 0]
                cost_without = self.calculator.calculate_route_cost(
                    route_without_customers, self.dist_matrix)['variable_cost']
            
            # 成本减少量越大，说明该节点成本贡献越高
            cost_reduction = base_cost - cost_without
            # 添加权重：时间窗越紧张、到达时间越晚的节点权重更高
            customer = self.id_to_customer[node]
            time_window_tightness = 1.0 / (customer['due_date'] - customer['ready_time'] + 1)
            weighted_cost = cost_reduction * (1 + time_window_tightness)
            
            all_nodes.append((ri, pi, node, weighted_cost))
    
    if not all_nodes:
        return solution, []
    
    total = len(all_nodes)
    k = max(1, int(total * ratio)) if ratio <= 1 else min(total, int(ratio))
    k = min(k, total)
    
    # 按成本贡献加权随机选择（成本贡献高的节点有更高概率被移除）
    weights = [node[3] for node in all_nodes]
    selected_indices = random.choices(range(len(all_nodes)), weights=weights, k=k)
    selected = [all_nodes[i] for i in selected_indices]
    
    selected.sort(key=lambda x: (x[0], x[1]), reverse=True)
    new_sol = [r[:] for r in solution]
    removed = []
    for ri, pi, node, _ in selected:
        if pi < len(new_sol[ri]):
            del new_sol[ri][pi]
            removed.append(node)
    
    new_sol = [r for r in new_sol if len(r) > 2]
    return new_sol, removed


def _route_removal(self, solution, ratio):
    """路径移除: 优先移除成本高或时间窗紧张的路径"""
    routes_info = []
    for ri, route in enumerate(solution):
        custs = [n for n in route if n != 0]
        if not custs:
            continue
            
        # 计算路径成本
        route_customers = [self.id_to_customer[n] for n in custs]
        route_cost = self.calculator.calculate_route_cost(route_customers, self.dist_matrix)
        
        # 计算路径的时间窗紧张度（平均时间窗宽度）
        total_tightness = 0
        for cust in route_customers:
            time_window_width = cust['due_date'] - cust['ready_time']
            total_tightness += 1.0 / (time_window_width + 1)
        avg_tightness = total_tightness / len(route_customers)
        
        # 路径评分 = 成本 × 时间窗紧张度
        route_score = route_cost['variable_cost'] * (1 + avg_tightness)
        routes_info.append((ri, custs, route_score))
    
    if not routes_info:
        return solution, []
    
    total_customers = sum(len(c) for _, c, _ in routes_info)
    target = max(1, int(total_customers * ratio)) if ratio <= 1 else int(ratio)
    
    # 按路径评分排序（评分高的优先移除）
    routes_info.sort(key=lambda x: x[2], reverse=True)
    
    removed = []
    to_remove_idx = []
    for ri, custs, _ in routes_info:
        if len(removed) + len(custs) > target and len(removed) > 0:
            # 如果添加这条路径会超过目标，且已有移除的节点，则停止
            break
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
    """连续节点移除: 优先移除时间窗连续的节点段"""
    # 收集所有非空路径
    valid_routes = [(i, r) for i, r in enumerate(solution) if len(r) > 2]
    if not valid_routes:
        return solution, []
    
    all_custs = [n for route in solution for n in route if n != 0]
    total = len(all_custs)
    n_remove = max(1, int(total * ratio)) if ratio <= 1 else min(total, int(ratio))
    n_remove = min(n_remove, total)
    
    removed = []
    new_sol = [r[:] for r in solution]
    
    while len(removed) < n_remove:
        # 选择一条路径
        ri, route = random.choice(valid_routes)
        
        # 计算路径中每个位置的时间窗连续性得分
        if len(route) <= 2:
            continue
            
        # 找到时间窗最连续的段
        best_start = 1
        best_score = -float('inf')
        best_length = 1
        
        for start in range(1, len(route) - 1):
            if route[start] == 0:
                continue
                
            # 计算从start开始的连续段的时间窗重叠度
            max_len = min(5, len(route) - start, n_remove - len(removed))
            for length in range(1, max_len + 1):
                if start + length >= len(route) or route[start + length - 1] == 0:
                    break
                    
                # 计算这段节点的时间窗重叠度
                segment = route[start:start + length]
                customers = [self.id_to_customer[n] for n in segment if n != 0]
                
                if len(customers) < 2:
                    continue
                    
                # 计算时间窗的重叠程度
                min_due = min(c['due_date'] for c in customers)
                max_ready = max(c['ready_time'] for c in customers)
                overlap = max(0, min_due - max_ready)
                overlap_score = overlap / (min_due - customers[0]['ready_time'] + 1) if min_due > customers[0]['ready_time'] else 0
                
                if overlap_score > best_score:
                    best_score = overlap_score
                    best_start = start
                    best_length = length
        
        # 移除最佳连续段
        slen = min(best_length, n_remove - len(removed))
        for i in range(slen):
            if best_start < len(new_sol[ri]):
                node = new_sol[ri][best_start]
                if node != 0:
                    removed.append(node)
                    del new_sol[ri][best_start]
        
        # 更新有效路径列表
        valid_routes = [(i, r) for i, r in enumerate(new_sol) if len(r) > 2]
        if not valid_routes:
            break
    
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
    route_costs = {}
    for ri, route in enumerate(new_sol):
        if len(route) > 2:
            route_customers = [self.id_to_customer[n] for n in route if n != 0]
            route_costs[ri] = self.calculator.calculate_route_cost(
                route_customers, self.dist_matrix)['variable_cost']
        else:
            route_costs[ri] = 0
    
    for node in removed_nodes:
        node_demand = self.customer_lookup[node].get('demand', 0)
        best_inc = float('inf')
        best_ri = None
        best_pos = None
        
        # 评估新路径成本
        new_route_cost = self.calculator.calculate_route_cost(
            [self.id_to_customer[node]], self.dist_matrix)['variable_cost']
        new_route_total = new_route_cost + self.calculator.f
        
        for ri, route in enumerate(new_sol):
            # 检查容量约束
            ld = sum(self.customer_lookup[n].get('demand', 0) for n in route if n != 0)
            if ld + node_demand > self.capacity:
                continue
            
            # 获取候选位置（最多10个）
            candidate_positions = self._candidate_positions(route, node)
            if not candidate_positions:
                continue
            
            c_before = route_costs.get(ri, 0)
            
            for pos in candidate_positions:
                # 创建插入后的路径
                route_after = route[:pos] + [node] + route[pos:]
                route_after_customers = [self.id_to_customer[n] for n in route_after if n != 0]
                
                # 计算插入后成本
                c_after = self.calculator.calculate_route_cost(
                    route_after_customers, self.dist_matrix)['variable_cost']
                inc = c_after - c_before
                
                if inc < best_inc:
                    best_inc = inc
                    best_ri = ri
                    best_pos = pos
        
        # 决策：插入现有路径还是创建新路径
        if best_ri is not None and best_inc <= new_route_total:
            new_sol[best_ri].insert(best_pos, node)
            # 更新缓存
            updated_route = new_sol[best_ri]
            updated_customers = [self.id_to_customer[n] for n in updated_route if n != 0]
            route_costs[best_ri] = self.calculator.calculate_route_cost(
                updated_customers, self.dist_matrix)['variable_cost']
        else:
            new_sol.append([0, node, 0])
            # 新路径成本缓存
            route_costs[len(new_sol)-1] = new_route_cost
    
    return new_sol


def _regret_insert(self, solution, removed_nodes):
    """后悔插入: 使用候选位置和Regret-3机制"""
    if not removed_nodes:
        return solution
    if not solution:
        return [[0, n, 0] for n in removed_nodes]
    
    new_sol = [r[:] for r in solution]
    remaining = list(removed_nodes)
    
    # 缓存路径成本
    route_costs = {}
    for ri, route in enumerate(new_sol):
        if len(route) > 2:
            route_customers = [self.id_to_customer[n] for n in route if n != 0]
            route_costs[ri] = self.calculator.calculate_route_cost(
                route_customers, self.dist_matrix)['variable_cost']
        else:
            route_costs[ri] = 0
    
    while remaining:
        best_regret = -float('inf')
        best_node = None
        best_ri = None
        best_pos = None
        
        for node in remaining:
            nd = self.customer_lookup[node].get('demand', 0)
            insertion_costs = []
            
            # 评估现有路径
            for ri, route in enumerate(new_sol):
                ld = sum(self.customer_lookup[n].get('demand', 0)
                        for n in route if n != 0)
                if ld + nd > self.capacity:
                    continue
                
                c_before = route_costs.get(ri, 0)
                best_inc_r = float('inf')
                best_pos_r = None
                
                # 使用候选位置
                candidate_positions = self._candidate_positions(route, node)
                if not candidate_positions:
                    continue
                
                for pos in candidate_positions:
                    route_after = route[:pos] + [node] + route[pos:]
                    route_after_customers = [self.id_to_customer[n] for n in route_after if n != 0]
                    c_after = self.calculator.calculate_route_cost(
                        route_after_customers, self.dist_matrix)['variable_cost']
                    inc = c_after - c_before
                    
                    if inc < best_inc_r:
                        best_inc_r = inc
                        best_pos_r = pos
                
                if best_pos_r is not None:
                    insertion_costs.append((best_inc_r, ri, best_pos_r))
            
            # 新路径选项
            new_route_cost = self.calculator.calculate_route_cost(
                [self.id_to_customer[node]], self.dist_matrix)['variable_cost']
            new_route_total = new_route_cost + self.calculator.f
            insertion_costs.append((new_route_total, None, None))
            
            # 计算Regret-3（考虑前3个最佳选择）
            if len(insertion_costs) >= 2:
                insertion_costs.sort(key=lambda x: x[0])
                # 取前3个或全部
                k = min(3, len(insertion_costs))
                top_k = insertion_costs[:k]
                
                # 计算后悔值：第2佳选择与最佳选择的成本差
                regret = 0
                for i in range(1, k):
                    regret += (top_k[i][0] - top_k[0][0]) * (1.0 / i)  # 权重递减
                
                if regret > best_regret:
                    best_regret = regret
                    best_node = node
                    best_ri = top_k[0][1]
                    best_pos = top_k[0][2]
            elif insertion_costs:
                # 只有一个选择
                if 0 > best_regret:  # 后悔值为0
                    best_regret = 0
                    best_node = node
                    best_ri = insertion_costs[0][1]
                    best_pos = insertion_costs[0][2]
        
        if best_node is None:
            # 无法插入任何现有路径，全部创建新路径
            for n in remaining:
                new_sol.append([0, n, 0])
            break
        
        # 执行插入
        if best_ri is not None:
            new_sol[best_ri].insert(best_pos, best_node)
            # 更新缓存
            updated_route = new_sol[best_ri]
            updated_customers = [self.id_to_customer[n] for n in updated_route if n != 0]
            route_costs[best_ri] = self.calculator.calculate_route_cost(
                updated_customers, self.dist_matrix)['variable_cost']
        else:
            new_sol.append([0, best_node, 0])
            # 新路径成本缓存
            new_route_cost = self.calculator.calculate_route_cost(
                [self.id_to_customer[best_node]], self.dist_matrix)['variable_cost']
            route_costs[len(new_sol)-1] = new_route_cost
        
        remaining.remove(best_node)
    
    return new_sol