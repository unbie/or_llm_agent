def _random_removal(self, solution, ratio):
    """成本敏感随机移除：优先移除在当前路径中边际成本高的节点"""
    # 计算每个节点的边际成本（移除后成本降低量）
    node_marginal_cost = {}
    for ri, route in enumerate(solution):
        if len(route) <= 2:  # 只有仓库
            continue
        
        route_nodes = [self.id_to_customer[n] for n in route]
        base_cost = self.calculator.calculate_route_cost(route_nodes, self.dist_matrix)['variable_cost']
        
        for pi in range(1, len(route)-1):
            node = route[pi]
            if node == 0:
                continue
                
            # 计算移除该节点后的成本
            route_without = route[:pi] + route[pi+1:]
            route_without_nodes = [self.id_to_customer[n] for n in route_without]
            cost_without = self.calculator.calculate_route_cost(route_without_nodes, self.dist_matrix)['variable_cost']
            
            marginal_cost = base_cost - cost_without  # 正值表示移除能降低成本
            node_marginal_cost[(ri, pi, node)] = max(0.01, marginal_cost)  # 避免零值
    
    if not node_marginal_cost:
        return solution, []
    
    all_nodes = list(node_marginal_cost.keys())
    total = len(all_nodes)
    k = max(1, int(total * ratio)) if ratio <= 1 else min(total, int(ratio))
    
    # 按边际成本加权随机选择（边际成本越高，被选中概率越大）
    weights = [node_marginal_cost[node] for node in all_nodes]
    selected_indices = random.choices(range(len(all_nodes)), weights=weights, k=k)
    selected = [all_nodes[idx] for idx in selected_indices]
    
    selected.sort(key=lambda x: (x[0], x[1]), reverse=True)
    new_sol = [r[:] for r in solution]
    removed = []
    
    for ri, pi, node in selected:
        del new_sol[ri][pi]
        removed.append(node)
    
    new_sol = [r for r in new_sol if len(r) > 2]
    return new_sol, removed


def _route_removal(self, solution, ratio):
    """货损敏感路径移除：优先移除货损成本高的路径"""
    if not solution:
        return solution, []
    
    # 计算每条路径的货损成本占比
    route_scores = []
    for ri, route in enumerate(solution):
        if len(route) <= 2:
            continue
        
        route_nodes = [self.id_to_customer[n] for n in route]
        cost_info = self.calculator.calculate_route_cost(route_nodes, self.dist_matrix)
        
        # 货损成本占总成本的比例（越高越应该被移除）
        total_cost = cost_info['variable_cost']
        freshness_cost = cost_info.get('c2', 0)
        
        if total_cost > 0:
            freshness_ratio = freshness_cost / total_cost
        else:
            freshness_ratio = 0
        
        # 同时考虑路径长度（短路径更容易被重新分配）
        customer_count = len([n for n in route if n != 0])
        length_penalty = 1.0 / max(1, customer_count)
        
        score = freshness_ratio * 0.7 + length_penalty * 0.3
        route_scores.append((ri, route, score))
    
    if not route_scores:
        return solution, []
    
    # 按分数排序（分数高的先移除）
    route_scores.sort(key=lambda x: x[2], reverse=True)
    
    total_customers = sum(len([n for n in route if n != 0]) for _, route, _ in route_scores)
    target = max(1, int(total_customers * ratio)) if ratio <= 1 else int(ratio)
    
    removed = []
    to_remove_idx = []
    
    for ri, route, _ in route_scores:
        custs = [n for n in route if n != 0]
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
    """时间窗连续移除：移除时间窗相近的连续节点"""
    all_custs = [n for route in solution for n in route if n != 0]
    if not all_custs:
        return solution, []
    
    total = len(all_custs)
    n_remove = max(1, int(total * ratio)) if ratio <= 1 else min(total, int(ratio))
    
    # 收集所有可能的连续段
    candidate_segments = []
    
    for ri, route in enumerate(solution):
        if len(route) <= 2:
            continue
        
        # 找到连续的时间窗相近节点
        for start in range(1, len(route)-1):
            if route[start] == 0:
                continue
            
            base_tw = self.customer_lookup[route[start]].get('ready_time', 0)
            
            # 尝试不同长度的段
            for length in range(1, min(5, len(route)-start)):
                if start + length >= len(route) or route[start+length] == 0:
                    break
                
                # 检查时间窗是否相近（在2小时内）
                current_tw = self.customer_lookup[route[start+length]].get('ready_time', 0)
                if abs(current_tw - base_tw) > 120:  # 2小时
                    break
                
                segment = route[start:start+length+1]
                customer_segment = [n for n in segment if n != 0]
                
                if customer_segment:
                    # 计算段的紧凑度（时间窗方差越小越紧凑）
                    tw_values = [self.customer_lookup[n].get('ready_time', 0) for n in customer_segment]
                    if len(tw_values) > 1:
                        mean_tw = sum(tw_values) / len(tw_values)
                        variance = sum((t - mean_tw) ** 2 for t in tw_values) / len(tw_values)
                        compactness = 1.0 / (1.0 + variance)
                    else:
                        compactness = 1.0
                    
                    candidate_segments.append((ri, start, length, customer_segment, compactness))
    
    if not candidate_segments:
        # 回退到原始策略
        return self._string_removal_fallback(solution, ratio)
    
    # 按紧凑度选择（越紧凑的段越容易被整体移除）
    candidate_segments.sort(key=lambda x: x[4], reverse=True)
    
    removed = []
    new_sol = [r[:] for r in solution]
    
    for ri, start, length, segment, _ in candidate_segments:
        if len(removed) >= n_remove:
            break
        
        # 检查这些节点是否还在原路径中
        if ri >= len(new_sol) or start >= len(new_sol[ri]):
            continue
        
        # 移除段
        actual_removed = []
        offset = 0
        
        for _ in range(length + 1):
            pos = start - offset
            if 0 < pos < len(new_sol[ri]) and new_sol[ri][pos] in segment:
                node = new_sol[ri][pos]
                if node != 0:
                    actual_removed.append(node)
                    del new_sol[ri][pos]
                    offset += 1
        
        removed.extend(actual_removed)
    
    # 如果移除不够，补充随机移除
    if len(removed) < n_remove:
        additional = n_remove - len(removed)
        all_remaining = [n for route in new_sol for n in route if n != 0]
        if all_remaining:
            extra_removed = random.sample(all_remaining, min(additional, len(all_remaining)))
            # 从new_sol中移除这些节点
            for node in extra_removed:
                for ri, route in enumerate(new_sol):
                    if node in route:
                        pos = route.index(node)
                        del new_sol[ri][pos]
                        break
            removed.extend(extra_removed)
    
    new_sol = [r for r in new_sol if len(r) > 2]
    return new_sol, removed[:n_remove]

def _string_removal_fallback(self, solution, ratio):
    """回退的连续移除策略"""
    all_custs = [n for route in solution for n in route if n != 0]
    if not all_custs:
        return solution, []
    
    total = len(all_custs)
    n_remove = max(1, int(total * ratio)) if ratio <= 1 else min(total, int(ratio))
    
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
    """增量贪心插入：使用候选位置和增量成本估算"""
    if not removed_nodes:
        return solution
    if not solution:
        return [[0, n, 0] for n in removed_nodes]
    
    new_sol = [r[:] for r in solution]
    remaining = list(removed_nodes)
    
    # 预计算每个节点的需求
    node_demands = {node: self.customer_lookup[node].get('demand', 0) for node in remaining}
    
    # 预计算每条路径的当前负载
    route_loads = []
    for ri, route in enumerate(new_sol):
        load = sum(self.customer_lookup[n].get('demand', 0) for n in route if n != 0)
        route_loads.append((ri, load))
    
    while remaining:
        best_node = None
        best_ri = None
        best_pos = None
        best_inc = float('inf')
        
        for node in remaining:
            nd = node_demands[node]
            
            # 检查现有路径
            for ri, load in route_loads:
                if load + nd > self.capacity:
                    continue
                
                route = new_sol[ri]
                candidate_positions = self._candidate_positions(route, node)
                
                if not candidate_positions:
                    continue
                
                # 使用增量估算（只计算插入点附近的成本变化）
                for pos in candidate_positions:
                    if pos < 1 or pos >= len(route):
                        continue
                    
                    # 获取插入点前后的节点
                    prev_node = route[pos-1]
                    next_node = route[pos] if pos < len(route) else 0
                    
                    # 计算增量距离变化
                    if prev_node == 0 and next_node == 0:
                        # 插入到空路径中
                        inc_dist = self.dist_matrix[0][node] + self.dist_matrix[node][0]
                        base_dist = 0
                    elif prev_node == 0:
                        # 插入到路径开头
                        inc_dist = self.dist_matrix[0][node] + self.dist_matrix[node][next_node]
                        base_dist = self.dist_matrix[0][next_node]
                    elif next_node == 0:
                        # 插入到路径末尾
                        inc_dist = self.dist_matrix[prev_node][node] + self.dist_matrix[node][0]
                        base_dist = self.dist_matrix[prev_node][0]
                    else:
                        # 插入到中间
                        inc_dist = self.dist_matrix[prev_node][node] + self.dist_matrix[node][next_node]
                        base_dist = self.dist_matrix[prev_node][next_node]
                    
                    dist_inc = inc_dist - base_dist
                    
                    # 估算时间窗惩罚增量（简化估算）
                    # 获取相关时间窗信息
                    node_info = self.id_to_customer[node]
                    prev_info = self.id_to_customer[prev_node] if prev_node != 0 else None
                    next_info = self.id_to_customer[next_node] if next_node != 0 else None
                    
                    # 简单估算：如果节点时间窗与前后节点兼容性差，增加惩罚
                    penalty_estimate = 0
                    if prev_info and node_info['ready_time'] < prev_info.get('due_date', 0):
                        penalty_estimate += 50
                    if next_info and node_info['due_date'] > next_info.get('ready_time', 0):
                        penalty_estimate += 50
                    
                    # 总增量估算（距离成本 + 制冷成本 + 惩罚估算）
                    inc_estimate = dist_inc * 3 + (dist_inc / 40) * 15 + penalty_estimate
                    
                    if inc_estimate < best_inc:
                        best_inc = inc_estimate
                        best_node = node
                        best_ri = ri
                        best_pos = pos
        
        # 如果没有找到合适位置，创建新路径
        if best_node is None:
            for node in remaining:
                new_sol.append([0, node, 0])
            break
        
        # 执行插入
        new_sol[best_ri].insert(best_pos, best_node)
        remaining.remove(best_node)
        
        # 更新路径负载
        for i, (ri, load) in enumerate(route_loads):
            if ri == best_ri:
                route_loads[i] = (ri, load + node_demands[best_node])
                break
    
    return new_sol


def _regret_insert(self, solution, removed_nodes):
    """增量后悔插入：使用候选位置和增量估算，考虑Regret-3"""
    if not removed_nodes:
        return solution
    if not solution:
        return [[0, n, 0] for n in removed_nodes]
    
    new_sol = [r[:] for r in solution]
    remaining = list(removed_nodes)
    
    # 预计算每个节点的需求
    node_demands = {node: self.customer_lookup[node].get('demand', 0) for node in remaining}
    
    while remaining:
        best_regret = -float('inf')
        best_node = None
        best_ri = None
        best_pos = None
        
        for node in remaining:
            nd = node_demands[node]
            insertion_costs = []
            
            # 检查现有路径
            for ri, route in enumerate(new_sol):
                load = sum(self.customer_lookup[n].get('demand', 0) for n in route if n != 0)
                if load + nd > self.capacity:
                    continue
                
                candidate_positions = self._candidate_positions(route, node)
                if not candidate_positions:
                    continue
                
                best_cost_ri = float('inf')
                best_pos_ri = None
                
                for pos in candidate_positions:
                    if pos < 1 or pos >= len(route):
                        continue
                    
                    # 增量成本估算
                    prev_node = route[pos-1]
                    next_node = route[pos] if pos < len(route) else 0
                    
                    # 计算增量距离
                    if prev_node == 0 and next_node == 0:
                        inc_dist = self.dist_matrix[0][node] + self.dist_matrix[node][0]
                        base_dist = 0
                    elif prev_node == 0:
                        inc_dist = self.dist_matrix[0][node] + self.dist_matrix[node][next_node]
                        base_dist = self.dist_matrix[0][next_node]
                    elif next_node == 0:
                        inc_dist = self.dist_matrix[prev_node][node] + self.dist_matrix[node][0]
                        base_dist = self.dist_matrix[prev_node][0]
                    else:
                        inc_dist = self.dist_matrix[prev_node][node] + self.dist_matrix[node][next_node]
                        base_dist = self.dist_matrix[prev_node][next_node]
                    
                    dist_inc = inc_dist - base_dist
                    cost_estimate = dist_inc * 3 + (dist_inc / 40) * 15  # 距离+制冷成本
                    
                    if cost_estimate < best_cost_ri:
                        best_cost_ri = cost_estimate
                        best_pos_ri = pos
            
                if best_pos_ri is not None:
                    insertion_costs.append((best_cost_ri, ri, best_pos_ri))
            
            # 新路径选项
            new_route_cost = (self.dist_matrix[0][node] + self.dist_matrix[node][0]) * 3
            new_route_cost += ((self.dist_matrix[0][node] + self.dist_matrix[node][0]) / 40) * 15
            new_route_cost += self.calculator.f  # 固定成本
            
            insertion_costs.append((new_route_cost, None, None))
            
            # 计算Regret-3（前3个最佳选择的后悔值）
            if len(insertion_costs) >= 2:
                insertion_costs.sort(key=lambda x: x[0])
                
                # 取前3个或全部
                k = min(3, len(insertion_costs))
                top_k = insertion_costs[:k]
                
                # 计算加权后悔值
                regret_sum = 0
                for i in range(1, k):
                    regret_sum += (top_k[i][0] - top_k[0][0]) * (1.0 / i)
                
                if regret_sum > best_regret:
                    best_regret = regret_sum
                    best_node = node
                    best_ri = top_k[0][1]
                    best_pos = top_k[0][2]
            else:
                # 只有一个选择，后悔值为0
                if 0 > best_regret:
                    best_regret = 0
                    best_node = node
                    best_ri = insertion_costs[0][1]
                    best_pos = insertion_costs[0][2]
        
        if best_node is None:
            # 创建新路径
            for node in remaining:
                new_sol.append([0, node, 0])
            break
        
        # 执行插入
        if best_ri is not None:
            new_sol[best_ri].insert(best_pos, best_node)
        else:
            new_sol.append([0, best_node, 0])
        
        remaining.remove(best_node)
    
    return new_sol