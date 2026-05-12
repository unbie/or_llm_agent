def _random_removal(self, solution, ratio):
    """随机移除: 但优先移除时间窗紧张或货损成本高的节点"""
    all_nodes = []
    for ri, route in enumerate(solution):
        if len(route) <= 2:
            continue
        # 计算路径中每个节点的成本贡献
        route_nodes = [self.id_to_customer[n] for n in route]
        cost_info = self.calculator.calculate_route_cost(route_nodes, self.dist_matrix)
        
        # 估算每个节点的边际成本（简化计算）
        for pi in range(1, len(route)-1):
            node = route[pi]
            if node == 0:
                continue
            cust = self.id_to_customer[node]
            # 时间窗紧迫度评分（越紧张越容易被移除）
            time_window_width = cust['due_date'] - cust['ready_time']
            time_pressure = 1.0 / (time_window_width + 1.0)  # 避免除零
            
            # 基于位置估算货损敏感度（配送时间越晚越敏感）
            # 简单使用在路径中的位置作为代理
            position_ratio = pi / len(route)
            
            # 综合评分 = 时间窗压力 * 位置权重
            score = time_pressure * (1.0 + position_ratio * 0.5)
            all_nodes.append((ri, pi, node, score))
    
    if not all_nodes:
        return solution, []
    
    total = len(all_nodes)
    k = max(1, int(total * ratio)) if ratio <= 1 else min(total, int(ratio))
    k = min(k, total)
    
    # 按评分加权随机选择（评分高的更可能被选中）
    weights = [score for _, _, _, score in all_nodes]
    selected_indices = random.choices(range(total), weights=weights, k=k)
    selected = [all_nodes[i] for i in selected_indices]
    
    # 按(路径索引, 位置)降序排序，确保删除时索引正确
    selected.sort(key=lambda x: (x[0], x[1]), reverse=True)
    
    new_sol = [r[:] for r in solution]
    removed = []
    for ri, pi, node, _ in selected:
        if pi < len(new_sol[ri]):
            del new_sol[ri][pi]
            removed.append(node)
    
    # 清理空路径
    new_sol = [r for r in new_sol if len(r) > 2]
    return new_sol, removed


def _route_removal(self, solution, ratio):
    """路径移除: 优先移除时间窗违规严重或货损成本高的路径"""
    routes_info = []
    for ri, route in enumerate(solution):
        custs = [n for n in route if n != 0]
        if not custs:
            continue
        
        # 计算路径完整成本
        route_nodes = [self.id_to_customer[n] for n in route]
        cost_info = self.calculator.calculate_route_cost(route_nodes, self.dist_matrix)
        
        # 计算路径的"不良度"评分
        # 1. 时间窗惩罚比例
        time_penalty_ratio = cost_info.get('c3', 0) / max(cost_info['variable_cost'], 1.0)
        
        # 2. 货损成本比例
        freshness_penalty_ratio = cost_info.get('c2', 0) / max(cost_info['variable_cost'], 1.0)
        
        # 3. 路径长度（过短的路径可能效率低）
        route_length_penalty = 1.0 / (len(custs) + 1.0)
        
        # 综合评分：不良路径得分更高
        badness_score = (time_penalty_ratio * 2.0 + 
                        freshness_penalty_ratio * 1.5 + 
                        route_length_penalty * 0.5)
        
        routes_info.append((ri, custs, badness_score))
    
    if not routes_info:
        return solution, []
    
    total_customers = sum(len(c) for _, c, _ in routes_info)
    target = max(1, int(total_customers * ratio)) if ratio <= 1 else int(ratio)
    
    # 按不良度降序排序，优先移除不良路径
    routes_info.sort(key=lambda x: x[2], reverse=True)
    
    removed = []
    to_remove_idx = []
    for ri, custs, _ in routes_info:
        if len(removed) + len(custs) > target and len(removed) > 0:
            # 如果添加这条路径会超过目标，跳过（除非这是第一条）
            continue
        
        removed.extend(custs)
        to_remove_idx.append(ri)
        if len(removed) >= target:
            break
    
    # 如果还没达到目标，从剩余路径中随机补充
    if len(removed) < target:
        remaining_routes = [(ri, custs) for ri, custs, _ in routes_info if ri not in to_remove_idx]
        if remaining_routes:
            ri, custs = random.choice(remaining_routes)
            removed.extend(custs[:target - len(removed)])
            to_remove_idx.append(ri)
    
    to_remove_idx.sort(reverse=True)
    new_sol = [r[:] for r in solution]
    for ri in to_remove_idx:
        if ri < len(new_sol):
            del new_sol[ri]
    
    return new_sol, removed[:target]


def _string_removal(self, solution, ratio):
    """连续节点移除: 优先移除时间窗连续紧张的节点段"""
    all_custs = [n for route in solution for n in route if n != 0]
    if not all_custs:
        return solution, []
    
    total = len(all_custs)
    n_remove = max(1, int(total * ratio)) if ratio <= 1 else min(total, int(ratio))
    n_remove = min(n_remove, total)
    
    removed = []
    new_sol = [r[:] for r in solution]
    
    # 尝试找到时间窗连续的节点段
    attempts = 0
    while len(removed) < n_remove and attempts < 10:
        valid_routes = [(i, r) for i, r in enumerate(new_sol) if len(r) > 2]
        if not valid_routes:
            break
        
        # 优先选择时间窗惩罚高的路径
        route_scores = []
        for ri, route in valid_routes:
            route_nodes = [self.id_to_customer[n] for n in route]
            cost_info = self.calculator.calculate_route_cost(route_nodes, self.dist_matrix)
            score = cost_info.get('c3', 0)  # 时间窗惩罚作为评分
            route_scores.append((ri, route, score))
        
        # 按评分降序排序
        route_scores.sort(key=lambda x: x[2], reverse=True)
        
        # 尝试前3条评分最高的路径
        for ri, route, _ in route_scores[:3]:
            if len(route) <= 2:
                continue
            
            # 寻找时间窗最紧张的区域
            if len(route) > 4:
                # 分析路径中每个节点的时间窗紧迫度
                tightness_scores = []
                for i in range(1, len(route)-1):
                    if route[i] == 0:
                        continue
                    cust = self.id_to_customer[route[i]]
                    time_width = cust['due_date'] - cust['ready_time']
                    if time_width <= 30:  # 时间窗宽度小于30分钟
                        tightness_scores.append((i, 2.0))
                    elif time_width <= 60:
                        tightness_scores.append((i, 1.0))
                    else:
                        tightness_scores.append((i, 0.5))
                
                if tightness_scores:
                    # 找到最紧张的区域
                    tightness_scores.sort(key=lambda x: x[1], reverse=True)
                    start_idx = tightness_scores[0][0]
                    
                    # 确定移除长度
                    max_len = min(3, len(route) - start_idx - 1, n_remove - len(removed))
                    if max_len > 0:
                        for offset in range(max_len):
                            pos = start_idx + offset
                            if pos < len(new_sol[ri]) and new_sol[ri][pos] != 0:
                                removed.append(new_sol[ri][pos])
                                del new_sol[ri][pos]
                        break
            
            # 如果没有找到紧张区域，使用原逻辑
            if len(removed) < n_remove:
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
                break
        
        attempts += 1
    
    # 清理空路径
    new_sol = [r for r in new_sol if len(r) > 2]
    return new_sol, removed


def _greedy_insert(self, solution, removed_nodes):
    """贪心插入: 使用候选位置并考虑时间窗兼容性"""
    if not removed_nodes:
        return solution
    if not solution:
        return [[0, n, 0] for n in removed_nodes]
    
    new_sol = [r[:] for r in solution]
    
    # 按时间窗紧迫度排序节点（紧迫的优先插入）
    node_scores = []
    for node in removed_nodes:
        cust = self.customer_lookup[node]
        time_width = cust['due_date'] - cust['ready_time']
        # 时间窗越窄，评分越高（越紧迫）
        score = 1.0 / (time_width + 1.0)
        node_scores.append((node, score))
    
    node_scores.sort(key=lambda x: x[1], reverse=True)
    sorted_nodes = [node for node, _ in node_scores]
    
    for node in sorted_nodes:
        node_demand = self.customer_lookup[node].get('demand', 0)
        best_inc = float('inf')
        best_ri = None
        best_pos = None
        
        # 检查现有路径
        for ri, route in enumerate(new_sol):
            # 容量检查
            ld = sum(self.customer_lookup[n].get('demand', 0) for n in route if n != 0)
            if ld + node_demand > self.capacity:
                continue
            
            # 使用候选位置而不是遍历所有位置
            candidate_positions = self._candidate_positions(route, node)
            if not candidate_positions:
                continue
            
            route_before = [self.id_to_customer[n] for n in route]
            c_before = self.calculator.calculate_route_cost(
                route_before, self.dist_matrix)['variable_cost']
            
            for pos in candidate_positions:
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
        
        # 决策：插入现有路径还是创建新路径
        if best_ri is not None and best_inc <= nr_cost:
            new_sol[best_ri].insert(best_pos, node)
        else:
            # 检查是否可以合并到其他单客户路径中
            merged = False
            for ri, route in enumerate(new_sol):
                if len(route) == 3:  # 单客户路径 [0, cust, 0]
                    other_node = route[1]
                    # 检查容量
                    other_demand = self.customer_lookup[other_node].get('demand', 0)
                    if node_demand + other_demand <= self.capacity:
                        # 尝试两种插入顺序
                        test_route1 = [0, other_node, node, 0]
                        test_route2 = [0, node, other_node, 0]
                        
                        rt1_nodes = [self.id_to_customer[n] for n in test_route1]
                        rt2_nodes = [self.id_to_customer[n] for n in test_route2]
                        
                        cost1 = self.calculator.calculate_route_cost(
                            rt1_nodes, self.dist_matrix)['variable_cost']
                        cost2 = self.calculator.calculate_route_cost(
                            rt2_nodes, self.dist_matrix)['variable_cost']
                        
                        if cost1 < cost2:
                            new_sol[ri] = test_route1
                        else:
                            new_sol[ri] = test_route2
                        merged = True
                        break
            
            if not merged:
                new_sol.append([0, node, 0])
    
    return new_sol


def _regret_insert(self, solution, removed_nodes):
    """后悔插入: 使用Regret-3并考虑时间窗兼容性"""
    if not removed_nodes:
        return solution
    if not solution:
        return [[0, n, 0] for n in removed_nodes]
    
    new_sol = [r[:] for r in solution]
    remaining = list(removed_nodes)
    
    # 按时间窗紧迫度预排序
    node_tightness = []
    for node in remaining:
        cust = self.customer_lookup[node]
        time_width = cust['due_date'] - cust['ready_time']
        node_tightness.append((node, time_width))
    
    node_tightness.sort(key=lambda x: x[1])  # 时间窗窄的优先
    
    while remaining:
        best_regret = -float('inf')
        best_node = None
        best_ri = None
        best_pos = None
        
        # 评估所有剩余节点（按紧迫度顺序）
        for node, _ in node_tightness:
            if node not in remaining:
                continue
                
            nd = self.customer_lookup[node].get('demand', 0)
            insertion_costs = []  # 存储(成本增量, 路径索引, 位置)
            
            # 检查现有路径
            for ri, route in enumerate(new_sol):
                # 容量检查
                ld = sum(self.customer_lookup[n].get('demand', 0) for n in route if n != 0)
                if ld + nd > self.capacity:
                    continue
                
                # 使用候选位置
                candidate_positions = self._candidate_positions(route, node)
                if not candidate_positions:
                    continue
                
                route_before = [self.id_to_customer[n] for n in route]
                c_before = self.calculator.calculate_route_cost(
                    route_before, self.dist_matrix)['variable_cost']
                
                best_inc_r = float('inf')
                best_pos_r = None
                
                for pos in candidate_positions:
                    ra = route[:pos] + [node] + route[pos:]
                    ra_n = [self.id_to_customer[n] for n in ra]
                    c_after = self.calculator.calculate_route_cost(
                        ra_n, self.dist_matrix)['variable_cost']
                    inc = c_after - c_before
                    
                    if inc < best_inc_r:
                        best_inc_r = inc
                        best_pos_r = pos
                
                if best_pos_r is not None:
                    insertion_costs.append((best_inc_r, ri, best_pos_r))
            
            # 新路径选项
            nr = [0, node, 0]
            nr_nodes = [self.id_to_customer[n] for n in nr]
            nr_cost = self.calculator.calculate_route_cost(
                nr_nodes, self.dist_matrix)['variable_cost']
            nr_cost += self.calculator.f
            insertion_costs.append((nr_cost, None, None))
            
            # 按成本排序
            insertion_costs.sort(key=lambda x: x[0])
            
            # 计算Regret-3（前3个最佳选项的后悔值）
            if len(insertion_costs) >= 3:
                # regret = (cost2 - cost1) + (cost3 - cost1)
                regret = (insertion_costs[1][0] - insertion_costs[0][0]) + \
                        (insertion_costs[2][0] - insertion_costs[0][0])
            elif len(insertion_costs) >= 2:
                regret = insertion_costs[1][0] - insertion_costs[0][0]
            else:
                regret = 0
            
            # 加权后悔值：时间窗紧迫的节点给予更高权重
            cust = self.customer_lookup[node]
            time_width = cust['due_date'] - cust['ready_time']
            time_weight = 2.0 if time_width <= 60 else 1.0
            weighted_regret = regret * time_weight
            
            if weighted_regret > best_regret:
                best_regret = weighted_regret
                best_node = node
                best_ri = insertion_costs[0][1]
                best_pos = insertion_costs[0][2]
        
        if best_node is None:
            # 为剩余节点创建新路径
            for n in remaining:
                new_sol.append([0, n, 0])
            break
        
        # 执行插入
        if best_ri is not None:
            new_sol[best_ri].insert(best_pos, best_node)
        else:
            new_sol.append([0, best_node, 0])
        
        remaining.remove(best_node)
        # 更新紧迫度列表
        node_tightness = [(n, w) for n, w in node_tightness if n in remaining]
    
    return new_sol