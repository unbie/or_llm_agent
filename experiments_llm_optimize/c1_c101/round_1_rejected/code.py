def _random_removal(self, solution, ratio):
    """增强版随机移除：基于成本贡献加权选择节点，优先移除高成本节点"""
    all_nodes = []
    node_costs = {}
    
    # 计算每个节点的边际成本贡献
    for ri, route in enumerate(solution):
        if len(route) <= 2:
            continue
            
        route_nodes = [self.id_to_customer[n] for n in route]
        route_cost = self.calculator.calculate_route_cost(route_nodes, self.dist_matrix)
        base_cost = route_cost['variable_cost']
        
        for pi, node in enumerate(route):
            if node == 0:
                continue
                
            # 计算移除该节点后的成本
            route_without = route[:pi] + route[pi+1:]
            if len(route_without) > 2:
                route_without_nodes = [self.id_to_customer[n] for n in route_without]
                cost_without = self.calculator.calculate_route_cost(route_without_nodes, self.dist_matrix)['variable_cost']
                cost_saving = base_cost - cost_without
            else:
                # 如果移除后路径只剩仓库，成本节省为整条路径成本
                cost_saving = base_cost + self.calculator.f
            
            # 时间窗违反和货损成本高的节点权重更高
            customer = self.id_to_customer[node]
            time_penalty_weight = 1.0
            if 'due_date' in customer and 'ready_time' in customer:
                # 估计到达时间（简化计算）
                if pi > 0:
                    prev_node = route[pi-1]
                    travel_time = self.dist_matrix[prev_node][node] / self.calculator.v * 60  # 分钟
                    # 使用前一个节点的服务时间作为估计
                    est_arrival = 0  # 简化估计
                else:
                    est_arrival = 0
                
                # 时间窗紧张度权重
                time_window_width = customer['due_date'] - customer['ready_time']
                if time_window_width < 60:  # 时间窗小于1小时
                    time_penalty_weight = 3.0
                elif time_window_width < 120:
                    time_penalty_weight = 2.0
            
            weighted_cost = cost_saving * time_penalty_weight
            all_nodes.append((ri, pi, node, weighted_cost))
    
    if not all_nodes:
        return solution, []
    
    total = len(all_nodes)
    k = max(1, int(total * ratio)) if ratio <= 1 else min(total, int(ratio))
    k = min(k, total)
    
    # 基于加权成本进行概率选择（成本越高，被选中的概率越大）
    weights = [cost for _, _, _, cost in all_nodes]
    min_weight = min(weights)
    max_weight = max(weights)
    
    if max_weight > min_weight:
        # 归一化并应用指数放大差异
        normalized = [(w - min_weight) / (max_weight - min_weight + 1e-6) for w in weights]
        amplified = [1.0 + n**2 for n in normalized]  # 平方放大差异
        selected_indices = random.choices(range(len(all_nodes)), weights=amplified, k=k)
    else:
        selected_indices = random.sample(range(len(all_nodes)), k)
    
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
    """智能路径移除：基于路径综合评分选择，优先移除低质量路径"""
    if not solution:
        return solution, []
    
    routes_info = []
    for ri, route in enumerate(solution):
        custs = [n for n in route if n != 0]
        if not custs:
            continue
            
        # 计算路径质量评分
        route_nodes = [self.id_to_customer[n] for n in route]
        route_cost = self.calculator.calculate_route_cost(route_nodes, self.dist_matrix)
        
        # 评分因素：时间窗违反、货损成本、距离效率
        score = 0.0
        
        # 1. 时间窗违反惩罚（硬时间窗违反路径应优先移除）
        time_penalty = route_cost.get('c3', 0)
        if time_penalty > 1000:  # 硬时间窗违反
            score += 10000
        
        # 2. 货损成本权重（货损高的路径质量差）
        freshness_cost = route_cost.get('c2', 0)
        score += freshness_cost * 0.5
        
        # 3. 距离效率（绕路程度）
        total_dist = route_cost.get('dist', 0)
        if len(custs) > 1:
            # 计算客户间的平均距离与最小生成树距离的比值
            # 简化：使用客户到仓库距离之和与总距离的比值
            depot_dist_sum = sum(self.dist_matrix[0][c] for c in custs)
            detour_ratio = total_dist / (depot_dist_sum * 2) if depot_dist_sum > 0 else 1.0
            if detour_ratio > 1.5:  # 绕路严重
                score += 500 * (detour_ratio - 1.5)
        
        # 4. 载重利用率低（车辆空跑）
        total_demand = sum(self.customer_lookup[n].get('demand', 0) for n in custs)
        load_ratio = total_demand / self.capacity
        if load_ratio < 0.3:  # 利用率低于30%
            score += 300 * (0.3 - load_ratio)
        
        routes_info.append((ri, custs, score))
    
    if not routes_info:
        return solution, []
    
    total_customers = sum(len(c) for _, c, _ in routes_info)
    target = max(1, int(total_customers * ratio)) if ratio <= 1 else int(ratio)
    
    # 按评分排序（评分高的优先移除）
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
    """时空连续移除：基于时间和空间连续性选择连续段"""
    all_custs = [n for route in solution for n in route if n != 0]
    if not all_custs:
        return solution, []
    
    total = len(all_custs)
    n_remove = max(1, int(total * ratio)) if ratio <= 1 else min(total, int(ratio))
    n_remove = min(n_remove, total)
    
    removed = []
    new_sol = [r[:] for r in solution]
    
    # 尝试找到时空连续的段
    attempts = 0
    while len(removed) < n_remove and attempts < 10:
        attempts += 1
        
        # 选择一条非空路径
        valid_routes = [(i, r) for i, r in enumerate(new_sol) if len(r) > 2]
        if not valid_routes:
            break
        
        # 基于路径质量选择（时间窗违反多的优先）
        route_scores = []
        for i, route in valid_routes:
            route_nodes = [self.id_to_customer[n] for n in route]
            cost = self.calculator.calculate_route_cost(route_nodes, self.dist_matrix)
            score = cost.get('c3', 0) + cost.get('c2', 0) * 0.5
            route_scores.append((i, route, score))
        
        route_scores.sort(key=lambda x: x[2], reverse=True)
        
        # 选择前3条中的一条
        selected_idx = min(2, len(route_scores) - 1)
        ri, route, _ = route_scores[selected_idx]
        
        if len(route) <= 2:
            continue
        
        # 在路径中寻找连续的高成本段
        best_start = 1
        best_score = -float('inf')
        max_len = min(5, len(route) - 2, n_remove - len(removed))
        
        for start in range(1, len(route) - max_len):
            # 计算这段的时空连续性得分
            segment_score = 0
            segment_nodes = []
            
            for i in range(max_len):
                if start + i >= len(route) - 1:
                    break
                node = route[start + i]
                if node == 0:
                    break
                
                customer = self.id_to_customer[node]
                segment_nodes.append(node)
                
                # 时间窗紧张度
                if 'due_date' in customer and 'ready_time' in customer:
                    time_width = customer['due_date'] - customer['ready_time']
                    if time_width < 60:
                        segment_score += 3.0
                    elif time_width < 120:
                        segment_score += 2.0
                
                # 空间连续性（与前后节点的距离）
                if i > 0:
                    prev_node = segment_nodes[i-1]
                    segment_score += 1.0 / (self.dist_matrix[prev_node][node] + 1e-6)
            
            if len(segment_nodes) >= 2 and segment_score > best_score:
                best_score = segment_score
                best_start = start
        
        # 移除选中的连续段
        slen = min(max_len, len(route) - best_start - 1, n_remove - len(removed))
        for offset in range(slen - 1, -1, -1):
            pos = best_start + offset
            if pos < len(new_sol[ri]) and new_sol[ri][pos] != 0:
                removed.append(new_sol[ri][pos])
                del new_sol[ri][pos]
    
    # 如果时空连续移除不够，补充随机移除
    if len(removed) < n_remove:
        remaining_needed = n_remove - len(removed)
        all_remaining = [(ri, pi, n) for ri, r in enumerate(new_sol) 
                        for pi, n in enumerate(r) if n != 0]
        
        if all_remaining:
            k = min(remaining_needed, len(all_remaining))
            selected = random.sample(all_remaining, k)
            selected.sort(key=lambda x: (x[0], x[1]), reverse=True)
            
            for ri, pi, node in selected:
                if pi < len(new_sol[ri]):
                    removed.append(node)
                    del new_sol[ri][pi]
    
    new_sol = [r for r in new_sol if len(r) > 2]
    return new_sol, removed


def _greedy_insert(self, solution, removed_nodes):
    """增强贪心插入：考虑时间窗和货损成本的增量计算"""
    if not removed_nodes:
        return solution
    if not solution:
        return [[0, n, 0] for n in removed_nodes]
    
    new_sol = [r[:] for r in solution]
    
    # 按时间窗紧迫度排序（时间窗窄的优先插入）
    sorted_nodes = []
    for node in removed_nodes:
        customer = self.customer_lookup[node]
        if 'due_date' in customer and 'ready_time' in customer:
            time_width = customer['due_date'] - customer['ready_time']
            sorted_nodes.append((node, time_width))
        else:
            sorted_nodes.append((node, float('inf')))
    
    sorted_nodes.sort(key=lambda x: x[1])  # 时间窗窄的优先
    removed_nodes_sorted = [n for n, _ in sorted_nodes]
    
    for node in removed_nodes_sorted:
        node_demand = self.customer_lookup[node].get('demand', 0)
        customer = self.customer_lookup[node]
        
        best_inc = float('inf')
        best_ri = None
        best_pos = None
        best_route_cost_after = None
        
        # 新路径成本
        new_route = [0, node, 0]
        new_route_nodes = [self.id_to_customer[n] for n in new_route]
        new_route_cost = self.calculator.calculate_route_cost(
            new_route_nodes, self.dist_matrix)['variable_cost']
        new_route_total = new_route_cost + self.calculator.f
        
        for ri, route in enumerate(new_sol):
            # 容量检查
            current_load = sum(self.customer_lookup[n].get('demand', 0) for n in route if n != 0)
            if current_load + node_demand > self.capacity:
                continue
            
            route_before = [self.id_to_customer[n] for n in route]
            cost_before = self.calculator.calculate_route_cost(
                route_before, self.dist_matrix)['variable_cost']
            
            # 获取候选位置
            candidate_positions = self._candidate_positions(route, node)
            
            # 如果没有候选位置，跳过
            if not candidate_positions:
                continue
            
            for pos in candidate_positions:
                # 检查插入后是否满足时间窗约束（快速检查）
                if pos > 0 and pos < len(route):
                    prev_node = route[pos-1]
                    next_node = route[pos] if pos < len(route) else 0
                    
                    # 快速时间可行性检查
                    if 'due_date' in customer and 'ready_time' in customer:
                        # 估计到达新节点的时间
                        prev_customer = self.id_to_customer[prev_node]
                        travel_time = self.dist_matrix[prev_node][node] / self.calculator.v * 60
                        
                        # 简化估计：使用前一个节点的离开时间
                        est_arrival = 0  # 这里需要实际计算，简化处理
                        
                        # 如果明显不可行，跳过
                        if est_arrival > customer['due_date'] + 60:  # 宽松检查
                            continue
                
                # 计算精确成本增量
                route_after = route[:pos] + [node] + route[pos:]
                route_after_nodes = [self.id_to_customer[n] for n in route_after]
                cost_after = self.calculator.calculate_route_cost(
                    route_after_nodes, self.dist_matrix)['variable_cost']
                
                inc = cost_after - cost_before
                
                # 特别关注时间窗和货损成本
                cost_details_before = self.calculator.calculate_route_cost(
                    route_before, self.dist_matrix)
                cost_details_after = self.calculator.calculate_route_cost(
                    route_after_nodes, self.dist_matrix)
                
                time_penalty_inc = cost_details_after.get('c3', 0) - cost_details_before.get('c3', 0)
                freshness_inc = cost_details_after.get('c2', 0) - cost_details_before.get('c2', 0)
                
                # 对时间窗违反和货损增加给予额外惩罚
                adjusted_inc = inc + time_penalty_inc * 2.0 + freshness_inc * 1.5
                
                if adjusted_inc < best_inc:
                    best_inc = adjusted_inc
                    best_ri = ri
                    best_pos = pos
                    best_route_cost_after = cost_after
        
        # 决策：插入现有路径还是创建新路径
        if best_ri is not None and best_inc <= new_route_total:
            new_sol[best_ri].insert(best_pos, node)
        else:
            new_sol.append([0, node, 0])
    
    return new_sol


def _regret_insert(self, solution, removed_nodes):
    """Regret-k插入：考虑前k个最佳位置的机会成本差异，k=3"""
    if not removed_nodes:
        return solution
    if not solution:
        return [[0, n, 0] for n in removed_nodes]
    
    new_sol = [r[:] for r in solution]
    remaining = list(removed_nodes)
    
    # 按时间窗紧迫度预排序
    node_urgency = {}
    for node in remaining:
        customer = self.customer_lookup[node]
        if 'due_date' in customer and 'ready_time' in customer:
            time_width = customer['due_date'] - customer['ready_time']
            urgency = 1.0 / (time_width + 1e-6)
        else:
            urgency = 0
        node_urgency[node] = urgency
    
    k_regret = 3  # Regret-3
    
    while remaining:
        best_regret = -float('inf')
        best_node = None
        best_ri = None
        best_pos = None
        
        for node in remaining:
            nd = self.customer_lookup[node].get('demand', 0)
            insertion_options = []  # (cost_inc, ri, pos)
            
            # 现有路径选项
            for ri, route in enumerate(new_sol):
                current_load = sum(self.customer_lookup[n].get('demand', 0) 
                                 for n in route if n != 0)
                if current_load + nd > self.capacity:
                    continue
                
                route_before = [self.id_to_customer[n] for n in route]
                cost_before = self.calculator.calculate_route_cost(
                    route_before, self.dist_matrix)['variable_cost']
                
                candidate_positions = self._candidate_positions(route, node)
                if not candidate_positions:
                    continue
                
                # 评估前k个最佳位置
                pos_costs = []
                for pos in candidate_positions:
                    route_after = route[:pos] + [node] + route[pos:]
                    route_after_nodes = [self.id_to_customer[n] for n in route_after]
                    cost_after = self.calculator.calculate_route_cost(
                        route_after_nodes, self.dist_matrix)['variable_cost']
                    
                    inc = cost_after - cost_before
                    
                    # 获取详细成本增量用于调整
                    cost_details_before = self.calculator.calculate_route_cost(
                        route_before, self.dist_matrix)
                    cost_details_after = self.calculator.calculate_route_cost(
                        route_after_nodes, self.dist_matrix)
                    
                    time_penalty_inc = cost_details_after.get('c3', 0) - cost_details_before.get('c3', 0)
                    freshness_inc = cost_details_after.get('c2', 0) - cost_details_before.get('c2', 0)
                    
                    # 调整后的增量成本
                    adjusted_inc = inc + time_penalty_inc * 2.0 + freshness_inc * 1.5
                    pos_costs.append((adjusted_inc, ri, pos))
                
                if pos_costs:
                    pos_costs.sort(key=lambda x: x[0])
                    # 取前min(k, len(pos_costs))个
                    for i in range(min(k_regret, len(pos_costs))):
                        insertion_options.append(pos_costs[i])
            
            # 新路径选项
            new_route = [0, node, 0]
            new_route_nodes = [self.id_to_customer[n] for n in new_route]
            new_route_cost = self.calculator.calculate_route_cost(
                new_route_nodes, self.dist_matrix)['variable_cost']
            new_route_total = new_route_cost + self.calculator.f
            insertion_options.append((new_route_total, None, None))
            
            # 排序并计算regret-k值
            insertion_options.sort(key=lambda x: x[0])
            
            if len(insertion_options) >= 2:
                # Regret-k计算：前k个最佳位置的成本差异加权和
                regret_value = 0
                for i in range(1, min(k_regret, len(insertion_options))):
                    regret_value += (insertion_options[i][0] - insertion_options[0][0]) * (1.0 / i)
                
                # 加入紧迫度权重
                urgency_weight = 1.0 + node_urgency.get(node, 0) * 0.5
                weighted_regret = regret_value * urgency_weight
                
                if weighted_regret > best_regret:
                    best_regret = weighted_regret
                    best_node = node
                    best_ri = insertion_options[0][1]
                    best_pos = insertion_options[0][2]
            elif insertion_options:
                # 只有一个选项
                if 0 > best_regret:  # 基础regret为0
                    best_regret = 0
                    best_node = node
                    best_ri = insertion_options[0][1]
                    best_pos = insertion_options[0][2]
        
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
    
    return new_sol