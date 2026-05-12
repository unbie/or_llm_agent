def _random_removal(self, solution, ratio):
    """智能随机移除：优先移除时间窗惩罚高或货损成本高的节点"""
    all_nodes = []
    
    # 计算每个节点的惩罚成本
    node_costs = {}
    for route in solution:
        if len(route) <= 2:
            continue
            
        route_nodes = [self.id_to_customer[n] for n in route]
        cost_info = self.calculator.calculate_route_cost(route_nodes, self.dist_matrix)
        
        # 为每个节点分配成本（按到达时间比例分配）
        for i, node_id in enumerate(route):
            if node_id == 0:
                continue
                
            # 节点的时间窗惩罚和货损成本占比
            customer = self.id_to_customer[node_id]
            ready_time = customer['ready_time']
            due_date = customer['due_date']
            
            # 估计到达时间（简化计算）
            arrival_time = cost_info.get('arrival_times', {}).get(node_id, (ready_time + due_date) / 2)
            
            # 计算时间窗偏离度
            time_deviation = 0
            if arrival_time < ready_time:
                time_deviation = ready_time - arrival_time
            elif arrival_time > due_date:
                time_deviation = arrival_time - due_date
            
            # 货损成本与时间正相关
            freshness_cost = arrival_time * 0.1  # 简化估计
            
            # 综合惩罚分数
            penalty_score = time_deviation * 10 + freshness_cost * 5
            node_costs[node_id] = penalty_score
            all_nodes.append((node_id, penalty_score))
    
    if not all_nodes:
        return solution, []
    
    total = len(all_nodes)
    k = max(1, int(total * ratio)) if ratio <= 1 else min(total, int(ratio))
    k = min(k, total)
    
    # 按惩罚成本加权随机选择
    nodes, weights = zip(*all_nodes)
    
    # 归一化权重并添加基础概率
    min_w, max_w = min(weights), max(weights)
    if max_w > min_w:
        normalized = [(w - min_w) / (max_w - min_w) + 0.1 for w in weights]
    else:
        normalized = [1.0] * len(weights)
    
    # 加权随机选择
    selected_ids = random.choices(nodes, weights=normalized, k=k)
    
    # 收集要移除的节点位置
    to_remove = []
    for ri, route in enumerate(solution):
        for pi, node in enumerate(route):
            if node != 0 and node in selected_ids:
                to_remove.append((ri, pi, node))
                selected_ids.remove(node)
                if not selected_ids:
                    break
        if not selected_ids:
            break
    
    to_remove.sort(key=lambda x: (x[0], x[1]), reverse=True)
    new_sol = [r[:] for r in solution]
    removed = []
    
    for ri, pi, node in to_remove:
        del new_sol[ri][pi]
        removed.append(node)
    
    new_sol = [r for r in new_sol if len(r) > 2]
    return new_sol, removed


def _route_removal(self, solution, ratio):
    """智能路径移除：优先移除平均惩罚成本高的路径"""
    routes_info = []
    
    for ri, route in enumerate(solution):
        custs = [n for n in route if n != 0]
        if not custs:
            continue
            
        # 计算路径的平均惩罚成本
        route_nodes = [self.id_to_customer[n] for n in route]
        cost_info = self.calculator.calculate_route_cost(route_nodes, self.dist_matrix)
        
        # 路径惩罚分数 = 时间惩罚 + 货损成本
        route_penalty = cost_info.get('c3', 0) + cost_info.get('c2', 0)
        avg_penalty = route_penalty / len(custs) if custs else 0
        
        routes_info.append((ri, custs, avg_penalty))
    
    if not routes_info:
        return solution, []
    
    total_customers = sum(len(c) for _, c, _ in routes_info)
    target = max(1, int(total_customers * ratio)) if ratio <= 1 else int(ratio)
    
    # 按惩罚成本降序排序
    routes_info.sort(key=lambda x: x[2], reverse=True)
    
    removed = []
    to_remove_idx = []
    
    for ri, custs, penalty in routes_info:
        if len(removed) + len(custs) > target and to_remove_idx:
            # 如果已经选了路径，尝试从当前路径中选部分节点
            need = target - len(removed)
            if need > 0:
                # 从当前路径中选择惩罚最高的节点
                node_penalties = []
                for node in custs:
                    customer = self.id_to_customer[node]
                    ready_time = customer['ready_time']
                    due_date = customer['due_date']
                    # 简化估计惩罚
                    penalty_est = abs((ready_time + due_date) / 2 - ready_time)
                    node_penalties.append((node, penalty_est))
                
                node_penalties.sort(key=lambda x: x[1], reverse=True)
                selected_nodes = [n for n, _ in node_penalties[:need]]
                
                # 从原路径中移除这些节点
                new_route = [n for n in solution[ri] if n == 0 or n not in selected_nodes]
                if len(new_route) > 2:
                    # 更新原路径
                    solution[ri] = new_route
                else:
                    # 路径太短，整条移除
                    to_remove_idx.append(ri)
                
                removed.extend(selected_nodes)
                break
        else:
            to_remove_idx.append(ri)
            removed.extend(custs)
            if len(removed) >= target:
                break
    
    to_remove_idx.sort(reverse=True)
    new_sol = [r[:] for r in solution]
    for ri in to_remove_idx:
        del new_sol[ri]
    
    return new_sol, removed[:target]


def _string_removal(self, solution, ratio):
    """智能连续移除：在时间窗惩罚高的区域移除连续节点"""
    all_custs = [n for route in solution for n in route if n != 0]
    if not all_custs:
        return solution, []
    
    total = len(all_custs)
    n_remove = max(1, int(total * ratio)) if ratio <= 1 else min(total, int(ratio))
    n_remove = min(n_remove, total)
    
    # 寻找高惩罚连续段
    high_penalty_segments = []
    
    for ri, route in enumerate(solution):
        if len(route) <= 2:
            continue
            
        route_nodes = [self.id_to_customer[n] for n in route]
        cost_info = self.calculator.calculate_route_cost(route_nodes, self.dist_matrix)
        
        # 寻找连续高惩罚区域
        for i in range(1, len(route) - 2):  # 至少保留首尾的0
            if route[i] == 0:
                continue
                
            # 检查连续3个节点的惩罚情况
            segment_penalty = 0
            segment_nodes = []
            
            for j in range(i, min(i + 3, len(route))):
                node_id = route[j]
                if node_id == 0:
                    continue
                    
                customer = self.id_to_customer[node_id]
                ready_time = customer['ready_time']
                due_date = customer['due_date']
                
                # 估计惩罚
                penalty_est = abs((ready_time + due_date) / 2 - ready_time)
                segment_penalty += penalty_est
                segment_nodes.append((ri, j, node_id))
            
            if segment_nodes and segment_penalty > 0:
                avg_penalty = segment_penalty / len(segment_nodes)
                high_penalty_segments.append((avg_penalty, segment_nodes))
    
    removed = []
    new_sol = [r[:] for r in solution]
    
    if high_penalty_segments:
        # 按惩罚降序排序
        high_penalty_segments.sort(key=lambda x: x[0], reverse=True)
        
        for penalty, segment in high_penalty_segments:
            if len(removed) >= n_remove:
                break
                
            # 移除这个连续段
            for ri, pi, node in segment:
                if node not in removed and len(removed) < n_remove:
                    # 确保索引有效
                    if pi < len(new_sol[ri]):
                        removed.append(node)
                        del new_sol[ri][pi]
    
    # 如果还不够，补充随机移除
    if len(removed) < n_remove:
        remaining = n_remove - len(removed)
        valid_nodes = []
        for ri, route in enumerate(new_sol):
            for pi, node in enumerate(route):
                if node != 0 and node not in removed:
                    valid_nodes.append((ri, pi, node))
        
        if valid_nodes and remaining > 0:
            selected = random.sample(valid_nodes, min(remaining, len(valid_nodes)))
            selected.sort(key=lambda x: (x[0], x[1]), reverse=True)
            for ri, pi, node in selected:
                del new_sol[ri][pi]
                removed.append(node)
    
    new_sol = [r for r in new_sol if len(r) > 2]
    return new_sol, removed


def _greedy_insert(self, solution, removed_nodes):
    """增量贪心插入：使用增量成本计算，避免重复完整路径计算"""
    if not removed_nodes:
        return solution
    if not solution:
        return [[0, n, 0] for n in removed_nodes]
    
    new_sol = [r[:] for r in solution]
    
    # 预计算每条路径的当前成本
    route_costs = {}
    route_capacity_used = {}
    
    for ri, route in enumerate(new_sol):
        if len(route) > 2:
            route_nodes = [self.id_to_customer[n] for n in route]
            cost_info = self.calculator.calculate_route_cost(route_nodes, self.dist_matrix)
            route_costs[ri] = cost_info['variable_cost']
            
            # 计算已使用容量
            used_cap = sum(self.customer_lookup[n].get('demand', 0) for n in route if n != 0)
            route_capacity_used[ri] = used_cap
    
    for node in removed_nodes:
        node_demand = self.customer_lookup[node].get('demand', 0)
        best_inc = float('inf')
        best_ri = None
        best_pos = None
        
        # 新路径选项的成本
        new_route_cost = self.calculator.calculate_route_cost(
            [self.id_to_customer[0], self.id_to_customer[node], self.id_to_customer[0]],
            self.dist_matrix
        )['variable_cost'] + self.calculator.f
        
        # 检查现有路径
        for ri, route in enumerate(new_sol):
            if ri not in route_costs:
                continue
                
            if route_capacity_used[ri] + node_demand > self.capacity:
                continue
            
            # 获取候选位置
            candidate_positions = self._candidate_positions(route, node)
            
            for pos in candidate_positions:
                # 增量计算插入成本
                if pos == 0 or pos == len(route):
                    continue
                    
                prev_node = route[pos-1]
                next_node = route[pos] if pos < len(route) else 0
                
                # 计算增量距离成本
                old_dist = self.dist_matrix[prev_node][next_node]
                new_dist = (self.dist_matrix[prev_node][node] + 
                          self.dist_matrix[node][next_node])
                dist_inc = (new_dist - old_dist) * 3  # C12系数
                
                # 估计时间增量（简化）
                time_inc = (new_dist - old_dist) / self.calculator.v * 15  # C13系数
                
                # 估计惩罚增量（基于节点时间窗）
                customer = self.id_to_customer[node]
                ready_time = customer['ready_time']
                due_date = customer['due_date']
                service_time = customer.get('service_time', 0)
                
                # 估计到达时间
                arrival_est = (ready_time + due_date) / 2
                
                # 估计惩罚
                penalty_est = 0
                if arrival_est < ready_time:
                    penalty_est = (ready_time - arrival_est) * 10
                elif arrival_est > due_date:
                    penalty_est = (arrival_est - due_date) * 100
                
                # 货损成本增量
                freshness_inc = arrival_est * 0.5
                
                # 总增量成本估计
                inc_est = dist_inc + time_inc + penalty_est + freshness_inc
                
                if inc_est < best_inc:
                    # 验证性计算（只在看起来最优时进行完整计算）
                    route_after = route[:pos] + [node] + route[pos:]
                    ra_nodes = [self.id_to_customer[n] for n in route_after]
                    c_after = self.calculator.calculate_route_cost(ra_nodes, self.dist_matrix)['variable_cost']
                    inc_actual = c_after - route_costs[ri]
                    
                    if inc_actual < best_inc:
                        best_inc = inc_actual
                        best_ri = ri
                        best_pos = pos
        
        # 决策
        if best_ri is not None and best_inc <= new_route_cost:
            # 插入到现有路径
            new_sol[best_ri].insert(best_pos, node)
            # 更新路径成本和容量
            route_after = new_sol[best_ri]
            ra_nodes = [self.id_to_customer[n] for n in route_after]
            route_costs[best_ri] = self.calculator.calculate_route_cost(ra_nodes, self.dist_matrix)['variable_cost']
            route_capacity_used[best_ri] += node_demand
        else:
            # 创建新路径
            new_sol.append([0, node, 0])
            ri = len(new_sol) - 1
            route_costs[ri] = new_route_cost - self.calculator.f  # 减去固定成本，因为variable_cost不包含固定成本
            route_capacity_used[ri] = node_demand
    
    return new_sol


def _regret_insert(self, solution, removed_nodes):
    """增量后悔插入：使用增量计算，评估所有剩余节点的k-regret"""
    if not removed_nodes:
        return solution
    if not solution:
        return [[0, n, 0] for n in removed_nodes]
    
    new_sol = [r[:] for r in solution]
    remaining = list(removed_nodes)
    
    # 预计算路径成本和容量
    route_costs = {}
    route_capacity_used = {}
    
    for ri, route in enumerate(new_sol):
        if len(route) > 2:
            route_nodes = [self.id_to_customer[n] for n in route]
            cost_info = self.calculator.calculate_route_cost(route_nodes, self.dist_matrix)
            route_costs[ri] = cost_info['variable_cost']
            used_cap = sum(self.customer_lookup[n].get('demand', 0) for n in route if n != 0)
            route_capacity_used[ri] = used_cap
    
    while remaining:
        node_regrets = []
        
        for node in remaining:
            node_demand = self.customer_lookup[node].get('demand', 0)
            insertion_costs = []
            
            # 检查现有路径
            for ri, route in enumerate(new_sol):
                if ri not in route_costs:
                    continue
                    
                if route_capacity_used[ri] + node_demand > self.capacity:
                    continue
                
                best_inc = float('inf')
                best_pos = None
                
                for pos in self._candidate_positions(route, node):
                    if pos == 0 or pos == len(route):
                        continue
                    
                    # 增量成本估计
                    prev_node = route[pos-1]
                    next_node = route[pos] if pos < len(route) else 0
                    
                    old_dist = self.dist_matrix[prev_node][next_node]
                    new_dist = (self.dist_matrix[prev_node][node] + 
                              self.dist_matrix[node][next_node])
                    dist_inc = (new_dist - old_dist) * 3
                    
                    time_inc = (new_dist - old_dist) / self.calculator.v * 15
                    
                    customer = self.id_to_customer[node]
                    ready_time = customer['ready_time']
                    due_date = customer['due_date']
                    arrival_est = (ready_time + due_date) / 2
                    
                    penalty_est = 0
                    if arrival_est < ready_time:
                        penalty_est = (ready_time - arrival_est) * 10
                    elif arrival_est > due_date:
                        penalty_est = (arrival_est - due_date) * 100
                    
                    freshness_inc = arrival_est * 0.5
                    inc_est = dist_inc + time_inc + penalty_est + freshness_inc
                    
                    if inc_est < best_inc:
                        # 完整验证
                        route_after = route[:pos] + [node] + route[pos:]
                        ra_nodes = [self.id_to_customer[n] for n in route_after]
                        c_after = self.calculator.calculate_route_cost(ra_nodes, self.dist_matrix)['variable_cost']
                        inc_actual = c_after - route_costs[ri]
                        
                        if inc_actual < best_inc:
                            best_inc = inc_actual
                            best_pos = pos
                
                if best_inc < float('inf'):
                    insertion_costs.append((best_inc, ri, best_pos))
            
            # 新路径选项
            new_route_cost = self.calculator.calculate_route_cost(
                [self.id_to_customer[0], self.id_to_customer[node], self.id_to_customer[0]],
                self.dist_matrix
            )['variable_cost'] + self.calculator.f
            insertion_costs.append((new_route_cost, None, None))
            
            # 计算k-regret (k=3)
            insertion_costs.sort(key=lambda x: x[0])
            
            if len(insertion_costs) >= 3:
                regret = (insertion_costs[1][0] - insertion_costs[0][0]) + \
                        (insertion_costs[2][0] - insertion_costs[0][0])
            elif len(insertion_costs) >= 2:
                regret = insertion_costs[1][0] - insertion_costs[0][0]
            else:
                regret = 0
            
            node_regrets.append((regret, node, insertion_costs[0]))
        
        if not node_regrets:
            # 创建新路径插入剩余节点
            for node in remaining:
                new_sol.append([0, node, 0])
            break
        
        # 选择后悔值最大的节点
        node_regrets.sort(key=lambda x: x[0], reverse=True)
        best_regret, best_node, (best_cost, best_ri, best_pos) = node_regrets[0]
        
        # 执行插入
        if best_ri is not None:
            new_sol[best_ri].insert(best_pos, best_node)
            # 更新路径成本和容量
            route_after = new_sol[best_ri]
            ra_nodes = [self.id_to_customer[n] for n in route_after]
            route_costs[best_ri] = self.calculator.calculate_route_cost(ra_nodes, self.dist_matrix)['variable_cost']
            route_capacity_used[best_ri] += self.customer_lookup[best_node].get('demand', 0)
        else:
            new_sol.append([0, best_node, 0])
            ri = len(new_sol) - 1
            route_costs[ri] = best_cost - self.calculator.f
            route_capacity_used[ri] = self.customer_lookup[best_node].get('demand', 0)
        
        remaining.remove(best_node)
    
    return new_sol