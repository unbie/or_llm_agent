## 分析

从运行数据看，当前算法存在以下关键问题：

1. **破坏算子成功率极低**（route_removal 13.3%，string_removal 21.2%）：说明随机性破坏无法有效打破局部最优，需要针对性移除高成本节点。
2. **修复算子效率低下**（greedy_insert 20%）：每次插入都计算完整路径成本，计算开销过大，且未考虑时间窗和货损的敏感性。
3. **成本结构失衡**：C2货损(29.3%)和C3时间惩罚(32.8%)占比过高，说明当前解的时间安排极差，但算子未针对性优化时间相关成本。
4. **收敛过早**：前25%迭代完成89.9%改善，后期缺乏有效探索机制。

**核心洞察**：历史优化尝试了"最差移除"和"增量计算"，但效果有限。必须采用更激进的策略：破坏算子应**主动识别并移除时间窗和货损成本高的节点**，修复算子应**优先考虑时间敏感性和货损衰减**，而非单纯的距离增量。

## 修改后的完整算子代码

```python
def _random_removal(self, solution, ratio):
    """智能随机移除：优先移除时间窗惩罚高或货损成本高的节点"""
    # 收集所有节点及其成本贡献
    node_costs = []
    for ri, route in enumerate(solution):
        if len(route) <= 2:
            continue
        # 计算路径完整成本
        route_nodes = [self.id_to_customer[n] for n in route]
        cost_info = self.calculator.calculate_route_cost(route_nodes, self.dist_matrix)
        # 估算每个节点的成本贡献（近似）
        for pi, node in enumerate(route):
            if node == 0:
                continue
            # 估算节点移除带来的成本变化（简化版）
            # 对于时间敏感节点，惩罚越高，越应该优先移除
            cust = self.id_to_customer[node]
            # 计算时间窗惩罚倾向（越接近边界惩罚越高）
            time_penalty_tendency = 0
            if 'due_date' in cust and 'ready_time' in cust:
                # 假设当前到达时间在中间（简化）
                mid_time = (cust['ready_time'] + cust['due_date']) / 2
                time_penalty_tendency = abs(mid_time - cust['ready_time']) / max(1, cust['due_date'] - cust['ready_time'])
            
            # 货损成本倾向（服务时间越长货损越高）
            freshness_tendency = cust.get('service_time', 0) / 100.0
            
            # 综合成本倾向（时间惩罚权重更高）
            cost_tendency = time_penalty_tendency * 0.7 + freshness_tendency * 0.3
            node_costs.append((cost_tendency, ri, pi, node))
    
    if not node_costs:
        return solution, []
    
    total = len(node_costs)
    k = max(1, int(total * ratio)) if ratio <= 1 else min(total, int(ratio))
    
    # 按成本倾向排序，高成本节点有更高概率被移除
    node_costs.sort(key=lambda x: x[0], reverse=True)
    
    # 使用轮盘赌选择，兼顾高成本节点和随机性
    selected = []
    weights = [cost**2 for cost, _, _, _ in node_costs]  # 平方放大差异
    total_weight = sum(weights)
    
    for _ in range(k):
        if total_weight <= 0:
            break
        r = random.random() * total_weight
        cumulative = 0
        for i, (cost, ri, pi, node) in enumerate(node_costs):
            if (ri, pi, node) in selected:
                continue
            cumulative += weights[i]
            if cumulative >= r:
                selected.append((ri, pi, node))
                total_weight -= weights[i]
                break
    
    # 按逆序删除，避免索引错乱
    selected.sort(key=lambda x: (x[0], x[1]), reverse=True)
    new_sol = [r[:] for r in solution]
    removed = []
    for ri, pi, node in selected:
        if pi < len(new_sol[ri]):
            del new_sol[ri][pi]
            removed.append(node)
    
    # 清理空路径
    new_sol = [r for r in new_sol if len(r) > 2]
    return new_sol, removed


def _route_removal(self, solution, ratio):
    """智能路径移除：优先移除时间窗惩罚高或货损成本高的路径"""
    if not solution:
        return solution, []
    
    # 评估每条路径的时间相关成本
    route_scores = []
    for ri, route in enumerate(solution):
        if len(route) <= 2:
            continue
        
        route_nodes = [self.id_to_customer[n] for n in route]
        cost_info = self.calculator.calculate_route_cost(route_nodes, self.dist_matrix)
        
        # 计算路径的时间敏感度得分
        # 时间窗惩罚和货损成本占比越高，得分越高
        total_cost = cost_info['variable_cost']
        time_related_cost = cost_info.get('c2', 0) + cost_info.get('c3', 0)
        
        if total_cost > 0:
            time_ratio = time_related_cost / total_cost
        else:
            time_ratio = 0
        
        # 路径长度也作为考虑因素（长路径可能包含更多优化机会）
        path_length = len([n for n in route if n != 0])
        
        # 综合得分：时间相关成本占比 + 路径长度权重
        score = time_ratio * 0.8 + (min(path_length, 10) / 10) * 0.2
        route_scores.append((score, ri, route))
    
    if not route_scores:
        return solution, []
    
    # 计算需要移除的客户总数
    total_customers = sum(len([n for n in route if n != 0]) for route in solution)
    target = max(1, int(total_customers * ratio)) if ratio <= 1 else int(ratio)
    
    # 按得分排序，高分路径优先考虑
    route_scores.sort(key=lambda x: x[0], reverse=True)
    
    removed = []
    to_remove_idx = []
    
    # 选择路径直到达到目标数量
    for score, ri, route in route_scores:
        custs = [n for n in route if n != 0]
        if not custs:
            continue
        
        # 有一定概率跳过高分路径，保持随机性
        if random.random() < 0.3:  # 30%概率跳过
            continue
            
        removed.extend(custs)
        to_remove_idx.append(ri)
        
        if len(removed) >= target:
            break
    
    # 如果没选够，从剩余路径中随机补足
    if len(removed) < target:
        remaining_routes = [(ri, route) for ri, route in enumerate(solution) 
                          if ri not in to_remove_idx and len(route) > 2]
        random.shuffle(remaining_routes)
        
        for ri, route in remaining_routes:
            custs = [n for n in route if n != 0]
            removed.extend(custs)
            to_remove_idx.append(ri)
            if len(removed) >= target:
                break
    
    # 按逆序删除路径
    to_remove_idx.sort(reverse=True)
    new_sol = [r[:] for r in solution]
    for ri in to_remove_idx:
        if ri < len(new_sol):
            del new_sol[ri]
    
    return new_sol, removed[:target]


def _string_removal(self, solution, ratio):
    """智能连续节点移除：优先移除时间窗紧密或货损敏感的区域"""
    all_custs = [n for route in solution for n in route if n != 0]
    if not all_custs:
        return solution, []
    
    total = len(all_custs)
    n_remove = max(1, int(total * ratio)) if ratio <= 1 else min(total, int(ratio))
    
    # 构建时间窗紧密度地图
    time_tightness = {}
    for route in solution:
        for i, node in enumerate(route):
            if node == 0:
                continue
            cust = self.id_to_customer[node]
            if 'due_date' in cust and 'ready_time' in cust:
                time_window = cust['due_date'] - cust['ready_time']
                # 时间窗越窄，紧密度越高
                tightness = 1.0 / max(1, time_window)
                time_tightness[node] = tightness
    
    removed = []
    new_sol = [r[:] for r in solution]
    
    while len(removed) < n_remove:
        # 选择有潜力的路径（包含时间窗紧的节点）
        candidate_routes = []
        for ri, route in enumerate(new_sol):
            if len(route) <= 2:
                continue
            # 计算路径的时间窗平均紧密度
            tight_nodes = [time_tightness.get(n, 0) for n in route if n != 0]
            if tight_nodes:
                avg_tightness = sum(tight_nodes) / len(tight_nodes)
                candidate_routes.append((avg_tightness, ri, route))
        
        if not candidate_routes:
            break
        
        # 优先选择时间窗紧的路径
        candidate_routes.sort(key=lambda x: x[0], reverse=True)
        
        # 80%概率选最紧的路径，20%概率随机选
        if random.random() < 0.8:
            _, ri, route = candidate_routes[0]
        else:
            _, ri, route = random.choice(candidate_routes)
        
        # 在路径中选择起始位置（优先靠近时间窗紧的节点）
        if len(route) <= 2:
            continue
        
        # 找到时间窗最紧的节点位置
        tight_positions = []
        for pos in range(1, len(route) - 1):
            node = route[pos]
            if node != 0:
                tightness = time_tightness.get(node, 0)
                tight_positions.append((tightness, pos))
        
        if tight_positions:
            tight_positions.sort(key=lambda x: x[0], reverse=True)
            # 从最紧节点附近开始移除
            start_pos = tight_positions[0][1]
        else:
            start_pos = random.randint(1, len(route) - 2)
        
        # 确定移除长度（1-3个节点，但不超过剩余需要）
        max_len = min(3, len(route) - start_pos - 1, n_remove - len(removed))
        if max_len <= 0:
            # 如果当前位置不合适，尝试其他路径
            continue
        
        slen = random.randint(1, max_len)
        
        # 执行移除
        for _ in range(slen):
            if start_pos < len(new_sol[ri]) and new_sol[ri][start_pos] != 0:
                node = new_sol[ri][start_pos]
                removed.append(node)
                del new_sol[ri][start_pos]
    
    # 清理空路径
    new_sol = [r for r in new_sol if len(r) > 2]
    return new_sol, removed


def _greedy_insert(self, solution, removed_nodes):
    """时间敏感的贪心插入：优先考虑时间窗和货损成本"""
    if not removed_nodes:
        return solution
    if not solution:
        return [[0, n, 0] for n in removed_nodes]
    
    new_sol = [r[:] for r in solution]
    
    # 按节点时间窗紧密度排序，紧的优先插入
    node_tightness = []
    for node in removed_nodes:
        cust = self.id_to_customer[node]
        if 'due_date' in cust and 'ready_time' in cust:
            time_window = cust['due_date'] - cust['ready_time']
            tightness = 1.0 / max(1, time_window)
        else:
            tightness = 0
        node_tightness.append((tightness, node))
    
    node_tightness.sort(key=lambda x: x[0], reverse=True)
    sorted_nodes = [node for _, node in node_tightness]
    
    for node in sorted_nodes:
        node_demand = self.customer_lookup[node].get('demand', 0)
        cust = self.id_to_customer[node]
        
        best_inc = float('inf')
        best_ri = None
        best_pos = None
        best_time_feasible = False
        
        # 检查是否有时窗要求
        has_time_window = 'due_date' in cust and 'ready_time' in cust
        
        for ri, route in enumerate(new_sol):
            # 容量检查
            ld = sum(self.customer_lookup[n].get('demand', 0) for n in route if n != 0)
            if ld + node_demand > self.capacity:
                continue
            
            route_before = [self.id_to_customer[n] for n in route]
            c_before = self.calculator.calculate_route_cost(
                route_before, self.dist_matrix)['variable_cost']
            
            # 获取候选位置
            candidate_positions = self._candidate_positions(route, node)
            
            for pos in candidate_positions:
                route_after = route[:pos] + [node] + route[pos:]
                ra_nodes = [self.id_to_customer[n] for n in route_after]
                c_after = self.calculator.calculate_route_cost(
                    ra_nodes, self.dist_matrix)['variable_cost']
                inc = c_after - c_before
                
                # 检查时间可行性（如果有时窗要求）
                time_feasible = True
                if has_time_window:
                    # 通过成本计算中的时间惩罚来间接判断
                    cost_info = self.calculator.calculate_route_cost(ra_nodes, self.dist_matrix)
                    if cost_info.get('c3', 0) > 10000:  # 硬时间窗惩罚阈值
                        time_feasible = False
                
                # 优先选择时间可行的插入
                if time_feasible and not best_time_feasible:
                    best_inc = inc
                    best_ri = ri
                    best_pos = pos
                    best_time_feasible = True
                elif (time_feasible and best_time_feasible and inc < best_inc) or \
                     (not time_feasible and not best_time_feasible and inc < best_inc):
                    best_inc = inc
                    best_ri = ri
                    best_pos = pos
                    best_time_feasible = time_feasible
        
        # 新路径成本
        nr = [0, node, 0]
        nr_nodes = [self.id_to_customer[n] for n in nr]
        nr_cost_info = self.calculator.calculate_route_cost(nr_nodes, self.dist_matrix)
        nr_cost = nr_cost_info['variable_cost'] + self.calculator.f
        
        # 检查新路径的时间可行性
        new_route_time_feasible = True
        if has_time_window and nr_cost_info.get('c3', 0) > 10000:
            new_route_time_feasible = False
        
        # 决策：优先时间可行，其次成本低
        if best_ri is not None:
            if best_time_feasible and (not new_route_time_feasible or best_inc <= nr_cost):
                new_sol[best_ri].insert(best_pos, node)
            elif not best_time_feasible and not new_route_time_feasible and best_inc <= nr_cost:
                new_sol[best_ri].insert(best_pos, node)
            else:
                new_sol.append([0, node, 0])
        else:
            new_sol.append([0, node, 0])
    
    return new_sol


def _regret_insert(self, solution, removed_nodes):
    """时间敏感的多重后悔插入：考虑前k个最佳位置的机会成本"""
    if not removed_nodes:
        return solution
    if not solution:
        return [[0, n, 0] for n in removed_nodes]
    
    new_sol = [r[:] for r in solution]
    remaining = list(removed_nodes)
    
    # 按时间窗紧密度排序节点
    node_tightness = []
    for node in remaining:
        cust = self.id_to_customer[node]
        if 'due_date' in cust and 'ready_time' in cust:
            time_window = cust['due_date'] - cust['ready_time']
            tightness = 1.0 / max(1, time_window)
        else:
            tightness = 0
        node_tightness.append((tightness, node))
    
    node_tightness.sort(key=lambda x: x[0], reverse=True)
    remaining = [node for _, node in node_tightness]
    
    while remaining:
        best_regret = -float('inf')
        best_node = None
        best_ri = None
        best_pos = None
        
        for node in remaining:
            nd = self.customer_lookup[node].get('demand', 0)
            cust = self.id_to_customer[node]
            has_time_window = 'due_date' in cust and 'ready_time' in cust
            
            # 收集所有可行插入位置的成本
            insert_options = []
            
            # 现有路径选项
            for ri, route in enumerate(new_sol):
                ld = sum(self.customer_lookup[n].get('demand', 0)
                        for n in route if n != 0)
                if ld + nd > self.capacity:
                    continue
                
                route_before = [self.id_to_customer[n] for n in route]
                c_before = self.calculator.calculate_route_cost(
                    route_before, self.dist_matrix)['variable_cost']
                
                # 检查所有候选位置
                candidate_positions = self._candidate_positions(route, node)
                for pos in candidate_positions:
                    ra = route[:pos] + [node] + route[pos:]
                    ra_n = [self.id_to_customer[n] for n in ra]
                    cost_info = self.calculator.calculate_route_cost(
                        ra_n, self.dist_matrix)
                    c_after = cost_info['variable_cost']
                    inc = c_after - c_before
                    
                    # 时间可行性标记
                    time_feasible = True
                    if has_time_window and cost_info.get('c3', 0) > 10000:
                        time_feasible = False
                    
                    insert_options.append((inc, ri, pos, time_feasible))
            
            # 新路径选项
            nr = [0, node, 0]
            nr_nodes = [self.id_to_customer[n] for n in nr]
            nr_cost_info = self.calculator.calculate_route_cost(nr_nodes, self.dist_matrix)
            nr_cost = nr_cost_info['variable_cost'] + self.calculator.f
            new_route_time_feasible = not (has_time_window and nr_cost_info.get('c3', 0) > 10000)
            insert_options.append((nr_cost, None, None, new_route_time_feasible))
            
            # 按成本排序
            insert_options.sort(key=lambda x: x[0])
            
            # 计算后悔值（考虑前3个最佳位置）
            k = min(3, len(insert_options))
            if k >= 2:
                # 分离时间可行和不可行的选项
                feasible_options = [opt for opt in insert_options if opt[3]]
                infeasible_options = [opt for opt in insert_options if not opt[3]]
                
                if feasible_options:
                    # 优先考虑时间可行的
                    best_feasible = feasible_options[0]
                    second_best = feasible_options[1] if len(feasible_options) > 1 else feasible_options[0]
                    regret = second_best[0] - best_feasible[0]
                    
                    # 如果有时间不可行的选项，增加额外惩罚
                    if infeasible_options:
                        regret += 1000  # 时间不可行的额外惩罚
                else:
                    # 只有时间不可行的选项
                    best_infeasible = infeasible_options[0]
                    second_best = infeasible_options[1] if len(infeasible_options) > 1 else infeasible_options[0]
                    regret = second_best[0] - best_infeasible[0] - 500  # 时间不可行的惩罚
            else:
                regret = 0
            
            # 调整后悔值：时间窗紧的节点有更高优先级
            if has_time_window:
                time_window = cust['due_date'] - cust['ready_time']
                tightness = 1.0 / max(1, time_window)
                regret *= (1 + tightness)  # 紧时间窗节点后悔值放大
            
            if regret > best_regret:
                best_regret = regret
                best_node = node
                # 选择最佳插入选项（优先时间可行）
                if insert_options[0][3]:  # 第一个时间可行
                    best_ri = insert_options[0][1]
                    best_pos = insert_options[0][2]
                else:
                    # 找第一个时间可行的
                    for inc, ri, pos, feasible in insert_options:
                        if feasible:
                            best_ri = ri
                            best_pos = pos
                            break
                    if best_ri is None:  # 没有时间可行的
                        best_ri = insert_options[0][1]
                        best_pos = insert_options[0][2]
        
        if best_node is None:
            # 剩余节点都创建新路径
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
```

## 核心改进点

1. **破坏算子智能化**：
   - `_random_removal`：优先移除时间窗惩罚高和货损成本高的节点
   - `_route_removal`：优先移除时间相关成本占比高的路径
   - `_string_removal`：优先移除时间窗紧密的连续区域

2. **修复算子时间敏感性**：
   - 按时间窗紧密度排序节点，紧的优先处理
   - 插入时优先考虑时间可行性，其次才是成本
   - 后悔值计算考虑前k个最佳位置，并放大紧时间窗节点的优先级

3. **成本结构优化**：
   - 通过针对性移除高时间成本节点，打破不良时间安排
   - 通过时间敏感的插入策略，改善时间窗遵守和货损控制

4. **跳出局部最优**：
   - 破坏算子引入轮盘赌选择，保持一定随机性
   - 修复算子考虑多重后悔值，避免短视决策