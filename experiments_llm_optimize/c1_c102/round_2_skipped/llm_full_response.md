## 性能瓶颈分析

从运行数据可以看出：

1. **修复算子成功率低**：`greedy_insert`仅18.6%成功率，`regret_insert`仅54.8%，说明插入策略过于保守，无法有效重组解结构
2. **破坏算子不平衡**：`route_removal`成功率仅11.8%，对于聚类数据过于激进；`string_removal`32.9%成功率，但连续移除长度固定(1-3)限制了探索能力
3. **成本计算开销大**：修复算子中频繁调用`calculate_route_cost`，但未充分利用候选位置筛选
4. **缺乏时间窗敏感度**：当前破坏算子未考虑时间窗约束，导致修复时难以满足紧时间窗

## 修改后代码

```python
def _random_removal(self, solution, ratio):
    """随机移除: 增加权重机制，优先移除高成本节点"""
    all_nodes = []
    for ri, route in enumerate(solution):
        if len(route) <= 2:
            continue
        # 计算路径中各节点的边际成本贡献
        route_customers = [self.id_to_customer[n] for n in route if n != 0]
        if not route_customers:
            continue
            
        base_cost = self.calculator.calculate_route_cost(
            route_customers, self.dist_matrix)['variable_cost']
        
        for pi, node in enumerate(route):
            if node == 0:
                continue
            # 计算移除该节点后的成本变化
            temp_route = [n for n in route if n != node]
            if len(temp_route) > 2:
                temp_customers = [self.id_to_customer[n] for n in temp_route if n != 0]
                if temp_customers:
                    new_cost = self.calculator.calculate_route_cost(
                        temp_customers, self.dist_matrix)['variable_cost']
                    cost_saving = base_cost - new_cost
                    # 成本节约越大，权重越高（越可能被移除）
                    weight = max(0.1, 1.0 + cost_saving / 1000)
                else:
                    weight = 1.0
            else:
                weight = 1.0
            all_nodes.append((ri, pi, node, weight))
    
    if not all_nodes:
        return solution, []
    
    total = len(all_nodes)
    k = max(1, int(total * ratio)) if ratio <= 1 else min(total, int(ratio))
    k = min(k, total)
    
    # 按权重概率选择节点
    weights = [item[3] for item in all_nodes]
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
    """路径移除: 优先移除低质量路径（高成本、低利用率）"""
    routes_info = []
    for ri, route in enumerate(solution):
        custs = [n for n in route if n != 0]
        if not custs:
            continue
            
        # 计算路径质量指标
        route_customers = [self.id_to_customer[n] for n in custs]
        cost_info = self.calculator.calculate_route_cost(route_customers, self.dist_matrix)
        total_cost = cost_info['variable_cost'] + self.calculator.f
        
        # 计算载重利用率
        total_demand = sum(self.customer_lookup[n].get('demand', 0) for n in custs)
        utilization = total_demand / self.capacity if self.capacity > 0 else 0
        
        # 路径质量评分：成本越高、利用率越低，评分越高（越可能被移除）
        # 同时考虑时间窗惩罚（C3）和货损成本（C2）的占比
        penalty_ratio = (cost_info.get('c3', 0) + cost_info.get('c2', 0)) / max(1, total_cost)
        quality_score = (total_cost / 1000) * (1.5 - utilization) * (1.0 + penalty_ratio)
        
        routes_info.append((ri, custs, quality_score))
    
    if not routes_info:
        return solution, []
    
    total_customers = sum(len(c) for _, c, _ in routes_info)
    target = max(1, int(total_customers * ratio)) if ratio <= 1 else int(ratio)
    
    # 按质量评分排序，优先移除低质量路径
    routes_info.sort(key=lambda x: x[2], reverse=True)
    
    removed = []
    to_remove_idx = []
    for ri, custs, _ in routes_info:
        if len(removed) + len(custs) > target and len(removed) > 0:
            # 如果添加当前路径会超过目标，跳过
            continue
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
    """连续节点移除: 自适应移除长度，优先移除时间窗冲突严重的连续段"""
    all_custs = [n for route in solution for n in route if n != 0]
    if not all_custs:
        return solution, []
    
    total = len(all_custs)
    n_remove = max(1, int(total * ratio)) if ratio <= 1 else min(total, int(ratio))
    n_remove = min(n_remove, total)
    
    removed = []
    new_sol = [r[:] for r in solution]
    
    # 计算各路径的时间窗冲突程度
    route_conflicts = []
    for ri, route in enumerate(new_sol):
        if len(route) <= 2:
            continue
        custs = [n for n in route if n != 0]
        if not custs:
            continue
        route_customers = [self.id_to_customer[n] for n in custs]
        cost_info = self.calculator.calculate_route_cost(route_customers, self.dist_matrix)
        # 时间窗惩罚越高，冲突越严重
        conflict_score = cost_info.get('c3', 0) / max(1, len(custs))
        route_conflicts.append((ri, conflict_score))
    
    # 优先从冲突严重的路径中移除
    route_conflicts.sort(key=lambda x: x[1], reverse=True)
    conflict_routes = [ri for ri, _ in route_conflicts]
    
    while len(removed) < n_remove:
        valid_routes = [(i, r) for i, r in enumerate(new_sol) if len(r) > 2]
        if not valid_routes:
            break
        
        # 优先选择冲突严重的路径
        available_routes = []
        for ri, route in valid_routes:
            if ri in conflict_routes:
                # 冲突路径有更高优先级
                available_routes.extend([(ri, route)] * 3)
            else:
                available_routes.append((ri, route))
        
        ri, route = random.choice(available_routes)
        
        # 自适应移除长度：基于剩余需要移除的数量和路径长度
        max_len = min(len(route) - 2, n_remove - len(removed))
        if max_len <= 0:
            continue
        
        # 动态调整移除长度：剩余越多，可能移除越长
        if n_remove - len(removed) >= 5:
            slen = random.randint(2, min(5, max_len))
        else:
            slen = random.randint(1, min(3, max_len))
        
        start = random.randint(1, len(route) - slen - 1)
        
        for i in range(slen):
            pos = start + i
            if pos < len(new_sol[ri]) and new_sol[ri][pos] != 0:
                removed.append(new_sol[ri][pos])
                del new_sol[ri][pos]
    
    new_sol = [r for r in new_sol if len(r) > 2]
    return new_sol, removed


def _greedy_insert(self, solution, removed_nodes):
    """贪心插入: 使用候选位置筛选，加入噪声机制避免局部最优"""
    if not removed_nodes:
        return solution
    if not solution:
        return [[0, n, 0] for n in removed_nodes]
    
    new_sol = [r[:] for r in solution]
    
    # 打乱节点顺序，增加随机性
    nodes_to_insert = list(removed_nodes)
    random.shuffle(nodes_to_insert)
    
    for node in nodes_to_insert:
        node_demand = self.customer_lookup[node].get('demand', 0)
        best_inc = float('inf')
        best_ri = None
        best_pos = None
        
        # 评估新路径的成本
        nr = [0, node, 0]
        nr_nodes = [self.id_to_customer[n] for n in nr]
        nr_cost = self.calculator.calculate_route_cost(
            nr_nodes, self.dist_matrix)['variable_cost']
        nr_cost += self.calculator.f
        
        # 先快速筛选可行路径（容量约束）
        feasible_routes = []
        for ri, route in enumerate(new_sol):
            ld = sum(self.customer_lookup[n].get('demand', 0) for n in route if n != 0)
            if ld + node_demand <= self.capacity:
                feasible_routes.append((ri, route))
        
        # 如果没有可行路径，创建新路径
        if not feasible_routes:
            new_sol.append([0, node, 0])
            continue
        
        for ri, route in feasible_routes:
            route_before = [self.id_to_customer[n] for n in route]
            c_before = self.calculator.calculate_route_cost(
                route_before, self.dist_matrix)['variable_cost']
            
            # 使用候选位置而不是遍历所有位置
            candidate_positions = self._candidate_positions(route, node)
            
            for pos in candidate_positions:
                if pos < 1 or pos >= len(route):
                    continue
                    
                route_after = route[:pos] + [node] + route[pos:]
                ra_nodes = [self.id_to_customer[n] for n in route_after]
                c_after = self.calculator.calculate_route_cost(
                    ra_nodes, self.dist_matrix)['variable_cost']
                
                inc = c_after - c_before
                
                # 加入小幅度随机噪声（±5%），避免总是选择局部最优
                noise = random.uniform(0.95, 1.05)
                noisy_inc = inc * noise
                
                if noisy_inc < best_inc:
                    best_inc = inc  # 决策时仍用原始成本
                    best_ri = ri
                    best_pos = pos
        
        # 决策：比较插入现有路径和创建新路径
        if best_ri is not None and best_inc <= nr_cost * 0.9:  # 设置10%的偏好阈值
            new_sol[best_ri].insert(best_pos, node)
        else:
            new_sol.append([0, node, 0])
    
    return new_sol


def _regret_insert(self, solution, removed_nodes):
    """后悔插入: 扩展为Regret-3，考虑前3个最佳位置的机会成本差异"""
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
            all_options = []
            
            # 检查现有路径
            for ri, route in enumerate(new_sol):
                ld = sum(self.customer_lookup[n].get('demand', 0)
                        for n in route if n != 0)
                if ld + nd > self.capacity:
                    continue
                
                route_before = [self.id_to_customer[n] for n in route]
                c_before = self.calculator.calculate_route_cost(
                    route_before, self.dist_matrix)['variable_cost']
                
                # 使用候选位置
                candidate_positions = self._candidate_positions(route, node)
                route_options = []
                
                for pos in candidate_positions:
                    if pos < 1 or pos >= len(route):
                        continue
                    
                    ra = route[:pos] + [node] + route[pos:]
                    ra_n = [self.id_to_customer[n] for n in ra]
                    c_after = self.calculator.calculate_route_cost(
                        ra_n, self.dist_matrix)['variable_cost']
                    inc = c_after - c_before
                    route_options.append((inc, ri, pos))
                
                # 取该路径上的前3个最佳位置
                route_options.sort(key=lambda x: x[0])
                all_options.extend(route_options[:3])
            
            # 新路径选项
            nr = [0, node, 0]
            nr_nodes = [self.id_to_customer[n] for n in nr]
            nr_cost = self.calculator.calculate_route_cost(
                nr_nodes, self.dist_matrix)['variable_cost']
            nr_cost += self.calculator.f
            all_options.append((nr_cost, None, None))
            
            # 计算Regret-3：前3个最佳选项的加权后悔值
            all_options.sort(key=lambda x: x[0])
            if len(all_options) >= 3:
                # 加权后悔值：考虑前3个选项的差异
                regret = (all_options[1][0] - all_options[0][0]) * 2.0 + \
                        (all_options[2][0] - all_options[0][0]) * 1.0
            elif len(all_options) >= 2:
                regret = (all_options[1][0] - all_options[0][0]) * 3.0
            else:
                regret = 0
            
            # 加入节点本身的启发式：时间窗紧的节点应优先插入
            customer_info = self.id_to_customer[node]
            time_window_width = customer_info['due_date'] - customer_info['ready_time']
            if time_window_width < 60:  # 紧时间窗（小于1小时）
                regret *= 1.5  # 提高紧时间窗节点的优先级
            
            if regret > best_regret:
                best_regret = regret
                best_node = node
                best_ri = all_options[0][1]
                best_pos = all_options[0][2]
        
        if best_node is None:
            # 为剩余节点创建新路径
            for n in remaining:
                new_sol.append([0, n, 0])
            break
        
        if best_ri is not None:
            new_sol[best_ri].insert(best_pos, best_node)
        else:
            new_sol.append([0, best_node, 0])
        
        remaining.remove(best_node)
    
    return new_sol
```

## 主要改进点

1. **破坏算子智能化**：
   - `_random_removal`：引入成本权重，优先移除高成本节点
   - `_route_removal`：基于路径质量（成本、利用率、时间窗惩罚）选择移除路径
   - `_string_removal`：自适应移除长度，优先处理时间窗冲突严重的路径段

2. **修复算子优化**：
   - 使用`self._candidate_positions`减少位置评估数量
   - `_greedy_insert`：加入噪声机制避免局部最优，设置新路径偏好阈值
   - `_regret_insert`：扩展为Regret-3，考虑前3个最佳位置，加权计算后悔值

3. **时间窗敏感度**：
   - 在破坏和修复中都考虑时间窗约束，优先处理紧时间窗节点

4. **计算效率**：
   - 通过候选位置筛选减少`calculate_route_cost`调用次数
   - 在修复前快速筛选满足容量约束的路径

这些改进保持了算子签名不变，但通过引入成本感知、自适应机制和更智能的决策逻辑，有望突破当前局部最优，提高算子成功率。