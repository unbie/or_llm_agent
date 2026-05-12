## 性能瓶颈分析

1. **破坏算子成功率低**：`_route_removal` 成功率仅10%，`_string_removal` 36.2%，说明当前破坏策略不够智能，无法有效打破局部最优
2. **修复算子计算开销大**：`_greedy_insert` 和 `_regret_insert` 对每个节点遍历所有路径所有位置，`calculate_route_cost` 调用过于频繁
3. **缺乏针对性破坏**：当前破坏算子未考虑生鲜物流特有的时间窗、货损成本等关键因素
4. **修复策略单一**：未充分利用候选位置机制，且 regret 计算只考虑前2个最佳位置

## 优化策略

1. **破坏算子**：引入基于成本贡献的智能移除策略，优先移除高成本/时间窗紧张的节点
2. **修复算子**：使用候选位置大幅减少计算量，同时引入 regret-k 机制
3. **计算优化**：缓存路径成本，避免重复计算

## 修改后的完整代码

```python
def _random_removal(self, solution, ratio):
    """增强版随机移除：引入基于成本贡献的加权随机选择"""
    all_nodes = []
    node_costs = {}
    
    # 计算每个节点的成本贡献
    for ri, route in enumerate(solution):
        if len(route) <= 2:
            continue
            
        # 计算完整路径成本
        route_nodes = [self.id_to_customer[n] for n in route]
        route_cost = self.calculator.calculate_route_cost(route_nodes, self.dist_matrix)
        
        # 估算每个节点的边际成本贡献
        for pi, node in enumerate(route):
            if node == 0:
                continue
                
            # 简单估算：节点前后距离 + 时间窗惩罚影响
            prev_node = route[pi-1] if pi > 0 else 0
            next_node = route[pi+1] if pi < len(route)-1 else 0
            
            # 移除该节点后节省的距离
            saved_dist = (self.dist_matrix[prev_node][node] + 
                         self.dist_matrix[node][next_node] - 
                         self.dist_matrix[prev_node][next_node])
            
            # 时间窗惩罚估算（基于节点的时间窗紧张程度）
            cust = self.customer_lookup[node]
            time_window_width = cust['due_date'] - cust['ready_time']
            time_penalty_factor = 1.0 / max(1.0, time_window_width) * 100
            
            # 综合成本贡献
            cost_contribution = saved_dist * 3 + time_penalty_factor
            all_nodes.append((ri, pi, node, cost_contribution))
            node_costs[node] = cost_contribution
    
    if not all_nodes:
        return solution, []
    
    total = len(all_nodes)
    k = max(1, int(total * ratio)) if ratio <= 1 else min(total, int(ratio))
    
    # 基于成本贡献的加权随机选择（高成本节点有更高概率被移除）
    weights = [node_costs[node] for _, _, node, _ in all_nodes]
    total_weight = sum(weights)
    if total_weight > 0:
        probs = [w/total_weight for w in weights]
        selected_indices = random.choices(range(len(all_nodes)), weights=probs, k=min(k, len(all_nodes)))
    else:
        selected_indices = random.sample(range(len(all_nodes)), min(k, len(all_nodes)))
    
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
    """智能路径移除：优先移除成本高或时间窗紧张的路径"""
    routes_info = []
    
    for ri, route in enumerate(solution):
        custs = [n for n in route if n != 0]
        if not custs:
            continue
            
        # 计算路径成本
        route_nodes = [self.id_to_customer[n] for n in route]
        route_cost = self.calculator.calculate_route_cost(route_nodes, self.dist_matrix)
        
        # 计算路径的时间窗紧张程度
        time_window_violation = 0
        for node in custs:
            cust = self.customer_lookup[node]
            time_window_width = cust['due_date'] - cust['ready_time']
            if time_window_width < 60:  # 时间窗宽度小于60分钟视为紧张
                time_window_violation += 1
        
        # 路径评分：成本高、时间窗紧张、客户数少的路径优先移除
        score = (route_cost['variable_cost'] * 0.5 + 
                time_window_violation * 100 + 
                (10 - len(custs)) * 50)  # 鼓励移除小路径
        
        routes_info.append((ri, custs, score))
    
    if not routes_info:
        return solution, []
    
    total_customers = sum(len(c) for _, c, _ in routes_info)
    target = max(1, int(total_customers * ratio)) if ratio <= 1 else int(ratio)
    
    # 按评分排序，高分（应优先移除）在前
    routes_info.sort(key=lambda x: x[2], reverse=True)
    
    removed = []
    to_remove_idx = []
    for ri, custs, _ in routes_info:
        if len(removed) + len(custs) > target and len(removed) > 0:
            # 如果添加这条路径会超过目标，跳过
            continue
            
        removed.extend(custs)
        to_remove_idx.append(ri)
        if len(removed) >= target:
            break
    
    # 如果还没达到目标，从剩余路径中随机移除一些节点
    if len(removed) < target:
        remaining_routes = [(ri, custs) for ri, custs, _ in routes_info if ri not in to_remove_idx]
        if remaining_routes:
            ri, custs = random.choice(remaining_routes)
            need = target - len(removed)
            selected = random.sample(custs, min(need, len(custs)))
            removed.extend(selected)
            
            # 从路径中移除选中的节点
            new_sol = [r[:] for r in solution]
            for node in selected:
                if node in new_sol[ri]:
                    new_sol[ri].remove(node)
            new_sol = [r for r in new_sol if len(r) > 2]
            return new_sol, removed
    
    to_remove_idx.sort(reverse=True)
    new_sol = [r[:] for r in solution]
    for ri in to_remove_idx:
        del new_sol[ri]
    
    return new_sol, removed[:target]


def _string_removal(self, solution, ratio):
    """时空连续节点移除：优先移除时间或空间上连续的节点"""
    all_custs = [n for route in solution for n in route if n != 0]
    if not all_custs:
        return solution, []
    
    total = len(all_custs)
    n_remove = max(1, int(total * ratio)) if ratio <= 1 else min(total, int(ratio))
    
    removed = []
    new_sol = [r[:] for r in solution]
    
    # 尝试按时间连续性移除
    time_based_removed = self._time_based_string_removal(new_sol, n_remove)
    if time_based_removed:
        removed = time_based_removed
    else:
        # 回退到原始的空间连续性移除，但加入智能选择
        while len(removed) < n_remove:
            valid_routes = [(i, r) for i, r in enumerate(new_sol) if len(r) > 2]
            if not valid_routes:
                break
            
            # 优先选择时间窗紧张的路径
            route_scores = []
            for i, route in valid_routes:
                custs = [n for n in route if n != 0]
                time_window_violation = sum(1 for n in custs 
                                          if self.customer_lookup[n]['due_date'] - 
                                             self.customer_lookup[n]['ready_time'] < 60)
                route_scores.append((i, route, time_window_violation))
            
            route_scores.sort(key=lambda x: x[2], reverse=True)
            ri, route, _ = route_scores[0] if route_scores[0][2] > 0 else random.choice(valid_routes)
            
            if len(route) <= 2:
                continue
                
            # 选择起始位置：优先从时间窗紧张的节点开始
            tight_positions = []
            for pos in range(1, len(route)-1):
                node = route[pos]
                if node != 0:
                    cust = self.customer_lookup[node]
                    if cust['due_date'] - cust['ready_date'] < 60:
                        tight_positions.append(pos)
            
            start = random.choice(tight_positions) if tight_positions else random.randint(1, len(route)-2)
            slen = min(random.randint(2, 4),  # 增加连续移除长度
                      len(route) - start - 1,
                      n_remove - len(removed))
            
            for offset in range(slen):
                pos = start + offset
                if pos < len(new_sol[ri]) and new_sol[ri][pos] != 0:
                    removed.append(new_sol[ri][pos])
                    del new_sol[ri][pos]
    
    new_sol = [r for r in new_sol if len(r) > 2]
    return new_sol, removed

def _time_based_string_removal(self, solution, n_remove):
    """按时间连续性移除节点"""
    removed = []
    
    for ri, route in enumerate(solution):
        if len(route) <= 2:
            continue
            
        # 提取客户节点及其时间窗
        cust_positions = []
        for pos, node in enumerate(route):
            if node != 0:
                cust = self.customer_lookup[node]
                mid_time = (cust['ready_time'] + cust['due_date']) / 2
                cust_positions.append((pos, node, mid_time))
        
        if len(cust_positions) < 2:
            continue
            
        # 按时间排序
        cust_positions.sort(key=lambda x: x[2])
        
        # 寻找时间上连续的段
        for i in range(len(cust_positions) - 1):
            j = i + 1
            while j < len(cust_positions) and (cust_positions[j][2] - cust_positions[i][2]) < 60:  # 1小时内
                j += 1
            
            if j - i >= 2:  # 找到至少2个连续节点
                segment = cust_positions[i:j]
                if len(removed) + len(segment) <= n_remove:
                    # 按位置逆序移除
                    segment.sort(key=lambda x: x[0], reverse=True)
                    for pos, node, _ in segment:
                        removed.append(node)
                        del solution[ri][pos]
                    break
        
        if removed:
            break
    
    return removed


def _greedy_insert(self, solution, removed_nodes):
    """增强贪心插入：使用候选位置 + 成本增量估算"""
    if not removed_nodes:
        return solution
    if not solution:
        return [[0, n, 0] for n in removed_nodes]
    
    new_sol = [r[:] for r in solution]
    
    # 缓存路径的当前成本
    route_costs = {}
    for ri, route in enumerate(new_sol):
        if len(route) > 2:
            route_nodes = [self.id_to_customer[n] for n in route]
            route_costs[ri] = self.calculator.calculate_route_cost(
                route_nodes, self.dist_matrix)['variable_cost']
    
    for node in removed_nodes:
        node_demand = self.customer_lookup[node].get('demand', 0)
        node_info = self.id_to_customer[node]
        
        best_inc = float('inf')
        best_ri = None
        best_pos = None
        
        # 计算新开路径的成本
        new_route = [0, node, 0]
        new_route_nodes = [self.id_to_customer[0], node_info, self.id_to_customer[0]]
        new_route_cost = self.calculator.calculate_route_cost(
            new_route_nodes, self.dist_matrix)['variable_cost'] + self.calculator.f
        
        for ri, route in enumerate(new_sol):
            # 容量检查
            current_load = sum(self.customer_lookup[n].get('demand', 0) for n in route if n != 0)
            if current_load + node_demand > self.capacity:
                continue
            
            # 使用候选位置而不是遍历所有位置
            candidate_positions = self._candidate_positions(route, node)
            if not candidate_positions:
                continue
            
            # 获取当前路径成本
            if ri not in route_costs:
                route_nodes = [self.id_to_customer[n] for n in route]
                route_costs[ri] = self.calculator.calculate_route_cost(
                    route_nodes, self.dist_matrix)['variable_cost']
            c_before = route_costs[ri]
            
            # 评估候选位置
            for pos in candidate_positions:
                # 快速成本增量估算（距离部分）
                prev_node = route[pos-1]
                next_node = route[pos] if pos < len(route) else 0
                
                dist_inc = (self.dist_matrix[prev_node][node] + 
                           self.dist_matrix[node][next_node] - 
                           self.dist_matrix[prev_node][next_node])
                
                # 时间窗惩罚估算
                time_penalty_est = 0
                if node_info['due_date'] - node_info['ready_time'] < 60:
                    time_penalty_est = 50  # 时间窗紧张的惩罚估计
                
                inc_estimate = dist_inc * 3 + time_penalty_est
                
                # 如果估算值已经比当前最佳差很多，跳过精确计算
                if inc_estimate > best_inc * 1.5 and best_ri is not None:
                    continue
                
                # 精确计算
                route_after = route[:pos] + [node] + route[pos:]
                ra_nodes = [self.id_to_customer[n] for n in route_after]
                c_after = self.calculator.calculate_route_cost(
                    ra_nodes, self.dist_matrix)['variable_cost']
                inc = c_after - c_before
                
                if inc < best_inc:
                    best_inc = inc
                    best_ri = ri
                    best_pos = pos
        
        # 决策：插入现有路径还是新开路径
        if best_ri is not None and best_inc <= new_route_cost:
            new_sol[best_ri].insert(best_pos, node)
            # 更新缓存成本
            route_after = new_sol[best_ri]
            ra_nodes = [self.id_to_customer[n] for n in route_after]
            route_costs[best_ri] = self.calculator.calculate_route_cost(
                ra_nodes, self.dist_matrix)['variable_cost']
        else:
            new_sol.append([0, node, 0])
            # 新路径的成本缓存
            ri = len(new_sol) - 1
            route_costs[ri] = new_route_cost - self.calculator.f
    
    return new_sol


def _regret_insert(self, solution, removed_nodes):
    """Regret-k 插入：计算前k个最佳位置的机会成本差异"""
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
            route_nodes = [self.id_to_customer[n] for n in route]
            route_costs[ri] = self.calculator.calculate_route_cost(
                route_nodes, self.dist_matrix)['variable_cost']
    
    # Regret-k 的 k 值
    K = 3  # 考虑前3个最佳位置
    
    while remaining:
        best_regret = -float('inf')
        best_node = None
        best_ri = None
        best_pos = None
        
        for node in remaining:
            node_demand = self.customer_lookup[node].get('demand', 0)
            node_info = self.id_to_customer[node]
            
            all_options = []
            
            # 1. 现有路径的插入选项
            for ri, route in enumerate(new_sol):
                current_load = sum(self.customer_lookup[n].get('demand', 0) for n in route if n != 0)
                if current_load + node_demand > self.capacity:
                    continue
                
                if ri not in route_costs:
                    route_nodes = [self.id_to_customer[n] for n in route]
                    route_costs[ri] = self.calculator.calculate_route_cost(
                        route_nodes, self.dist_matrix)['variable_cost']
                c_before = route_costs[ri]
                
                # 使用候选位置
                candidate_positions = self._candidate_positions(route, node)
                if not candidate_positions:
                    continue
                
                pos_costs = []
                for pos in candidate_positions:
                    route_after = route[:pos] + [node] + route[pos:]
                    ra_nodes = [self.id_to_customer[n] for n in route_after]
                    c_after = self.calculator.calculate_route_cost(
                        ra_nodes, self.dist_matrix)['variable_cost']
                    inc = c_after - c_before
                    pos_costs.append((inc, ri, pos))
                
                # 取成本最低的K个位置
                pos_costs.sort(key=lambda x: x[0])
                all_options.extend(pos_costs[:K])
            
            # 2. 新开路径选项
            new_route = [0, node, 0]
            new_route_nodes = [self.id_to_customer[0], node_info, self.id_to_customer[0]]
            new_route_cost = self.calculator.calculate_route_cost(
                new_route_nodes, self.dist_matrix)['variable_cost'] + self.calculator.f
            all_options.append((new_route_cost, None, None))
            
            # 按成本排序
            all_options.sort(key=lambda x: x[0])
            
            # 计算Regret-k值：第k个最佳选项与最佳选项的差异
            if len(all_options) >= K:
                regret_sum = 0
                for k in range(1, min(K, len(all_options))):
                    regret_sum += all_options[k][0] - all_options[0][0]
                regret = regret_sum / (K-1) if K > 1 else 0
            elif len(all_options) >= 2:
                regret = all_options[1][0] - all_options[0][0]
            else:
                regret = 0
            
            # 添加节点本身的启发式权重（时间窗紧张的节点应优先插入）
            time_window_width = node_info['due_date'] - node_info['ready_time']
            if time_window_width < 60:
                regret *= 1.5  # 时间窗紧张的节点有更高优先级
            
            if regret > best_regret:
                best_regret = regret
                best_node = node
                best_ri = all_options[0][1]
                best_pos = all_options[0][2]
        
        if best_node is None:
            # 回退到贪心插入
            for n in remaining:
                new_sol.append([0, n, 0])
            break
        
        # 执行插入
        if best_ri is not None:
            new_sol[best_ri].insert(best_pos, best_node)
            # 更新缓存成本
            route_after = new_sol[best_ri]
            ra_nodes = [self.id_to_customer[n] for n in route_after]
            route_costs[best_ri] = self.calculator.calculate_route_cost(
                ra_nodes, self.dist_matrix)['variable_cost']
        else:
            new_sol.append([0, best_node, 0])
        
        remaining.remove(best_node)
    
    return new_sol
```

## 主要改进点

1. **智能破坏策略**：
   - `_random_removal`：基于节点成本贡献的加权随机选择
   - `_route_removal`：优先移除成本高、时间窗紧张的路径
   - `_string_removal`：新增时间连续性移除策略

2. **计算优化**：
   - 使用`_candidate_positions`大幅减少插入位置评估
   - 引入成本增量估算，避免不必要的精确计算
   - 缓存路径成本，减少重复计算

3. **Regret-k机制**：
   - 考虑前k个最佳位置的机会成本差异
   - 对时间窗紧张的节点给予更高优先级

4. **生鲜物流针对性优化**：
   - 时间窗紧张度作为重要决策因素
   - 综合考虑距离成本、时间窗惩罚和货损成本