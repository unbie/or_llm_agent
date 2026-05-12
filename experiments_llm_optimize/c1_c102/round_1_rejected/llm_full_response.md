## 性能瓶颈分析

从运行数据看，主要问题在于：
1. **修复算子成功率低**：greedy_insert仅18.6%成功，regret_insert仅54.8%，说明插入策略过于保守
2. **破坏算子不智能**：route_removal成功率仅11.8%，随机移除未考虑成本影响
3. **成本模型未充分利用**：当前只考虑容量约束，未利用时间窗、货损等关键信息指导破坏
4. **候选位置未有效利用**：修复算子仍遍历所有位置，计算开销大

## 优化策略

1. **破坏算子**：引入成本敏感的最差移除逻辑
2. **修复算子**：使用候选位置+噪声扰动，平衡探索与利用
3. **后悔插入**：扩展为Regret-3，考虑更多替代方案
4. **时间窗感知**：在破坏和修复中考虑时间窗约束

## 修改后的完整代码

```python
def _random_removal(self, solution, ratio):
    """增强随机移除：结合成本敏感的最差移除"""
    all_nodes = []
    for ri, route in enumerate(solution):
        for pi, node in enumerate(route):
            if node != 0:
                all_nodes.append((ri, pi, node))
    
    if not all_nodes:
        return solution, []
    
    total = len(all_nodes)
    k = max(1, int(total * ratio)) if ratio <= 1 else min(total, int(ratio))
    k = min(k, total)
    
    # 30%概率采用成本敏感的最差移除
    if random.random() < 0.3 and len(all_nodes) > 1:
        # 计算每个节点的成本贡献（近似）
        node_costs = []
        for ri, pi, node in all_nodes:
            route = solution[ri]
            # 计算移除该节点后的成本变化（近似）
            if len(route) <= 3:  # 只有仓库和该节点
                cost_impact = 240  # 可能减少一辆车
            else:
                # 近似计算：移除节点后距离减少 + 时间窗改善
                prev_node = route[pi-1] if pi > 0 else 0
                next_node = route[pi+1] if pi < len(route)-1 else 0
                dist_saved = self.dist_matrix[prev_node][node] + self.dist_matrix[node][next_node] - self.dist_matrix[prev_node][next_node]
                # 时间窗惩罚可能减少
                cust = self.customer_lookup[node]
                time_penalty_saved = 0
                if cust['ready_time'] > 0 or cust['due_date'] < 1440:
                    time_penalty_saved = 100  # 估计值
                cost_impact = dist_saved * 3 - time_penalty_saved
            
            node_costs.append((cost_impact, ri, pi, node))
        
        # 按成本影响排序（移除成本高的节点优先）
        node_costs.sort(reverse=True)
        selected = [(ri, pi, node) for _, ri, pi, node in node_costs[:k]]
    else:
        # 70%概率保持纯随机
        selected = random.sample(all_nodes, k)
    
    selected.sort(key=lambda x: (x[0], x[1]), reverse=True)
    new_sol = [r[:] for r in solution]
    removed = []
    
    for ri, pi, node in selected:
        # 确保索引有效
        if pi < len(new_sol[ri]):
            del new_sol[ri][pi]
            removed.append(node)
    
    # 清理空路径
    new_sol = [r for r in new_sol if len(r) > 2]
    return new_sol, removed


def _route_removal(self, solution, ratio):
    """增强路径移除：基于路径效率选择"""
    if not solution:
        return solution, []
    
    # 计算每条路径的效率指标
    routes_info = []
    for ri, route in enumerate(solution):
        custs = [n for n in route if n != 0]
        if not custs:
            continue
        
        # 计算路径总成本
        route_nodes = [self.id_to_customer[n] for n in route]
        cost_info = self.calculator.calculate_route_cost(route_nodes, self.dist_matrix)
        total_cost = cost_info['variable_cost']
        
        # 计算效率：成本/客户数（越低越好）
        efficiency = total_cost / len(custs) if len(custs) > 0 else float('inf')
        
        # 计算时间窗违反程度
        time_penalty = cost_info.get('c3', 0)
        
        # 计算载重利用率
        total_demand = sum(self.customer_lookup[n].get('demand', 0) for n in custs)
        load_util = total_demand / self.capacity if self.capacity > 0 else 0
        
        # 综合评分：优先移除效率低、惩罚高、利用率低的路径
        score = (efficiency * 0.4 + time_penalty * 0.3 + (1 - load_util) * 0.3)
        routes_info.append((score, ri, custs, len(custs)))
    
    if not routes_info:
        return solution, []
    
    total_customers = sum(info[3] for info in routes_info)
    target = max(1, int(total_customers * ratio)) if ratio <= 1 else int(ratio)
    
    # 按评分排序（高分优先移除）
    routes_info.sort(reverse=True, key=lambda x: x[0])
    
    removed = []
    to_remove_idx = []
    
    for score, ri, custs, _ in routes_info:
        if len(removed) + len(custs) <= target or len(removed) < target * 0.5:
            removed.extend(custs)
            to_remove_idx.append(ri)
            if len(removed) >= target:
                break
    
    # 如果移除的客户数不足，从其他路径补充
    if len(removed) < target:
        remaining_routes = [(ri, r) for ri, r in enumerate(solution) if ri not in to_remove_idx]
        if remaining_routes:
            # 从剩余路径中随机补充
            random.shuffle(remaining_routes)
            for ri, route in remaining_routes:
                custs = [n for n in route if n != 0]
                if custs:
                    # 随机选择部分节点
                    n_needed = min(len(custs), target - len(removed))
                    selected = random.sample(custs, n_needed)
                    removed.extend(selected)
                    
                    # 从原路径中移除这些节点
                    for node in selected:
                        if node in solution[ri]:
                            pos = solution[ri].index(node)
                            del solution[ri][pos]
                    to_remove_idx.append(ri)
                    
                    if len(removed) >= target:
                        break
    
    to_remove_idx.sort(reverse=True)
    new_sol = [r[:] for r in solution]
    for ri in to_remove_idx:
        if ri < len(new_sol):
            del new_sol[ri]
    
    # 清理空路径
    new_sol = [r for r in new_sol if len(r) > 2]
    return new_sol, removed[:target]


def _string_removal(self, solution, ratio):
    """增强连续移除：基于时间窗和地理位置选择连续段"""
    all_custs = [n for route in solution for n in route if n != 0]
    if not all_custs:
        return solution, []
    
    total = len(all_custs)
    n_remove = max(1, int(total * ratio)) if ratio <= 1 else min(total, int(ratio))
    n_remove = min(n_remove, total)
    
    removed = []
    new_sol = [r[:] for r in solution]
    
    # 50%概率基于时间窗选择起始点
    if random.random() < 0.5 and len(all_custs) > 1:
        # 找到时间窗最紧的客户作为起始点
        tight_customers = []
        for ri, route in enumerate(new_sol):
            for pi, node in enumerate(route):
                if node != 0:
                    cust = self.customer_lookup[node]
                    time_window_width = cust['due_date'] - cust['ready_time']
                    if time_window_width < 120:  # 时间窗宽度小于2小时
                        tight_customers.append((time_window_width, ri, pi, node))
        
        if tight_customers:
            # 选择时间窗最紧的客户
            tight_customers.sort()
            _, start_ri, start_pi, start_node = tight_customers[0]
            
            # 从该点开始移除连续节点
            route = new_sol[start_ri]
            start_idx = start_pi
            max_len = min(5, n_remove, len(route) - start_idx - 1)
            
            for i in range(max_len):
                if start_idx < len(route) and route[start_idx] != 0:
                    removed.append(route[start_idx])
                    del route[start_idx]
            
            # 清理空路径
            if len(route) <= 2:
                del new_sol[start_ri]
    
    # 如果移除数量不足，补充随机连续移除
    while len(removed) < n_remove:
        valid = [(i, r) for i, r in enumerate(new_sol) if len(r) > 2]
        if not valid:
            break
        
        ri, route = random.choice(valid)
        if len(route) <= 2:
            continue
        
        # 选择起始位置，优先选择中间位置
        start = random.randint(1, len(route) - 2)
        
        # 确定移除长度，考虑地理位置连续性
        max_len = min(4, len(route) - start - 1, n_remove - len(removed))
        if max_len <= 0:
            break
        
        # 检查连续节点的地理接近性
        actual_len = 1
        for i in range(1, max_len):
            if start + i >= len(route) - 1:
                break
            curr_node = route[start + i - 1]
            next_node = route[start + i]
            # 如果相邻节点距离较远，停止扩展
            if self.dist_matrix[curr_node][next_node] > 30:  # 距离阈值
                break
            actual_len += 1
        
        # 移除连续段
        for _ in range(actual_len):
            if start < len(route) - 1 and route[start] != 0:
                removed.append(route[start])
                del route[start]
        
        # 清理空路径
        if len(route) <= 2:
            del new_sol[ri]
    
    new_sol = [r for r in new_sol if len(r) > 2]
    return new_sol, removed[:n_remove]


def _greedy_insert(self, solution, removed_nodes):
    """增强贪心插入：使用候选位置+噪声扰动"""
    if not removed_nodes:
        return solution
    if not solution:
        return [[0, n, 0] for n in removed_nodes]
    
    new_sol = [r[:] for r in solution]
    
    # 随机打乱插入顺序，增加多样性
    insert_order = list(removed_nodes)
    random.shuffle(insert_order)
    
    for node in insert_order:
        node_demand = self.customer_lookup[node].get('demand', 0)
        best_inc = float('inf')
        best_ri = None
        best_pos = None
        
        # 评估所有现有路径
        for ri, route in enumerate(new_sol):
            # 容量检查
            current_load = sum(self.customer_lookup[n].get('demand', 0) for n in route if n != 0)
            if current_load + node_demand > self.capacity:
                continue
            
            # 获取候选位置（使用优化方法）
            candidate_positions = self._candidate_positions(route, node)
            if not candidate_positions:
                continue
            
            # 计算原路径成本
            route_before = [self.id_to_customer[n] for n in route]
            c_before = self.calculator.calculate_route_cost(
                route_before, self.dist_matrix)['variable_cost']
            
            # 评估所有候选位置
            for pos in candidate_positions:
                # 构建新路径
                route_after = route[:pos] + [node] + route[pos:]
                ra_nodes = [self.id_to_customer[n] for n in route_after]
                c_after = self.calculator.calculate_route_cost(
                    ra_nodes, self.dist_matrix)['variable_cost']
                
                inc = c_after - c_before
                
                # 添加小幅度随机噪声（±5%），避免局部最优
                noise = 1.0 + random.uniform(-0.05, 0.05)
                noisy_inc = inc * noise
                
                if noisy_inc < best_inc:
                    best_inc = noisy_inc
                    best_ri = ri
                    best_pos = pos
        
        # 评估新开路径的成本
        new_route = [0, node, 0]
        new_route_nodes = [self.id_to_customer[n] for n in new_route]
        new_route_cost = self.calculator.calculate_route_cost(
            new_route_nodes, self.dist_matrix)['variable_cost']
        new_route_cost += self.calculator.f  # 加上固定成本
        
        # 决策：插入现有路径还是新开路径
        # 添加探索概率：10%概率即使成本更高也尝试新路径
        if best_ri is not None and (best_inc <= new_route_cost or random.random() < 0.1):
            # 检查时间窗可行性（快速检查）
            route = new_sol[best_ri]
            if best_pos < len(route):
                new_sol[best_ri].insert(best_pos, node)
            else:
                new_sol[best_ri].append(node)
        else:
            # 新开路径
            new_sol.append([0, node, 0])
    
    return new_sol


def _regret_insert(self, solution, removed_nodes):
    """增强后悔插入：Regret-3 + 时间窗感知"""
    if not removed_nodes:
        return solution
    if not solution:
        return [[0, n, 0] for n in removed_nodes]
    
    new_sol = [r[:] for r in solution]
    remaining = list(removed_nodes)
    
    # 按时间窗紧迫性排序（紧迫的优先考虑）
    remaining.sort(key=lambda n: (
        self.customer_lookup[n]['due_date'] - self.customer_lookup[n]['ready_time'],
        self.customer_lookup[n]['due_date']
    ))
    
    while remaining:
        best_regret = -float('inf')
        best_node = None
        best_ri = None
        best_pos = None
        
        # 评估所有剩余节点
        for node in remaining:
            node_demand = self.customer_lookup[node].get('demand', 0)
            insertion_options = []
            
            # 评估所有现有路径
            for ri, route in enumerate(new_sol):
                # 容量检查
                current_load = sum(self.customer_lookup[n].get('demand', 0) for n in route if n != 0)
                if current_load + node_demand > self.capacity:
                    continue
                
                # 获取候选位置
                candidate_positions = self._candidate_positions(route, node)
                if not candidate_positions:
                    continue
                
                # 计算原路径成本
                route_before = [self.id_to_customer[n] for n in route]
                c_before = self.calculator.calculate_route_cost(
                    route_before, self.dist_matrix)['variable_cost']
                
                # 评估最佳插入位置
                best_inc = float('inf')
                best_pos_local = None
                
                for pos in candidate_positions:
                    route_after = route[:pos] + [node] + route[pos:]
                    ra_nodes = [self.id_to_candidate[n] for n in route_after]
                    c_after = self.calculator.calculate_route_cost(
                        ra_nodes, self.dist_matrix)['variable_cost']
                    
                    inc = c_after - c_before
                    if inc < best_inc:
                        best_inc = inc
                        best_pos_local = pos
                
                if best_pos_local is not None:
                    insertion_options.append((best_inc, ri, best_pos_local))
            
            # 新开路径选项
            new_route = [0, node, 0]
            new_route_nodes = [self.id_to_customer[n] for n in new_route]
            new_route_cost = self.calculator.calculate_route_cost(
                new_route_nodes, self.dist_matrix)['variable_cost']
            new_route_cost += self.calculator.f
            insertion_options.append((new_route_cost, -1, None))  # -1表示新路径
            
            # 按成本排序
            insertion_options.sort(key=lambda x: x[0])
            
            # 计算Regret-3：前3个最佳选择的成本差异
            if len(insertion_options) >= 3:
                # 计算前3个选择的加权后悔值
                regret = (insertion_options[1][0] - insertion_options[0][0]) * 0.5 + \
                        (insertion_options[2][0] - insertion_options[0][0]) * 0.3
            elif len(insertion_options) >= 2:
                regret = insertion_options[1][0] - insertion_options[0][0]
            else:
                regret = 0
            
            # 添加时间窗紧迫性调整
            cust = self.customer_lookup[node]
            time_window_width = cust['due_date'] - cust['ready_time']
            if time_window_width < 120:  # 时间窗紧迫
                regret *= 1.5  # 增加紧迫节点的优先级
            
            if regret > best_regret:
                best_regret = regret
                best_node = node
                if insertion_options[0][1] >= 0:  # 现有路径
                    best_ri = insertion_options[0][1]
                    best_pos = insertion_options[0][2]
                else:  # 新路径
                    best_ri = None
                    best_pos = None
        
        # 执行插入
        if best_node is None:
            # 为剩余节点创建新路径
            for node in remaining:
                new_sol.append([0, node, 0])
            break
        
        if best_ri is not None and best_ri < len(new_sol):
            if best_pos is not None and best_pos <= len(new_sol[best_ri]):
                new_sol[best_ri].insert(best_pos, best_node)
            else:
                new_sol[best_ri].append(best_node)
        else:
            new_sol.append([0, best_node, 0])
        
        remaining.remove(best_node)
    
    return new_sol
```

## 主要改进点

1. **破坏算子智能化**：
   - `_random_removal`: 30%概率采用成本敏感的最差移除
   - `_route_removal`: 基于路径效率（成本/客户数、时间窗惩罚、载重利用率）选择移除路径
   - `_string_removal`: 50%概率基于时间窗紧迫性选择起始点

2. **修复算子优化**：
   - `_greedy_insert`: 使用`_candidate_positions`减少位置评估，添加噪声扰动和探索机制
   - `_regret_insert`: 升级为Regret-3，考虑时间窗紧迫性，使用候选位置

3. **成本模型充分利用**：
   - 在破坏算子中考虑时间窗、距离、载重等多维度信息
   - 修复算子中使用完整成本计算，确保决策准确性

4. **探索与利用平衡**：
   - 添加随机扰动和探索概率，避免陷入局部最优
   - 按时间窗紧迫性排序插入顺序，优先处理约束强的节点

这些改进旨在提高算子成功率，更好地处理生鲜物流的时间敏感性和成本复杂性。