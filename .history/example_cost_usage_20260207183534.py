"""
ALNS算子中使用成本计算的示例
展示如何在插入算子中使用 utils.py 的 calculate_route_cost() 函数
"""

# ============================================================================
# 示例1：简化版贪心插入（只使用距离成本）
# ============================================================================
def greedy_insert_simple(self, solution, removed_nodes):
    """
    简化版：只使用距离增量作为成本指标
    优点：速度快，适合大规模问题
    """
    if not removed_nodes:
        return solution
    if not solution:
        return [[0, node, 0] for node in removed_nodes]
    
    new_solution = [route[:] for route in solution]
    
    for node in removed_nodes:
        node_demand = self.customer_lookup[node]['demand']
        best_cost = float('inf')
        best_route_idx = None
        best_position = None
        
        # 尝试插入到现有路径
        for route_idx, route in enumerate(new_solution):
            route_demand = sum(self.customer_lookup[n]['demand'] for n in route if n != 0)
            if route_demand + node_demand > self.capacity:
                continue
            
            for pos in range(1, len(route)):
                prev, next_n = route[pos-1], route[pos]
                # 方法1：只计算距离增量（快速）
                cost_inc = (self.dist_matrix[prev][node] + 
                           self.dist_matrix[node][next_n] - 
                           self.dist_matrix[prev][next_n])
                
                if cost_inc < best_cost:
                    best_cost = cost_inc
                    best_route_idx = route_idx
                    best_position = pos
        
        # 考虑创建新路径
        new_route_cost = self.dist_matrix[0][node] * 2
        
        if best_route_idx is not None and best_cost <= new_route_cost:
            new_solution[best_route_idx].insert(best_position, node)
        else:
            new_solution.append([0, node, 0])
    
    return new_solution


# ============================================================================
# 示例2：完整版贪心插入（使用 calculate_route_cost）
# ============================================================================
def greedy_insert_full_cost(self, solution, removed_nodes):
    """
    完整版：使用 calculate_route_cost() 计算真实成本
    优点：考虑时间窗、货损等所有成本因素，决策更精确
    缺点：计算量大，适合小规模问题或最终优化阶段
    """
    if not removed_nodes:
        return solution
    if not solution:
        return [[0, node, 0] for node in removed_nodes]
    
    new_solution = [route[:] for route in solution]
    
    for node in removed_nodes:
        node_demand = self.customer_lookup[node]['demand']
        best_cost_increase = float('inf')
        best_route_idx = None
        best_position = None
        
        # 尝试插入到现有路径
        for route_idx, route in enumerate(new_solution):
            route_demand = sum(self.customer_lookup[n]['demand'] for n in route if n != 0)
            if route_demand + node_demand > self.capacity:
                continue
            
            # 计算当前路径的成本
            route_nodes_before = [self.solver.id_to_customer[n] for n in route]
            cost_before = self.solver.calculator.calculate_route_cost(
                route_nodes_before, self.dist_matrix
            )['variable_cost']
            
            for pos in range(1, len(route)):
                # 插入节点后的新路径
                new_route = route[:pos] + [node] + route[pos:]
                route_nodes_after = [self.solver.id_to_customer[n] for n in new_route]
                
                # 计算插入后的成本（包括C12+C13+C2+C3）
                cost_after = self.solver.calculator.calculate_route_cost(
                    route_nodes_after, self.dist_matrix
                )['variable_cost']
                
                # 成本增量
                cost_inc = cost_after - cost_before
                
                if cost_inc < best_cost_increase:
                    best_cost_increase = cost_inc
                    best_route_idx = route_idx
                    best_position = pos
        
        # 考虑创建新路径
        new_route = [0, node, 0]
        route_nodes_new = [self.solver.id_to_customer[n] for n in new_route]
        new_route_cost = self.solver.calculator.calculate_route_cost(
            route_nodes_new, self.dist_matrix
        )['variable_cost']
        # 加上固定成本
        new_route_cost += self.solver.calculator.f
        
        if best_route_idx is not None and best_cost_increase <= new_route_cost:
            new_solution[best_route_idx].insert(best_position, node)
        else:
            new_solution.append([0, node, 0])
    
    return new_solution


# ============================================================================
# 示例3：混合策略（初筛+精确评估）
# ============================================================================
def greedy_insert_hybrid(self, solution, removed_nodes):
    """
    混合策略：先用距离快速筛选候选位置，再用完整成本精确评估
    平衡速度和精度
    """
    if not removed_nodes:
        return solution
    if not solution:
        return [[0, node, 0] for node in removed_nodes]
    
    new_solution = [route[:] for route in solution]
    CANDIDATE_SIZE = 5  # 保留前5个候选位置
    
    for node in removed_nodes:
        node_demand = self.customer_lookup[node]['demand']
        
        # 阶段1：用距离快速收集候选位置
        candidates = []  # (cost_estimate, route_idx, position)
        
        for route_idx, route in enumerate(new_solution):
            route_demand = sum(self.customer_lookup[n]['demand'] for n in route if n != 0)
            if route_demand + node_demand > self.capacity:
                continue
            
            for pos in range(1, len(route)):
                prev, next_n = route[pos-1], route[pos]
                dist_inc = (self.dist_matrix[prev][node] + 
                           self.dist_matrix[node][next_n] - 
                           self.dist_matrix[prev][next_n])
                candidates.append((dist_inc, route_idx, pos))
        
        # 只保留前K个候选
        candidates.sort(key=lambda x: x[0])
        candidates = candidates[:CANDIDATE_SIZE]
        
        # 阶段2：用完整成本精确评估候选位置
        best_cost_increase = float('inf')
        best_route_idx = None
        best_position = None
        
        for _, route_idx, pos in candidates:
            route = new_solution[route_idx]
            
            # 计算插入前后的成本
            route_nodes_before = [self.solver.id_to_customer[n] for n in route]
            cost_before = self.solver.calculator.calculate_route_cost(
                route_nodes_before, self.dist_matrix
            )['variable_cost']
            
            new_route = route[:pos] + [node] + route[pos:]
            route_nodes_after = [self.solver.id_to_customer[n] for n in new_route]
            cost_after = self.solver.calculator.calculate_route_cost(
                route_nodes_after, self.dist_matrix
            )['variable_cost']
            
            cost_inc = cost_after - cost_before
            
            if cost_inc < best_cost_increase:
                best_cost_increase = cost_inc
                best_route_idx = route_idx
                best_position = pos
        
        # 考虑新路径
        new_route = [0, node, 0]
        route_nodes_new = [self.solver.id_to_customer[n] for n in new_route]
        new_route_cost = self.solver.calculator.calculate_route_cost(
            route_nodes_new, self.dist_matrix
        )['variable_cost'] + self.solver.calculator.f
        
        if best_route_idx is not None and best_cost_increase <= new_route_cost:
            new_solution[best_route_idx].insert(best_position, node)
        else:
            new_solution.append([0, node, 0])
    
    return new_solution


# ============================================================================
# 建议
# ============================================================================
"""
选择哪种方法？

1. **小规模问题（<100客户）**：
   - 推荐：完整版 (greedy_insert_full_cost)
   - 理由：问题规模小，计算量可接受，精确决策能显著提升解质量

2. **中等规模问题（100-500客户）**：
   - 推荐：混合策略 (greedy_insert_hybrid)
   - 理由：平衡速度和精度，只对筛选出的候选位置进行精确评估

3. **大规模问题（>500客户）**：
   - 推荐：简化版 (greedy_insert_simple)
   - 理由：距离是成本的主要因素，简化方法已经能提供良好的启发式指导
   
4. **实际应用建议**：
   - ALNS迭代过程中：使用简化版（速度优先）
   - 最终局部优化阶段：使用完整版（精度优先）
   - 或者：前N次迭代用简化版，后续迭代用完整版

注意：无论算子中使用哪种方法，框架的 cost() 函数都会使用完整的
     calculate_route_cost() 来评估最终解的真实成本！
"""
