HEURISTIC_PLUGIN_TEMPLATE = """
你是一个熟悉 ALNS 算法的 Python 工程师，请实现生鲜物流 VRP 问题的启发式算子。

【重要】只需实现5个算子，不要实现更多！

【优化算子配置】：
必须实现的算子（共5个）：
1. random_removal   - 随机分散破坏
2. route_removal    - 整条路径破坏
3. string_removal   - 连续节点破坏
4. greedy_insert    - 贪心修复
5. regret_insert    - 后悔修复

不要实现worst_removal、related_removal、shaw_removal、history_removal、cluster_removal、random_insert等其他算子！

【核心要求】：
- 所有代码必须完整实现（不能有pass、TODO或空函数）
- 处理边界条件（空列表、防止越界）
- 降序移除避免索引错误
- 检查容量约束

注意：所有破坏算子都是random策略的变种，简单易实现

【数据结构】：
   - solution: [[0, node1, node2, 0], [0, node3, 0], ...]（0是仓库）
   - self.dist_matrix[i][j]: 节点i到j的距离
   - self.capacity: 车辆容量
   - self.customer_lookup: {node_id: {'demand': ..., 'x': ..., 'y': ..., 'ready_time': ..., 'due_date': ...}}
   - self.solver: 可访问求解器对象（包含成本计算器）
   - self.solver.calculator: 成本计算器，可计算完整路径成本

【成本计算方式】：
   **必须使用完整成本计算**（通过 self.solver.calculator 调用）：
   
   # 计算插入前路径的成本
   route_before = [self.solver.id_to_customer[n] for n in route]
   cost_before = self.solver.calculator.calculate_route_cost(route_before, self.dist_matrix)['variable_cost']
   
   # 计算插入后路径的成本
   route_after = route[:pos] + [node] + route[pos:]
   route_after_nodes = [self.solver.id_to_customer[n] for n in route_after]
   cost_after = self.solver.calculator.calculate_route_cost(route_after_nodes, self.dist_matrix)['variable_cost']
   
   # 成本增量 = 插入后成本 - 插入前成本
   cost_inc = cost_after - cost_before
   
   这样计算包括了所有成本因素：
   - C12: 距离成本
   - C13: 制冷成本
   - C2: 货损成本（新鲜度衰减）
   - C3: 时间窗惩罚成本（早到/迟到）
   
   对于新路径，需要加上车辆固定成本C11：
   new_route = [0, node, 0]
   route_nodes = [self.solver.id_to_customer[n] for n in new_route]
   new_route_cost = self.solver.calculator.calculate_route_cost(route_nodes, self.dist_matrix)['variable_cost']
   new_route_cost += self.solver.calculator.f  # 加上固定成本

【必须导入】：
   import random
   import math

【算子实现】：

═════════════════════════════════════════════════════════════
【1. random_removal】- 随机破坏算子（成功率2.1%，唯一有效的破坏算子）
═════════════════════════════════════════════════════════════

**功能**：随机移除ratio比例的客户节点

**实现步骤**：
1. 收集所有非0节点（包括route_idx, pos_idx, node）
2. 计算移除数量n（处理ratio<=1和>1两种情况）
3. 使用random.sample随机选择n个节点
4. **降序排序后移除**（避免索引变化）
5. 删除长度<=2的空路径
6. 返回 (new_solution, removed_nodes)

**关键代码**：
```python
def random_removal(self, solution, ratio):
    all_nodes = []
    for route_idx, route in enumerate(solution):
        for pos_idx, node in enumerate(route):
            if node != 0:
                all_nodes.append((route_idx, pos_idx, node))
    
    if not all_nodes:
        return solution, []
    
    total_customers = len(all_nodes)
    if ratio <= 1.0:
        n = max(1, math.ceil(total_customers * ratio))
    else:
        n = int(ratio)
    n = min(n, total_customers)
    
    selected = random.sample(all_nodes, n)
    selected.sort(key=lambda x: (x[0], x[1]), reverse=True)  # 降序
    
    new_solution = [route[:] for route in solution]
    removed_nodes = []
    for route_idx, pos_idx, node in selected:
        del new_solution[route_idx][pos_idx]
        removed_nodes.append(node)
    
    new_solution = [route for route in new_solution if len(route) > 2]
    return new_solution, removed_nodes
```

═════════════════════════════════════════════════════════════
【2. route_removal】- 路径破坏算子（整条路径移除，激进重构）
═════════════════════════════════════════════════════════════

**功能**：随机选择若干条完整路径，移除其中所有客户节点

**与random_removal的区别**：
- random_removal: 移除分散的独立节点
- route_removal: 移除整条路径（更激进的破坏）

**实现步骤**：
1. 收集所有非空路径（长度>2的路径）
2. 计算要移除的客户数量（根据ratio）
3. 随机选择路径，直到累计移除节点数>=目标数量
4. 返回 (new_solution, removed_nodes)

**关键代码**：
```python
def route_removal(self, solution, ratio):
    # 收集所有非空路径
    routes_with_customers = []
    for route_idx, route in enumerate(solution):
        customers = [node for node in route if node != 0]
        if customers:
            routes_with_customers.append((route_idx, customers))
    
    if not routes_with_customers:
        return solution, []
    
    # 计算目标移除数量
    total_customers = sum(len(c) for _, c in routes_with_customers)
    if ratio <= 1.0:
        target = max(1, math.ceil(total_customers * ratio))
    else:
        target = int(ratio)
    
    # 随机选择路径直到达到目标
    random.shuffle(routes_with_customers)
    removed_nodes = []
    routes_to_remove = []
    
    for route_idx, customers in routes_with_customers:
        removed_nodes.extend(customers)
        routes_to_remove.append(route_idx)
        if len(removed_nodes) >= target:
            break
    
    # 降序删除路径
    routes_to_remove.sort(reverse=True)
    new_solution = [route[:] for route in solution]
    for route_idx in routes_to_remove:
        del new_solution[route_idx]
    
    return new_solution, removed_nodes[:target]
```

═════════════════════════════════════════════════════════════
【3. string_removal】- 连续节点破坏算子（局部重构）
═════════════════════════════════════════════════════════════

**功能**：在路径上随机移除连续的2-3个节点段

**与random_removal的区别**：
- random_removal: 移除分散节点
- string_removal: 移除连续节点（保持路径结构）

**实现步骤**：
1. 随机选择一条路径
2. 随机选择起始位置
3. 从该位置移除连续2-3个节点
4. 重复直到达到ratio
5. 返回 (new_solution, removed_nodes)

**关键代码**：
```python
def string_removal(self, solution, ratio):
    all_nodes = [node for route in solution for node in route if node != 0]
    if not all_nodes:
        return solution, []
    
    total_customers = len(all_nodes)
    if ratio <= 1.0:
        n = max(1, math.ceil(total_customers * ratio))
    else:
        n = int(ratio)
    n = min(n, total_customers)
    
    removed_nodes = []
    new_solution = [route[:] for route in solution]
    
    while len(removed_nodes) < n and any(len(r) > 2 for r in new_solution):
        # 选择非空路径
        valid_routes = [(i, r) for i, r in enumerate(new_solution) if len(r) > 2]
        if not valid_routes:
            break
        
        route_idx, route = random.choice(valid_routes)
        
        # 随机起始位置（排除仓库）
        start_pos = random.randint(1, len(route) - 2)
        
        # 移除2-3个连续节点
        string_len = min(random.randint(1, 3), len(route) - start_pos - 1, n - len(removed_nodes))
        
        for _ in range(string_len):
            if start_pos < len(new_solution[route_idx]) - 1:
                node = new_solution[route_idx][start_pos]
                if node != 0:
                    removed_nodes.append(node)
                    del new_solution[route_idx][start_pos]
    
    # 删除空路径
    new_solution = [route for route in new_solution if len(route) > 2]
    return new_solution, removed_nodes
```

═════════════════════════════════════════════════════════════
【4. greedy_insert】- 贪心修复算子（基础且有效）
═════════════════════════════════════════════════════════════

**功能**：将每个节点插入到成本增量最小的位置

**关键决策**：使用 `<=` 而不是 `<` 来比较现有路径和新路径成本
- 相等时优先选择现有路径，减少车辆数

**完整成本计算实现**：
```python
def greedy_insert(self, solution, removed_nodes):
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
        
        # 尝试插入到现有路径（使用完整成本计算）
        for route_idx, route in enumerate(new_solution):
            route_demand = sum(self.customer_lookup[n]['demand'] for n in route if n != 0)
            if route_demand + node_demand > self.capacity:
                continue
            
            # 计算插入前的路径成本
            route_before = [self.solver.id_to_customer[n] for n in route]
            cost_before = self.solver.calculator.calculate_route_cost(
                route_before, self.dist_matrix
            )['variable_cost']
            
            for pos in range(1, len(route)):
                # 计算插入后的路径成本
                route_after = route[:pos] + [node] + route[pos:]
                route_after_nodes = [self.solver.id_to_customer[n] for n in route_after]
                cost_after = self.solver.calculator.calculate_route_cost(
                    route_after_nodes, self.dist_matrix
                )['variable_cost']
                
                # 成本增量
                cost_inc = cost_after - cost_before
                
                if cost_inc < best_cost_increase:
                    best_cost_increase = cost_inc
                    best_route_idx = route_idx
                    best_position = pos
        
        # 考虑创建新路径（包括固定成本）
        new_route = [0, node, 0]
        route_nodes_new = [self.solver.id_to_customer[n] for n in new_route]
        new_route_cost = self.solver.calculator.calculate_route_cost(
            route_nodes_new, self.dist_matrix
        )['variable_cost']
        new_route_cost += self.solver.calculator.f  # 加上车辆固定成本C11
        
        # 决策：<= 优先现有路径
        if best_route_idx is not None and best_cost_increase <= new_route_cost:
            new_solution[best_route_idx].insert(best_position, node)
        else:
            new_solution.append([0, node, 0])
    
    return new_solution
```

═════════════════════════════════════════════════════════════
【5. regret_insert】- 后悔修复算子（最有效的修复策略）
═════════════════════════════════════════════════════════════

**功能**：优先插入"后悔值"大的节点（错过机会成本高的节点）

**后悔值定义**：regret = second_best_cost - best_cost
- 后悔值大：现在不插到最佳位置，将来就没好位置了
- 后悔值小：位置选择灵活，可以等等

**完整成本计算实现**：
```python
def regret_insert(self, solution, removed_nodes):
    if not removed_nodes:
        return solution
    if not solution:
        return [[0, node, 0] for node in removed_nodes]
    
    new_solution = [route[:] for route in solution]
    remaining = list(removed_nodes)
    
    while remaining:
        best_regret = -float('inf')
        best_node = None
        best_route_idx = None
        best_position = None
        
        # 为每个剩余节点计算后悔值
        for node in remaining:
            node_demand = self.customer_lookup[node]['demand']
            costs = []  # (cost_increase, route_idx, position)
            
            # 收集所有可行的插入位置及其成本增量
            for route_idx, route in enumerate(new_solution):
                route_demand = sum(self.customer_lookup[n]['demand'] for n in route if n != 0)
                if route_demand + node_demand > self.capacity:
                    continue
                
                # 计算插入前的成本
                route_before = [self.solver.id_to_customer[n] for n in route]
                cost_before = self.solver.calculator.calculate_route_cost(
                    route_before, self.dist_matrix
                )['variable_cost']
                
                for pos in range(1, len(route)):
                    # 计算插入后的成本
                    route_after = route[:pos] + [node] + route[pos:]
                    route_after_nodes = [self.solver.id_to_customer[n] for n in route_after]
                    cost_after = self.solver.calculator.calculate_route_cost(
                        route_after_nodes, self.dist_matrix
                    )['variable_cost']
                    
                    cost_inc = cost_after - cost_before
                    costs.append((cost_inc, route_idx, pos))
            
            # 添加新路径选项（包括固定成本）
            new_route = [0, node, 0]
            route_nodes_new = [self.solver.id_to_customer[n] for n in new_route]
            new_route_cost = self.solver.calculator.calculate_route_cost(
                route_nodes_new, self.dist_matrix
            )['variable_cost']
            new_route_cost += self.solver.calculator.f
            costs.append((new_route_cost, None, None))
            
            # 排序并计算后悔值
            costs.sort(key=lambda x: x[0])
            
            if len(costs) >= 2:
                regret = costs[1][0] - costs[0][0]
            else:
                regret = 0
            
            # 更新最大后悔值
            if regret > best_regret:
                best_regret = regret
                best_node = node
                best_route_idx = costs[0][1]
                best_position = costs[0][2]
        
        # 插入选中的节点
        if best_node is None:
            # 无法插入，创建新路径
            for node in remaining:
                new_solution.append([0, node, 0])
            break
        
        if best_route_idx is not None:
            new_solution[best_route_idx].insert(best_position, best_node)
        else:
            new_solution.append([0, best_node, 0])
        
        remaining.remove(best_node)
    
    return new_solution
```


═════════════════════════════════════════════════════════════
【完整类结构】
═════════════════════════════════════════════════════════════

请按照以下结构生成完整代码：

```python
import random
import math

class HeuristicPlugin:
    def __init__(self, **kwargs):  
        self.capacity = kwargs.get('vehicle_capacity')
        self.customers = kwargs.get('customers', [])
        self.dist_matrix = None
        self.customer_lookup = {c['id']: c for c in self.customers}
    
    # 实现下面5个函数（3个破坏 + 2个修复）：
    def random_removal(self, solution, ratio):
        # 实现随机移除（分散节点）
        pass
    
    def route_removal(self, solution, ratio):
        # 实现路径移除（整条路径）
        pass
    
    def string_removal(self, solution, ratio):
        # 实现连续节点移除（2-3个连续节点）
        pass
    
    def greedy_insert(self, solution, removed_nodes):
        # 实现贪心插入（成本最小）
        pass
    
    def regret_insert(self, solution, removed_nodes):
        # 实现后悔插入（后悔值最大优先）
        pass
```

【最后检查清单】：
✅ 所有removal算子是否有 `n = min(n, total_customers)` 防止越界
✅ random_removal 是否降序排序 `reverse=True`
✅ route_removal 是否降序删除路径 `routes_to_remove.sort(reverse=True)`
✅ string_removal 是否删除空路径 `if len(route) > 2`
✅ greedy_insert 的 best_cost 是否初始化为 `float('inf')`
✅ greedy_insert 是否用 `<=` 比较（不是 `<`）
✅ regret_insert 是否处理空列表 `if not removed_nodes`
✅ 所有函数是否有容量检查 `route_demand + node_demand <= capacity`

【算子效果对比】：

破坏算子策略：
- **random_removal**: 分散破坏，全局探索（基础策略）
- **route_removal**: 激进破坏，整条路径重构（大规模优化）
- **string_removal**: 温和破坏，局部微调（保持结构）

修复算子策略：
- **greedy_insert**: 贪心策略，快速插入（计算简单）
- **regret_insert**: 后悔策略，智能插入（效果最好）

【注意事项】：
1. 不要实现 cost()、validate()、check_feasible() 方法（框架已提供）
2. 算子内部可以直接修改solution（框架已做深拷贝）
3. 确保所有边界条件都有检查
4. 代码必须完整可运行，不能有TODO或pass

【关于成本计算函数的说明】：
必须在算子实现中使用 self.solver.calculator.calculate_route_cost() 进行完整成本计算。
这样能精确考虑所有成本因素（C12距离+C13制冷+C2货损+C3时间惩罚），
使得算子的决策基于真实的成本增量，而不是简化的距离估算。
"""
