HEURISTIC_PLUGIN_TEMPLATE = """
import random
import math

class HeuristicPlugin:
    def __init__(self, **kwargs):  
        self.capacity = kwargs.get('vehicle_capacity')
        self.customers = kwargs.get('customers', [])
        self.dist_matrix = None
        self.customer_lookup = {c['id']: c for c in self.customers}
        self.node_history = {}
    
    def random_removal(self, solution, ratio):
根据实际测试优化，保留有效算子并增加多样性：
- 破坏算子（Destroy）：3个 - random_removal、route_removal、string_removal
- 修复算子（Repair）：2个 - greedy_insert、regret_insert

注意：虽然只有random类算子有效，但通过不同的移除模式增加搜索多样性

【数据结构】：
   - solution: [[0, node1, node2, 0], [0, node3, 0], ...]（0是仓库）
   - self.dist_matrix[i][j]: 节点i到j的距离
   - self.capacity: 车辆容量
   - self.customer_lookup: {node_id: {'demand': ..., 'x': ..., 'y': ...}}

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
【2. route_removal】- 路径破坏算子（随机变种，增加多样性）
═════════════════════════════════════════════════════════════

**功能**：随机选择几条完整路径，移除其中所有客户

**与random_removal的区别**：
- random_removal: 随机移除分散的节点
- route_removal: 移除整条路径（更激进的破坏）

**实现步骤**：
1. 收集所有非空路径（长度>2的路径）
2. 计算要移除的路径数量（根据ratio和总客户数）
3. 随机选择若干条路径
4. 移除这些路径中的所有客户
5. 返回 (new_solution, removed_nodes)

**关键代码**：
```python
def route_removal(self, solution, ratio):
    # 收集所有非空路径及其客户
    routes_with_customers = []
    for route_idx, route in enumerate(solution):
        customers = [node for node in route if node != 0]
        if customers:
            routes_with_customers.append((route_idx, customers))
    
    if not routes_with_customers:
        return solution, []
    
    # 计算要移除的客户总数
    total_customers = sum(len(customers) for _, customers in routes_with_customers)
    if ratio <= 1.0:
        target_remove = max(1, math.ceil(total_customers * ratio))
    else:
        target_remove = int(ratio)
    
    # 随机选择路径直到移除足够客户
    random.shuffle(routes_with_customers)
    removed_nodes = []
    routes_to_remove = []
    
    for route_idx, customers in routes_with_customers:
        removed_nodes.extend(customers)
        routes_to_remove.append(route_idx)
        if len(removed_nodes) >= target_remove:
            break
    
    # 降序删除路径（避免索引变化）
    routes_to_remove.sort(reverse=True)
    new_solution = [route[:] for route in solution]
    for route_idx in routes_to_remove:
        del new_solution[route_idx]
    
    return new_solution, removed_nodes[:target_remove]
```

═════════════════════════════════════════════════════════════
【3. string_removal】- 字符串破坏算子（随机变种，移除连续节点）
═════════════════════════════════════════════════════════════

**功能**：在每条路径上随机移除一段连续的节点（像"字符串"一样）

**与random_removal的区别**：
- random_removal: 移除分散的独立节点
- string_removal: 移除连续的节点串（局部重构）

**实现步骤**：
1. 遍历所有路径
2. 在每条路径上随机选择起始位置
3. 从该位置开始连续移除若干节点
4. 直到移除总数达到ratio
5. 返回 (new_solution, removed_nodes)

**关键代码**：
```python
def string_removal(self, solution, ratio):
    # 收集所有客户节点
    all_nodes = []
    for route_idx, route in enumerate(solution):
        for pos_idx, node in enumerate(route):
            if node != 0:
                all_nodes.append(node)
    
    if not all_nodes:
        return solution, []
    
    # 计算移除数量
    total_customers = len(all_nodes)
    if ratio <= 1.0:
        n = max(1, math.ceil(total_customers * ratio))
    else:
        n = int(ratio)
    n = min(n, total_customers)
    
    removed_nodes = []
    new_solution = [route[:] for route in solution]
    
    # 随机选择路径并移除连续节点
    while len(removed_nodes) < n and any(len(r) > 2 for r in new_solution):
        # 找出所有非空路径
        valid_routes = [(i, r) for i, r in enumerate(new_solution) if len(r) > 2]
        if not valid_routes:
            break
        
        # 随机选择一条路径
        route_idx, route = random.choice(valid_routes)
        
        # 随机选择起始位置（排除仓库）
        start_pos = random.randint(1, len(route) - 2)
        
        # 移除连续的2-3个节点（或到路径末尾）
        string_length = min(random.randint(1, 3), len(route) - start_pos - 1, n - len(removed_nodes))
        
        for _ in range(string_length):
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
【4. greedy_insert】- 贪心修复算子（成功率0.5%，基础但必需）
═════════════════════════════════════════════════════════════

**功能**：将每个节点插入到成本增量最小的位置

**实现步骤**：
对每个待插入节点：
1. best_cost = float('inf')  # 必须是无穷大
2. 遍历所有现有路径，找成本增量最小的位置
   - 检查容量：route_demand + node_demand <= capacity
   - 计算成本增量：cost_inc = dist[prev][node] + dist[node][next] - dist[prev][next]
3. 计算新建路径成本：new_route_cost = dist[0][node] * 2
4. **决策**：if best_cost <= new_route_cost（用<=优先现有路径）
5. 插入到选定位置

**关键代码**：
```python
def greedy_insert(self, solution, removed_nodes):
    if not removed_nodes:
        return solution
    
    new_solution = [route[:] for route in solution]
    
    for node in removed_nodes:
        node_demand = self.customer_lookup.get(node, {}).get('demand', 0)
        best_cost = float('inf')  # 必须是无穷大！
        best_route_idx = None
        best_pos = None
        
        # 搜索现有路径
        for route_idx, route in enumerate(new_solution):
            route_demand = sum(self.customer_lookup.get(n, {}).get('demand', 0) 
                              for n in route if n != 0)
            if self.capacity and route_demand + node_demand > self.capacity:
                continue
            
            for pos in range(1, len(route)):
                prev = route[pos-1]
                next_node = route[pos]
                
                cost_inc = (self.dist_matrix[prev][node] + 
                           self.dist_matrix[node][next_node] - 
                           self.dist_matrix[prev][next_node])
                
                if cost_inc < best_cost:
                    best_cost = cost_inc
                    best_route_idx = route_idx
                    best_pos = pos
        
        # 计算新建路径成本
        new_route_cost = self.dist_matrix[0][node] + self.dist_matrix[node][0]
        
        # 决策：用 <= 优先现有路径
        if best_cost <= new_route_cost and best_route_idx is not None:
            new_solution[best_route_idx].insert(best_pos, node)
        else:
            new_solution.append([0, node, 0])
    
    return new_solution
```

═════════════════════════════════════════════════════════════
【5. regret_insert】- 后悔修复算子（成功率1%，最有效的修复算子）
═════════════════════════════════════════════════════════════

**功能**：优先插入"后悔值"大的节点（最优与次优位置差距大的）

**后悔值定义**：regret = second_best_cost - best_cost
- 后悔值大 → 如果不现在插入到最佳位置，后面就没好位置了
- 后悔值小 → 位置选择灵活，可以等等

**实现步骤**：
1. 对每个待插入节点，找出所有可行位置的成本
2. 排序后取最佳和次佳，计算后悔值
3. 选择后悔值最大的节点，插入其最佳位置
4. 重复直到所有节点插入完毕

**关键代码**：
```python
def regret_insert(self, solution, removed_nodes):
    if not removed_nodes:
        return solution
    
    new_solution = [route[:] for route in solution]
    remaining = list(removed_nodes)
    
    while remaining:
        best_regret = -float('inf')
        best_node = None
        best_insert_info = None
        
        # 对每个节点计算后悔值
        for node in remaining:
            node_demand = self.customer_lookup.get(node, {}).get('demand', 0)
            costs = []
            
            # 收集所有可行位置的成本
            for route_idx, route in enumerate(new_solution):
                route_demand = sum(self.customer_lookup.get(n, {}).get('demand', 0) 
                                  for n in route if n != 0)
                if self.capacity and route_demand + node_demand > self.capacity:
                    continue
                
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    next_node = route[pos]
                    cost_inc = (self.dist_matrix[prev][node] + 
                               self.dist_matrix[node][next_node] - 
                               self.dist_matrix[prev][next_node])
                    costs.append((cost_inc, route_idx, pos))
            
            # 新建路径也是一个选项
            new_route_cost = self.dist_matrix[0][node] + self.dist_matrix[node][0]
            costs.append((new_route_cost, None, None))
            
            # 计算后悔值
            if len(costs) >= 2:
                costs.sort(key=lambda x: x[0])
                regret = costs[1][0] - costs[0][0]  # 次佳 - 最佳
            else:
                regret = 0
            
            # 更新最大后悔值
            if regret > best_regret:
                best_regret = regret
                best_node = node
                best_insert_info = costs[0] if costs else None
        
        # 插入后悔值最大的节点
        if best_insert_info:
            cost, route_idx, pos = best_insert_info
            if route_idx is not None:
                new_solution[route_idx].insert(pos, best_node)
            else:
                new_solution.append([0, best_node, 0])
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
    
    # 实现上面5个函数：
    def random_removal(self, solution, ratio):
        # 复制上面的代码
        pass
    
    def route_removal(self, solution, ratio):
        # 复制上面的代码
        pass
    
    def string_removal(self, solution, ratio):
        # 复制上面的代码
        pass
    
    def greedy_insert(self, solution, removed_nodes):
        # 复制上面的代码
        pass
    
    def regret_insert(self, solution, removed_nodes):
        # 复制上面的代码
        pass
```

【最后检查】：
✅ 所有removal算子是否有 `n = min(n, total_customers)` 防止越界
✅ random_removal 是否降序排序 `reverse=True`
✅ route_removal 是否降序删除路径 `routes_to_remove.sort(reverse=True)`
✅ string_removal 是否删除空路径 `if len(route) > 2`
✅ greedy_insert 的 best_cost 是否初始化为 `float('inf')`
✅ greedy_insert 是否用 `<=` 比较（不是 `<`）
✅ regret_insert 是否处理空列表情况 `if not removed_nodes`
✅ 所有函数是否有容量检查 `route_demand + node_demand <= capacity`

【注意事项】：
1. 不要实现 cost()、validate()、check_feasible() 方法（框架已提供）
2. 算子内部可以直接修改solution（框架已做深拷贝）
3. 确保所有边界条件都有检查
4. 代码必须完整可运行，不能有TODO或pass
"""
