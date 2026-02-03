HEURISTIC_PLUGIN_TEMPLATE = """
你是一个熟悉 ALNS (Adaptive Large Neighborhood Search) 算法的 Python 工程师，请帮我实现生鲜物流 VRP 问题中的启发式算子。

根据ALNS最佳实践，算子配置为：
- 破坏算子（Destroy）：6个 - random_removal、worst_removal、related_removal、shaw_removal、history_removal、cluster_removal
- 修复算子（Repair）：3个 - greedy_insert、regret_insert、random_insert

要求如下：

1. 数据结构：
   - solution 是列表，每条路径是一个列表，例如 [0, node1, node2, 0]。
   - 0 表示仓库，node 表示客户节点（ID > 0）。
   - self.dist_matrix[i][j] 表示节点 i 到节点 j 的距离（二维列表，已初始化）。
   - 移除节点后，如果路径长度 <= 2（即只剩 [0,0] 或 [0]），则删除该路径。
   - 插入节点时必须考虑新建路径的情况：新路径为 [0, node, 0]。

2. 算子要求：

【random_removal】：
   - 随机移除 solution 中 ratio 比例的客户节点（非 0 节点），至少移除一个节点。
   - **关键**：ratio 可能是比例（0-1）或具体数量（>1的整数），需要兼容处理：
       if ratio <= 1.0:
           n = max(1, math.ceil(total_customers * ratio))
       else:
           n = int(ratio)
   - **重要**：必须确保 n 不超过可用节点总数，使用 n = min(n, len(all_nodes))
   - 移除节点时按路径索引和位置索引降序排序（使用 key=lambda x: (x[0], x[1]), reverse=True），避免索引变化引发错误。
   - 移除完成后统一删除长度 <= 2 的路径：solution = [route for route in solution if len(route) > 2]
   - **边界情况处理**：
       * 如果 solution 中没有客户节点（all_nodes 为空），直接返回 solution, []
       * 使用 random.sample(all_nodes, n) 前必须确保 n <= len(all_nodes)
   - 返回修改后的 solution 和 removed_nodes 列表。

【worst_removal】：
   - 对每条路径中非0节点计算边际贡献：
       contrib = dist_matrix[prev][node] + dist_matrix[node][next] - dist_matrix[prev][next]
     其中 prev = route[pos_idx-1], next = route[pos_idx+1]
   - 按贡献从大到小选择 ratio 比例的节点移除，至少移除一个节点。
   - **关键**：同样需要处理 ratio 的两种情况（比例或数量）。
   - **重要**：计算贡献时只考虑路径长度 >= 3 的路径（跳过长度 < 3 的路径）。
   - **必须**：确保移除数量 n <= len(node_contributions)，使用 n = min(n, len(node_contributions))
   - 移除顺序按路径索引和位置索引降序排序（key=lambda x: (x[0], x[1]), reverse=True）。
   - 移除完成后统一删除长度 <= 2 的路径。
   - **边界情况处理**：
       * 如果没有可计算贡献的节点（node_contributions 为空），返回 solution, []
   - 返回修改后的 solution 和 removed_nodes 列表。

【related_removal】（关联破坏）：
   - **核心思想**：移除地理位置相近或属于同一区域的客户节点，便于重新规划整个区域
   - 随机选择一个种子节点 seed_node
   - 计算所有其他节点与 seed_node 的"关联度"（使用距离作为度量）
   - 按关联度从高到低（距离从小到大）选择 ratio 比例的节点移除
   - **关联度计算**：relatedness = dist_matrix[seed_node][other_node]
   - 距离越小，关联度越高，越应该一起移除
   - 同样需要处理 ratio 的两种情况（比例或数量）
   - 移除顺序按路径索引和位置索引降序排序
   - 返回修改后的 solution 和 removed_nodes 列表

【shaw_removal】（Shaw破坏）：
   - **核心思想**：综合考虑距离、时间窗、需求的相似性来选择移除节点
   - 随机选择一个种子节点 seed_node
   - 计算每个节点与 seed_node 的 Shaw 相似度：
       shaw_score = φ * normalized_distance + χ * time_diff + ψ * demand_diff
     其中：
       - normalized_distance = dist_matrix[seed][node] / max_distance
       - time_diff = |ready_time[seed] - ready_time[node]| / max_time_diff
       - demand_diff = |demand[seed] - demand[node]| / max_demand
       - φ=9, χ=3, ψ=2 为权重（距离最重要）
   - shaw_score 越小表示越相似，优先移除相似的节点
   - 按 shaw_score 从小到大选择 ratio 比例的节点移除
   - 返回修改后的 solution 和 removed_nodes 列表

【history_removal】（历史破坏）：
   - **核心思想**：移除长期未被修改过的节点，探索搜索盲区
   - 需要维护 self.node_history = {} 字典，记录每个节点最后被修改的迭代次数
   - 如果 self.node_history 未初始化或为空，则使用 random_removal 作为回退
   - 优先移除 history 值最小（最久未改动）的节点
   - 选择 ratio 比例的节点移除
   - 返回修改后的 solution 和 removed_nodes 列表

【cluster_removal】（聚类破坏）：
   - **核心思想**：基于位置聚类，移除属于同一聚类的客户群
   - 使用简单的网格聚类：将地图划分为 grid_size × grid_size 的网格
   - 每个客户根据坐标 (x, y) 分配到对应网格
   - 随机选择一个非空网格
   - 移除该网格中的所有客户（最多 ratio 比例）
   - 如果网格客户数不足，可以移除相邻网格的客户
   - 返回修改后的 solution 和 removed_nodes 列表

【greedy_insert】- 最关键的算子，直接影响解的质量
   
   **目标**：将节点插入成本最小的位置，优先利用现有路径，减少车辆数
   
   **算法步骤**（必须自己实现）：
   1. 对每个待插入节点：
      - 先搜索所有现有路径中的最佳插入位置（计算插入成本增量）
      - 记录最小成本和对应位置
      - 然后计算新建路径的成本（dist[0][node] + dist[node][0]）
      - 比较：如果现有路径成本 ≤ 新建成本，插入现有路径；否则新建路径
   
   2. **关键点**：
      - best_cost初始值必须是 float('inf')（不是0）
      - 必须先搜索完所有现有路径，再计算新建路径成本
      - 比较时用 <=（相等时优先现有路径，减少车辆数）
      - 插入时必须检查容量约束（路径总需求不超过vehicle_capacity）
   
   3. **容量约束检查**：
      ```python
      route_demand = sum(self.customer_lookup.get(n, {}).get('demand', 0) 
                        for n in route if n != 0)
      node_demand = self.customer_lookup.get(node, {}).get('demand', 0)
      if route_demand + node_demand <= self.capacity:
          # 可行，可以插入
      ```

【regret_insert】（后悔修复）- 更智能的插入策略：
   - **核心思想**：优先插入"后悔值"大的节点（最优与次优位置成本差距大）
   - 后悔值 regret = second_best_cost - best_cost
   - 后悔值大意味着：如果不现在插入到最佳位置，后面可能就没好位置了
   - **实现步骤**：
     1. 对每个待插入节点，计算插入到每个可能位置的成本增量
     2. 找出最佳位置（best_cost）和次佳位置（second_best_cost）
     3. 计算后悔值 regret = second_best_cost - best_cost
     4. 所有节点按后悔值从大到小排序
     5. 优先插入后悔值最大的节点到其最佳位置
     6. 重复直到所有节点插入完毕
   - **变种**：k-regret 插入，考虑第 k 好位置的差距
   - 返回插入所有节点后的完整解

【random_insert】（随机修复）- 增加解的多样性：
   - **核心思想**：随机选择可行位置插入，有时能发现意外的好方案
   - 对每个待插入节点：
     1. 收集所有可行的插入位置（包括现有路径和新建路径）
     2. 从可行位置中随机选择一个
     3. 执行插入
   - **注意**：虽然是随机的，但必须保证解的可行性
   - 如果没有可行位置，则新建路径 [0, node, 0]
   - 返回插入所有节点后的完整解

3. 注意事项：
   - 移除节点和插入节点时都要确保 solution 的合法性。
   - 避免在循环中直接删除路径导致索引错误（使用列表推导式过滤空路径）。
   - **框架已做深拷贝**，算子内部可以直接修改 solution，无需再次深拷贝。
   - **禁止实现** cost()、validate()、check_feasible() 方法。
   - 请生成完整可运行的 Python 函数，补充以下的TODO部分。

4. 必须导入的模块：
   import random
   import math  # 用于 math.ceil()

5. 代码质量要求：
   - 所有边界条件必须检查（空列表、越界、采样数量等）
   - 降序移除避免索引变化
   - 使用 min() 确保不超出范围
   - 添加必要的 if 判断处理边界情况

【示例：边界检查模板】
```python
# 示例1：防止 random.sample 报错
if not all_nodes:
    return solution, []
n = min(n, len(all_nodes))  # 关键：确保 n 不超过总数

# 示例2：删除空路径
solution = [route for route in solution if len(route) > 2]
```

import random
import copy
import math  # 必须导入


class HeuristicPlugin:
    def __init__(self, **kwargs):  
        # 从关键字参数中提取所需数据
        self.data = kwargs
        self.capacity = kwargs.get('vehicle_capacity')
        self.customers = kwargs.get('customers', [])
        self.dist_matrix = None  # 由 Solver 注入
        # ID → customer dict
        self.customer_lookup = {c['id']: c for c in self.customers}

    # ==============================
    # 禁止实现的函数（由 Solver 提供）
    # ==============================

    def cost(self, solution):
        raise RuntimeError("LLM must not define cost()")

    def validate(self, solution):
        raise RuntimeError("LLM must not define validate()")

    def check_feasible(self, route):
        raise RuntimeError("LLM must not define feasibility")

    # ===================================================
    # TODO: 请在下方实现具体的 Destroy 和 Insert 算子
    # ===================================================

    # -----------------------------
    # Destroy 算子要求
    # -----------------------------
    # random_removal:
    #   - 收集所有非0节点位置 (route_idx, pos_idx, node_id)
    #   - 计算移除数量 n（处理比例和数量两种情况）
    #   - **关键**：确保 n = min(n, len(all_nodes))，防止 random.sample 报错
    #   - 随机选择 n 个节点
    #   - 按 (route_idx, pos_idx) 降序排序后移除
    #   - 删除长度 <= 2 的空路径
    #   - 返回 new_solution, removed_nodes
    
    def random_removal(self, solution, ratio):
        '''
        从 solution 中随机移除 ratio 比例的客户节点。
        
        Args:
            solution: 当前解（路径列表）
            ratio: 移除比例（0-1）或数量（>1的整数）
        
        Returns:
            (移除后的解, 被移除的节点ID列表)
        '''
        # 收集所有非0节点
        all_nodes = []
        for route_idx, route in enumerate(solution):
            for pos_idx, node in enumerate(route):
                if node != 0:
                    all_nodes.append((route_idx, pos_idx, node))
        
        if not all_nodes:
            return solution, []
        
        # 计算移除数量
        total_customers = len(all_nodes)
        if ratio <= 1.0:
            n = max(1, math.ceil(total_customers * ratio))
        else:
            n = int(ratio)
        n = min(n, total_customers)
        
        # 随机选择节点
        selected = random.sample(all_nodes, n)
        selected.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        # 创建副本并移除节点
        new_solution = [route[:] for route in solution]
        removed_nodes = []
        for route_idx, pos_idx, node in selected:
            del new_solution[route_idx][pos_idx]
            removed_nodes.append(node)
        
        # 删除空路径
        new_solution = [route for route in new_solution if len(route) > 2]
        return new_solution, removed_nodes

    # worst_removal:
    #   - 遍历每条路径（跳过长度 < 3 的路径）
    #   - 计算每个非0节点的边际贡献 contrib
    #   - 按 contrib 降序排序，选择前 n 个
    #   - **关键**：确保 n = min(n, len(node_contributions))
    #   - 按 (route_idx, pos_idx) 降序排序后移除
    #   - 删除长度 <= 2 的空路径
    #   - 返回 new_solution, removed_nodes
    
    def worst_removal(self, solution, ratio):
        '''
        移除边际贡献最大的节点。
        
        【代码框架 - 必须按此结构实现】：
        '''
        # 步骤1: 收集所有非0节点及其边际贡献
        node_contributions = []
        for route_idx, route in enumerate(solution):
            if len(route) < 3:  # 跳过长度<3的路径
                continue
            for pos_idx in range(1, len(route)-1):
                node = route[pos_idx]
                prev_node = route[pos_idx-1]
                next_node = route[pos_idx+1]
                
                # 计算距离贡献
                dist_contrib = (self.dist_matrix[prev_node][node] + 
                               self.dist_matrix[node][next_node] - 
                               self.dist_matrix[prev_node][next_node])
                
                # 添加需求因素
                demand = self.customer_lookup.get(node, {}).get('demand', 0)
                contrib = dist_contrib + demand * 0.01
                
                node_contributions.append((route_idx, pos_idx, node, contrib))
        
        if not node_contributions:
            return solution, []
        
        # 步骤2: 按贡献降序排序
        node_contributions.sort(key=lambda x: x[3], reverse=True)
        
        # 步骤3: 计算移除数量
        total_nodes = len(node_contributions)
        if ratio <= 1.0:
            n = max(1, math.ceil(total_nodes * ratio))
        else:
            n = int(ratio)
        n = min(n, total_nodes)
        
        # 步骤4: 选择并移除
        selected = node_contributions[:n]
        selected.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        new_solution = [route[:] for route in solution]
        removed_nodes = []
        for route_idx, pos_idx, node, _ in selected:
            del new_solution[route_idx][pos_idx]
            removed_nodes.append(node)
        
        # 步骤5: 删除空路径
        new_solution = [route for route in new_solution if len(route) > 2]
        return new_solution, removed_nodes

    # related_removal:
    #   - 选择种子节点，移除与其地理位置相近的节点
    #   - 关联度基于距离计算
    
    def related_removal(self, solution, ratio):
        '''
        关联破坏：移除地理位置相近的客户节点。
        
        【代码框架 - 必须按此结构实现】：
        '''
        # 步骤1: 收集所有非0节点 - 必须实现
        all_nodes = []
        for route_idx, route in enumerate(solution):
            for pos_idx, node in enumerate(route):
                if node != 0:
                    all_nodes.append((route_idx, pos_idx, node))
        
        if not all_nodes:
            return solution, []
        
        # 步骤2: 随机选择种子节点 - 必须实现
        seed = random.choice(all_nodes)
        seed_node = seed[2]
        
        # 步骤3: 计算距离并排序 - 必须实现
        node_distances = []
        for route_idx, pos_idx, node in all_nodes:
            dist = self.dist_matrix[seed_node][node]
            node_distances.append((route_idx, pos_idx, node, dist))
        
        # 按距离升序排序（距离小的更相关）
        node_distances.sort(key=lambda x: x[3])
        
        # 步骤4: 计算移除数量 - 必须实现
        total_customers = len(all_nodes)
        if ratio <= 1.0:
            n = max(1, math.ceil(total_customers * ratio))
        else:
            n = int(ratio)
        n = min(n, total_customers)
        
        # 步骤5: 选择并移除 - 必须实现
        selected = node_distances[:n]
        selected.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        new_solution = [route[:] for route in solution]
        removed_nodes = []
        for route_idx, pos_idx, node, _ in selected:
            del new_solution[route_idx][pos_idx]
            removed_nodes.append(node)
        
        # 步骤6: 删除空路径
        new_solution = [route for route in new_solution if len(route) > 2]
        return new_solution, removed_nodes

    # shaw_removal:
    #   - 综合距离、时间窗、需求相似性选择移除节点
    
    def shaw_removal(self, solution, ratio):
        '''
        Shaw破坏：基于距离、时间窗、需求的综合相似性移除节点。
        
        【代码框架 - 必须按此结构实现】：
        '''
        # 步骤1: 收集节点并选择种子
        all_nodes = []
        for route_idx, route in enumerate(solution):
            for pos_idx, node in enumerate(route):
                if node != 0:
                    all_nodes.append((route_idx, pos_idx, node))
        
        if not all_nodes:
            return solution, []
        
        seed = random.choice(all_nodes)
        seed_node = seed[2]
        seed_customer = self.customer_lookup[seed_node]
        
        # 步骤2: 计算归一化因子（安全归一化）
        distances = [self.dist_matrix[seed_node][x[2]] for x in all_nodes]
        max_distance = max(distances) if distances else 1.0
        if max_distance < 0.001:
            max_distance = 1.0
        
        times = [self.customer_lookup[x[2]].get('ready_time', 0) for x in all_nodes]
        max_time_diff = max(times) - min(times) if times else 1.0
        if max_time_diff < 0.001:
            max_time_diff = 1.0
        
        demands = [self.customer_lookup[x[2]].get('demand', 0) for x in all_nodes]
        max_demand = max(demands) if demands else 1.0
        if max_demand < 0.001:
            max_demand = 1.0
        
        # 步骤3: 计算Shaw相似度
        shaw_scores = []
        for route_idx, pos_idx, node in all_nodes:
            customer = self.customer_lookup[node]
            normalized_dist = self.dist_matrix[seed_node][node] / max_distance
            time_diff = abs(seed_customer.get('ready_time', 0) - customer.get('ready_time', 0)) / max_time_diff
            demand_diff = abs(seed_customer.get('demand', 0) - customer.get('demand', 0)) / max_demand
            shaw_score = 9 * normalized_dist + 3 * time_diff + 2 * demand_diff
            shaw_scores.append((route_idx, pos_idx, node, shaw_score))
        
        # 步骤4: 按shaw_score升序排序
        shaw_scores.sort(key=lambda x: x[3])
        
        # 步骤5: 计算移除数量并选择
        total_customers = len(all_nodes)
        if ratio <= 1.0:
            n = max(1, math.ceil(total_customers * ratio))
        else:
            n = int(ratio)
        n = min(n, total_customers)
        
        selected = shaw_scores[:n]
        selected.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        # 步骤6: 移除并返回
        new_solution = [route[:] for route in solution]
        removed_nodes = []
        for route_idx, pos_idx, node, _ in selected:
            del new_solution[route_idx][pos_idx]
            removed_nodes.append(node)
        
        new_solution = [route for route in new_solution if len(route) > 2]
        return new_solution, removed_nodes

    # history_removal:
    #   - 移除长期未被修改的节点，探索搜索盲区
    
    def history_removal(self, solution, ratio):
        '''
        历史破坏：移除长期未被修改的节点。
        
        【代码框架 - 必须按此结构实现】：
        '''
        # 步骤1: 收集所有非0节点
        all_nodes = []
        for route_idx, route in enumerate(solution):
            for pos_idx, node in enumerate(route):
                if node != 0:
                    all_nodes.append((route_idx, pos_idx, node))
        
        if not all_nodes:
            return solution, []
        
        # 步骤2: 检查历史记录
        if not hasattr(self, 'node_history') or not self.node_history:
            return self.random_removal(solution, ratio)
        
        # 步骤3: 获取历史值
        history_values = []
        for route_idx, pos_idx, node in all_nodes:
            history = self.node_history.get(node, 0)
            history_values.append((route_idx, pos_idx, node, history))
        
        # 步骤4: 按历史值升序排序（值越小越久未改动）
        history_values.sort(key=lambda x: x[3])
        
        # 步骤5: 计算移除数量
        total_customers = len(all_nodes)
        if ratio <= 1.0:
            n = max(1, math.ceil(total_customers * ratio))
        else:
            n = int(ratio)
        n = min(n, total_customers)
        
        selected = history_values[:n]
        selected.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        # 步骤6: 移除并返回
        new_solution = [route[:] for route in solution]
        removed_nodes = []
        for route_idx, pos_idx, node, _ in selected:
            del new_solution[route_idx][pos_idx]
            removed_nodes.append(node)
        
        new_solution = [route for route in new_solution if len(route) > 2]
        return new_solution, removed_nodes

    # cluster_removal:
    #   - 基于网格聚类，移除同一区域的客户
    
    def cluster_removal(self, solution, ratio):
        '''
        聚类破坏：基于位置网格聚类，移除同一聚类的客户。
        
        【代码框架 - 必须按此结构实现】：
        '''
        # 步骤1: 收集节点及坐标
        nodes_with_coords = []
        for route_idx, route in enumerate(solution):
            for pos_idx, node in enumerate(route):
                if node != 0:
                    customer = self.customer_lookup[node]
                    x = customer.get('x', 0)
                    y = customer.get('y', 0)
                    nodes_with_coords.append((route_idx, pos_idx, node, x, y))
        
        if not nodes_with_coords:
            return solution, []
        
        # 步骤2: 计算坐标范围
        xs = [item[3] for item in nodes_with_coords]
        ys = [item[4] for item in nodes_with_coords]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        if max_x - min_x < 0.001:
            max_x = min_x + 1.0
        if max_y - min_y < 0.001:
            max_y = min_y + 1.0
        
        # 步骤3: 创建5x5网格
        grid = {}
        for route_idx, pos_idx, node, x, y in nodes_with_coords:
            grid_x = int(5 * (x - min_x) / (max_x - min_x))
            grid_y = int(5 * (y - min_y) / (max_y - min_y))
            grid_x = min(grid_x, 4)
            grid_y = min(grid_y, 4)
            key = (grid_x, grid_y)
            if key not in grid:
                grid[key] = []
            grid[key].append((route_idx, pos_idx, node))
        
        # 步骤4: 随机选择非空网格
        non_empty = [nodes for nodes in grid.values() if nodes]
        if not non_empty:
            return solution, []
        
        cluster_nodes = random.choice(non_empty)
        
        # 步骤5: 计算移除数量
        total_customers = len(nodes_with_coords)
        if ratio <= 1.0:
            n = max(1, math.ceil(total_customers * ratio))
        else:
            n = int(ratio)
        n = min(n, total_customers)
        
        # 步骤6: 如果节点不足，扩展到相邻网格
        if len(cluster_nodes) < n:
            for cell_key, cell_nodes in grid.items():
                if any(node in cluster_nodes for node in cell_nodes):
                    gx, gy = cell_key
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            neighbor_key = (gx + dx, gy + dy)
                            if neighbor_key in grid:
                                cluster_nodes.extend(grid[neighbor_key])
                    break
        
        # 步骤7: 选择并移除
        selected = cluster_nodes[:n]
        selected.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        new_solution = [route[:] for route in solution]
        removed_nodes = []
        for route_idx, pos_idx, node in selected:
            del new_solution[route_idx][pos_idx]
            removed_nodes.append(node)
        
        new_solution = [route for route in new_solution if len(route) > 2]
        return new_solution, removed_nodes

    # -----------------------------
    # Insert 算子
    # -----------------------------
    
    def greedy_insert(self, solution, removed_nodes):
        '''
        贪心插入节点到成本增量最小的位置。
        
        Args:
            solution: 当前解（路径列表）
            removed_nodes: 待插入的节点ID列表
        
        Returns:
            插入所有节点后的完整解
        
        【必须自己实现以下算法】：
        对每个节点：
        1. best_cost 初始化为 float('inf')  # 不是0！
        2. 搜索所有现有路径，找出成本增量最小的位置
           - **关键**：必须检查容量约束：
             current_load = sum(demand for n in route if n != 0)
             if current_load + node_demand > self.capacity: continue
        3. 搜索完成后，计算新建路径成本：
           new_route_cost = dist[0][node] + dist[node][0]
        4. 比较：如果 best_cost <= new_route_cost，插入现有路径；否则新建路径
        
        注意：
        - 必须先搜索完所有现有路径，再计算新建成本
        - 使用 <= 比较（相等时优先现有路径，减少车辆数）
        '''
        # TODO: 在此处实现算法
        pass

    # regret_insert:
    #   - 优先插入后悔值大的节点
    #   - 后悔值 = 次佳位置成本 - 最佳位置成本
    
    def regret_insert(self, solution, removed_nodes):
        '''
        后悔修复：优先插入后悔值大的节点。
        
        【代码框架 - 必须按此结构实现】：
        '''
        if not removed_nodes:
            return solution
        
        new_solution = [route[:] for route in solution]
        remaining = list(removed_nodes)
        
        # 循环直到所有节点插入完毕
        while remaining:
            best_regret = -float('inf')
            best_node = None
            best_insert_info = None
            
            # 对每个待插入节点计算后悔值
            for node in remaining:
                node_demand = self.customer_lookup.get(node, {}).get('demand', 0)
                costs = []  # (cost, route_idx, pos)
                
                # 搜索现有路径中的可行位置
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
                
                # 添加新建路径选项
                new_route_cost = self.dist_matrix[0][node] + self.dist_matrix[node][0]
                costs.append((new_route_cost, None, None))
                
                # 计算后悔值
                if len(costs) >= 2:
                    costs.sort(key=lambda x: x[0])
                    regret = costs[1][0] - costs[0][0]
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

    # random_insert:
    #   - 随机选择可行位置插入，增加解的多样性
    
    def random_insert(self, solution, removed_nodes):
        '''
        随机修复：随机选择可行位置插入节点。
        
        【代码框架 - 必须按此结构实现】：
        '''
        if not removed_nodes:
            return solution
        
        new_solution = [route[:] for route in solution]
        
        for node in removed_nodes:
            node_demand = self.customer_lookup.get(node, {}).get('demand', 0)
            
            # 收集所有可行位置
            positions = []  # (route_idx, pos)
            
            for route_idx, route in enumerate(new_solution):
                # 检查容量约束
                route_demand = sum(self.customer_lookup.get(n, {}).get('demand', 0) 
                                  for n in route if n != 0)
                if self.capacity and route_demand + node_demand > self.capacity:
                    continue
                
                # 收集该路径中的所有可行位置
                for pos in range(1, len(route)):
                    positions.append((route_idx, pos))
            
            # 加入新建路径选项
            positions.append((None, None))
            
            # 随机选择一个位置
            choice = random.choice(positions)
            
            if choice[0] is not None:
                # 插入到现有路径
                route_idx, pos = choice
                new_solution[route_idx].insert(pos, node)
            else:
                # 新建路径
                new_solution.append([0, node, 0])
        
        return new_solution
"""