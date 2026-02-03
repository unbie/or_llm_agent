HEURISTIC_PLUGIN_TEMPLATE = """
你是一个熟悉 ALNS (Adaptive Large Neighborhood Search) 算法的 Python 工程师，请帮我实现生鲜物流 VRP 问题中的启发式算子。

【ALNS核心原则 - 必读】：
1. **破坏算子目标**：智能地移除节点，为后续优化创造空间
   - 不是随机破坏，而是移除"有问题"的节点（成本高、位置不佳、难服务）
   - 移除比例要适中（通常10-30%），太少没效果，太多难恢复
   
2. **修复算子目标**：以更优方式重新插入节点
   - 贪心插入：找最便宜的位置（快速但可能局部最优）
   - 后悔插入：优先插入"选择少"的节点（平衡全局）
   - 随机插入：增加多样性（探索新可能）

3. **成功的关键**：破坏后的修复要比原来更好
   - 示例：移除一个远离路径的节点，修复时插入到更近的路径中
   - 移除需求大的节点，修复时可能减少车辆数
   
4. **代码质量要求**：
   - 所有代码必须完整实现（不能有pass、TODO或空函数）
   - 边界条件必须处理（空列表、除零等）
   - 降序移除避免索引错误
   - 容量约束必须检查

【生成代码后必须自检】：
✅ worst_removal: 搜索你的代码中 "* 0." 确认是 "* 0.5" 不是 "* 0.01"
✅ greedy_insert: 搜索 "best_cost" 确认是 "<= new_route_cost" 不是 "< new_route_cost"
✅ greedy_insert: 搜索 "best_cost =" 确认初始化为 "float('inf')"
✅ 所有removal: 确认有 ".sort(key=lambda x: (x[0], x[1]), reverse=True)"
✅ 所有算子: 确认开头有 "if not" 边界检查
✅ 所有算子: 确认有 "n = min(n," 防止越界

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

【worst_removal】- 成功率提升的关键算子：
   - **━━━ 核心公式（逐字复制，一个字都不能改）━━━**：
       ```python
       contrib = dist_contrib + demand * 0.5
       ```
       ⚠️ 警告：如果你写成 0.01 或其他值，算子将失效！
   
   - **完整实现公式**：
       dist_contrib = dist_matrix[prev][node] + dist_matrix[node][next] - dist_matrix[prev][next]
       demand = customer_lookup[node]['demand']
       contrib = dist_contrib + demand * 0.5  # 复制这一行！不要改0.5！
   
   - **为什么必须用0.5**：测试证明0.01效果差（所有运行成功率0%），0.5是最优值
   - 按贡献从大到小排序，选择贡献最大的 ratio 比例节点移除
   - **关键**：处理 ratio 的两种情况（比例0-1或数量>1）
   - **重要**：只处理路径长度 >= 3 的路径
   - **必须**：确保移除数量 n <= len(node_contributions)
   - 移除顺序按路径索引和位置索引降序排序
   - 移除完成后统一删除长度 <= 2 的路径
   - **边界情况**：如果没有可计算贡献的节点，返回 solution, []
   - 返回修改后的 solution 和 removed_nodes 列表

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

【greedy_insert】- 最关键的算子，直接影响解的质量和成功率
   
   **目标**：将节点插入成本最小的位置，优先利用现有路径，减少车辆数
   
   **算法步骤（必须严格按此顺序实现）**：
   1. 对每个待插入节点：
      - **步骤A**：先遍历所有现有路径，找出成本增量最小的插入位置
        * 初始化：best_cost = float('inf')  # 必须是无穷大，不是0！
        * 检查容量：如果 route_demand + node_demand > capacity，跳过该路径
        * **优化策略**：优先考虑容量利用率高的路径（减少碎片化）
        * 计算成本增量：cost_inc = dist[prev][node] + dist[node][next] - dist[prev][next]
        * 更新最佳：if cost_inc < best_cost，记录该位置
      
      - **步骤B**：搜索完所有现有路径后，计算新建路径成本
        * new_route_cost = dist[0][node] + dist[0][node]
        * 注意：必须在步骤A完成后才计算！
      
      - **步骤C**：决策（关键逻辑）
        * if best_cost <= new_route_cost and best_route_idx is not None:
            插入现有路径  # 用 <=，相等时优先现有路径！
        * else:
            新建路径 [0, node, 0]
   
   2. **为什么用 <= 而不是 <**：
      - 当成本相等时，插入现有路径可以减少车辆数
      - 减少车辆数通常能降低总成本（减少往返仓库的距离）
      - 这是贪心算子成功的关键！
   
   3. **容量约束检查（必须严格执行）**：
      ```python
      route_demand = sum(self.customer_lookup.get(n, {}).get('demand', 0) 
                        for n in route if n != 0)
      node_demand = self.customer_lookup.get(node, {}).get('demand', 0)
      if self.capacity and route_demand + node_demand > self.capacity:
          continue  # 跳过该路径
      ```
   
   4. **常见错误（必须避免）**：
      - ❌ best_cost初始化为0或new_route_cost
      - ❌ 先计算new_route_cost再搜索现有路径
      - ❌ 用 < 而不是 <=
      - ❌ 忘记检查容量约束
      - ❌ best_route_idx为None时仍然插入现有路径

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
        self.node_history = {}  # 用于 history_removal 算子

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
                
                # ━━━ 复制下面这两行，0.5不能改！━━━
                demand = self.customer_lookup.get(node, {}).get('demand', 0)
                contrib = dist_contrib + demand * 0.5  # ⚠️ 0.5不能改！
                
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
        
        【代码框架 - 必须按此结构实现】：
        对每个节点：
        1. best_cost 初始化为 float('inf')
        2. 搜索所有现有路径，找出成本增量最小的位置（检查容量约束）
        3. 计算新建路径成本
        4. 比较并插入（优先现有路径）
        '''
        if not removed_nodes:
            return solution
        
        new_solution = [route[:] for route in solution]
        
        for node in removed_nodes:
            node_demand = self.customer_lookup.get(node, {}).get('demand', 0)
            best_cost = float('inf')
            best_route_idx = None
            best_pos = None
            
            # 搜索现有路径中的最佳位置
            for route_idx, route in enumerate(new_solution):
                # 检查容量约束
                route_demand = sum(self.customer_lookup.get(n, {}).get('demand', 0) 
                                  for n in route if n != 0)
                if self.capacity and route_demand + node_demand > self.capacity:
                    continue
                
                # 搜索路径中的最佳插入位置
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
            
            # 【关键决策】用 <= 不是 <，相等时优先现有路径！这能减少车辆数
            if best_cost <= new_route_cost and best_route_idx is not None:
                new_solution[best_route_idx].insert(best_pos, node)  # 插入现有路径
            else:
                new_solution.append([0, node, 0])  # 新建路径
        
        return new_solution

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

【代码完成后务必检查以下关键点】：
1. ⚠️ 最重要！在你的 worst_removal 代码中搜索 "demand *"，确认后面是 "0.5" 不是 "0.01"
2. greedy_insert 的比较是否使用 `<=`（不是 `<`）
3. greedy_insert 的 best_cost 是否初始化为 `float('inf')`
4. 所有 removal 算子删除节点时是否降序排序
5. 所有算子是否有边界检查（空列表、n不超过总数等）

【如果测试显示算子成功率0%，说明你没有使用0.5！回去修改！】
"""