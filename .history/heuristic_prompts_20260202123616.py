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
   
   **强制执行的代码结构**（必须严格按此顺序）：
   
   ```
   for node in removed_nodes:
       # ===== 检查点1: 必须先初始化为无穷大 =====
       best_cost = float('inf')   # 不是0，不是new_route_cost，必须是inf
       best_route_idx = None
       best_position = None
       
       # ===== 检查点2: 先搜索所有现有路径 =====
       for route_idx, route in enumerate(solution):
           for pos in range(1, len(route)):
               prev = route[pos-1]
               next_node = route[pos]
               cost_increase = dist[prev][node] + dist[node][next_node] - dist[prev][next_node]
               if cost_increase < best_cost:  # 用 < 更新最佳
                   best_cost = cost_increase
                   best_route_idx = route_idx
                   best_position = pos
       
       # ===== 检查点3: 搜索完成后才计算新建成本 =====
       new_route_cost = dist[0][node] + dist[node][0]
       
       # ===== 检查点4: 用 <= 比较，优先现有路径 =====
       if best_route_idx is not None and best_cost <= new_route_cost:
           solution[best_route_idx].insert(best_position, node)
       else:
           solution.append([0, node, 0])
   ```
   
   **自检问题**（生成代码前回答）：
   Q1: best_cost 初始值是什么？ → 必须是 float('inf')
   Q2: 何时计算 new_route_cost？ → 搜索完所有现有路径之后
   Q3: 比较时用 < 还是 <=？ → 必须用 <=（相等时优先现有路径）

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
        # TODO: 实现随机移除逻辑
        # 提示：
        # 1. 收集所有非0节点：all_nodes = [(route_idx, pos_idx, node), ...]
        # 2. 计算 n，确保 n = min(n, len(all_nodes))
        # 3. random.sample(all_nodes, n)
        # 4. 降序排序：selected.sort(key=lambda x: (x[0], x[1]), reverse=True)
        # 5. 移除节点并记录到 removed_nodes
        # 6. 过滤空路径：solution = [route for route in solution if len(route) > 2]
        pass

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
        
        Args:
            solution: 当前解（路径列表）
            ratio: 移除比例（0-1）或数量（>1的整数）
        
        Returns:
            (移除后的解, 被移除的节点ID列表)
        '''
        # TODO: 实现最差移除逻辑
        # 提示：
        # 1. 遍历路径，跳过 len(route) < 3 的路径
        # 2. 计算贡献：contrib = dist_matrix[prev][node] + dist_matrix[node][next] - dist_matrix[prev][next]
        # 3. 存储 (route_idx, pos_idx, node, contrib)
        # 4. 按 contrib 降序排序
        # 5. 选择前 n 个（确保 n = min(n, len(node_contributions))）
        # 6. 按 (route_idx, pos_idx) 降序排序后移除
        # 7. 过滤空路径
        pass

    # related_removal:
    #   - 选择种子节点，移除与其地理位置相近的节点
    #   - 关联度基于距离计算
    
    def related_removal(self, solution, ratio):
        '''
        关联破坏：移除地理位置相近的客户节点。
        
        Args:
            solution: 当前解（路径列表）
            ratio: 移除比例（0-1）或数量（>1的整数）
        
        Returns:
            (移除后的解, 被移除的节点ID列表)
        '''
        # TODO: 实现关联移除逻辑
        # 1. 收集所有非0节点
        # 2. 随机选择种子节点 seed_node
        # 3. 计算其他节点与种子节点的距离
        # 4. 按距离升序排序（距离越近关联度越高）
        # 5. 选择前 n 个最近的节点移除（包含种子节点）
        # 6. 按 (route_idx, pos_idx) 降序排序后移除
        # 7. 过滤空路径
        pass

    # shaw_removal:
    #   - 综合距离、时间窗、需求相似性选择移除节点
    
    def shaw_removal(self, solution, ratio):
        '''
        Shaw破坏：基于距离、时间窗、需求的综合相似性移除节点。
        
        Args:
            solution: 当前解（路径列表）
            ratio: 移除比例（0-1）或数量（>1的整数）
        
        Returns:
            (移除后的解, 被移除的节点ID列表)
        '''
        # TODO: 实现Shaw移除逻辑
        # 1. 收集所有非0节点
        # 2. 随机选择种子节点
        # 3. 对每个节点计算 Shaw 相似度分数：
        #    shaw_score = 9*normalized_dist + 3*time_diff + 2*demand_diff
        # 4. 按 shaw_score 升序排序（分数越低越相似）
        # 5. 选择前 n 个移除
        # 6. 降序移除，过滤空路径
        pass

    # history_removal:
    #   - 移除长期未被修改的节点，探索搜索盲区
    
    def history_removal(self, solution, ratio):
        '''
        历史破坏：移除长期未被修改的节点。
        
        Args:
            solution: 当前解（路径列表）
            ratio: 移除比例（0-1）或数量（>1的整数）
        
        Returns:
            (移除后的解, 被移除的节点ID列表)
        '''
        # TODO: 实现历史移除逻辑
        # 1. 检查 self.node_history 是否存在，不存在则初始化为空字典
        # 2. 收集所有非0节点
        # 3. 如果 node_history 为空，回退到 random_removal
        # 4. 按 history 值升序排序（值越小表示越久未改动）
        # 5. 选择前 n 个移除
        # 6. 降序移除，过滤空路径
        pass

    # cluster_removal:
    #   - 基于网格聚类，移除同一区域的客户
    
    def cluster_removal(self, solution, ratio):
        '''
        聚类破坏：基于位置网格聚类，移除同一聚类的客户。
        
        Args:
            solution: 当前解（路径列表）
            ratio: 移除比例（0-1）或数量（>1的整数）
        
        Returns:
            (移除后的解, 被移除的节点ID列表)
        '''
        # TODO: 实现聚类移除逻辑
        # 1. 收集所有非0节点及其坐标
        # 2. 计算网格大小（可用 grid_size = 5）
        # 3. 将节点分配到网格 grid[(gx, gy)] = [nodes...]
        # 4. 随机选择一个非空网格
        # 5. 移除该网格中的节点（最多 n 个）
        # 6. 如果不足，考虑相邻网格
        # 7. 降序移除，过滤空路径
        pass

    # -----------------------------
    # Insert 算子
    # -----------------------------
    
    def greedy_insert(self, solution, removed_nodes):
        '''
        贪心插入节点到成本增量最小的位置。
        优先插入现有路径，新建路径仅作为兜底。
        
        Args:
            solution: 部分解（已移除节点的路径列表）
            removed_nodes: 需要插入的节点ID列表
        
        Returns:
            插入所有节点后的完整解
        '''
        # TODO: 按以下顺序实现
        #
        # 边界处理:
        #   - removed_nodes 为空 → return solution
        #   - solution 为空 → 为每个节点创建 [0, node, 0]
        #
        # 对每个 node 循环:
        #   步骤1: best_cost = float('inf')  # 必须初始化为无穷大！
        #   步骤2: 遍历所有现有路径的插入位置，找成本增量最小的
        #   步骤3: new_route_cost = dist[0][node] + dist[node][0]
        #   步骤4: if best_cost <= new_route_cost: 插入现有路径
        #          else: 新建路径
        #
        # ⚠️ 错误示例（不要这样做）：
        #   new_route_cost = dist[0][node] + dist[node][0]
        #   best_cost = new_route_cost  # 错！会导致永远新建路径
        #
        # ✓ 正确顺序: 初始化∞ → 搜索现有 → 计算新建 → 比较(<=)决策
        # 4. return solution
        pass
"""