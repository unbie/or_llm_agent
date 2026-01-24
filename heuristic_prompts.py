HEURISTIC_PLUGIN_TEMPLATE = """
你是一个熟悉 ALNS (Adaptive Large Neighborhood Search) 算法的 Python 工程师，请帮我实现生鲜物流 VRP 问题中的三个启发式算子：random_removal、worst_removal 和 greedy_insert。要求如下：

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

【greedy_insert】：
   - 对每个待插入节点，优先插入现有路径，新建路径仅作为兜底
   
   - **算法思路**：
     1. 在所有现有路径中搜索最佳插入位置（成本增量最小）
     2. 将最佳插入成本与新建路径成本比较
     3. 如果现有路径更优或相同，插入现有路径；否则新建路径
   
   - **插入位置**：range(1, len(route))
     增量 = dist[prev][node] + dist[node][next] - dist[prev][next]
   
   - **新建路径成本**：dist[0][node] + dist[node][0]（作为兜底比较）
   
   - **选择规则**：best_cost <= new_route_cost 时插入现有路径
     （用 <= 确保成本相同时优先现有路径，减少车辆数）
   
   - **边界情况**：
     * removed_nodes 为空：返回 solution
     * solution 为空：为每个节点新建 [0, node, 0]

【greedy_insert 参考实现 - 必须严格按此逻辑】：
```python
def greedy_insert(self, solution, removed_nodes):
    if not removed_nodes:
        return solution
    if not solution:
        return [[0, node, 0] for node in removed_nodes]
    
    for node in removed_nodes:
        best_cost = float('inf')  # 步骤a: 初始化为无穷大
        best_route_idx = None
        best_position = None
        
        # 步骤b: 遍历所有现有路径的插入位置
        for route_idx, route in enumerate(solution):
            for pos in range(1, len(route)):
                prev, next_node = route[pos-1], route[pos]
                cost_increase = (self.dist_matrix[prev][node] + 
                                self.dist_matrix[node][next_node] - 
                                self.dist_matrix[prev][next_node])
                if cost_increase < best_cost:
                    best_cost = cost_increase
                    best_route_idx = route_idx
                    best_position = pos
        
        # 步骤c: 计算新建路径成本
        new_route_cost = self.dist_matrix[0][node] + self.dist_matrix[node][0]
        
        # 步骤d: 比较并选择（<= 优先现有路径）
        if best_route_idx is not None and best_cost <= new_route_cost:
            solution[best_route_idx].insert(best_position, node)
        else:
            solution.append([0, node, 0])
    
    return solution
```

3. 注意事项：
   - 移除节点和插入节点时都要确保 solution 的合法性。
   - 避免在循环中直接删除路径导致索引错误（使用列表推导式过滤空路径）。
   - **框架已做深拷贝**，算子内部可以直接修改 solution，无需再次深拷贝。
   - **禁止实现** cost()、validate()、check_feasible() 方法。
   - 请生成完整可运行的 Python 函数，补充以下的TODO部分。4. 必须导入的模块：
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

    # -----------------------------
    # Insert 算子
    # -----------------------------
    # greedy_insert 核心逻辑：
    #   1. 边界检查
    #   2. 对每个节点，遍历所有候选位置，记录最小成本及位置
    #   3. 候选位置包括：现有路径插入位置 + 新建路径
    #   4. 选择成本最小的位置执行插入
    #   5. 成本相同时优先现有路径（用 <= 比较）
    
    def greedy_insert(self, solution, removed_nodes):
        '''
        贪心插入节点到成本增量最小的位置。
        
        Args:
            solution: 部分解（已移除节点的路径列表）
            removed_nodes: 需要插入的节点ID列表
        
        Returns:
            插入所有节点后的完整解
        '''
        # TODO: 实现贪心插入逻辑
        # 
        # 实现要点：
        # 1. if not removed_nodes: return solution
        # 2. if not solution: 为每个节点创建 [0, node, 0]
        # 3. 对每个 node:
        #    a. best_cost = float('inf'), best_route_idx = None, best_position = None
        #    b. 遍历所有路径的所有插入位置 range(1, len(route))
        #       计算 cost_increase，如果 < best_cost 则更新
        #    c. 计算 new_route_cost = dist[0][node] + dist[node][0]
        #    d. 如果 best_route_idx is not None 且 best_cost <= new_route_cost:
        #          插入现有路径
        #       否则:
        #          新建路径 [0, node, 0]
        # 4. return solution
        pass
"""