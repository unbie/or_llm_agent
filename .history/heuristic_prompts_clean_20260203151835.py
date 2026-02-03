HEURISTIC_PLUGIN_TEMPLATE = """
你是一个熟悉ALNS算法的Python工程师，请帮我实现VRP问题的启发式算子。

【算子配置】
- 破坏算子：6个 - random_removal、worst_removal、related_removal、shaw_removal、history_removal、cluster_removal  
- 修复算子：3个 - greedy_insert、regret_insert、random_insert

【核心要求】
1. 所有函数必须完整实现，不能有pass或TODO
2. 处理所有边界条件（空列表、越界等）
3. 移除节点时必须降序排序避免索引错误
4. 必须检查容量约束

【数据结构】
- solution: [[0, node1, node2, 0], ...] （0是仓库）
- self.dist_matrix[i][j]: 距离矩阵
- self.capacity: 车辆容量
- self.customer_lookup: {id: {'demand': ..., 'x': ..., 'y': ..., 'ready_time': ...}}
- self.node_history: {node_id: last_iteration} (用于history_removal)

【必须导入】
import random
import math

【算子实现要求】

1. random_removal(self, solution, ratio):
   - 随机移除ratio比例的客户节点
   - ratio可能是0-1的比例或>1的数量，需要兼容处理
   - 必须降序移除避免索引变化
   - 删除空路径 (len <= 2)
   - 返回 (new_solution, removed_nodes)

2. worst_removal(self, solution, ratio):
   - 计算每个节点的边际贡献：contrib = dist[prev][node] + dist[node][next] - dist[prev][next]
   - 按贡献降序移除ratio个节点
   - 只处理长度>=3的路径
   - 降序移除，删除空路径
   - 返回 (new_solution, removed_nodes)

3. related_removal(self, solution, ratio):
   - 随机选择seed节点
   - 计算其他节点到seed的距离
   - 按距离升序移除最近的ratio个节点
   - 降序移除，删除空路径
   - 返回 (new_solution, removed_nodes)

4. shaw_removal(self, solution, ratio):
   - 随机选择seed节点
   - 计算Shaw相似度：shaw_score = 9*normalized_dist + 3*time_diff + 2*demand_diff
   - 归一化时防止除零：if max_value < 0.001: max_value = 1.0
   - 按shaw_score升序移除相似的ratio个节点
   - 降序移除，删除空路径
   - 返回 (new_solution, removed_nodes)

5. history_removal(self, solution, ratio):
   - 如果self.node_history为空，回退到random_removal
   - 按history值升序选择最久未改动的ratio个节点
   - 降序移除，删除空路径
   - 返回 (new_solution, removed_nodes)

6. cluster_removal(self, solution, ratio):
   - 创建5x5网格聚类
   - 随机选择非空网格
   - 移除该网格中的ratio个节点
   - 如果不足则扩展到相邻网格
   - 降序移除，删除空路径
   - 返回 (new_solution, removed_nodes)

7. greedy_insert(self, solution, removed_nodes):
   - 对每个节点：
     * best_cost初始化为float('inf')
     * 搜索所有现有路径的最佳插入位置
     * 检查容量约束
     * 计算新建路径成本
     * **关键**：用 <= 比较（相等时优先现有路径）
   - 返回完整solution

8. regret_insert(self, solution, removed_nodes):
   - 计算每个节点的最佳和次佳位置
   - regret = second_best - best
   - 优先插入regret值大的节点
   - 检查容量约束
   - 返回完整solution

9. random_insert(self, solution, removed_nodes):
   - 收集所有可行位置（检查容量）
   - 随机选择一个位置插入
   - 如果无可行位置则新建路径
   - 返回完整solution

【关键边界检查】
```python
# 防止空列表
if not all_nodes:
    return solution, []

# 防止越界
n = min(n, len(all_nodes))

# ratio处理
if ratio <= 1.0:
    n = max(1, math.ceil(total * ratio))
else:
    n = int(ratio)

# 降序移除
selected.sort(key=lambda x: (x[0], x[1]), reverse=True)

# 删除空路径
new_solution = [route for route in new_solution if len(route) > 2]
```

请生成完整的HeuristicPlugin类代码，包含__init__和所有9个算子函数。
"""
