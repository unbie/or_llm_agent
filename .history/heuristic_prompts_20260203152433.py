HEURISTIC_PLUGIN_TEMPLATE = """
你是一个熟悉 ALNS 算法的 Python 工程师，请实现生鲜物流 VRP 问题的启发式算子。

【核心要求】：
- 所有代码必须完整实现（不能有pass、TODO或空函数）
- 处理边界条件（空列表、防止越界）
- 降序移除避免索引错误
- 检查容量约束

【优化算子配置】：
ALNS标准配置（框架要求）：
- 破坏算子（Destroy）：6个 - random_removal、worst_removal、related_removal、shaw_removal、history_removal、cluster_removal
- 修复算子（Repair）：3个 - greedy_insert、regret_insert、random_insert

注意：根据测试，random类算子最有效，但需要实现所有算子以保证框架正常运行

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
【其他破坏算子简要说明】
═════════════════════════════════════════════════════════════

**2. worst_removal** - 移除成本贡献最大的节点
**3. related_removal** - 移除地理/时间相邻的相关节点
**4. shaw_removal** - 基于相似性（距离+时间+需求）移除节点簇
**5. history_removal** - 移除历史上较少出现在好解中的节点
**6. cluster_removal** - 基于聚类移除整个区域的节点

═════════════════════════════════════════════════════════════
【修复算子简要说明】
═════════════════════════════════════════════════════════════

**7. greedy_insert** - 贪心插入（成本增量最小）
**8. regret_insert** - 后悔插入（优先插入后悔值大的节点）
**9. random_insert** - 随机插入（增加多样性）
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
    
    # 实现下面9个函数（6个破坏 + 3个修复）：
    def random_removal(self, solution, ratio):
        # 实现随机移除
        pass
    
    def worst_removal(self, solution, ratio):
        # 实现最差节点移除
        pass
    
    def related_removal(self, solution, ratio):
        # 实现关联移除
        pass
    
    def shaw_removal(self, solution, ratio):
        # 实现Shaw移除
        pass
    
    def history_removal(self, solution, ratio):
        # 实现历史移除
        pass
    
    def cluster_removal(self, solution, ratio):
        # 实现聚类移除
        pass
    
    def greedy_insert(self, solution, removed_nodes):
        # 实现贪心插入
        pass
    
    def regret_insert(self, solution, removed_nodes):
        # 实现后悔插入
        pass
    
    def random_insert(self, solution, removed_nodes):
        # 实现随机插入
        pass
```

【最后检查】：
✅ 所有removal算子是否有 `n = min(n, total_customers)` 防止越界
✅ random_removal 是否降序排序 `reverse=True`
✅ worst_removal 是否计算边际成本贡献
✅ related/shaw/history/cluster 是否处理空节点情况
✅ greedy_insert 的 best_cost 是否初始化为 `float('inf')`
✅ greedy_insert 是否用 `<=` 比较（不是 `<`）
✅ regret_insert 是否处理空列表情况 `if not removed_nodes`
✅ random_insert 是否收集所有可行位置
✅ 所有函数是否有容量检查 `route_demand + node_demand <= capacity`

【其他必需算子简要说明】：

2. **worst_removal**: 移除边际成本贡献最大的节点
   - 计算每个节点的detour距离: dist[prev][node] + dist[node][next] - dist[prev][next]
   - 按贡献降序排序，移除前ratio个

3. **related_removal**: 移除地理位置相近的节点
   - 随机选seed节点，计算所有节点到seed的距离
   - 按距离升序排序，移除最近的ratio个

4. **shaw_removal**: 基于多维相似度移除
   - 综合考虑距离、时间窗、需求的相似性
   - shaw_score = 9*dist + 3*time_diff + 2*demand_diff
   - 移除相似度最高的ratio个

5. **history_removal**: 移除长期未改动的节点
   - 使用self.node_history记录最后修改迭代
   - 按历史值升序排序，移除最久未改动的ratio个
   - 如果history为空，回退到random_removal

6. **cluster_removal**: 基于网格聚类移除
   - 将节点分配到5x5网格
   - 随机选择一个非空网格
   - 移除该网格及相邻网格的节点

7. **greedy_insert**: 贪心插入到成本增量最小位置
   - 遍历所有位置，计算插入成本增量
   - 优先现有路径（用<=比较），相等时选现有路径减少车辆数

8. **regret_insert**: 后悔值优先插入
   - regret = second_best_cost - best_cost
   - 优先插入后悔值大的节点（错过机会成本高）

9. **random_insert**: 随机插入增加多样性
   - 收集所有可行位置
   - 随机选择一个位置插入

【注意事项】：
1. 不要实现 cost()、validate()、check_feasible() 方法（框架已提供）
2. 算子内部可以直接修改solution（框架已做深拷贝）
3. 确保所有边界条件都有检查
4. 代码必须完整可运行，不能有TODO或pass
"""
