# -*- coding: utf-8 -*-
"""
optimize_prompts.py — LLM 迭代优化算子的 Prompt 模板
=====================================================
核心思想：不是让 LLM "从头生成" 算子，而是让 LLM "分析已有算子的性能数据后进行针对性修改"。
"""

# ============================================================
# 数据集特征描述（帮助 LLM 理解不同问题类型）
# ============================================================
DATASET_CHARACTERISTICS = {
    "c1": "C1类（聚类-紧时间窗）：客户按簇状分布，时间窗较窄。适合路径级破坏（同簇客户应在同路径）。",
    "c2": "C2类（聚类-宽时间窗）：客户按簇状分布，时间窗较宽。灵活度高，可用更激进的破坏比例。",
    "r1": "R1类（随机-紧时间窗）：客户随机分布，时间窗较窄。客户分散，插入位置选择困难，需要精确的时间窗感知。",
    "r2": "R2类（随机-宽时间窗）：客户随机分布，时间窗较宽。搜索空间大，需要有效的大范围探索。",
    "rc1": "RC1类（混合-紧时间窗）：部分客户聚类+部分随机，时间窗较窄。需要兼顾聚类和分散客户的策略。",
    "rc2": "RC2类（混合-宽时间窗）：部分客户聚类+部分随机，时间窗较宽。最灵活但搜索空间最大。",
}

# ============================================================
# 成本模型说明
# ============================================================
COST_MODEL_DESCRIPTION = """
【成本模型】本问题是生鲜物流VRP，总成本 = C11 + C12 + C13 + C2 + C3：
- C11 = 车辆数 × 240（固定成本，减少车辆数可显著降低成本）
- C12 = 总距离 × 3（距离成本）
- C13 = 总行驶时长 × 15（制冷成本，与时间正相关）
- C2 = 货损成本（基于新鲜度衰减函数，配送越晚损失越大）
- C3 = 时间窗惩罚（早到/迟到惩罚，违反硬时间窗极高惩罚）

关键洞察：不是单纯最小化距离！配送顺序影响到达时间，进而影响 C2 和 C3。
"""

# ============================================================
# 可用 API 说明
# ============================================================
SOLVER_API_DESCRIPTION = """
【算子可用的 self 属性】
- self.dist_matrix[i][j]: 节点i到节点j的欧几里得距离矩阵
- self.capacity: 车辆容量 (int)
- self.customer_lookup: {node_id: {'id': ..., 'demand': ..., 'x': ..., 'y': ..., 'ready_time': ..., 'due_date': ..., 'service_time': ...}}
- self.id_to_customer: 同 customer_lookup
- self.calculator: FreshnessAndPenaltyCalculator 实例
  - self.calculator.calculate_route_cost(route_nodes, dist_matrix) -> {'variable_cost': ..., 'c2': ..., 'c3': ..., 'dist': ...}
  - self.calculator.f: 车辆固定成本 (240)
  - self.calculator.v: 车辆速度 (40 km/h)
- self._candidate_positions(route, node_id) -> list[int]
  返回将 node_id 插入 route 的候选位置列表（按插入点前后节点到 node 的距离之和排序，返回最近的前 K 个位置，默认 K=10）。
  在修复算子中优先用此方法替代 range(1, len(route))，可显著降低计算开销，同时保留大部分高质量插入位置。
  注意：如果路径长度 <= 2 或 K 足够大，它会退化为 range(1, len(route))。

【数据结构】
- solution: [[0, node1, node2, ..., 0], [0, node3, node4, ..., 0], ...]
  每条路径以 0（仓库）开头和结尾
- route_nodes: [customer_dict, customer_dict, ...] 用于 calculate_route_cost

【破坏算子签名】
def _random_removal(self, solution, ratio) -> (new_solution, removed_nodes_list)
def _route_removal(self, solution, ratio) -> (new_solution, removed_nodes_list)
def _string_removal(self, solution, ratio) -> (new_solution, removed_nodes_list)

【修复算子签名】
def _greedy_insert(self, solution, removed_nodes) -> new_solution
def _regret_insert(self, solution, removed_nodes) -> new_solution
"""

OPTIMIZE_PROMPT_TEMPLATE = """你是 ALNS（自适应大邻域搜索）算法优化专家。
我有一个用于求解生鲜物流 VRP 问题的 ALNS 求解器，其中包含 5 个算子（3个破坏 + 2个修复）。
我已经运行了这些算子并收集了详细的性能数据。

请你：
1. 分析当前算子代码中的性能瓶颈
2. 根据运行数据，提出针对性的代码修改建议（必须有实质性的算法逻辑改进，而不是简单的代码风格调整）
3. 输出修改后的完整算子代码

{cost_model}

{solver_api}

═══════════════════════════════════════════════════════════════
【当前算子代码（第 {round_num} 轮优化）】
═══════════════════════════════════════════════════════════════

```python
{current_operator_code}
```

═══════════════════════════════════════════════════════════════
【运行性能报告】
═══════════════════════════════════════════════════════════════

数据集: {dataset_name} ({dataset_desc})
客户数量: {n_customers}
车辆容量: {vehicle_capacity}

--- 基本指标 ---
最优成本: {best_cost:.2f}
初始成本: {initial_cost:.2f}
改善率: {improvement_pct:.1f}%
车辆数: {num_routes}
求解耗时: {elapsed_time:.1f}秒

--- 算子使用统计 ---
{operator_stats}

--- 收敛分析 ---
最后改善迭代: 第 {last_improve_iter} 次（共 {total_iters} 次迭代）
前25%迭代的改善占比: {early_improvement_pct:.1f}%（如果占比很高且后期无改善，说明陷入了局部最优）
收敛后无改善迭代数: {stagnation_iters}

--- 成本分解 ---
C11 固定成本: {c11:.2f} ({c11_pct:.1f}%)
C12 距离成本: {c12:.2f} ({c12_pct:.1f}%)
C13 制冷成本: {c13:.2f} ({c13_pct:.1f}%)
C2  货损成本: {c2:.2f} ({c2_pct:.1f}%)
C3  时间惩罚: {c3:.2f} ({c3_pct:.1f}%)

--- 路径统计 ---
平均每路径客户数: {avg_customers_per_route:.1f}
平均载重利用率: {avg_load_utilization:.1f}%
{extra_analysis}

═══════════════════════════════════════════════════════════════
【高级优化方向建议（如果遇到瓶颈可以参考）】
═══════════════════════════════════════════════════════════════
如果前几轮优化未能显著降低成本或陷入了同样的瓶颈，请**务必转换思路，尝试更高级的策略**，例如：
1. **破坏算子进阶**：引入 "最差移除 (Worst Removal)" 逻辑（移除距离偏离度大或惩罚成本高的节点）；引入 "时空约束破坏"（针对时间窗或地理位置进行集中移除）。
2. **修复算子进阶**：
   - 插入位置探索时引入 "候选列表 (Candidate List)" 机制或局部启发式，不盲目遍历整条路径的所有位置。
   - 扩展 regret_insert 为 "Regret-k"（k>2），计算前k个最有位置的机会成本差异。
   - 在贪心计算中加入随机噪声机制 (Noise) 或者闪避惩罚机制，强迫模型跳出局部最优。
3. **避免低效操作**：如果某个破坏算子成功率极低，不要只是微调代码，必须引入如上提出的全新机制让它变得更"聪明"。
4. **拒绝重复工作**：严禁给出与历史记录中已经被证明无效的相同思路的微调！必须大刀阔斧地改变逻辑。

═══════════════════════════════════════════════════════════════
【优化要求】
═══════════════════════════════════════════════════════════════

1. 基于上述性能数据，分析每个算子的不足之处
2. 针对性地修改算子代码（必须是实质性逻辑改变，突破之前的思维定势）
3. 保持 5 个方法名和签名完全不变
4. 修复算子中必须使用 self.calculator.calculate_route_cost() 进行完整成本计算
5. 不要修改 __init__ 或添加新方法，只修改上述 5 个方法的内部逻辑
6. 修复算子中强烈推荐使用 self._candidate_positions(route, node_id) 代替 range(1, len(route)) 遍历全部位置

【严禁的常见错误（历史教训）】
以下写法会导致算法退化，绝对不允许出现：
- ❌ 用带噪声的 inc 做位置比较：`best_inc = inc * noise`（噪声只能作参考，决策必须用原始 inc）
- ❌ 截断 regret 评估节点：`for node in remaining[:10]`（regret 必须评估所有剩余节点）
- ❌ 随机采样插入位置：`random.sample(positions, 5)`（应遍历所有候选位置）
- ❌ 用 `j * 30` 或常数估计到达时间（时间必须基于 dist_matrix 和 calculator.v 实际计算）
- ❌ 修改候选位置后不同步更新路径偏移量（删除节点后索引会变化）

请先简要分析（3-5句话），然后输出完整的修改后代码：

```python
def _random_removal(self, solution, ratio):
    ...

def _route_removal(self, solution, ratio):
    ...

def _string_removal(self, solution, ratio):
    ...

def _greedy_insert(self, solution, removed_nodes):
    ...

def _regret_insert(self, solution, removed_nodes):
    ...
```
"""

# ============================================================
# 后续轮次的累积优化 Prompt（包含历史优化记录）
# ============================================================
HISTORY_SECTION_TEMPLATE = """
--- 历史优化记录（极为重要） ---
【重要警告】请仔细阅读以下历史记录。如果你看到了与历史记录中相同思路但导致成本上升（或没变）的尝试，请**绝对避免重复该思路**，想出完全不同的优化策略。
{history_entries}
"""

HISTORY_ENTRY_TEMPLATE = """第 {round_num} 轮: 成本 {cost:.2f} (vs 上一轮 {prev_cost:.2f}, {change_direction}{change_pct:.1f}%)
  修改内容: {modification_summary}"""

