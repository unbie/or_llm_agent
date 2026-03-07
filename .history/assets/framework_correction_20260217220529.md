# OR-LLM-Agent 框架图修正建议

## 正确的算子配置

### Intelligent Agent Layer (智能体层)
✅ 无需修改 - 当前版本正确

### Algorithm Runtime Layer (ALNS框架执行层)

#### Modular Interface Slots
应分为两类算子：

**破坏算子 (Destroy Operators)**
- 主要算子 (3个，由LLM生成):
  1. `random_removal` - 随机分散破坏
  2. `route_removal` - 整条路径破坏
  3. `string_removal` - 连续节点破坏

- 备用算子 (6个，框架预设):
  1. `_fallback_random_removal`
  2. `_fallback_worst_removal`
  3. `_fallback_related_removal`
  4. `_fallback_shaw_removal`
  5. `_fallback_history_removal`
  6. `_fallback_cluster_removal`

**修复算子 (Repair Operators)**
- 主要算子 (2个，由LLM生成):
  1. `greedy_insert` - 贪心修复
  2. `regret_insert` - 后悔修复

- 备用算子 (2个，框架预设):
  1. `_fallback_greedy_insert` - 简化贪心插入
  2. `_fallback_regret_insert` - 委托给贪心

### Evaluation Layer (评估层)
只保留数学公式部分即可：
$$C_{total} = C_{fixed} + C_{dist} + C_{freshness} + C_{penalty}$$

## 代码证据

见以下文件：
1. 算子定义：`heuristic_prompts.py` 第1-100行
2. Fallback机制：`heuristic_skeleton.py` 第470-500行
3. 轮盘赌选择：`heuristic_skeleton.py` 第470行
4. 成本计算：`utils.py` 第47-110行
