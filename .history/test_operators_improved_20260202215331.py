"""
测试改进后的算子 - 验证容量约束和完整实现
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from heuristic_prompts import HEURISTIC_PLUGIN_PROMPT_PREFIX
import re

# 检查关键改进点
code = HEURISTIC_PLUGIN_PROMPT_PREFIX

print("=" * 70)
print("算子改进验证")
print("=" * 70)

# 1. 检查related_removal是否完整实现
if "seed = random.choice(all_nodes)" in code and "node_distances.append" in code:
    print("✅ related_removal: 完整实现（非回退）")
else:
    print("❌ related_removal: 仍是stub实现")

# 2. 检查shaw_removal是否完整实现  
if "shaw_score = 9 * normalized_dist" in code:
    print("✅ shaw_removal: 完整实现（综合相似度）")
else:
    print("❌ shaw_removal: 仍是stub实现")

# 3. 检查cluster_removal是否完整实现
if "grid_size = 5" in code and "grid.setdefault" in code:
    print("✅ cluster_removal: 完整实现（网格聚类）")
else:
    print("❌ cluster_removal: 仍是stub实现")

# 4. 检查greedy_insert容量约束
if "current_load = sum(self.customer_lookup" in code and "greedy_insert" in code:
    print("✅ greedy_insert: 有容量约束检查")
else:
    print("❌ greedy_insert: 缺少容量约束")

# 5. 检查regret_insert容量约束
regret_section = code[code.find("def regret_insert"):code.find("def random_insert")] if "def regret_insert" in code else ""
if "current_load" in regret_section and "capacity" in regret_section:
    print("✅ regret_insert: 有容量约束检查")
else:
    print("❌ regret_insert: 缺少容量约束")

# 6. 检查random_insert容量约束
random_section = code[code.find("def random_insert"):] if "def random_insert" in code else ""
if "current_load" in random_section and "capacity" in random_section:
    print("✅ random_insert: 有容量约束检查")
else:
    print("❌ random_insert: 缺少容量约束")

# 7. 检查worst_removal改进
worst_section = code[code.find("def worst_removal"):code.find("def related_removal")] if "def worst_removal" in code else ""
if "demand * 0.01" in worst_section:
    print("✅ worst_removal: 增强版（考虑需求）")
else:
    print("⚠️  worst_removal: 基础版（仅距离）")

print("\n" + "=" * 70)
print("改进建议:")
print("=" * 70)
print("如果上述检查都通过，但测试结果仍显示0%成功率，")
print("请确保Notebook中执行的代码使用的是文件中的最新版本。")
print("\n具体操作：")
print("1. 重新运行notebook，确保加载最新的heuristic_prompts.py")
print("2. 或者直接将改进后的代码粘贴到notebook的代码块中")
print("3. 检查skeleton.py中的fallback算子是否也有容量约束")
print("=" * 70)
