# ========================================
# 在notebook中执行此代码，使用完整的ALNS算子
# ========================================

# 直接读取完整模板文件
with open(r'd:\pythonProject\or_llm_agent\heuristic_prompts.py', 'r', encoding='utf-8') as f:
    full_template = f.read()

# 提取HeuristicPlugin类代码（从class定义到文件末尾）
import re
match = re.search(r'(class HeuristicPlugin:.*)', full_template, re.DOTALL)
if match:
    plugin_code = match.group(1)
    # 去除多余的引号（如果有）
    plugin_code = plugin_code.strip('"""').strip("'''")
    
    print("=" * 60)
    print("✓ 成功加载完整算子模板")
    print(f"✓ 代码长度: {len(plugin_code)} 字符")
    print("=" * 60)
    
    # 执行代码创建类
    exec(plugin_code, globals())
    print("✓ HeuristicPlugin类已创建")
    
    # 创建插件实例（假设data已定义）
    # plugin = HeuristicPlugin(**data)
    # print("✓ 插件实例已初始化")
else:
    print("❌ 未找到HeuristicPlugin类定义")

# 现在可以正常使用plugin进行求解了
# solver = HeuristicSolver(data, plugin)
# solver.solve()
