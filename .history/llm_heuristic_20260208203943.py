import openai
from dotenv import load_dotenv
import os
import re
import subprocess
import sys
import tempfile
import copy
import json
import shutil
import wcwidth
import json
from rich.console import Console
from rich.markdown import Markdown
import io
from contextlib import redirect_stdout
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import numpy as np
from utils import (
    is_number_string,
    convert_to_number,
    extract_best_objective,
    extract_and_execute_python_code,
    eval_model_result,

    )
from heuristic_skeleton import HEURISTIC_SKELETON
from heuristic_prompts import HEURISTIC_PLUGIN_TEMPLATE

# Load environment variables from .env file
load_dotenv()

# api_data = dict(
# api_key = 'sk-sbxihlsgrzsjfknusvrxiokzdwxofzbhjdyfznqgqifguclu', #os.getenv("OPENAI_API_KEY")
# base_url = 'https://api.siliconflow.cn/v1' #os.getenv("OPENAI_API_BASE")
# )

# api_data = dict(
# api_key = os.getenv("OPENAI_API_KEY"),
# base_url = os.getenv("OPENAI_API_BASE")
# )

# api_data = dict(
#     api_key = os.getenv("OPENAI_API_KEY")
# )


api_data = dict(
    # 配置 API Key，这里直接填入您的火山引擎 Key，不再从环境变量读取以避免报错
    api_key="3b2262fa-c113-4f64-90db-10ed2659a583",
    # 配置 Base URL，指定为火山引擎的接口地址
    base_url="https://ark.cn-beijing.volces.com/api/v3"
)

# Initialize OpenAI client
# client = openai.OpenAI(
#     api_key=api_data['api_key'],
# )
# 初始化 OpenAI 客户端
client = openai.OpenAI(
    api_key=api_data['api_key'],  # 传入 API Key
    base_url=api_data['base_url']  # 传入 Base URL
)


def get_display_width(text):
    """
    Calculate the display width of a string, accounting for wide characters like Chinese.
    Uses the wcwidth module for accurate width calculation.

    Args:
        text (str): The text to calculate the width for.

    Returns:
        int: The display width of the text.
    """
    return wcwidth.wcswidth(text)


def print_header(text="", add_newline_before=True, add_newline_after=True,
                 border_char="=", side_char="||"):
    """
    Print a header with customizable text in the middle, adjusted to the console window width.
    Properly handles wide characters like Chinese.

    Args:
        text (str): The text to display in the middle of the header.
        add_newline_before (bool): Whether to add a newline before the header.
        add_newline_after (bool): Whether to add a newline after the header.
        border_char (str): Character to use for the top and bottom borders.
        side_char (str): Character to use for the side borders.
    """
    # Add a newline before the header if requested
    if add_newline_before:
        print()

    # Get terminal width
    # try:
    terminal_width = shutil.get_terminal_size().columns
    # except Exception:
    #     # Fallback width if terminal size cannot be determined
    #     terminal_width = 80

    # Ensure minimum width
    terminal_width = max(terminal_width, 40)

    # Calculate side character padding
    side_char_len = len(side_char)

    # Print the top border
    print(border_char * terminal_width)

    # Print the empty line
    print(side_char + " " * (terminal_width - 2 * side_char_len) + side_char)

    # Print the middle line with text
    text_display_width = get_display_width(text)
    available_space = terminal_width - 2 * side_char_len

    if text_display_width <= available_space:
        left_padding = (available_space - text_display_width) // 2
        right_padding = available_space - text_display_width - left_padding
        # print(terminal_width, text_display_width, available_space, left_padding, right_padding)
        print(side_char + " " * left_padding + text + " " * right_padding + side_char)
    else:
        # If text is too long, we need to truncate it
        # This is more complex with wide characters, so we'll do it character by character
        truncated_text = ""
        truncated_width = 0
        for char in text:
            char_width = get_display_width(char)
            if truncated_width + char_width + 3 > available_space:  # +3 for the "..."
                break
            truncated_text += char
            truncated_width += char_width

        truncated_text += "..."
        right_padding = available_space - get_display_width(truncated_text)
        print(side_char + truncated_text + " " * right_padding + side_char)

    # Print the empty line
    print(side_char + " " * (terminal_width - 2 * side_char_len) + side_char)

    # Print the bottom border
    print(border_char * terminal_width)

    # Add a newline after the header if requested
    if add_newline_after:
        print()


def query_llm(messages, model_name="ep-20260106214023-k4p8b", temperature=0):
    """
    调用 LLM 获取响应结果，使用流式输出方式。
    兼容推理模型(DeepSeek R1)的 reasoning_content 输出。
    """
    try:
        # 使用stream=True启用流式输出
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            stream=True
        )

        # 用于累积完整响应
        full_response = ""

        # 用于控制打印格式
        print("[LLM Response] ", end="", flush=True)

        # 逐块处理流式响应
        for chunk in response:
            # 检查是否有 choices
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # 1. 优先尝试获取 reasoning_content (推理模型的思考过程)
            # 注意：不同版本的 SDK 或 API，字段可能叫 reasoning_content
            reasoning = getattr(delta, 'reasoning_content', None)
            if reasoning:
                # 可以选择用灰色打印思考过程，或者直接打印
                print(reasoning, end="", flush=True)

            # 2. 获取正常的 content (最终回复)
            content = getattr(delta, 'content', None)
            if content:
                print(content, end="", flush=True)
                full_response += content

        # 输出完成后换行
        print()
        return full_response

    except Exception as e:
        print(f"\n[API Exception] {e}")
        # 如果流式失败，返回空字符串以免后续逻辑崩溃
        return ""



def generate_or_code_solver(messages_bak, model_name, data, max_attempts=3):
    messages = copy.deepcopy(messages_bak)
    
    current_project_dir = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")

    # === 使用更精确的 Prompt ===
    prompt = HEURISTIC_PLUGIN_TEMPLATE  # 使用 heuristic_prompts.py 中定义的模板
    
    messages.append({"role": "user", "content": prompt})
    attempt = 0

    while attempt < max_attempts:
        llm_response = query_llm(messages, model_name)
        
        # === 提取 LLM 生成的代码 ===
        code_match = re.search(r"```python\n(.*?)```", llm_response, re.DOTALL)
        llm_plugin_code = code_match.group(1).strip() if code_match else llm_response
        
        # === 检查是否使用了完整成本计算 ===
        uses_full_cost = "self.solver.calculator.calculate_route_cost" in llm_plugin_code
        uses_simple_dist = "self.dist_matrix[" in llm_plugin_code and "- self.dist_matrix[" in llm_plugin_code
        print(f"\n[成本计算检查] 使用完整成本: {uses_full_cost}, 使用简化距离: {uses_simple_dist}")
        if not uses_full_cost:
            print("⚠️ 警告：LLM生成的代码未使用完整成本计算函数！")
        
        # === 强制提取 HeuristicPlugin 类（防止 LLM 生成其他内容）===
        print("\n[Code Generation] Extracting HeuristicPlugin class...")
        
        # === 统一的 __init__ 方法（与提示模板和执行代码一致）===
        unified_init = (
            "    def __init__(self, *args, **kwargs):\n"
            "        # 支持 HeuristicPlugin(data=data) 调用方式\n"
            "        _data = kwargs.get('data', {})\n"
            "        self.capacity = _data.get('vehicle_capacity', 200)\n"
            "        self.customers = _data.get('customers', [])\n"
            "        self.dist_matrix = None  # 由 Solver 注入\n"
            "        self.solver = None  # 由 Solver 注入\n"
            "        self.customer_lookup = {c['id']: c for c in self.customers}\n"
        )
        
        # 方案 1: 如果 LLM 生成了完整的类定义，只保留方法部分（跳过 __init__）
        if "class HeuristicPlugin:" in llm_plugin_code:
            # 提取类中的方法（去掉类定义行和 __init__）
            lines = llm_plugin_code.split('\n')
            method_lines = []
            in_class = False
            skip_init = False
            init_indent = 0
            
            for line in lines:
                if 'class HeuristicPlugin:' in line:
                    in_class = True
                    continue  # 跳过类定义行
                
                if in_class:
                    # 如果遇到新的类定义（顶层），停止
                    if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                        if 'class ' in line:
                            break
                    
                    # 跳过 LLM 生成的 __init__ 方法
                    stripped = line.lstrip()
                    if stripped.startswith('def __init__'):
                        skip_init = True
                        init_indent = len(line) - len(stripped)
                        continue
                    if skip_init:
                        # 如果遇到同级别或更高级别的 def，停止跳过
                        if stripped.startswith('def ') and (len(line) - len(stripped)) <= init_indent:
                            skip_init = False
                        else:
                            continue
                    
                    method_lines.append(line)
            
            # 重新构建完整的 Plugin 类（使用统一的 __init__）
            plugin_class_code = "class HeuristicPlugin:\n"
            plugin_class_code += unified_init + "\n"
            plugin_class_code += '\n'.join(method_lines)
            
            print(f"[Code Generation] HeuristicPlugin class reconstructed ({len(plugin_class_code)} chars)")
        else:
            # 方案 2: 如果只生成了方法，手动添加类框架
            plugin_class_code = "class HeuristicPlugin:\n"
            plugin_class_code += unified_init + "\n"
            
            # 给 LLM 生成的方法添加缩进
            indented_methods = '\n'.join('    ' + line if line.strip() else line 
                                         for line in llm_plugin_code.split('\n'))
            plugin_class_code += indented_methods
            
            print(f"[Code Generation] HeuristicPlugin class constructed ({len(plugin_class_code)} chars)")

        # === 自动补全缺失方法（防止 AttributeError）===
        required_destroy_methods = ['random_removal', 'route_removal', 'string_removal']
        required_insert_methods = ['greedy_insert', 'regret_insert']
        
        for method_name in required_destroy_methods:
            if f'def {method_name}' not in plugin_class_code:
                print(f"[Auto-Fix] 补全缺失的破坏算子: {method_name}")
                if method_name == 'route_removal':
                    stub = (
                        "\n    def route_removal(self, solution, ratio):\n"
                        "        \"\"\"路径移除：随机移除整条路径\"\"\"\n"
                        "        if not solution:\n"
                        "            return solution, []\n"
                        "        non_empty = [(i, [n for n in r if n != 0]) for i, r in enumerate(solution) if any(n != 0 for n in r)]\n"
                        "        if not non_empty:\n"
                        "            return solution, []\n"
                        "        idx, customers = random.choice(non_empty)\n"
                        "        new_sol = [r[:] for i, r in enumerate(solution) if i != idx]\n"
                        "        return new_sol, customers\n"
                    )
                elif method_name == 'string_removal':
                    stub = (
                        "\n    def string_removal(self, solution, ratio):\n"
                        "        \"\"\"连续节点移除：委托给random_removal\"\"\"\n"
                        "        return self.random_removal(solution, ratio)\n"
                    )
                else:  # random_removal
                    stub = (
                        "\n    def random_removal(self, solution, ratio):\n"
                        "        \"\"\"随机移除\"\"\"\n"
                        "        all_n = [(ri,pi,n) for ri,r in enumerate(solution) for pi,n in enumerate(r) if n!=0]\n"
                        "        if not all_n: return solution, []\n"
                        "        total = len(all_n)\n"
                        "        k = min(total, max(1, int(ratio) if ratio>1 else math.ceil(total*ratio)))\n"
                        "        sel = random.sample(all_n, k)\n"
                        "        sel.sort(key=lambda x:(x[0],x[1]), reverse=True)\n"
                        "        ns = [r[:] for r in solution]\n"
                        "        rm = []\n"
                        "        for ri,pi,n in sel: del ns[ri][pi]; rm.append(n)\n"
                        "        ns = [r for r in ns if len(r)>2]\n"
                        "        return ns, rm\n"
                    )
                plugin_class_code += stub
        
        for method_name in required_insert_methods:
            if f'def {method_name}' not in plugin_class_code:
                print(f"[Auto-Fix] 补全缺失的修复算子: {method_name}")
                if method_name == 'regret_insert':
                    stub = (
                        "\n    def regret_insert(self, solution, removed_nodes):\n"
                        "        \"\"\"后悔插入：委托给greedy_insert\"\"\"\n"
                        "        return self.greedy_insert(solution, removed_nodes)\n"
                    )
                else:  # greedy_insert - use fallback
                    stub = (
                        "\n    def greedy_insert(self, solution, removed_nodes):\n"
                        "        \"\"\"贪心插入：随机插入\"\"\"\n"
                        "        if not removed_nodes: return solution\n"
                        "        ns = [r[:] for r in solution] if solution else []\n"
                        "        for node in removed_nodes: ns.append([0, node, 0])\n"
                        "        return ns\n"
                    )
                plugin_class_code += stub
        
        # === 拼接最终脚本（关键修改）===
        full_code = (
            "# -*- coding: utf-8 -*-\n"
            "import math, json, random, copy, sys, traceback\n"
            f"sys.path.append('{current_project_dir}')\n"
            "from utils import FreshnessAndPenaltyCalculator\n\n"
            f"data = {json.dumps(data)}\n\n"
            # === 第一部分：Skeleton（包含 HeuristicSolver）===
            f"{HEURISTIC_SKELETON}\n\n"
            # === 第二部分：LLM 生成的 HeuristicPlugin ===
            f"{plugin_class_code}\n\n"
            # === 第三部分：执行代码 ===
            "if __name__ == '__main__':\n"
            "    try:\n"
            "        plugin = HeuristicPlugin(data=data)\n"
            "        print('[Initialization] Plugin initialized')\n"
            "        solver = HeuristicSolver(data, plugin)\n"
            "        print('[Initialization] Solver initialized')\n"
            "        best_sol, best_cost = solver.solve_multi_run(max_iters=800, num_runs=3)\n"
            "        print(f'BEST_COST: {best_cost}')\n"
            "        print(f'BEST_SOLUTION: {best_sol}')\n"
            "    except Exception as e:\n"
            "        print(f'[Runtime Exception] {e}')\n"
            "        traceback.print_exc(file=sys.stdout)\n"
        )

        # === 验证代码结构 ===
        solver_count = full_code.count('class HeuristicSolver')
        plugin_count = full_code.count('class HeuristicPlugin')
        init_debug = '[初始化]' in full_code
        
        print(f"[Code Validation] Solver classes: {solver_count}, Plugin classes: {plugin_count}")
        
        if solver_count != 1 or plugin_count != 1:
            print(f"[Warning] Unexpected class count - Solver={solver_count}, Plugin={plugin_count}")
        
     
        # === 执行代码 ===
        success, result_msg = extract_and_execute_python_code(f"```python\n{full_code}\n```")

        if success and "BEST_COST:" in result_msg:
            print("\n[Solver Status] Algorithm execution completed")
            return True, result_msg, messages_bak

        print(f"\n[Attempt {attempt+1}/{max_attempts}] Code refinement needed")
        print(f"[Debug Info]\n{result_msg}\n")
        messages.append({"role": "assistant", "content": llm_response})
        messages.append({
            "role": "user", 
            "content": f"Code execution encountered issues. Please refine the implementation.\n\nDebug information:\n{result_msg}\n\n"
            f"【重要】你的类必须恰好包含以下5个方法：\n"
            f"  1. random_removal(self, solution, ratio)\n"
            f"  2. route_removal(self, solution, ratio)\n"
            f"  3. string_removal(self, solution, ratio)\n"
            f"  4. greedy_insert(self, solution, removed_nodes)\n"
            f"  5. regret_insert(self, solution, removed_nodes)\n"
            f"只实现这5个，不多不少。regret_insert 中只需比较每条路径的最佳插入位置即可。"
        })
        attempt += 1

    return False, None, messages_bak

def extract_solution_from_output(output):
    """从求解输出中提取路线信息"""
    solution = []
    
    print("[调试] 开始提取路线信息...")
    
    # 尝试多种格式提取路线
    # 格式1: Route 1: [0, 1, 2, 0]
    route_pattern1 = r'Route \d+: \[(.*?)\]'
    matches = re.findall(route_pattern1, output, re.IGNORECASE)
    
    print(f"[调试] 格式1匹配到 {len(matches)} 条路线")
    
    if matches:
        for i, match in enumerate(matches):
            nodes_str = match.strip()
            if nodes_str:
                try:
                    # 更宽松的数字提取
                    nodes = [int(x.strip()) for x in nodes_str.split(',') if x.strip().lstrip('-').isdigit()]
                    if nodes:
                        solution.append(nodes)
                        print(f"[调试] 路线 {i+1}: {nodes[:5]}{'...' if len(nodes) > 5 else ''}")
                except Exception as e:
                    print(f"[调试] 解析路线 {i+1} 失败: {e}")
    
    # 格式2: BEST_SOLUTION: [[0,1,2,0], [0,3,4,0]]
    if not solution:
        print("[调试] 尝试格式2...")
        solution_pattern = r'BEST_SOLUTION:\s*(\[\[.*?\]\])'
        match = re.search(solution_pattern, output, re.DOTALL)
        if match:
            try:
                import ast
                solution = ast.literal_eval(match.group(1))
                print(f"[调试] 格式2提取到 {len(solution)} 条路线")
            except Exception as e:
                print(f"[调试] 格式2解析失败: {e}")
    
    # 如果还是没有，尝试更宽松的匹配
    if not solution:
        print("[调试] 尝试宽松匹配...")
        # 查找任何包含节点列表的行
        route_lines = re.findall(r'路线\s*\d+[：:]\s*\[(.*?)\]', output, re.IGNORECASE)
        if route_lines:
            print(f"[调试] 宽松匹配找到 {len(route_lines)} 条路线")
            for line in route_lines:
                try:
                    nodes = [int(x.strip()) for x in line.split(',') if x.strip().lstrip('-').isdigit()]
                    if nodes:
                        solution.append(nodes)
                except:
                    pass
    
    print(f"[调试] 最终提取到 {len(solution)} 条路线")
    return solution

def extract_iteration_history(output):
    """从输出中提取迭代历史 - 只提取最后一次运行的数据"""
    iterations = []
    cost_history = []
    best_history = []
    
    # 查找迭代信息，格式如: "Iter 10: Current=1234.56, Best=1200.00"
    iter_pattern = r'Iter\s+(\d+):\s+Current=([\d.]+),\s+Best=([\d.]+)'
    matches = re.findall(iter_pattern, output)
    
    if not matches:
        return iterations, cost_history, best_history
    
    # 检测运行边界：当迭代次数重置（从大到小）时，说明开始新一轮
    all_runs = []
    current_run = []
    
    for match in matches:
        iteration, current_cost, best_cost = match
        iter_num = int(iteration)
        
        # 如果迭代次数变小，说明开始了新一轮
        if current_run and iter_num <= current_run[-1][0]:
            all_runs.append(current_run)
            current_run = []
        
        current_run.append((iter_num, float(current_cost), float(best_cost)))
    
    # 添加最后一轮
    if current_run:
        all_runs.append(current_run)
    
    # 只返回最后一次运行的数据（通常是最优的）
    if all_runs:
        last_run = all_runs[-1]
        iterations = [item[0] for item in last_run]
        cost_history = [item[1] for item in last_run]
        best_history = [item[2] for item in last_run]
    
    return iterations, cost_history, best_history

def visualize_results(dataset, solution, best_cost, output):
    """生成学术风格的可视化图表"""
    # === 学术风格全局设置 ===
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'SimSun'],  # 英文用 TNR，中文回退宋体
        'mathtext.fontset': 'stix',
        'axes.unicode_minus': False,
        'axes.linewidth': 0.8,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 600,
        'lines.linewidth': 1.2,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'grid.linestyle': '--',
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    
    # ═══════════════════════════════════════════════════
    # (a) Convergence Curve
    # ═══════════════════════════════════════════════════
    ax1 = axes[0]
    iterations, cost_history, best_history = extract_iteration_history(output)
    
    if cost_history and best_history and iterations:
        # 移动平均平滑
        window_size = max(5, len(cost_history) // 10)
        
        def moving_average(data, window):
            if len(data) < window:
                return data
            smoothed = []
            for i in range(len(data)):
                start = max(0, i - window // 2)
                end = min(len(data), i + window // 2 + 1)
                smoothed.append(sum(data[start:end]) / (end - start))
            return smoothed
        
        cost_smooth = moving_average(cost_history, window_size)
        
        # 学术配色：灰 + 蓝 + 红
        ax1.plot(iterations, cost_history, color='#B0B0B0', linewidth=0.6, alpha=0.4, label='Current cost (raw)')
        ax1.plot(iterations, cost_smooth, color='#2166AC', linewidth=1.4, label='Current cost (smoothed)')
        ax1.plot(iterations, best_history, color='#B2182B', linewidth=1.6, linestyle='-', label='Best cost')
        
        # y 轴范围
        min_cost = min(min(best_history), min(cost_smooth))
        max_cost = max(max(best_history), max(cost_smooth))
        y_range = max_cost - min_cost
        if y_range > 0:
            ax1.set_ylim(min_cost - y_range * 0.05, max_cost + y_range * 0.08)
        
        # 标注最优解（简洁学术箭头）
        min_idx = best_history.index(min(best_history))
        ax1.annotate(
            f'{best_history[min_idx]:.2f}',
            xy=(iterations[min_idx], best_history[min_idx]),
            xytext=(30, 20), textcoords='offset points',
            fontsize=9, color='#B2182B',
            arrowprops=dict(arrowstyle='->', color='#B2182B', lw=1.0),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#B2182B', linewidth=0.8, alpha=0.9)
        )
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Objective Value')
        ax1.set_title('(a) ALNS Convergence Curve', fontweight='bold')
        ax1.legend(loc='upper right', frameon=True, edgecolor='gray', fancybox=False, framealpha=0.9)
        ax1.grid(True)
        ax1.tick_params(direction='in', top=True, right=True)
    else:
        ax1.text(0.5, 0.5, 'No iteration data', ha='center', va='center', transform=ax1.transAxes, fontsize=11)
        ax1.set_title('(a) ALNS Convergence Curve', fontweight='bold')
    
    # ═══════════════════════════════════════════════════
    # (b) Vehicle Routing Map
    # ═══════════════════════════════════════════════════
    ax2 = axes[1]
    customers = dataset['customers']
    depot = customers[0]
    depot_x, depot_y = depot['x'], depot['y']
    
    # 学术配色方案（区分度高、打印友好）
    academic_colors = [
        '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173',
        '#5254a3', '#6b6ecf', '#9c9ede', '#bd9e39', '#ad494a',
    ]
    
    # 绘制路线
    if solution and len(solution) > 0:
        for idx, route in enumerate(solution):
            try:
                if not route or len(route) == 0:
                    continue
                complete_route = []
                if route[0] != 0:
                    complete_route.append(0)
                complete_route.extend(route)
                if route[-1] != 0:
                    complete_route.append(0)
                
                route_x = [customers[n]['x'] for n in complete_route if n < len(customers)]
                route_y = [customers[n]['y'] for n in complete_route if n < len(customers)]
                
                if len(route_x) >= 2:
                    color = academic_colors[idx % len(academic_colors)]
                    ax2.plot(route_x, route_y, '-', color=color, linewidth=1.0, alpha=0.7, zorder=2)
                    # 客户点
                    if len(route_x) > 2:
                        ax2.scatter(route_x[1:-1], route_y[1:-1], c=color, s=20, alpha=0.85,
                                   zorder=4, edgecolors='black', linewidths=0.3)
            except Exception:
                continue
    
    # 仓库（黑色方块，学术常见标记）
    ax2.plot(depot_x, depot_y, 's', color='black', markersize=10, zorder=6, label='Depot')
    
    # 所有客户点（灰色底层，在有路线时不重复绘制）
    if not solution or len(solution) == 0:
        customer_coords = [(c['x'], c['y']) for c in customers[1:]]
        if customer_coords:
            cx, cy = zip(*customer_coords)
            ax2.scatter(cx, cy, c='#666666', s=15, alpha=0.5, zorder=3, label='Customer')
    
    ax2.set_xlabel('$x$ coordinate')
    ax2.set_ylabel('$y$ coordinate')
    route_count = len(solution) if solution else 0
    ax2.set_title(f'(b) Vehicle Routes ($K$={route_count})', fontweight='bold')
    ax2.grid(True)
    ax2.set_aspect('equal')
    ax2.tick_params(direction='in', top=True, right=True)
    # 简化图例：只显示仓库标记
    ax2.legend(loc='upper right', frameon=True, edgecolor='gray', fancybox=False, framealpha=0.9, markerscale=0.8)
    
    # ═══════════════════════════════════════════════════
    # (c) Multi-run Cost Comparison
    # ═══════════════════════════════════════════════════
    ax3 = axes[2]
    
    multi_run_pattern = r'各次成本: \[(.*?)\]'
    multi_run_match = re.search(multi_run_pattern, output)
    
    if multi_run_match:
        costs_str = multi_run_match.group(1)
        costs = [float(c.strip().strip("'\"")) for c in costs_str.split(',')]
        
        runs = list(range(1, len(costs) + 1))
        min_cost_val = min(costs)
        avg_cost_val = sum(costs) / len(costs)
        
        # 学术柱状图：灰色 + 最优高亮
        bar_colors = ['#B2182B' if c == min_cost_val else '#4393C3' for c in costs]
        bars = ax3.bar(runs, costs, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.6, width=0.6)
        
        # 参考线
        ax3.axhline(y=min_cost_val, color='#B2182B', linestyle='--', linewidth=0.8, alpha=0.7,
                    label=f'Best = {min_cost_val:.2f}')
        ax3.axhline(y=avg_cost_val, color='#2166AC', linestyle=':', linewidth=0.8, alpha=0.7,
                    label=f'Mean = {avg_cost_val:.2f}')
        
        # 数值标注
        for bar, cost in zip(bars, costs):
            ax3.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                    f'{cost:.1f}', ha='center', va='bottom', fontsize=8)
        
        # y 轴截断显示（避免从 0 开始导致差异不明显）
        cost_range = max(costs) - min(costs)
        if cost_range > 0:
            y_bottom = min(costs) - cost_range * 1.5
            y_top = max(costs) + cost_range * 1.0
            ax3.set_ylim(y_bottom, y_top)
        
        ax3.set_xlabel('Run')
        ax3.set_ylabel('Objective Value')
        ax3.set_xticks(runs)
        ax3.set_title('(c) Multi-run Comparison', fontweight='bold')
        ax3.legend(loc='upper right', frameon=True, edgecolor='gray', fancybox=False, framealpha=0.9)
        ax3.grid(True, axis='y')
        ax3.tick_params(direction='in', top=True, right=True)
    else:
        ax3.text(0.5, 0.5, 'No multi-run data', ha='center', va='center', transform=ax3.transAxes, fontsize=11)
        ax3.set_title('(c) Multi-run Comparison', fontweight='bold')
    
    plt.tight_layout(w_pad=2.5)
    
    # 保存
    import os
    project_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(project_dir, 'vrp_result.png')
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white', pad_inches=0.15)
    # 同时保存 PDF（学术投稿用矢量图）
    pdf_path = os.path.join(project_dir, 'vrp_result.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', pad_inches=0.15)
    plt.close()
    print(f"[可视化] 图表已保存到: {output_path}")
    print(f"[可视化] 矢量图已保存到: {pdf_path}")

def extract_operator_stats(output):
    """从输出中提取算子统计信息"""
    stats = {}
    
    # 查找算子统计，格式如: "random_removal: 100 uses, 5 success (5.0%)"
    pattern = r'(\w+):\s+(\d+)\s+uses?,\s+(\d+)\s+success.*?\((\d+\.?\d*)%\)'
    matches = re.findall(pattern, output)
    
    for match in matches:
        op_name, uses, successes, rate = match
        if int(uses) > 0:  # 只显示使用过的算子
            stats[op_name] = {
                'uses': int(uses),
                'successes': int(successes),
                'rate': float(rate)
            }
    
    return stats

def load_solomon_data(file_path):
    data = {}
    customers = []
    vehicle_capacity = None
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        line = line.strip()
        if line == "" or line.startswith("C") or line.startswith("VEHICLE") or line.startswith("NUMBER") or line.startswith("CUSTOMER"):
            continue
        parts = line.split()
        if len(parts) == 2 and vehicle_capacity is None:
            vehicle_capacity = int(parts[1])
            break
    if vehicle_capacity is None:
        vehicle_capacity = 200
    for line in lines:
        line = line.strip()
        if line == "" or line.startswith("C") or line.startswith("VEHICLE") or line.startswith("NUMBER") or line.startswith("CUSTOMER") or line.startswith("CUST"):
            continue
        parts = line.split()
        if parts[0].isdigit() and len(parts) >= 7:
            cust_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            demand = int(parts[3])
            ready_time = float(parts[4])
            due_date = float(parts[5])
            service_time = float(parts[6])
            customers.append({
                "id": cust_id, "demand": demand, "x": x, "y": y,
                "ready_time": ready_time, "due_date": due_date, "service_time": service_time
            })
    data["vehicle_capacity"] = vehicle_capacity
    data["customers"] = customers
    return data



if __name__ == "__main__":
    import sys
    import os
    import io
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    console = Console()
    model_name = 'ep-20260106214023-k4p8b'
    messages_bak = []

    solomon_file = r"D:\pythonProject\or_llm_agent\data\1 Solomon Benchmark\c1\c105.txt"

    dataset = load_solomon_data(solomon_file)
    # Patch time windows for depot
    if dataset['customers']:
        depot = dataset['customers'][0]
        for cust in dataset['customers']:
            cust['E_i'] = max(0, cust['ready_time']) if cust['id'] != 0 else 0
            cust['L_i'] = min(depot['due_date'], cust['due_date']) if cust['id'] != 0 else depot['due_date']

    print_header("ALNS-based VRP Solver with LLM-generated Heuristics")
    print(f"Dataset: Solomon C101")
    print(f"Model: DeepSeek R1")
    print(f"Algorithm: Adaptive Large Neighborhood Search\n")
    
    is_solve_success, output, _ = generate_or_code_solver(
        messages_bak, model_name, dataset, max_attempts=5
    )

    if is_solve_success:
        print("\n" + "="*80)
        print("SOLUTION SUMMARY")
        print("="*80)
        
        # Extract and display key metrics
        cost_match = re.search(r'BEST_COST:\s*([\d.]+)', output)
        best_cost = None
        if cost_match:
            best_cost = float(cost_match.group(1))
            print(f"\nObjective Value: {best_cost:.2f}")
        
        route_matches = re.findall(r'Route \d+', output)
        if route_matches:
            print(f"Number of Routes: {len(route_matches)}")
        
        # Extract solution for visualization
        solution = extract_solution_from_output(output)
        
        print(f"\n[调试] 提取到的路线数量: {len(solution) if solution else 0}")
        if solution and len(solution) > 0:
            print(f"[调试] 第一条路线示例: {solution[0][:10]}{'...' if len(solution[0]) > 10 else ''}")
        
        # Visualize results - 即使没有solution也尝试生成图表
        if best_cost:
            print("[可视化] 正在生成图表...")
            try:
                visualize_results(dataset, solution, best_cost, output)
            except Exception as e:
                print(f"[可视化错误] {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "="*80)
        print("DETAILED OUTPUT")
        print("="*80)
        print(output)
    else:
        print("\n" + "="*80)
        print("EXECUTION TERMINATED")
        print("="*80)
        print("\nNote: Maximum iteration limit reached.")
        print("The algorithm has completed all attempted refinements.")

