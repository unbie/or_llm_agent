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
from heuristic_prompts_o import HEURISTIC_PLUGIN_TEMPLATE

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
        
        # === 强制提取 HeuristicPlugin 类（防止 LLM 生成其他内容）===
        print("\n[Code Generation] Extracting HeuristicPlugin class...")
        
        # 方案 1: 如果 LLM 生成了完整的类定义，只保留方法部分
        if "class HeuristicPlugin:" in llm_plugin_code:
            # 提取类中的方法（去掉类定义行）
            lines = llm_plugin_code.split('\n')
            method_lines = []
            in_class = False
            indent_level = 0
            
            for line in lines:
                if 'class HeuristicPlugin:' in line:
                    in_class = True
                    indent_level = len(line) - len(line.lstrip())
                    continue  # 跳过类定义行
                
                if in_class:
                    # 如果遇到新的类定义（顶层），停止
                    if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                        if 'class ' in line:
                            break
                    method_lines.append(line)
            
            # 重新构建完整的 Plugin 类
            plugin_class_code = "class HeuristicPlugin:\n"
            plugin_class_code += "    def __init__(self, *args, **kwargs):\n"
            plugin_class_code += "        self.dist_matrix = kwargs.get('dist_matrix')\n"
            plugin_class_code += "        self.vehicle_capacity = kwargs.get('vehicle_capacity', 200)\n"
            plugin_class_code += "        self.nodes_dict = kwargs.get('nodes_dict')\n\n"
            plugin_class_code += '\n'.join(method_lines)
            
            print(f"[Code Generation] HeuristicPlugin class reconstructed ({len(plugin_class_code)} chars)")
        else:
            # 方案 2: 如果只生成了方法，手动添加类框架
            plugin_class_code = "class HeuristicPlugin:\n"
            plugin_class_code += "    def __init__(self, *args, **kwargs):\n"
            plugin_class_code += "        self.dist_matrix = kwargs.get('dist_matrix')\n"
            plugin_class_code += "        self.vehicle_capacity = kwargs.get('vehicle_capacity', 200)\n"
            plugin_class_code += "        self.nodes_dict = kwargs.get('nodes_dict')\n\n"
            
            # 给 LLM 生成的方法添加缩进
            indented_methods = '\n'.join('    ' + line if line.strip() else line 
                                         for line in llm_plugin_code.split('\n'))
            plugin_class_code += indented_methods
            
            print(f"[Code Generation] HeuristicPlugin class constructed ({len(plugin_class_code)} chars)")

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
            "        best_sol, best_cost = solver.solve_multi_run(max_iters=150, num_runs=5, base_seed=42)\n"
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
            "content": f"Code execution encountered issues. Please refine the implementation.\n\nDebug information:\n{result_msg}\n\nNote: Implement random_removal, worst_removal, and greedy_insert methods."
        })
        attempt += 1

    return False, None, messages_bak

def extract_solution_from_output(output):
    """从求解输出中提取路线信息"""
    solution = []
    
    # 尝试多种格式提取路线
    # 格式1: Route 1: [0, 1, 2, 0]
    route_pattern1 = r'Route \d+: \[(.*?)\]'
    matches = re.findall(route_pattern1, output)
    
    if matches:
        for match in matches:
            nodes_str = match.strip()
            if nodes_str:
                try:
                    nodes = [int(x.strip()) for x in nodes_str.split(',') if x.strip().replace('-','').isdigit()]
                    if nodes:
                        solution.append(nodes)
                except:
                    pass
    
    # 格式2: BEST_SOLUTION: [[0,1,2,0], [0,3,4,0]]
    if not solution:
        solution_pattern = r'BEST_SOLUTION:\s*(\[.*?\])'
        match = re.search(solution_pattern, output, re.DOTALL)
        if match:
            try:
                import ast
                solution = ast.literal_eval(match.group(1))
            except:
                pass
    
    return solution

def extract_iteration_history(output):
    """从输出中提取迭代历史"""
    cost_history = []
    best_history = []
    
    # 查找迭代信息，格式如: "Iter 10: Current=1234.56, Best=1200.00"
    iter_pattern = r'Iter\s+(\d+):\s+Current=([\d.]+),\s+Best=([\d.]+)'
    matches = re.findall(iter_pattern, output)
    
    for match in matches:
        iteration, current_cost, best_cost = match
        cost_history.append(float(current_cost))
        best_history.append(float(best_cost))
    
    return cost_history, best_history

def visualize_results(dataset, solution, best_cost, output):
    """生成可视化图表"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文
    except:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig = plt.figure(figsize=(18, 6))
    
    # 子图1: 迭代历史
    ax1 = plt.subplot(1, 3, 1)
    cost_history, best_history = extract_iteration_history(output)
    
    if cost_history and best_history:
        iterations = [i * 40 for i in range(len(cost_history))]  # 每40次迭代输出一次
        # 只显示最优成本曲线
        ax1.plot(iterations, best_history, 'r-', linewidth=2.5, label='最优成本', marker='o', markersize=4, markevery=2)
        ax1.fill_between(iterations, best_history, alpha=0.2, color='red')
        
        # 缩小y轴范围，显示曲线起伏
        min_cost = min(best_history)
        max_cost = max(best_history)
        y_range = max_cost - min_cost
        if y_range > 0:
            # 设置上下边界，留入一定边距
            y_margin = y_range * 0.15
            ax1.set_ylim(min_cost - y_margin, max_cost + y_margin)
        
        ax1.set_xlabel('迭代次数', fontsize=12)
        ax1.set_ylabel('成本', fontsize=12)
        ax1.set_title('ALNS算法收敛曲线', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # 标注最优值
        min_idx = best_history.index(min_cost)
        ax1.plot(iterations[min_idx], min_cost, 'r*', markersize=18, zorder=5, markeredgecolor='darkred', markeredgewidth=2)
        ax1.annotate(f'最优: {min_cost:.2f}', 
                    xy=(iterations[min_idx], min_cost),
                    xytext=(15, 15), textcoords='offset points',
                    fontsize=11, color='darkred', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.6', facecolor='yellow', alpha=0.8, edgecolor='red', linewidth=2),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
    else:
        ax1.text(0.5, 0.5, '无迭代数据', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=14)
        ax1.set_title('ALNS算法收敛曲线', fontsize=14, fontweight='bold')
    
    # 子图2: 车辆路线图
    ax2 = plt.subplot(1, 3, 2)
    customers = dataset['customers']
    
    # 提取坐标
    depot = customers[0]
    depot_x, depot_y = depot['x'], depot['y']
    
    # 绘制仓库
    ax2.plot(depot_x, depot_y, 'rs', markersize=15, label='配送中心', zorder=5, markeredgecolor='darkred', markeredgewidth=2)
    
    # 绘制客户点
    customer_coords = [(c['x'], c['y']) for c in customers[1:]]
    if customer_coords:
        cx, cy = zip(*customer_coords)
        ax2.scatter(cx, cy, c='blue', s=50, alpha=0.6, label='客户点', zorder=3, edgecolors='navy', linewidths=0.5)
    
    # 绘制路线 - 只有当solution存在且有效时
    if solution and len(solution) > 0:
        colors = plt.cm.tab20(np.linspace(0, 1, max(20, len(solution))))
        
        for idx, route in enumerate(solution):
            try:
                # 确保路线包含仓库（节点0）作为起点和终点
                if not route or len(route) == 0:
                    continue
                
                # 构建完整路线：仓库 -> 客户们 -> 仓库
                complete_route = []
                
                # 添加起始仓库
                if route[0] != 0:
                    complete_route.append(0)
                
                # 添加路线中的所有节点
                complete_route.extend(route)
                
                # 添加终止仓库
                if route[-1] != 0:
                    complete_route.append(0)
                
                # 提取路线所有节点的坐标
                route_x = []
                route_y = []
                for node in complete_route:
                    if node < len(customers):
                        route_x.append(customers[node]['x'])
                        route_y.append(customers[node]['y'])
                
                if len(route_x) >= 2:
                    color = colors[idx % len(colors)]
                    
                    # 绘制完整路线（从仓库出发到每个客户再回到仓库）
                    ax2.plot(route_x, route_y, '-', color=color, linewidth=2.5, alpha=0.75, zorder=2)
                    
                    # 绘制中间的客户点（不包括起始和结束的仓库）
                    if len(route_x) > 2:
                        # 只绘制客户点，不绘制仓库点
                        customer_x = route_x[1:-1]
                        customer_y = route_y[1:-1]
                        ax2.scatter(customer_x, customer_y, c=[color]*len(customer_x), s=80, alpha=0.9, 
                                   zorder=4, edgecolors='black', linewidths=1.5)
                    
                    # 标注路线编号
                    mid_idx = len(route_x) // 2
                    ax2.text(route_x[mid_idx], route_y[mid_idx], f'{idx+1}', 
                            fontsize=9, fontweight='bold', color='white',
                            bbox=dict(boxstyle='circle,pad=0.3', facecolor=color, edgecolor='black', linewidth=1.5),
                            ha='center', va='center', zorder=5)
                    
            except Exception as e:
                print(f"[警告] 绘制路线 {idx+1} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    ax2.set_xlabel('X 坐标', fontsize=12)
    ax2.set_ylabel('Y 坐标', fontsize=12)
    route_count = len(solution) if solution else 0
    ax2.set_title(f'车辆路线规划图 (共{route_count}条路线)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.axis('equal')
    
    # 子图3: 成本分解
    ax3 = plt.subplot(1, 3, 3)
    
    # 提取多次运行的成本
    multi_run_pattern = r'各次成本: \[(.*?)\]'
    multi_run_match = re.search(multi_run_pattern, output)
    
    if multi_run_match:
        costs_str = multi_run_match.group(1)
        costs = [float(c.strip().strip("'\"")) for c in costs_str.split(',')]
        
        runs = [f'第{i+1}次' for i in range(len(costs))]
        colors_bar = ['#FF6B6B' if c == min(costs) else '#4ECDC4' for c in costs]
        
        bars = ax3.bar(runs, costs, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax3.axhline(y=min(costs), color='red', linestyle='--', linewidth=2, label=f'最优: {min(costs):.2f}')
        ax3.axhline(y=sum(costs)/len(costs), color='green', linestyle='--', linewidth=2, label=f'平均: {sum(costs)/len(costs):.2f}')
        
        ax3.set_ylabel('总成本', fontsize=12)
        ax3.set_title('多次运行成本对比', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, axis='y', alpha=0.3, linestyle='--')
        
        # 添加数值标签
        for bar, cost in zip(bars, costs):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{cost:.1f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, '无多次运行数据', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=14)
        ax3.set_title('多次运行成本对比', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存到项目根目录
    import os
    project_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(project_dir, 'vrp_result.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[可视化] 图表已保存到: {output_path}")

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

    solomon_file = r"D:\pythonProject\or_llm_agent\data\1 Solomon Benchmark\c1\c101.txt"

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

