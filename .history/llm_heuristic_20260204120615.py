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
            "        best_sol, best_cost = solver.solve_multi_run(max_iters=1000, num_runs=5, base_seed=42)\n"
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
    temp_history = []
    accept_rate_history = []
    
    # 查找迭代信息，格式如: "Iter 10: Current=1234.56, Best=1200.00, Temp=50.0, AcceptRate=0.123"
    iter_pattern = r'Iter\s+(\d+):\s+Current=([\d.]+),\s+Best=([\d.]+),\s+Temp=([\d.]+),\s+AcceptRate=([\d.]+)'
    matches = re.findall(iter_pattern, output)
    
    if not matches:
        # 尝试旧格式（没有AcceptRate）
        iter_pattern_old = r'Iter\s+(\d+):\s+Current=([\d.]+),\s+Best=([\d.]+)'
        matches_old = re.findall(iter_pattern_old, output)
        if matches_old:
            for match in matches_old:
                iteration, current_cost, best_cost = match
                iter_num = int(iteration)
                iterations.append(iter_num)
                cost_history.append(float(current_cost))
                best_history.append(float(best_cost))
            return iterations, cost_history, best_history, [], []
        return iterations, cost_history, best_history, temp_history, accept_rate_history
    
    # 检测运行边界：当迭代次数重置（从大到小）时，说明开始新一轮
    all_runs = []
    current_run = []
    
    for match in matches:
        iteration, current_cost, best_cost, temp, accept_rate = match
        iter_num = int(iteration)
        
        # 如果迭代次数变小，说明开始了新一轮
        if current_run and iter_num <= current_run[-1][0]:
            all_runs.append(current_run)
            current_run = []
        
        current_run.append((iter_num, float(current_cost), float(best_cost), float(temp), float(accept_rate)))
    
    # 添加最后一轮
    if current_run:
        all_runs.append(current_run)
    
    # 只返回最后一次运行的数据（通常是最优的）
    if all_runs:
        last_run = all_runs[-1]
        iterations = [item[0] for item in last_run]
        cost_history = [item[1] for item in last_run]
        best_history = [item[2] for item in last_run]
        temp_history = [item[3] for item in last_run]
        accept_rate_history = [item[4] for item in last_run]
    
    return iterations, cost_history, best_history, temp_history, accept_rate_history

def visualize_results(dataset, solution, best_cost, output):
    """生成可视化图表 - 完整改进版（4子图布局）"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文
    except:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建2x2子图布局
    fig = plt.figure(figsize=(20, 12))
    
    # 提取数据
    iterations, cost_history, best_history, temp_history, accept_rate_history = extract_iteration_history(output)
    
    # ========== 子图1: 成本收敛曲线（左上） ==========
    ax1 = plt.subplot(2, 2, 1)
    
    if cost_history and best_history and iterations:
        # 计算移动平均，平滑曲线
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
        
        # 计算优化率
        initial_cost = cost_history[0]
        final_best = min(best_history)
        improvement_rate = (initial_cost - final_best) / initial_cost * 100 if initial_cost > 0 else 0
        
        # 绘制三条曲线
        ax1.plot(iterations, cost_history, color='steelblue', linewidth=1.0, alpha=0.2, label='探索成本（原始）')
        ax1.plot(iterations, cost_smooth, 'b-', linewidth=2.5, label='探索成本（平滑）', alpha=0.85)
        ax1.plot(iterations, best_history, 'r-', linewidth=2.8, label='历史最优成本', zorder=5)
        
        # 优化y轴范围
        min_cost = min(min(best_history), min(cost_smooth))
        max_cost = max(max(best_history), max(cost_smooth))
        y_range = max_cost - min_cost
        if y_range > 0:
            y_margin = y_range * 0.1
            ax1.set_ylim(min_cost - y_margin, max_cost + y_margin)
        
        # 标注关键点
        start_idx = 0
        end_idx = len(iterations) - 1
        min_idx = best_history.index(min(best_history))
        
        # 初始探索成本
        ax1.plot(iterations[start_idx], cost_smooth[start_idx], 'o', color='navy', markersize=8, zorder=6)
        ax1.annotate(f'初始解: {cost_smooth[start_idx]:.2f}',
                xy=(iterations[start_idx], cost_smooth[start_idx]),
                xytext=(15, 20), textcoords='offset points',
                fontsize=10, color='navy', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8, edgecolor='navy', linewidth=1.5),
                arrowprops=dict(arrowstyle='->', color='navy', lw=1.5))
        
        # 最终探索成本（可能比初始高）
        ax1.plot(iterations[end_idx], cost_smooth[end_idx], 's', color='darkblue', markersize=8, zorder=6)
        ax1.annotate(f'最终探索: {cost_smooth[end_idx]:.2f}',
                xy=(iterations[end_idx], cost_smooth[end_idx]),
                xytext=(15, -25), textcoords='offset points',
                fontsize=10, color='darkblue', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.8, edgecolor='darkblue', linewidth=1.5))
        
        # 最优解（红色五角星）
        ax1.plot(iterations[min_idx], best_history[min_idx], 'r*', markersize=22, zorder=7, 
                markeredgecolor='darkred', markeredgewidth=2.5)
        ax1.annotate(f'全局最优: {best_history[min_idx]:.2f}\n(优化率: {improvement_rate:.1f}%)', 
                xy=(iterations[min_idx], best_history[min_idx]),
                xytext=(20, 25), textcoords='offset points',
                fontsize=11, color='darkred', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='yellow', alpha=0.9, edgecolor='red', linewidth=2.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
        
        ax1.set_xlabel('迭代次数', fontsize=13, fontweight='bold')
        ax1.set_ylabel('成本', fontsize=13, fontweight='bold')
        ax1.set_title('ALNS成本收敛曲线\n（U形探索 + 单调最优）', fontsize=14, fontweight='bold', pad=15)
        ax1.legend(fontsize=11, loc='upper right', frameon=True, shadow=True, fancybox=True)
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
        
        # 添加说明文本
        explanation_text = (
            "说明：蓝色曲线（探索成本）呈U形是正常的\n"
            "算法会接受更差的解以跳出局部最优\n"
            "红色曲线（最优成本）只减不增"
        )
        ax1.text(0.02, 0.02, explanation_text, transform=ax1.transAxes,
                fontsize=9, verticalalignment='bottom', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    else:
        ax1.text(0.5, 0.5, '无迭代数据', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=14)
        ax1.set_title('ALNS成本收敛曲线', fontsize=14, fontweight='bold')
    
    # ========== 子图2: 温度与接受率（右上） ==========
    ax2 = plt.subplot(2, 2, 2)
    
    if temp_history and accept_rate_history:
        # 创建双y轴
        ax2_twin = ax2.twinx()
        
        # 绘制温度曲线（左y轴）
        line1 = ax2.plot(iterations, temp_history, 'orange', linewidth=2.5, label='温度', alpha=0.8)
        ax2.set_xlabel('迭代次数', fontsize=13, fontweight='bold')
        ax2.set_ylabel('温度 (T)', fontsize=13, fontweight='bold', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        
        # 绘制接受率曲线（右y轴，每10次迭代一个点）
        # 接受率对应的迭代点：第10, 20, 30, ... 次迭代
        accept_iters = [iterations[min(i*10, len(iterations)-1)] for i in range(len(accept_rate_history))]
        # 确保长度匹配
        accept_rate_to_plot = accept_rate_history[:len(accept_iters)]
        line2 = ax2_twin.plot(accept_iters, accept_rate_to_plot, 'green', linewidth=2.5, 
                              label='接受率', alpha=0.8, marker='o', markersize=4)
        ax2_twin.set_ylabel('接受率 (%)', fontsize=13, fontweight='bold', color='green')
        ax2_twin.tick_params(axis='y', labelcolor='green')
        ax2_twin.set_ylim(0, 1)
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, fontsize=11, loc='upper right', frameon=True, shadow=True)
        
        ax2.set_title('温度衰减与接受率变化\n（解释为何接受更差的解）', fontsize=14, fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
        
        # 添加说明
        avg_accept_rate = sum(accept_rate_to_plot)/len(accept_rate_to_plot) if accept_rate_to_plot else 0
        temp_explanation = (
            "说明：温度控制接受更差解的概率\n"
            f"初始温度: {temp_history[0]:.1f}\n"
            f"最终温度: {temp_history[-1]:.1f}\n"
            f"平均接受率: {avg_accept_rate*100:.1f}%"
        )
        ax2.text(0.02, 0.02, temp_explanation, transform=ax2.transAxes,
                fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    elif temp_history:
        # 只有温度数据
        ax2.plot(iterations, temp_history, 'orange', linewidth=2.5, label='温度')
        ax2.set_xlabel('迭代次数', fontsize=13, fontweight='bold')
        ax2.set_ylabel('温度 (T)', fontsize=13, fontweight='bold')
        ax2.set_title('温度衰减曲线', fontsize=14, fontweight='bold', pad=15)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, linestyle='--')
    else:
        ax2.text(0.5, 0.5, '无温度数据', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_title('温度与接受率', fontsize=14, fontweight='bold')
    
    # ========== 子图3: 车辆路线图（左下） ==========
    ax3 = plt.subplot(2, 2, 3)
    customers = dataset['customers']
    
    # 提取坐标
    depot = customers[0]
    depot_x, depot_y = depot['x'], depot['y']
    
    # 绘制仓库
    ax3.plot(depot_x, depot_y, 'rs', markersize=18, label='配送中心', zorder=5, 
            markeredgecolor='darkred', markeredgewidth=2.5)
    
    # 绘制客户点
    customer_coords = [(c['x'], c['y']) for c in customers[1:]]
    if customer_coords:
        cx, cy = zip(*customer_coords)
        ax3.scatter(cx, cy, c='blue', s=60, alpha=0.6, label='客户点', zorder=3, 
                   edgecolors='navy', linewidths=0.8)
    
    # 绘制路线
    if solution and len(solution) > 0:
        colors = plt.cm.tab20(np.linspace(0, 1, max(20, len(solution))))
        
        for idx, route in enumerate(solution):
            try:
                if not route or len(route) == 0:
                    continue
                
                # 构建完整路线
                complete_route = []
                if route[0] != 0:
                    complete_route.append(0)
                complete_route.extend(route)
                if route[-1] != 0:
                    complete_route.append(0)
                
                # 提取坐标
                route_x = []
                route_y = []
                for node in complete_route:
                    if node < len(customers):
                        route_x.append(customers[node]['x'])
                        route_y.append(customers[node]['y'])
                
                if len(route_x) >= 2:
                    color = colors[idx % len(colors)]
                    ax3.plot(route_x, route_y, '-', color=color, linewidth=2.8, alpha=0.75, 
                            zorder=2, label=f'路线{idx+1}')
                    
                    if len(route_x) > 2:
                        customer_x = route_x[1:-1]
                        customer_y = route_y[1:-1]
                        ax3.scatter(customer_x, customer_y, c=[color]*len(customer_x), s=90, 
                                   alpha=0.9, zorder=4, edgecolors='black', linewidths=1.8)
                    
            except Exception as e:
                print(f"[警告] 绘制路线 {idx+1} 时出错: {e}")
                continue
    
    ax3.set_xlabel('X 坐标', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Y 坐标', fontsize=13, fontweight='bold')
    route_count = len(solution) if solution else 0
    ax3.set_title(f'车辆路线规划图\n(共{route_count}条路线)', fontsize=14, fontweight='bold', pad=15)
    ax3.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1, 
              frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.axis('equal')
    
    # ========== 子图4: 多次运行统计（右下） ==========
    ax4 = plt.subplot(2, 2, 4)
    
    # 提取多次运行的成本
    multi_run_pattern = r'各次成本: \[(.*?)\]'
    multi_run_match = re.search(multi_run_pattern, output)
    
    if multi_run_match:
        costs_str = multi_run_match.group(1)
        costs = [float(c.strip().strip("'\"")) for c in costs_str.split(',')]
        
        if len(costs) > 1:
            # 使用箱线图 + 散点图
            positions = [1]
            bp = ax4.boxplot([costs], positions=positions, widths=0.6, patch_artist=True,
                            boxprops=dict(facecolor='lightblue', alpha=0.7, linewidth=2),
                            medianprops=dict(color='red', linewidth=2.5),
                            whiskerprops=dict(linewidth=1.5),
                            capprops=dict(linewidth=1.5))
            
            # 叠加散点
            x_positions = np.random.normal(1, 0.04, size=len(costs))
            colors_scatter = ['red' if c == min(costs) else 'darkgreen' if c == max(costs) else 'gray' 
                             for c in costs]
            ax4.scatter(x_positions, costs, c=colors_scatter, s=120, alpha=0.8, zorder=3, 
                       edgecolors='black', linewidths=1.5)
            
            # 统计信息
            min_cost = min(costs)
            max_cost = max(costs)
            mean_cost = sum(costs) / len(costs)
            std_cost = (sum((c - mean_cost)**2 for c in costs) / len(costs)) ** 0.5
            
            ax4.axhline(y=min_cost, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'最优: {min_cost:.2f}')
            ax4.axhline(y=mean_cost, color='blue', linestyle='--', linewidth=2, alpha=0.7, label=f'均值: {mean_cost:.2f}')
            
            ax4.set_ylabel('总成本', fontsize=13, fontweight='bold')
            ax4.set_title(f'多次运行成本分布统计\n({len(costs)}次运行)', fontsize=14, fontweight='bold', pad=15)
            ax4.set_xticks([1])
            ax4.set_xticklabels([f'{len(costs)}次运行'])
            ax4.legend(fontsize=11, loc='upper right', frameon=True, shadow=True)
            ax4.grid(True, axis='y', alpha=0.3, linestyle='--')
            
            # 添加统计信息文本
            stats_text = (
                f"最优成本: {min_cost:.2f}\n"
                f"最差成本: {max_cost:.2f}\n"
                f"平均成本: {mean_cost:.2f}\n"
                f"标准差: {std_cost:.2f}\n"
                f"变异系数: {std_cost/mean_cost*100:.1f}%"
            )
            ax4.text(0.98, 0.02, stats_text, transform=ax4.transAxes,
                    fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7, edgecolor='darkgreen', linewidth=1.5))
        else:
            # 单次运行，使用条形图
            runs = ['运行1']
            colors_bar = ['#FF6B6B']
            bars = ax4.bar(runs, costs, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=2)
            ax4.set_ylabel('总成本', fontsize=13, fontweight='bold')
            ax4.set_title('单次运行成本', fontsize=14, fontweight='bold', pad=15)
            ax4.grid(True, axis='y', alpha=0.3, linestyle='--')
            
            for bar, cost in zip(bars, costs):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{cost:.1f}',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, '无多次运行数据', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        ax4.set_title('多次运行统计', fontsize=14, fontweight='bold')
    
    plt.tight_layout(pad=2.0)
    
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

