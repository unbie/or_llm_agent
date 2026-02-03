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
        print("LLM Output: ", end="", flush=True)

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
        print(f"\n[API Error] 调用出错: {e}")
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
        print("\n[处理] 提取 HeuristicPlugin 类...")
        
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
            
            print(f"[提取] 重构了 HeuristicPlugin 类（{len(plugin_class_code)} 字符）")
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
            
            print(f"[提取] 手动构建 HeuristicPlugin 类（{len(plugin_class_code)} 字符）")

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
            "        print('Plugin 初始化成功')\n"
            "        solver = HeuristicSolver(data, plugin)\n"
            "        print('Solver 初始化成功')\n"
            "        best_sol, best_cost = solver.solve_multi_run(max_iters=150, num_runs=5, base_seed=42)\n"
            "        print(f'BEST_COST: {best_cost}')\n"
            "        print(f'BEST_SOLUTION: {best_sol}')\n"
            "    except Exception as e:\n"
            "        print(f'Runtime Error: {e}')\n"
            "        traceback.print_exc(file=sys.stdout)\n"
        )

        # === 验证代码结构 ===
        solver_count = full_code.count('class HeuristicSolver')
        plugin_count = full_code.count('class HeuristicPlugin')
        init_debug = '[初始化]' in full_code
        
        print(f"[验证] Solver类: {solver_count}, Plugin类: {plugin_count}, 调试信息: {init_debug}")
        
        if solver_count != 1 or plugin_count != 1:
            print(f"[警告] 类定义数量异常！Solver={solver_count}, Plugin={plugin_count}")
        
     
        # === 执行代码 ===
        success, result_msg = extract_and_execute_python_code(f"```python\n{full_code}\n```")

        if success and "BEST_COST:" in result_msg:
            print("\n=== 求解成功 ===")
            return True, result_msg, messages_bak

        print(f"\n[Attempt {attempt+1} Failed]\n错误日志：\n{result_msg}\n")
        messages.append({"role": "assistant", "content": llm_response})
        messages.append({
            "role": "user", 
            "content": f"代码执行报错，请修复。\n错误信息：\n{result_msg}\n\n注意：只需要实现 random_removal, worst_removal, greedy_insert 三个方法！"
        })
        attempt += 1

    return False, None, messages_bak

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

    print_header("生鲜物流 ALNS 求解")
    
    is_solve_success, output, _ = generate_or_code_solver(
        messages_bak, model_name, dataset, max_attempts=5
    )

    if is_solve_success:
        print("\n=== 求解成功 ===\n", output)
    else:
        print("\n=== 求解失败 ===")

