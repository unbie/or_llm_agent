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
    eval_model_result
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
    api_key="0a1ccd3d-2e96-4770-9e70-ace5d0c5bd66",
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


# def query_llm(messages, model_name="ep-20251202173916-9j664", temperature=0.2):
#     """
#     调用 LLM 获取响应结果，使用流式输出方式。
#
#     Args:
#         messages (list): 对话上下文列表。
#         model_name (str): LLM模型名称，默认为"gpt-4"。
#         temperature (float): 控制输出的随机性，默认为 0.2。
#
#     Returns:
#         str: LLM 生成的响应内容。
#     """
#     # 使用stream=True启用流式输出
#     response = client.chat.completions.create(
#         model=model_name,
#         messages=messages,
#         temperature=temperature,
#         stream=True
#     )
#
#     # 用于累积完整响应
#     full_response = ""
#
#     # 用于控制打印格式
#     print("LLM Output: ", end="", flush=True)
#
#     # 逐块处理流式响应
#     for chunk in response:
#         # 首先检查choices列表是否非空
#         if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
#             # 然后检查是否有delta和content
#             if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
#                 content = chunk.choices[0].delta.content
#                 if content:
#                     print(content, end="", flush=True)
#                     full_response += content
#
#     # 输出完成后换行
#     print()
#
#     return full_response
def query_llm(messages, model_name="ep-20260106214023-k4p8b", temperature=0.2):
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


# def generate_or_code_solver(messages_bak, model_name, max_attempts):
#     messages = copy.deepcopy(messages_bak)
#
#     print_header("LLM生成Python Gurobi 代码")
#
#     gurobi_code = query_llm(messages, model_name)
#
#     print_header("自动执行python代码")
#     # 4. 代码执行 & 修复
#     text = f"{gurobi_code}"
#     attempt = 0
#     while attempt < max_attempts:
#         buffer2 = io.StringIO()
#         with redirect_stdout(buffer2):
#             success, error_msg = extract_and_execute_python_code(text)
#         captured_output2 = buffer2.getvalue()
#         for c in captured_output2:
#             print(c, end="", flush=True)
#             time.sleep(0.005)
#
#         if success:
#             messages_bak.append({"role": "assistant", "content": gurobi_code})
#             return True, error_msg, messages_bak
#
#         print(f"\n第 {attempt + 1} 次尝试失败，请求 LLM 修复代码...\n")
#
#         # 构建修复请求
#         messages.append({"role": "assistant", "content": gurobi_code})
#         messages.append({"role": "user",
#                          "content": f"代码执行出现错误，错误信息如下:\n{error_msg}\n请修复代码并重新提供完整的可执行代码。"})
#
#         # 获取修复后的代码
#         gurobi_code = query_llm(messages, model_name)
#         text = f"{gurobi_code}"
#
#         print("\n获取到修复后的代码，准备重新执行...\n")
#         attempt += 1
#     # not add gurobi code
#     messages_bak.append({"role": "assistant", "content": gurobi_code})
#     print(f"达到最大尝试次数 ({max_attempts})，未能成功执行代码。")
#     return False, None, messages_bak
#


def generate_or_code_solver(messages_bak, model_name, data, max_attempts=3):
    messages = copy.deepcopy(messages_bak)

    # 1. 【动态描述】自动分析数据结构，而不是硬编码键名
    # 获取第一个客户节点的样本，以此展示数据格式
    customer_sample = data['customers'][0] if data['customers'] else {}
    data_schema = {k: type(v).__name__ for k, v in customer_sample.items()}

    print_header(f"LLM 启发式生成 - 动态数据适配模式")

    # 2. 构造通用的 Prompt
    # 我们不再说“不要用 time_window”，而是说“请严格参考以下 Schema”
    prompt = (
        "请编写一个完整的 Python 脚本来解决 VRPTW 问题。请严格基于提供的骨架进行补全。\n\n"
        "### 1. 运行环境与注入：\n"
        "- 代码第一行必须是：# -*- coding: utf-8 -*-\n"
        "- 全局变量 `data` 已注入。`data['customers']` 包含所有节点。\n\n"
        "### 2. 动态数据结构参考 (Data Schema)：\n"
        f"每个客户节点（customer node）的键值对结构如下，请在编写逻辑时准确引用这些键：\n"
        f"{json.dumps(customer_sample, indent=2, ensure_ascii=False)}\n"
        f"对应数据类型参考: {data_schema}\n\n"
        "### 3. 任务要求：\n"
        f"请补全 `HeuristicPlugin` 类中的所有 TODO：\n"
        f"{HEURISTIC_PLUGIN_TEMPLATE}\n\n"
        "### 4. 编写指南：\n"
        "- **强制排序插入 (EDD)**：在 `greedy_insert` 中，**必须**先将 `removed_nodes` 按其 `due_date` 升序排列。\n"
        "- **彻底清理空路径**：在破坏算子（removal）移除节点后，如果路径只剩下仓库 `[0, 0]`，**必须立即丢弃该路径**。\n"
        "- 必须遍历 `range(1, len(data['customers']))` 以处理所有客户。\n"
        "仅输出类代码块，包裹在 ```python ... ``` 中。"
    )

    messages.append({"role": "user", "content": prompt})

    # 3. 自动注入代码逻辑（保持不变）
    full_data_json = json.dumps(data, ensure_ascii=True)
    full_data_code = f"data = {full_data_json}"

    attempt = 0
    while attempt < max_attempts:
        llm_response = query_llm(messages, model_name)
        code_match = re.search(r"```python\n(.*?)```", llm_response, re.DOTALL)
        llm_plugin_code = code_match.group(1).strip() if code_match else llm_response

        # 拼接最终执行脚本
        final_code_to_run = (
            "# -*- coding: utf-8 -*-\n"
            "import math, json, random, copy, time\n\n"
            f"{full_data_code}\n\n"
            f"{HEURISTIC_SKELETON}\n\n"
            f"{llm_plugin_code}\n\n"
            "if __name__ == '__main__':\n"
            "    try:\n"
            "        plugin = HeuristicPlugin(data)\n"
            "        solver = HeuristicSolver(data, plugin)\n"
            "        best_sol, best_cost = solver.solve(max_iters=200)\n"
            "        print(f'BEST_COST: {best_cost}')\n"
            "        print(f'BEST_SOLUTION: {best_sol}')\n"
            "    except Exception as e:\n"
            # 捕获异常并打印详细 traceback，让 LLM 自己看哪里错了
            "        import traceback\n"
            "        traceback.print_exc()\n"
        )

        formatted_for_exec = f"```python\n{final_code_to_run}\n```"
        success, result_msg = extract_and_execute_python_code(formatted_for_exec)

        if success and "BEST_COST:" in result_msg:
            messages_bak.append({"role": "assistant", "content": llm_response})
            return True, result_msg, messages_bak

        # 4. 【通用反馈】不再硬编码错误提示，而是直接把 Traceback 喂给 LLM
        print(f"\n[执行失败] 正在将错误堆栈反馈给 LLM...\n")
        feedback = (
            f"执行过程中出现错误。请阅读以下 Traceback 并参考前述 Schema 进行修正：\n"
            f"```\n{result_msg}\n```\n"
            "请重新检查你对 data['customers'] 属性的访问是否正确。"
        )

        messages.append({"role": "assistant", "content": llm_response})
        messages.append({"role": "user", "content": feedback})
        attempt += 1

    return False, None, messages_bak

# def or_llm_agent(user_question, model_name="ep-20251202173916-9j664", max_attempts=3):
#     """
#     向 LLM 请求 Gurobi 代码解决方案并执行，如果失败则尝试修复。
#
#     Args:
#         user_question (str): 用户的问题描述。
#         model_name (str): 使用的 LLM 模型名称，默认为"gpt-4"。
#         max_attempts (int): 最大尝试次数，默认为3。
#
#     Returns:
#         tuple: (success: bool, best_objective: float or None, final_code: str)
#     """
#     # 初始化对话记录
#     messages = [
#         {"role": "system", "content": (
#             "你是一个运筹优化专家。请根据用户提供的运筹优化问题构建数学模型，以数学（线性规划）模型对原问题进行有效建模。"
#             "尽量关注获得一个正确的数学模型表达式，无需太关注解释。"
#             "该模型后续用作指导生成gurobi代码，这一步主要用作生成有效的线性规模表达式。"
#         )},
#         {"role": "user", "content": user_question}
#     ]
#
#     # 1. 生成数学模型
#     print_header("LLM推理构建线性规划模型")
#     math_model = query_llm(messages, model_name)
#     # print("【数学模型】:\n", math_model)
#
#     # # 2. 校验数学模型
#     # messages.append({"role": "assistant", "content": math_model})
#     # messages.append({"role": "user", "content": (
#     #     "请基于上面的数学模型是否符合问题描述，如果存在错误，则进行修正；如果不存在错误则检查是否能进行优化。"
#     #     "无论何种情况，最终请重新输出该数学模型。"
#     # )})
#
#     # validate_math_model = query_llm(messages, model_name)
#     # print("【校验后的数学模型】:\n", validate_math_model)
#
#     validate_math_model = math_model
#     messages.append({"role": "assistant", "content": validate_math_model})
#
#     # ------------------------------
#     messages.append({"role": "user", "content": (
#         "请基于以上的数学模型，写出完整、可靠的 Python 代码，使用 Gurobi 求解该运筹优化问题。"
#         "代码中请包含必要的模型构建、变量定义、约束添加、目标函数设定以及求解和结果输出。"
#         "以 ```python\n{code}\n``` 形式输出，无需输出代码解释。"
#     )})
#     # copy msg; solve; add the laset gurobi code
#     is_solve_success, result, messages = generate_or_code_solver(messages, model_name, max_attempts)
#     print(f'Stage result: {is_solve_success}, {result}')
#     if is_solve_success:
#         if not is_number_string(result):
#             print('!![No available solution warning]!!')
#             # no solution
#             messages.append({"role": "user", "content": (
#                 "现有模型运行结果为*无可行解*，请认真仔细地检查数学模型和gurobi代码，是否存在错误，以致于造成无可行解"
#                 "检查完成后，最终请重新输出gurobi python代码"
#                 "以 ```python\n{code}\n``` 形式输出，无需输出代码解释。"
#             )})
#             is_solve_success, result, messages = generate_or_code_solver(messages, model_name, max_attempts=1)
#     else:
#         print('!![Max attempt debug error warning]!!')
#         messages.append({"role": "user", "content": (
#             "现在模型代码多次调试仍然报错，请认真仔细地检查数学模型是否存在错误"
#             "检查后最终请重新构建gurobi python代码"
#             "以 ```python\n{code}\n``` 形式输出，无需输出代码解释。"
#         )})
#         is_solve_success, result, messages = generate_or_code_solver(messages, model_name, max_attempts=2)
#
#     return is_solve_success, result
#
def generate_or_code_solver(messages_bak, model_name, data, max_attempts=3):
    messages = copy.deepcopy(messages_bak)

    # 1. 【动态描述】自动分析数据结构，而不是硬编码键名
    # 获取第一个客户节点的样本，以此展示数据格式
    customer_sample = data['customers'][0] if data['customers'] else {}
    data_schema = {k: type(v).__name__ for k, v in customer_sample.items()}

    print_header(f"LLM 启发式生成")

    # 2. 构造通用的 Prompt
    prompt = (
        "请编写一个完整的 Python 脚本来解决 VRPTW 问题。请严格基于提供的骨架进行补全。\n\n"
        "### 1. 运行环境与注入：\n"
        "- 代码第一行必须是：# -*- coding: utf-8 -*-\n"
        "- 全局变量 `data` 已注入。`data['customers']` 包含所有节点。\n\n"
        "### 2. 动态数据结构参考 (Data Schema)：\n"
        f"每个客户节点（customer node）的键值对结构如下，请在编写逻辑时准确引用这些键：\n"
        f"{json.dumps(customer_sample, indent=2, ensure_ascii=False)}\n"
        f"对应数据类型参考: {data_schema}\n\n"
        "### 3. 任务要求：\n"
        f"请补全 `HeuristicPlugin` 类中的所有 TODO：\n"
        f"{HEURISTIC_PLUGIN_TEMPLATE}\n\n"
        "### 4. 编写指南：\n"
        "- 必须遍历 `range(1, len(data['customers']))` 以处理所有客户。\n"
        "- 严禁使用中文字符，结尾打印 'BEST_COST' 和 'BEST_SOLUTION'。\n"
        "仅输出类代码块，包裹在 ```python ... ``` 中。"
    )

    messages.append({"role": "user", "content": prompt})

    # 3. 自动注入代码逻辑（保持不变）
    full_data_json = json.dumps(data, ensure_ascii=True)
    full_data_code = f"data = {full_data_json}"

    attempt = 0
    while attempt < max_attempts:
        llm_response = query_llm(messages, model_name)
        code_match = re.search(r"```python\n(.*?)```", llm_response, re.DOTALL)
        llm_plugin_code = code_match.group(1).strip() if code_match else llm_response

        # 拼接最终执行脚本
        final_code_to_run = (
            "# -*- coding: utf-8 -*-\n"
            "import math, json, random, copy, time\n\n"
            f"{full_data_code}\n\n"
            f"{HEURISTIC_SKELETON}\n\n"
            f"{llm_plugin_code}\n\n"
            "if __name__ == '__main__':\n"
            "    try:\n"
            "        plugin = HeuristicPlugin(data)\n"
            "        solver = HeuristicSolver(data, plugin)\n"
            "        best_sol, best_cost = solver.solve(max_iters=200)\n"
            "        print(f'BEST_COST: {best_cost}')\n"
            "        print(f'BEST_SOLUTION: {best_sol}')\n"
            "    except Exception as e:\n"
            # 捕获异常并打印详细 traceback，让 LLM 自己看哪里错了
            "        import traceback\n"
            "        traceback.print_exc()\n"
        )

        formatted_for_exec = f"```python\n{final_code_to_run}\n```"
        success, result_msg = extract_and_execute_python_code(formatted_for_exec)

        if success and "BEST_COST:" in result_msg:
            messages_bak.append({"role": "assistant", "content": llm_response})
            return True, result_msg, messages_bak

        # 4. 【通用反馈】不再硬编码错误提示，而是直接把 Traceback 喂给 LLM
        print(f"\n[执行失败] 正在将错误堆栈反馈给 LLM...\n")
        feedback = (
            f"执行过程中出现错误。请阅读以下 Traceback 并参考前述 Schema 进行修正：\n"
            f"```\n{result_msg}\n```\n"
            "请重新检查你对 data['customers'] 属性的访问是否正确。"
        )

        messages.append({"role": "assistant", "content": llm_response})
        messages.append({"role": "user", "content": feedback})
        attempt += 1

    return False, None, messages_bak


#
#
# def load_dataset(data_path):
#     """
#     Load dataset from either JSONL format (IndustryOR.json, BWOR.json) or regular JSON format
#     """
#     dataset = {}
#
#     with open(data_path, 'r', encoding='utf-8') as f:
#         # Try to detect format by reading first line
#         first_line = f.readline().strip()
#         f.seek(0)  # Reset file pointer
#
#         if first_line.startswith('{"en_question"') or first_line.startswith('{"cn_question"'):
#             # JSONL format (IndustryOR.json, BWOR.json)
#             for line_num, line in enumerate(f, 1):
#                 line = line.strip()
#                 if line:
#                     try:
#                         item = json.loads(line)
#                         # Convert to expected format
#                         dataset_item = {
#                             'question': item.get('en_question', item.get('cn_question', '')),
#                             'answer': item.get('en_answer', item.get('cn_answer', '')),
#                             'difficulty': item.get('difficulty', 'Unknown'),
#                             'id': item.get('id', line_num - 1)
#                         }
#                         # Use id as string key
#                         dataset[str(dataset_item['id'])] = dataset_item
#                     except json.JSONDecodeError as e:
#                         print(f"Warning: Could not parse line {line_num}: {line}")
#                         continue
#         else:
#             # Regular JSON format (legacy)
#             dataset = json.load(f)
#
#     return dataset
#
def load_solomon_data(file_path):
    """
    读取 Solomon Benchmark txt 文件（C101 结构），返回字典：
    {
        "vehicle_capacity": 200,
        "customers": [
            {"id": 0, "demand": 0, "x": 40, "y": 50, "ready_time":0, "due_date":1236, "service_time":0},
            ...
        ]
    }
    """
    data = {}
    customers = []
    vehicle_capacity = None

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 先提取车辆容量
    for idx, line in enumerate(lines):
        line = line.strip()
        if line == "" or line.startswith("C") or line.startswith("VEHICLE") or line.startswith(
                "NUMBER") or line.startswith("CUSTOMER"):
            continue
        parts = line.split()
        # 第一个找到的行，车辆信息行
        if len(parts) == 2 and vehicle_capacity is None:
            # VEHICLE CAPACITY
            vehicle_capacity = int(parts[1])
            break
    if vehicle_capacity is None:
        vehicle_capacity = 200  # 默认值

    # 提取客户数据：行首是数字且长度>=7
    for line in lines:
        line = line.strip()
        if line == "" or line.startswith("C") or line.startswith("VEHICLE") or line.startswith(
                "NUMBER") or line.startswith("CUSTOMER") or line.startswith("CUST"):
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
                "id": cust_id,
                "demand": demand,
                "x": x,
                "y": y,
                "ready_time": ready_time,
                "due_date": due_date,
                "service_time": service_time
            })

    data["vehicle_capacity"] = vehicle_capacity
    data["customers"] = customers
    return data


if __name__ == "__main__":
    # # Import the load_dataset function from the async script
    # import sys
    # import os
    #
    # sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    #
    # dataset = load_dataset('data/datasets/IndustryOR.json')
    # # print(dataset['0'])
    # console = Console()
    #
    # model_name = 'ep-20251202173916-9j664'
    # # model_name = ''
    #
    # # model_name = 'Pro/deepseek-ai/DeepSeek-R1'
    # # model_name = 'deepseek-reasoner'
    #
    # pass_count = 0
    # correct_count = 0
    # for i, d in dataset.items():
    #     # print(i)
    #     # if int(i) in [0]:
    #     print_header("运筹优化问题")
    #     user_question, answer = d['question'], d['answer']
    #     # print(user_question)
    #     buffer2 = io.StringIO()
    #     with redirect_stdout(buffer2):
    #         md = Markdown(user_question)
    #         console.print(md)
    #         print('-------------')
    #
    #     captured_output2 = buffer2.getvalue()
    #     for c in captured_output2:
    #         print(c, end="", flush=True)
    #         time.sleep(0.005)
    #     is_solve_success, best_solution = or_llm_heuristic_agent(user_question, model_name)
    #     # is_solve_success, llm_result = gpt_code_agent_simple(user_question, model_name)
    #     if is_solve_success:
    #         print(f"成功执行代码，最优启发式成本: {best_solution}")
    #     else:
    #         print("执行代码失败。")
    import sys
    import os
    import io
    import time
    from rich.console import Console
    from rich.markdown import Markdown
    from contextlib import redirect_stdout

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    console = Console()
    model_name = 'ep-20260106214023-k4p8b'
    messages_bak = []

    # 指定 Solomon Benchmark 文件路径
    solomon_file = r"D:\pythonProject\or_llm_agent\data\1 Solomon Benchmark\c1\c101.txt"

    # 加载算例数据
    dataset = load_solomon_data(solomon_file)  # 返回字典格式

    # 打印问题描述
    print_header("生鲜物流问题")
    buffer2 = io.StringIO()
    with redirect_stdout(buffer2):
        md = Markdown(f"### VRPSTWSDBCCFLC 算例: {os.path.basename(solomon_file)}")
        console.print(md)
        print('------------------------------------')
    captured_output2 = buffer2.getvalue()
    for c in captured_output2:
        print(c, end="", flush=True)
        time.sleep(0.005)

    # 调用启发式求解器
    is_solve_success, best_solution_output, messages_bak = generate_or_code_solver(
        messages_bak, model_name, dataset, max_attempts=3
    )

    # 输出结果
    if is_solve_success:
        print("成功执行启发式算法，结果如下：")
        print(best_solution_output)
    else:
        print("启发式算法执行失败。")
