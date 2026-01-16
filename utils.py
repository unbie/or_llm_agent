import re
import subprocess
import sys
import tempfile
import os

# util.py
import math


class FreshnessAndPenaltyCalculator:
    def __init__(self, config):
        # --- C1 基础参数 ---
        self.f = config.get("vehicle_fixed_cost", 240)  # C11: 每辆车固定成本
        self.c = config.get("vehicle_distance_cost_per_km", 3)  # C12: 单位距离成本
        self.ct = config.get("cooling_cost_per_hour", 15)  # C13: 单位时间制冷成本
        self.v = config.get("vehicle_speed_kmph", 40)  # 速度 v

        # --- C2 货损参数 ---
        self.p = config.get("product_price_per_ton", 5000)  # 单价 p
        self.theta1 = config.get("theta_transport", 0.002)  # θ1
        self.theta2 = config.get("theta_service", 0.005)  # θ2
        self.delta1 = config.get("customer_loss_threshold", 0.02)  # δ1

        # --- C3 惩罚参数 ---
        self.z1 = config.get("early_penalty_per_hour", 20)  # Z1
        self.z2 = config.get("late_penalty_per_hour", 40)  # Z2

    def calculate_route_cost(self, route_nodes, dist_matrix):
        """
        计算单条路径的变动成本 (C12 + C13 + C2 + C3)
        注意：固定成本 C11 在 Plugin 层级统一根据路径数计算
        """
        route_dist = 0.0
        c2_freshness = 0.0
        c3_penalty = 0.0

        curr_time = 0.0  # t_ik (分钟)
        cum_service_h = 0.0  # t'_ik (累计装卸小时)

        for i in range(1, len(route_nodes)):
            prev = route_nodes[i - 1]
            curr = route_nodes[i]

            # --- 1. 物理计算 ---
            d = dist_matrix[prev['id']][curr['id']]
            route_dist += d
            drive_min = d * (60.0 / self.v)
            curr_time += drive_min  # 到达时刻 t_ik

            # --- 2. 货损成本 C2 (式4, 式6) ---
            if curr['id'] != 0:
                tik_h = curr_time / 60.0
                # r_i = 1 - exp(-theta1*(tik - t'ik) - theta2*t'ik)
                ri = 1 - math.exp(-self.theta1 * (tik_h - cum_service_h) - self.theta2 * cum_service_h)
                # Di * max(ri - delta1, 0)
                c2_freshness += self.p * curr['demand'] * max(ri - self.delta1, 0)

            # --- 3. 时间惩罚 C3 (式7) ---
            if curr['id'] != 0:
                fi_t = 0.0
                ei, li = curr['ready_time'], curr['due_date']
                Ei, Li = curr['E_i'], curr['L_i']

                if Ei <= curr_time < ei:
                    fi_t = self.z1 * (ei - curr_time) / (ei - Ei + 1e-6)
                elif li < curr_time <= Li:
                    fi_t = self.z2 * (curr_time - li) / (Li - li + 1e-6)
                elif curr_time < Ei or curr_time > Li:
                    fi_t = 300.0  # 极高惩罚项
                c3_penalty += fi_t

            # --- 4. 状态更新 ---
            # 离开时刻逻辑
            curr_time = max(curr_time, curr['ready_time']) + curr['service_time']
            cum_service_h += curr['service_time'] / 60.0

        # C12: 距离成本
        c12 = route_dist * self.c
        # C13: 制冷成本 (基于总行驶时长)
        total_drive_h = (curr_time / 60.0) - cum_service_h
        c13 = total_drive_h * self.ct

        return {
            "variable_cost": c12 + c13 + c2_freshness + c3_penalty,  # C12+C13+C2+C3
            "c2": c2_freshness,
            "c3": c3_penalty,
            "dist": route_dist
        }

def is_number_string(s):
    """
    Determine if a string is a numeric string, including integers and decimals.

    Args:
    s: The string to be checked.

    Returns:
    True if the string is a numeric string, otherwise False.
    """
    pattern = r"^[-+]?\d+(\.\d+)?$"  # Regular expression to match integers or decimals
    return re.match(pattern, s) is not None


def convert_to_number(s):
    """
    Convert a string to a number (integer or float).

    Args:
        s: The string to be converted.

    Returns:
        int or float: Returns int if the string represents an integer, float if it represents a decimal.
        Returns None if conversion fails.
    """
    try:
        # Try to convert to integer
        if s.isdigit() or (s.startswith('-') and s[1:].isdigit()):
            return int(s)
        # Try to convert to float
        num = float(s)
        return num
    except (ValueError, TypeError):
        return None


def extract_best_objective(output_text):
    """
    Extract Best objective or Optimal objective value from Gurobi output.

    Args:
        output_text: Gurobi output text

    Returns:
        float or None: Optimal solution value, returns None if not found
    """
    # First check if model is infeasible
    if "Model is infeasible" in output_text:
        return None

    # Try to find Best objective
    match = re.search(r'Best objective\s+([\d.e+-]+)', output_text)
    if not match:
        # If not found, try to find Optimal objective
        match = re.search(r'Optimal objective\s+([\d.e+-]+)', output_text)

    if not match:
        # If not found, try to find Optimal cost
        match = re.search(r'Optimal cost\s+([\d.e+-]+)', output_text)

    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None

    return None


# def extract_and_execute_python_code(text_content):
#     """
#     Extract Python code blocks from text and execute them.
#
#     Args:
#         text_content: Text content containing code blocks.
#
#     Returns:
#         bool: True if execution was successful, False otherwise
#         str: Error message if execution failed, best objective if successful
#     """
#     python_code_blocks = re.findall(r'```python\s*([\s\S]*?)```', text_content)
#
#     if not python_code_blocks:
#         print("No Python code blocks found.")
#         return False, "No Python code blocks found"
#
#     for code_block in python_code_blocks:
#         code_block = code_block.strip()
#         if not code_block:
#             print("Found an empty Python code block, skipped.")
#             continue
#
#         print("Found Python code block, starting execution...")
#         try:
#             with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
#                 tmp_file.write(code_block)
#                 temp_file_path = tmp_file.name
#
#             result = subprocess.run([sys.executable, temp_file_path], capture_output=True, text=True, check=False)
#
#             if result.returncode == 0:
#                 print("Python code executed successfully, output:\n")
#                 print(result.stdout)
#
#                 # best_obj = extract_best_objective(result.stdout)
#                 # if best_obj is not None:
#                 #     print(f"\nOptimal solution value (Best objective): {best_obj}")
#                 # else:
#                 #     print("\nOptimal solution value not found")
#                 # return True, result.stdout
#
#                 if result.returncode == 0:
#                     print("Python code executed successfully.")
#                     # 这里是关键：在返回前尝试提取 objective 值
#                     best_obj = extract_best_objective(result.stdout)
#                     if best_obj is not None:
#                         # 如果提取到了数字，直接返回这个数字的字符串，方便后续 eval_model_result
#                         return True, str(best_obj)
#                     else:
#                         # 如果没提取到数字但运行成功，返回原始输出
#                         return True, result.stdout
#
#             else:
#                 print(f"Python code execution error, error message:\n")
#                 print(result.stderr)
#                 return False, result.stderr
#
#         except Exception as e:
#             print(f"Error occurred while executing Python code block: {e}")
#             return False, str(e)
#         finally:
#             if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
#                 os.remove(temp_file_path)
#         print("-" * 30)
#
#     return False, "No valid code blocks executed"
def extract_and_execute_python_code(text_content):
    """
    Extract Python code blocks from text and execute them.

    Args:
        text_content: Text content containing code blocks.

    Returns:
        bool: True if execution was successful, False otherwise
        str: Error message if execution failed, best objective if successful
    """
    python_code_blocks = re.findall(r'```python\s*([\s\S]*?)```', text_content)

    if not python_code_blocks:
        # 增加一个兜底：如果没有 Markdown 标记，尝试执行整个文本（防止 LLM 没带标记）
        if "import" in text_content and "def" in text_content:
            python_code_blocks = [text_content]
        else:
            return False, "No Python code blocks found"

    for code_block in python_code_blocks:
        code_block = code_block.strip()
        if not code_block: continue

        print("Found Python code block, starting execution...")
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding='utf-8') as tmp_file:
                tmp_file.write(code_block)
                temp_file_path = tmp_file.name

            result = subprocess.run([sys.executable, temp_file_path], capture_output=True, text=True, encoding='utf-8', check=False)

            if result.returncode == 0:
                print("Python code executed successfully.")
                # 在返回前尝试提取 objective 值
                best_obj = extract_best_objective(result.stdout)
                if best_obj is not None:
                    # 如果提取到了数字，直接返回这个数字的字符串
                    return True, str(best_obj)
                else:
                    # 如果没提取到数字但运行成功，返回原始输出
                    return True, result.stdout
            else:
                return False, result.stderr

        except Exception as e:
            return False, str(e)
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    return False, "No valid code blocks executed"


def eval_model_result(success, result, ground_truth, err_range=0.1):
    pass_flag = False
    correct_flag = False
    if success:
        pass_flag = True
        if is_number_string(str(result)) and ground_truth is not None:
            result_num = convert_to_number(str(result))
            ground_truth_num = convert_to_number(str(ground_truth))
            # Check that both conversions were successful before comparing
            if result_num is not None and ground_truth_num is not None:
                if abs(result_num - ground_truth_num) < err_range:
                    correct_flag = True
        elif result == 'None':  # no available solution
            if ground_truth is None or ground_truth == 'None':
                correct_flag = True
    return pass_flag, correct_flag 