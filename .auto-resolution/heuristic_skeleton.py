HEURISTIC_SKELETON = r"""
import math
import random
import copy
import time

# ==========================================
# 启发式求解器框架 (Driver)
# ==========================================

class HeuristicSolver:
    def __init__(self, data, plugin):
        self.data = data
        self.plugin = plugin
        
        # 预计算距离矩阵，供 Plugin 和 Solver 使用
        self.customers = data['customers']
        self.n = len(self.customers)
        self.dist_matrix = [[0]*self.n for _ in range(self.n)]
        for i in range(self.n):
            xi, yi = self.customers[i]['x'], self.customers[i]['y']
            for j in range(self.n):
                xj, yj = self.customers[j]['x'], self.customers[j]['y']
                self.dist_matrix[i][j] = math.hypot(xi - xj, yi - yj)
        
        # 将距离矩阵注入插件（方便插件计算成本）
        self.plugin.dist_matrix = self.dist_matrix

    def construct_initial_solution(self):
        solution = []
        for c in self.customers:
            cid = c['id']
            if cid == 0: # 跳过仓库
                continue
            # 路径格式：[仓库, 客户, 仓库]
            solution.append([0, cid, 0])
        return solution

    def solve(self, max_iters=200):
        # 1. 生成初始解
        current_solution = self.construct_initial_solution()
        best_solution = copy.deepcopy(current_solution)
        
        current_cost = self.plugin.cost(current_solution)
        best_cost = current_cost
        
        # 2. ALNS 参数
        T = current_cost * 0.05  # 初始温度
        alpha = 0.995            # 降温系数
        
        # 3. 迭代主循环
        for it in range(max_iters):
            # 3.1 破坏 (Destroy)
            # 传入当前的 destroy 算子权重，由插件内部选择算子并记录
            partial_solution, removed_nodes = self.plugin.destroy(current_solution)
            
            # 3.2 修复 (Insert)
            # 传入当前的 insert 算子权重，由插件内部选择算子并记录
            new_solution = self.plugin.insert(partial_solution, removed_nodes)
            
            # 3.3 验证可行性
            if not self.plugin.validate(new_solution):
                continue
            
            # 3.4 计算成本与接受准则
            new_cost = self.plugin.cost(new_solution)
            
            accepted = False
            # 模拟退火接受准则
            if new_cost < current_cost:
                accepted = True
            else:
                diff = new_cost - current_cost
                # 防止溢出
                prob = math.exp(-diff / T) if T > 1e-6 else 0
                if random.random() < prob:
                    accepted = True
            
            # 3.5 更新状态
            if accepted:
                current_solution = copy.deepcopy(new_solution)
                current_cost = new_cost
                
                if new_cost < best_cost:
                    best_solution = copy.deepcopy(new_solution)
                    best_cost = new_cost
                    
            # 3.6 自适应权重更新
            # 奖励机制: 新最优=1.0, 接受但非最优=0.5, 拒绝=0.0
            reward = 0.0
            if accepted:
                reward = 0.5
                if new_cost < best_cost:
                    reward = 1.0
            
            self.plugin.update_weights(reward)
            
            # 3.7 降温
            T *= alpha
            
        return best_solution, best_cost
"""
