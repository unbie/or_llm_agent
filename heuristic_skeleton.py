HEURISTIC_SKELETON = r"""
import math
import random
import copy
from utils import FreshnessAndPenaltyCalculator

class HeuristicSolver:
    def __init__(self, data, plugin):
        self.data = data
        self.customers = data['customers']
        self.plugin = plugin 
        self.n = len(self.customers)

        # 全局距离矩阵
        self.dist_matrix = [[0]*self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                self.dist_matrix[i][j] = math.hypot(
                    self.customers[i]['x'] - self.customers[j]['x'],
                    self.customers[i]['y'] - self.customers[j]['y']
                )

        # 初始化计算器 (config为空则用默认)
        self.calculator = FreshnessAndPenaltyCalculator({})
        
        # === 关键修复：注入能力给插件 ===
        self.plugin.dist_matrix = self.dist_matrix
        self.plugin.calculator = self.calculator  # 让插件能调用 calculate_route_cost
        self.plugin.capacity = self.data.get('vehicle_capacity', 200)

    def construct_initial_solution(self):
        # 改进：简单的节约算法构造初始解，而不是一人一车
        # 这里为了简化，我们先生成一人一车，然后尝试随机合并几次
        routes = [[0, c['id'], 0] for c in self.customers if c['id'] != 0]
        # 简单洗牌后尝试拼接，减少初始车辆数
        random.shuffle(routes)
        return routes

    def cost(self, solution):
        if not solution: return float('inf')
        total = 0
        for route in solution:
            route_nodes = [self.customers[i] for i in route]
            # 使用主程序的复杂Cost计算（含时间窗惩罚）
            res = self.calculator.calculate_route_cost(route_nodes, self.dist_matrix)
            total += res['total_cost']
        return total

    def solve(self, max_iters=2000): # 增加默认迭代次数
        current_solution = self.construct_initial_solution()
        best_solution = copy.deepcopy(current_solution)
        current_cost = self.cost(current_solution)
        best_cost = current_cost
        
        print(f"初始成本: {current_cost:.2f} (车辆数: {len(current_solution)})")

        T = current_cost * 0.05
        alpha = 0.998 # 降温慢一点

        for it in range(max_iters):
            temp_sol = copy.deepcopy(current_solution)
            
            # --- destroy ---
            # 50% 概率用 worst, 50% random
            q = random.randint(4, min(30, len(self.customers)//2)) # 破坏数量动态化
            
            if random.random() < 0.5 and hasattr(self.plugin, 'worst_removal'):
                 partial, removed = self.plugin.worst_removal(temp_sol, q)
            else:
                 partial, removed = self.plugin.random_removal(temp_sol, q)
            
            # --- repair ---
            # 传入 self.customers 给 insert 用
            new_solution = self.plugin.greedy_insert(partial, removed, self.customers)
            
            new_cost = self.cost(new_solution)
            
            # Acceptance
            if new_cost < current_cost:
                current_solution = new_solution
                current_cost = new_cost
                if new_cost < best_cost:
                    best_solution = copy.deepcopy(new_solution)
                    best_cost = new_cost
            else:
                if random.random() < math.exp(-(new_cost - current_cost) / (T + 1e-6)):
                    current_solution = new_solution
                    current_cost = new_cost
            
            T *= alpha
            
            if it % 200 == 0:
                print(f"Iter {it}: Best Cost {best_cost:.2f}, Cur Cost {current_cost:.2f}")

        return best_solution, best_cost
"""