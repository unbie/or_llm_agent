HEURISTIC_SKELETON = r"""
import math
import random
import copy
from utils import FreshnessAndPenaltyCalculator

class HeuristicSolver:
    def __init__(self, data, config=None):
        self.data = data
        self.customers = data['customers']
        self.n = len(self.customers)

        # 预计算距离矩阵
        self.dist_matrix = [[0]*self.n for _ in range(self.n)]
        for i in range(self.n):
            xi, yi = self.customers[i]['x'], self.customers[i]['y']
            for j in range(self.n):
                xj, yj = self.customers[j]['x'], self.customers[j]['y']
                self.dist_matrix[i][j] = math.hypot(xi - xj, yi - yj)

        # 初始化 FreshnessAndPenaltyCalculator
        self.calculator = FreshnessAndPenaltyCalculator(config or {})

    def construct_initial_solution(self):
        # 生成初始解，每个客户单独一条路径
        solution = []
        for c in self.customers:
            cid = c['id']
            if cid == 0:  # 跳过仓库
                continue
            solution.append([0, cid, 0])
        return solution

    def cost(self, solution):
        #计算整个解的总成本
        total_cost = 0.0
        for route in solution:
            route_nodes = [self.customers[i] for i in route]
            cost_info = self.calculator.calculate_route_cost(route_nodes, self.dist_matrix)
            total_cost += cost_info['variable_cost']
        # 加上固定车辆成本
        total_cost += len(solution) * self.calculator.f
        return total_cost

    def solve(self, max_iters=1500):
        #ALNS + 模拟退火迭代求解
        current_solution = self.construct_initial_solution()
        best_solution = copy.deepcopy(current_solution)
        current_cost = self.cost(current_solution)
        best_cost = current_cost

        # 模拟退火参数
        T = current_cost * 0.05
        alpha = 0.995

        for it in range(max_iters):
            # 破坏操作：随机移除一些客户
            removed_nodes = random.sample([c['id'] for c in self.customers if c['id'] != 0],
                                          k=max(1, len(self.customers)//10))
            partial_solution = []
            for route in current_solution:
                new_route = [i for i in route if i not in removed_nodes]
                if len(new_route) > 2:  # 保留至少仓库+客户+仓库
                    partial_solution.append(new_route)

            # 修复操作：随机插入移除的客户
            for cid in removed_nodes:
                insert_route = random.choice(partial_solution) if partial_solution else [0, cid, 0]
                # 插入到随机位置（仓库之间）
                pos = random.randint(1, len(insert_route)-1)
                insert_route.insert(pos, cid)
                if insert_route not in partial_solution:
                    partial_solution.append(insert_route)

            new_solution = partial_solution
            new_cost = self.cost(new_solution)

            # 接受准则：模拟退火
            accepted = False
            if new_cost < current_cost:
                accepted = True
            else:
                diff = new_cost - current_cost
                prob = math.exp(-diff / T) if T > 1e-6 else 0
                if random.random() < prob:
                    accepted = True

            if accepted:
                current_solution = copy.deepcopy(new_solution)
                current_cost = new_cost
                if new_cost < best_cost:
                    best_solution = copy.deepcopy(new_solution)
                    best_cost = new_cost

            # 简单自适应权重更新示例（可以扩展）
            T *= alpha

        return best_solution, best_cost

"""
