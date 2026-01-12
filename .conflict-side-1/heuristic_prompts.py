HEURISTIC_PLUGIN_TEMPLATE = r"""
import random
import math
import copy

# =======================================================
# LLM 必须基于以下类模板实现启发式逻辑
# 注意：
# 1. 路径(Route)表示为: [0, customer_id, ..., 0]，即必须包含起终点仓库0。
# 2. 距离矩阵 self.dist_matrix 由 Solver 注入，直接使用即可。
# =======================================================

class HeuristicPlugin:
    def __init__(self, data):
        self.data = data
        self.capacity = data['vehicle_capacity']
        self.customers = data['customers']
        self.dist_matrix = [] # 由 Solver 注入

        self.customer_lookup = {c['id']: c for c in self.customers}
        
        # ALNS 算子配置
        # 定义 Destroy 和 Insert 算子方法名
        self.destroy_ops = [self.random_removal, self.worst_removal]
        self.insert_ops = [self.greedy_insert]

        # 算子权重与历史记录
        self.d_weights = [1.0] * len(self.destroy_ops)
        self.i_weights = [1.0] * len(self.insert_ops)
        self.last_d_idx = 0
        self.last_i_idx = 0
        self.rho = 0.1 # 权重更新学习率

    def cost(self, solution):
        
        total_dist = 0.0
        # TODO: 遍历所有路径，累加相邻节点间的距离
        return total_dist

    def validate(self, solution):
    # TODO: 实现约束检查
    return True

    def destroy(self, solution, remove_ratio=0.2):
        
        # 破坏算子：移除部分节点。
        # 返回: (partial_solution, removed_nodes)
        # 注意:
        # 1.不要移除节点
        # 2. 如果某路径移除节点后只剩[0, 0]，应将其从解中删除。
        
        solution = copy.deepcopy(solution)
        # 轮盘赌选择算子
        self.last_d_idx = random.choices(range(len(self.destroy_ops)), weights=self.d_weights)[0]
        op = self.destroy_ops[self.last_d_idx]

        # 执行具体算子逻辑
        return op(solution, remove_ratio)

    def insert(self, partial_solution, removed_nodes):
        
        # 修复算子：将移除的节点重新插入。
        # 返回: new_solution
        # 注意:
        # 1.尝试插入到现有路径的合法位置(0和0之间)。
        # 2.如果无法插入，创建新路径[0, node, 0]。
        # 
        partial_solution = copy.deepcopy(partial_solution)
        # 轮盘赌选择算子
        self.last_i_idx = random.choices(range(len(self.insert_ops)), weights=self.i_weights)[0]
        op = self.insert_ops[self.last_i_idx]
        
        return op(partial_solution, removed_nodes)
    
    def update_weights(self, reward):
        
        # 根据Solver传回的reward更新最近一次使用的算子权重
        
        # 更新 Destroy 权重
        w_d = self.d_weights[self.last_d_idx]
        self.d_weights[self.last_d_idx] = w_d * (1 - self.rho) + reward * self.rho
        
        # 更新 Insert 权重
        w_i = self.i_weights[self.last_i_idx]
        self.i_weights[self.last_i_idx] = w_i * (1 - self.rho) + reward * self.rho
        
    # ============================================
    # TODO: 请在下方实现具体的 Destroy 和 Insert 算子
    # ============================================
    
    def random_removal(self, solution, ratio):
    # TODO: 随机移除逻辑
        removed = []
        return solution, removed
    
    def worst_removal(self, solution, ratio):
    # TODO: 最差移除逻辑（移除成本贡献最大的节点）
        removed = []
        return solution, removed
        
    def greedy_insert(self, solution, removed_nodes):
    # TODO: 贪婪插入逻辑
    # 必须检查容量和时间窗约束 (check_feasible)
        return solution
    
    def check_feasible(self, route):
    # 辅助函数：检查单条路径是否满足容量和时间窗
    # TODO: 实现检查逻辑
        return True
"""