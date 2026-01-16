HEURISTIC_PLUGIN_TEMPLATE = """
你是一个熟悉 ALNS (Adaptive Large Neighborhood Search) 算法的 Python 工程师，请帮我实现生鲜物流 VRP 问题中的三个启发式算子：random_removal、worst_removal 和 greedy_insert。要求如下：

1. 数据结构：
   - solution 是列表，每条路径是一个列表，例如 [0, node1, node2, 0]。
   - 0 表示仓库，node 表示客户节点。
   - self.dist_matrix[i][j] 表示节点 i 到节点 j 的距离。
   - 移除节点后，如果路径长度 <= 2（即只剩 [0,0]），则删除该路径。
   - 插入节点时必须考虑新建路径的情况：新路径为 [0, node, 0]。

2. 算子要求：

【random_removal】：
   - 随机移除 solution 中 ratio 比例的客户节点（非 0 节点），至少移除一个节点。
   - 移除节点时按路径索引和位置索引降序，避免索引变化引发错误。
   - 移除完成后统一删除长度 <= 2 的路径。
   - 返回修改后的 solution 和 removed_nodes 列表。

【worst_removal】：
   - 对每条路径中非0节点计算边际贡献：
       contrib = dist(prev,node) + dist(node,next) - dist(prev,next)
   - 按贡献从大到小选择 ratio 比例的节点移除，至少移除一个节点。
   - 移除顺序按路径索引和位置索引降序。
   - 移除完成后统一删除长度 <= 2 的路径。
   - 返回修改后的 solution 和 removed_nodes 列表。

【greedy_insert】：
   - 遍历 removed_nodes 中的每个节点。
   - 对现有路径中每个可能插入位置计算增量成本：
       cost_increase = dist(prev,node) + dist(node,next) - dist(prev,next)
   - 同时考虑新建路径的成本：dist(0,node) + dist(node,0)
   - 选择增量成本最小的位置插入节点。
   - 返回修改后的 solution。

3. 注意事项：
   - 移除节点和插入节点时都要确保 solution 的合法性。
   - 避免在循环中直接删除路径导致索引错误。
   - 必须使用深拷贝或安全操作，保证不会破坏原 solution。
   - 请生成完整可运行的 Python 函数，补充以下的TODO部分。

import random
import copy


class HeuristicPlugin:
    def __init__(self, data):
        self.data = data
        self.capacity = data['vehicle_capacity']
        self.customers = data['customers']
        self.dist_matrix = None   # 由 Solver 注入

        # ID → customer dict
        self.customer_lookup = {c['id']: c for c in data['customers']}

        # ALNS 算子配置
        self.destroy_ops = [self.random_removal, self.worst_removal]
        self.insert_ops = [self.greedy_insert]

        self.d_weights = [1.0] * len(self.destroy_ops)
        self.i_weights = [1.0] * len(self.insert_ops)
        self.last_d_idx = 0
        self.last_i_idx = 0
        self.rho = 0.1

    # ==============================
    # 这些函数禁止 LLM 实现
    # ==============================

    def cost(self, solution):
        raise RuntimeError("LLM must not define cost()")

    def validate(self, solution):
        raise RuntimeError("LLM must not define validate()")

    def check_feasible(self, route):
        raise RuntimeError("LLM must not define feasibility")

    # ==============================
    # ALNS 框架函数（固定）
    # ==============================

    def destroy(self, solution, remove_ratio=0.2):
    
        # 调用 Destroy 算子移除解中的客户节点。
        # 返回 new_solution, removed_nodes
        # 
        solution = copy.deepcopy(solution)
        self.last_d_idx = random.choices(range(len(self.destroy_ops)), weights=self.d_weights)[0]
        op = self.destroy_ops[self.last_d_idx]
        return op(solution, remove_ratio)

    def insert(self, partial_solution, removed_nodes):
        
        # 调用 Insert 算子将 removed_nodes 插回解中。
        # 返回 new_solution
        
        partial_solution = copy.deepcopy(partial_solution)
        self.last_i_idx = random.choices(range(len(self.insert_ops)), weights=self.i_weights)[0]
        op = self.insert_ops[self.last_i_idx]
        return op(partial_solution, removed_nodes)

    def update_weights(self, reward):
        self.d_weights[self.last_d_idx] = (
            self.d_weights[self.last_d_idx] * (1 - self.rho) + reward * self.rho
        )
        self.i_weights[self.last_i_idx] = (
            self.i_weights[self.last_i_idx] * (1 - self.rho) + reward * self.rho
        )

    # ===================================================
    # TODO: 请在下方实现具体的 Destroy 和 Insert 算子
    # ===================================================

    # -----------------------------
    # Destroy 算子要求
    # -----------------------------
    # random_removal:
    #   - 随机移除 ratio 比例的客户节点（非 0 节点），至少移除一个。
    #   - 移除后，如果路径只剩 [0,0]，去掉空路径。
    #   - 返回 new_solution, removed_nodes
    
    def random_removal(self, solution, ratio):
        # TODO: 从 solution 中随机移除 ratio 比例的客户
        pass

    # worst_removal:
    #   - 计算每个客户节点的边际距离贡献：
    #       contrib = dist(prev,node) + dist(node,next) - dist(prev,next)
    #   - 按贡献从大到小移除 n 个节点（n=max(1,total_customer*ratio)）
    #   - 路径长度小于3的节点跳过
    #   - 返回 new_solution, removed_nodes
    
    def worst_removal(self, solution, ratio):
        # TODO: 根据 dist_matrix 计算每个客户的边际距离贡献
        pass

    # -----------------------------
    # Insert 算子要求
    # -----------------------------
    # greedy_insert:
    #   - 遍历每个 removed_nodes
    #   - 对每条路径的每个合法位置计算插入成本：
    #         cost_increase = dist(prev,node)+dist(node,next)-dist(prev,next)
    #   - 必须考虑新路径插入，增量成本 = dist(0,node)+dist(node,0)
    #   - 选择增量最小的位置插入
    #   - 返回修改后的 solution
    #   - 系统会检查容量和时间窗可行性，你无需实现
    def greedy_insert(self, solution, removed_nodes):
        # TODO: 把 removed_nodes 用最小距离增量插入到 solution
        pass
"""
