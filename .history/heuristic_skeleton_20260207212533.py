HEURISTIC_SKELETON = r"""
import random
import copy
import math


class HeuristicSolver:
    def __init__(self, data, plugin):
        self.data = data
        self.plugin = plugin
        self.customers = data['customers']
        self.n = len(self.customers)
        
        # 构建距离矩阵
        self.dist_matrix = self._build_distance_matrix()
        self.plugin.dist_matrix = self.dist_matrix  # 注入到 plugin
        
        # 构建客户索引映射
        self.id_to_customer = {c['id']: c for c in self.customers}
        
        # 成本计算器
        from utils import FreshnessAndPenaltyCalculator
        self.calculator = FreshnessAndPenaltyCalculator(data)
        
        # ALNS 算子权重管理 - 简化配置：3个破坏算子，2个修复算子
        self.destroy_ops = [
            self.plugin.random_removal,    # 分散破坏
            self.plugin.route_removal,     # 路径级破坏
            self.plugin.string_removal     # 连续节点破坏
        ]
        # 修复算子：贪心、后悔
        self.insert_ops = [
            self.plugin.greedy_insert,
            self.plugin.regret_insert
        ]
        self.d_weights = [1.0] * len(self.destroy_ops)
        self.i_weights = [1.0] * len(self.insert_ops)
        self.last_d_idx = 0
        self.last_i_idx = 0
        self.rho = 0.1  # 记忆系数（文档建议0.1-0.4）
        
        # 分数设置（文档推荐 σ₁=33, σ₂=9, σ₃=3, σ₄=0）
        self.sigma1 = 33  # 找到全局最优
        self.sigma2 = 9   # 找到改进方案
        self.sigma3 = 3   # 被接受但未改进
        self.sigma4 = 0   # 未被接受
        
        # 算子统计记录（实战技巧：记录和可视化）
        self.op_stats = {
            'destroy': {i: {'uses': 0, 'successes': 0, 'score': 0} for i in range(len(self.destroy_ops))},
            'insert': {i: {'uses': 0, 'successes': 0, 'score': 0} for i in range(len(self.insert_ops))}
        }
        
        # 历史记录（用于 history_removal）
        self.node_history = {c['id']: 0 for c in self.customers if c['id'] != 0}
        self.current_iteration = 0
        
        # 将历史记录注入到 plugin（供 history_removal 使用）
        self.plugin.node_history = self.node_history
        self.plugin.solver = self  # 让 plugin 可以访问 solver
        
        # 记录和可视化（技巧5）
        self.cost_history = []  # 目标函数值变化曲线
        self.best_cost_history = []  # 最优成本变化曲线
        self.weight_history = {'destroy': [], 'insert': []}  # 权重变化曲线
        
        # 候选集限制参数（技巧4：加速计算）
        self.candidate_list_size = 10  # 只考虑最近的K个位置
        
        print("Solver 初始化成功")
    
    def _fallback_greedy_insert(self, solution, removed_nodes):
        '''备用贪心插入算子（使用完整成本计算）'''
        if not removed_nodes:
            return solution
        if not solution:
            return [[0, node, 0] for node in removed_nodes]
        
        capacity = self.data.get('vehicle_capacity', 200)
        remaining_nodes = list(removed_nodes)
        
        for node in removed_nodes:
            node_demand = self.id_to_customer[node].get('demand', 0)
            best_cost_increase = float('inf')
            best_route_idx = None
            best_position = None
            
            # 尝试插入到现有路径（使用完整成本计算）
            for route_idx, route in enumerate(solution):
                current_load = sum(self.id_to_customer[n].get('demand', 0) 
                                  for n in route if n != 0)
                if current_load + node_demand > capacity:
                    continue
                
                # 计算插入前的路径成本
                route_before = [self.id_to_customer[n] for n in route]
                cost_before = self.calculator.calculate_route_cost(
                    route_before, self.dist_matrix
                )['variable_cost']
                
                for pos in range(1, len(route)):
                    # 计算插入后的路径成本
                    route_after = route[:pos] + [node] + route[pos:]
                    route_after_nodes = [self.id_to_customer[n] for n in route_after]
                    cost_after = self.calculator.calculate_route_cost(
                        route_after_nodes, self.dist_matrix
                    )['variable_cost']
                    
                    cost_inc = cost_after - cost_before
                    
                    if cost_inc < best_cost_increase:
                        best_cost_increase = cost_inc
                        best_route_idx = route_idx
                        best_position = pos
            
            # 考虑创建新路径（包括固定成本）
            new_route = [0, node, 0]
            route_nodes_new = [self.id_to_customer[n] for n in new_route]
            new_route_cost = self.calculator.calculate_route_cost(
                route_nodes_new, self.dist_matrix
            )['variable_cost']
            new_route_cost += self.calculator.f  # 固定成本
            
            # 决策：优先现有路径
            if best_route_idx is not None and best_cost_increase <= new_route_cost:
                solution[best_route_idx].insert(best_position, node)
                remaining_nodes.remove(node)
            else:
                solution.append([0, node, 0])
                remaining_nodes.remove(node)
        
        # 确保所有节点都被插入
        if remaining_nodes:
            print(f"[警告] {len(remaining_nodes)} 个节点未插入，创建独立路径")
            for node in remaining_nodes:
                solution.append([0, node, 0])
        
        return solution
    
    def _fallback_regret_insert(self, solution, removed_nodes):
        '''备用后悔插入算子（使用完整成本计算）'''
        if not removed_nodes:
            return solution
        if not solution:
            return [[0, node, 0] for node in removed_nodes]
        
        capacity = self.data.get('vehicle_capacity', 200)
        remaining = list(removed_nodes)
        
        while remaining:
            best_regret = -float('inf')
            best_node = None
            best_route_idx = None
            best_position = None
            
            for node in remaining:
                node_demand = self.id_to_customer[node].get('demand', 0)
                costs = []  # (cost_increase, route_idx, position)
                
                # 现有路径的位置
                for route_idx, route in enumerate(solution):
                    current_load = sum(self.id_to_customer[n].get('demand', 0) 
                                      for n in route if n != 0)
                    if current_load + node_demand > capacity:
                        continue
                    
                    # 计算插入前的成本
                    route_before = [self.id_to_customer[n] for n in route]
                    cost_before = self.calculator.calculate_route_cost(
                        route_before, self.dist_matrix
                    )['variable_cost']
                    
                    for pos in range(1, len(route)):
                        # 计算插入后的成本
                        route_after = route[:pos] + [node] + route[pos:]
                        route_after_nodes = [self.id_to_customer[n] for n in route_after]
                        cost_after = self.calculator.calculate_route_cost(
                            route_after_nodes, self.dist_matrix
                        )['variable_cost']
                        
                        cost_inc = cost_after - cost_before
                        costs.append((cost_inc, route_idx, pos))
                
                # 新建路径的成本（包括固定成本）
                new_route = [0, node, 0]
                route_nodes_new = [self.id_to_customer[n] for n in new_route]
                new_cost = self.calculator.calculate_route_cost(
                    route_nodes_new, self.dist_matrix
                )['variable_cost']
                new_cost += self.calculator.f  # 固定成本
                costs.append((new_cost, None, None))
                
                # 排序找最佳和次佳
                costs.sort(key=lambda x: x[0])
                
                if len(costs) >= 2:
                    regret = costs[1][0] - costs[0][0]
                else:
                    regret = 0
                
                if regret > best_regret:
                    best_regret = regret
                    best_node = node
                    best_route_idx = costs[0][1]
                    best_position = costs[0][2]
            
            # 如果找不到节点，强制插入剩余所有节点
            if best_node is None:
                print(f"[警告] 后悔插入失败，{len(remaining)} 个节点创建独立路径")
                for node in remaining:
                    solution.append([0, node, 0])
                break
            
            # 插入后悔值最大的节点
            if best_route_idx is not None:
                solution[best_route_idx].insert(best_position, best_node)
            else:
                solution.append([0, best_node, 0])
            
            remaining.remove(best_node)
        
        return solution
        
        return solution
    
    def _fallback_random_insert(self, solution, removed_nodes):
        '''备用随机插入算子'''
        if not removed_nodes:
            return solution
        if not solution:
            return [[0, node, 0] for node in removed_nodes]
        
        capacity = self.data.get('vehicle_capacity', 200)
        
        for node in removed_nodes:
            node_demand = self.id_to_customer[node].get('demand', 0)
            # 收集所有可行位置
            positions = []
            for route_idx, route in enumerate(solution):
                # 检查容量约束
                current_load = sum(self.id_to_customer[n].get('demand', 0) 
                                  for n in route if n != 0)
                if current_load + node_demand > capacity:
                    continue
                
                for pos in range(1, len(route)):
                    positions.append((route_idx, pos))
            
            # 加入新建路径选项
            positions.append((None, None))
            
            # 随机选择
            choice = random.choice(positions)
            
            if choice[0] is not None:
                solution[choice[0]].insert(choice[1], node)
            else:
                solution.append([0, node, 0])
        
        return solution
    
    def _fallback_related_removal(self, solution, ratio):
        '''备用关联破坏算子'''
        # 收集所有非0节点
        all_nodes = []
        for route_idx, route in enumerate(solution):
            for pos_idx, node in enumerate(route):
                if node != 0:
                    all_nodes.append((route_idx, pos_idx, node))
        
        if not all_nodes:
            return solution, []
        
        # 计算移除数量
        if ratio <= 1.0:
            n = max(1, math.ceil(len(all_nodes) * ratio))
        else:
            n = int(ratio)
        n = min(n, len(all_nodes))
        
        # 随机选择种子节点
        seed_info = random.choice(all_nodes)
        seed_node = seed_info[2]
        
        # 按与种子节点的距离排序
        all_nodes.sort(key=lambda x: self.dist_matrix[seed_node][x[2]])
        
        # 选择最近的n个节点
        selected = all_nodes[:n]
        
        # 降序移除
        selected.sort(key=lambda x: (x[0], x[1]), reverse=True)
        removed_nodes = []
        for route_idx, pos_idx, node in selected:
            solution[route_idx].pop(pos_idx)
            removed_nodes.append(node)
        
        # 过滤空路径
        solution = [route for route in solution if len(route) > 2]
        return solution, removed_nodes
    
    def _fallback_shaw_removal(self, solution, ratio):
        '''备用Shaw破坏算子'''
        # 收集所有非0节点
        all_nodes = []
        for route_idx, route in enumerate(solution):
            for pos_idx, node in enumerate(route):
                if node != 0:
                    all_nodes.append((route_idx, pos_idx, node))
        
        if not all_nodes:
            return solution, []
        
        # 计算移除数量
        if ratio <= 1.0:
            n = max(1, math.ceil(len(all_nodes) * ratio))
        else:
            n = int(ratio)
        n = min(n, len(all_nodes))
        
        # 随机选择种子节点
        seed_info = random.choice(all_nodes)
        seed_node = seed_info[2]
        seed_customer = self.id_to_customer[seed_node]
        
        # 计算最大值用于归一化
        max_dist = max(self.dist_matrix[seed_node][x[2]] for x in all_nodes) or 1
        max_demand = max(self.id_to_customer[x[2]].get('demand', 0) for x in all_nodes) or 1
        max_time = max(abs(self.id_to_customer[x[2]].get('ready_time', 0) - seed_customer.get('ready_time', 0)) for x in all_nodes) or 1
        
        # 计算Shaw分数
        def shaw_score(node_info):
            node = node_info[2]
            customer = self.id_to_customer[node]
            
            dist_norm = self.dist_matrix[seed_node][node] / max_dist
            time_diff = abs(customer.get('ready_time', 0) - seed_customer.get('ready_time', 0)) / max_time
            demand_diff = abs(customer.get('demand', 0) - seed_customer.get('demand', 0)) / max_demand
            
            # 权重: φ=9, χ=3, ψ=2
            return 9 * dist_norm + 3 * time_diff + 2 * demand_diff
        
        # 按Shaw分数排序（分数越低越相似）
        all_nodes.sort(key=shaw_score)
        
        # 选择前n个
        selected = all_nodes[:n]
        
        # 降序移除
        selected.sort(key=lambda x: (x[0], x[1]), reverse=True)
        removed_nodes = []
        for route_idx, pos_idx, node in selected:
            solution[route_idx].pop(pos_idx)
            removed_nodes.append(node)
        
        # 过滤空路径
        solution = [route for route in solution if len(route) > 2]
        return solution, removed_nodes
    
    def _fallback_history_removal(self, solution, ratio):
        '''备用历史破坏算子'''
        # 收集所有非0节点
        all_nodes = []
        for route_idx, route in enumerate(solution):
            for pos_idx, node in enumerate(route):
                if node != 0:
                    all_nodes.append((route_idx, pos_idx, node))
        
        if not all_nodes:
            return solution, []
        
        # 计算移除数量
        if ratio <= 1.0:
            n = max(1, math.ceil(len(all_nodes) * ratio))
        else:
            n = int(ratio)
        n = min(n, len(all_nodes))
        
        # 按历史值排序（值越小表示越久未改动）
        all_nodes.sort(key=lambda x: self.node_history.get(x[2], 0))
        
        # 选择前n个（最久未改动的）
        selected = all_nodes[:n]
        
        # 降序移除
        selected.sort(key=lambda x: (x[0], x[1]), reverse=True)
        removed_nodes = []
        for route_idx, pos_idx, node in selected:
            solution[route_idx].pop(pos_idx)
            removed_nodes.append(node)
        
        # 过滤空路径
        solution = [route for route in solution if len(route) > 2]
        return solution, removed_nodes
    
    def _fallback_cluster_removal(self, solution, ratio):
        '''备用聚类破坏算子'''
        # 收集所有非0节点及其坐标
        all_nodes = []
        for route_idx, route in enumerate(solution):
            for pos_idx, node in enumerate(route):
                if node != 0:
                    customer = self.id_to_customer[node]
                    all_nodes.append((route_idx, pos_idx, node, customer.get('x', 0), customer.get('y', 0)))
        
        if not all_nodes:
            return solution, []
        
        # 计算移除数量
        if ratio <= 1.0:
            n = max(1, math.ceil(len(all_nodes) * ratio))
        else:
            n = int(ratio)
        n = min(n, len(all_nodes))
        
        # 计算坐标范围
        xs = [node[3] for node in all_nodes]
        ys = [node[4] for node in all_nodes]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # 网格划分
        grid_size = 5
        cell_width = (max_x - min_x + 1) / grid_size
        cell_height = (max_y - min_y + 1) / grid_size
        
        # 分配节点到网格
        grid = {}
        for node_info in all_nodes:
            gx = min(grid_size - 1, int((node_info[3] - min_x) / cell_width)) if cell_width > 0 else 0
            gy = min(grid_size - 1, int((node_info[4] - min_y) / cell_height)) if cell_height > 0 else 0
            key = (gx, gy)
            if key not in grid:
                grid[key] = []
            grid[key].append(node_info)
        
        # 随机选择一个非空网格
        non_empty_cells = [k for k, v in grid.items() if v]
        if not non_empty_cells:
            return solution, []
        
        selected_cell = random.choice(non_empty_cells)
        selected = grid[selected_cell][:n]
        
        # 如果不足，从相邻网格补充
        if len(selected) < n:
            gx, gy = selected_cell
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    neighbor = (gx + dx, gy + dy)
                    if neighbor in grid:
                        for node_info in grid[neighbor]:
                            if node_info not in selected and len(selected) < n:
                                selected.append(node_info)
        
        # 降序移除
        selected.sort(key=lambda x: (x[0], x[1]), reverse=True)
        removed_nodes = []
        for route_idx, pos_idx, node, x, y in selected:
            solution[route_idx].pop(pos_idx)
            removed_nodes.append(node)
        
        # 过滤空路径
        solution = [route for route in solution if len(route) > 2]
        return solution, removed_nodes
    
    def destroy(self, solution, remove_ratio=0.2):
        '''调用 Destroy 算子移除解中的客户节点'''
        solution = copy.deepcopy(solution)
        self.last_d_idx = random.choices(range(len(self.destroy_ops)), weights=self.d_weights)[0]
        op = self.destroy_ops[self.last_d_idx]
        
        # 尝试调用 LLM 生成的算子，失败则使用备用
        try:
            result = op(solution, remove_ratio)
            if result is None or (isinstance(result, tuple) and result[0] is None):
                raise ValueError("算子返回None")
            return result
        except Exception as e:
            # 使用对应的备用算子
            fallback_ops = [
                lambda s, r: self._fallback_random_removal(s, r),
                lambda s, r: self._fallback_worst_removal(s, r),
                lambda s, r: self._fallback_related_removal(s, r),
                lambda s, r: self._fallback_shaw_removal(s, r),
                lambda s, r: self._fallback_history_removal(s, r),
                lambda s, r: self._fallback_cluster_removal(s, r),
            ]
            if self.last_d_idx < len(fallback_ops):
                return fallback_ops[self.last_d_idx](solution, remove_ratio)
            return solution, []
    
    def _fallback_random_removal(self, solution, ratio):
        '''备用随机破坏算子'''
        all_nodes = []
        for route_idx, route in enumerate(solution):
            for pos_idx, node in enumerate(route):
                if node != 0:
                    all_nodes.append((route_idx, pos_idx, node))
        
        if not all_nodes:
            return solution, []
        
        if ratio <= 1.0:
            n = max(1, math.ceil(len(all_nodes) * ratio))
        else:
            n = int(ratio)
        n = min(n, len(all_nodes))
        
        selected = random.sample(all_nodes, n)
        selected.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        removed_nodes = []
        for route_idx, pos_idx, node in selected:
            solution[route_idx].pop(pos_idx)
            removed_nodes.append(node)
        
        solution = [route for route in solution if len(route) > 2]
        return solution, removed_nodes
    
    def _fallback_worst_removal(self, solution, ratio):
        '''备用最差破坏算子'''
        node_contributions = []
        for route_idx, route in enumerate(solution):
            if len(route) < 3:
                continue
            for pos_idx in range(1, len(route) - 1):
                node = route[pos_idx]
                if node == 0:
                    continue
                prev = route[pos_idx - 1]
                next_n = route[pos_idx + 1]
                contrib = (self.dist_matrix[prev][node] + 
                          self.dist_matrix[node][next_n] - 
                          self.dist_matrix[prev][next_n])
                node_contributions.append((route_idx, pos_idx, node, contrib))
        
        if not node_contributions:
            return solution, []
        
        total = len(node_contributions)
        if ratio <= 1.0:
            n = max(1, math.ceil(total * ratio))
        else:
            n = int(ratio)
        n = min(n, total)
        
        node_contributions.sort(key=lambda x: x[3], reverse=True)
        selected = node_contributions[:n]
        selected.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        removed_nodes = []
        for route_idx, pos_idx, node, _ in selected:
            solution[route_idx].pop(pos_idx)
            removed_nodes.append(node)
        
        solution = [route for route in solution if len(route) > 2]
        return solution, removed_nodes

    def insert(self, partial_solution, removed_nodes):
        '''调用 Insert 算子将 removed_nodes 插回解中'''
        partial_solution = copy.deepcopy(partial_solution)
        self.last_i_idx = random.choices(range(len(self.insert_ops)), weights=self.i_weights)[0]
        op = self.insert_ops[self.last_i_idx]
        
        # 尝试调用 LLM 生成的算子，失败则使用备用
        try:
            result = op(partial_solution, removed_nodes)
            if result is None:
                raise ValueError("算子返回None")
            return result
        except Exception as e:
            # 使用对应的备用算子
            fallback_ops = [
                self._fallback_greedy_insert,
                self._fallback_regret_insert,
                self._fallback_random_insert,
            ]
            if self.last_i_idx < len(fallback_ops):
                return fallback_ops[self.last_i_idx](partial_solution, removed_nodes)
            return self._fallback_greedy_insert(partial_solution, removed_nodes)
    
    def update_weights(self, reward):
        '''更新算子权重（使用文档推荐的4档分数体系）
        
        Args:
            reward: 奖励类型
                - 'global_best': 找到全局最优（σ₁=33）
                - 'improved': 找到改进方案（σ₂=9）
                - 'accepted': 被接受但未改进（σ₃=3）
                - 'rejected': 未被接受（σ₄=0）
        '''
        # 根据奖励类型确定分数
        if reward == 'global_best':
            score = self.sigma1  # 33
        elif reward == 'improved':
            score = self.sigma2  # 9
        elif reward == 'accepted':
            score = self.sigma3  # 3
        else:  # rejected
            score = self.sigma4  # 0
        
        # 更新算子权重
        self.d_weights[self.last_d_idx] = (
            self.d_weights[self.last_d_idx] * (1 - self.rho) + score * self.rho
        )
        self.i_weights[self.last_i_idx] = (
            self.i_weights[self.last_i_idx] * (1 - self.rho) + score * self.rho
        )
        
        # 更新算子统计
        self.op_stats['destroy'][self.last_d_idx]['uses'] += 1
        self.op_stats['destroy'][self.last_d_idx]['score'] += score
        self.op_stats['insert'][self.last_i_idx]['uses'] += 1
        self.op_stats['insert'][self.last_i_idx]['score'] += score
        
        if reward in ['global_best', 'improved']:
            self.op_stats['destroy'][self.last_d_idx]['successes'] += 1
            self.op_stats['insert'][self.last_i_idx]['successes'] += 1
    
    def merge_short_routes(self, solution):
        '''尝试合并短路径以减少车辆数'''
        if len(solution) <= 1:
            return solution
        
        capacity = self.data.get('vehicle_capacity', 200)
        merged = True
        
        while merged:
            merged = False
            # 按路径客户数排序，优先合并短路径
            solution.sort(key=lambda r: len(r))
            
            for i in range(len(solution)):
                if merged:
                    break
                route_i = solution[i]
                nodes_i = [n for n in route_i if n != 0]
                if not nodes_i:
                    continue
                load_i = sum(self.id_to_customer[n].get('demand', 0) for n in nodes_i)
                
                best_merge_saving = 0
                best_j = -1
                best_new_route = None
                
                for j in range(i + 1, len(solution)):
                    route_j = solution[j]
                    nodes_j = [n for n in route_j if n != 0]
                    if not nodes_j:
                        continue
                    load_j = sum(self.id_to_customer[n].get('demand', 0) for n in nodes_j)
                    
                    # 检查容量约束
                    if load_i + load_j > capacity:
                        continue
                    
                    # 尝试多种合并方式，选择最优的
                    old_cost = self._route_distance(route_i) + self._route_distance(route_j)
                    
                    # 方式1: i + j
                    new_route1 = [0] + nodes_i + nodes_j + [0]
                    cost1 = self._route_distance(new_route1)
                    
                    # 方式2: j + i
                    new_route2 = [0] + nodes_j + nodes_i + [0]
                    cost2 = self._route_distance(new_route2)
                    
                    # 方式3: i + reverse(j)
                    new_route3 = [0] + nodes_i + nodes_j[::-1] + [0]
                    cost3 = self._route_distance(new_route3)
                    
                    # 方式4: reverse(i) + j
                    new_route4 = [0] + nodes_i[::-1] + nodes_j + [0]
                    cost4 = self._route_distance(new_route4)
                    
                    # 选择最优方式
                    options = [(cost1, new_route1), (cost2, new_route2), 
                              (cost3, new_route3), (cost4, new_route4)]
                    best_cost, best_route = min(options, key=lambda x: x[0])
                    saving = old_cost - best_cost
                    
                    if saving > best_merge_saving:
                        best_merge_saving = saving
                        best_j = j
                        best_new_route = best_route
                
                # 只有真正节省成本时才合并（saving > 0）
                if best_j >= 0 and best_merge_saving > 0:
                    solution[i] = best_new_route
                    solution.pop(best_j)
                    merged = True
        
        return solution
    
    def two_opt_route(self, route):
        '''对单条路径进行2-opt优化（只考虑距离）'''
        if len(route) < 4:  # 至少需要 [0, a, b, 0]
            return route
        
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    # 计算反转 route[i:j+1] 的收益
                    # 原: route[i-1]->route[i] 和 route[j]->route[j+1]
                    # 新: route[i-1]->route[j] 和 route[i]->route[j+1]
                    a, b = route[i-1], route[i]
                    c, d = route[j], route[j+1]
                    
                    old_dist = self.dist_matrix[a][b] + self.dist_matrix[c][d]
                    new_dist = self.dist_matrix[a][c] + self.dist_matrix[b][d]
                    
                    if new_dist < old_dist - 0.001:
                        # 反转 route[i:j+1]
                        route[i:j+1] = route[i:j+1][::-1]
                        improved = True
        
        return route
    
    def local_search(self, solution):
        '''对整个解进行局部搜索优化'''
        for idx in range(len(solution)):
            solution[idx] = self.two_opt_route(solution[idx])
        return solution
    
    def _route_distance(self, route):
        '''计算单条路径的总距离'''
        dist = 0
        for k in range(len(route) - 1):
            dist += self.dist_matrix[route[k]][route[k+1]]
        return dist

    def _build_distance_matrix(self):
        '''构建距离矩阵'''
        n = len(self.customers)
        dist_matrix = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    xi, yi = self.customers[i]['x'], self.customers[i]['y']
                    xj, yj = self.customers[j]['x'], self.customers[j]['y']
                    dist_matrix[i][j] = math.sqrt((xi - xj)**2 + (yi - yj)**2)
        
        return dist_matrix
    
    def cost(self, solution):
        '''
        计算解的总成本
         关键修复：调用 calculate_route_cost 而不是 calculate_total_cost
        '''
        if not solution:
            return float('inf')
        
        # C11: 固定成本 = 路径数量 × 每车固定成本
        num_vehicles = len(solution)
        fixed_cost = num_vehicles * self.calculator.f
        
        # 计算所有路径的变动成本 (C12 + C13 + C2 + C3)
        total_variable_cost = 0.0
        
        for route_ids in solution:
            # 将路径ID列表转换为客户节点对象列表
            route_nodes = [self.id_to_customer[node_id] for node_id in route_ids]
            
            # ⚠️ 调用正确的方法：calculate_route_cost
            route_cost_info = self.calculator.calculate_route_cost(route_nodes, self.dist_matrix)
            
            # 累加变动成本
            total_variable_cost += route_cost_info['variable_cost']
        
        return fixed_cost + total_variable_cost
    
    def validate(self, solution):
        '''验证解的可行性'''
        if not solution:
            return False
        
        capacity = self.data.get('vehicle_capacity', 200)
        
        # ===== 关键修复：检查客户覆盖完整性 =====
        visited_customers = set()
        for route_ids in solution:
            for node_id in route_ids:
                if node_id != 0:
                    visited_customers.add(node_id)
        
        # 必须覆盖所有非仓库客户
        all_customers = set(c['id'] for c in self.customers if c['id'] != 0)
        if visited_customers != all_customers:
            missing = all_customers - visited_customers
            # print(f"[验证失败] 缺少 {len(missing)} 个客户: {list(missing)[:10]}...")
            return False
        
        for route_ids in solution:
            load = 0
            curr_time = 0.0
            
            for i in range(len(route_ids)):
                node_id = route_ids[i]
                customer = self.id_to_customer[node_id]
                
                # 累加需求
                if node_id != 0:
                    load += customer.get('demand', 0)
                
                # 计算到达时间
                if i > 0:
                    prev_id = route_ids[i-1]
                    dist = self.dist_matrix[prev_id][node_id]
                    travel_time = dist * (60.0 / self.calculator.v)
                    curr_time += travel_time
                
                # 检查硬时间窗（使用 ready_time 和 due_date）
                if node_id != 0:
                    ready_time = customer.get('ready_time', 0)
                    due_date = customer.get('due_date', float('inf'))
                    
                    # 到达太晚：违反硬时间窗
                    if curr_time > due_date:
                        return False
                    
                    # 到达太早：等待到开始服务时间
                    if curr_time < ready_time:
                        curr_time = ready_time
                
                # 更新时间（加上服务时间）
                if node_id != 0:
                    curr_time += customer.get('service_time', 0)
            
            # 检查容量约束
            if load > capacity:
                return False
        
        return True
    
    def solve(self, max_iters=300, seed=None):
        '''ALNS 求解器
        
        Args:
            max_iters: 最大迭代次数
            seed: 随机种子（设置后结果可重复）
        '''
        if seed is not None:
            random.seed(seed)
            print(f"[初始化] 随机种子: {seed}")
        
        print(f"[初始化] 开始生成初始解... (迭代次数: {max_iters})")
        
        non_depot = [c for c in self.customers if c['id'] != 0]
        capacity = self.data.get('vehicle_capacity', 200)
        
        id_to_idx = {c['id']: i for i, c in enumerate(self.customers)}
        
        # ===== 技巧1：初始方案很重要 =====
        # 生成多个初始解，选择最好的那个
        num_initial_solutions = 3  # 生成3个初始解
        best_initial_solution = None
        best_initial_cost = float('inf')
        
        for init_attempt in range(num_initial_solutions):
            current_solution = []
            remaining = list(non_depot)
            
            # 不同的初始排序策略增加多样性
            if init_attempt == 0:
                # 策略1: 按需求降序（优先处理大需求客户）
                remaining.sort(key=lambda c: c.get('demand', 0), reverse=True)
            elif init_attempt == 1:
                # 策略2: 按时间窗紧迫度（优先处理时间紧的客户）
                remaining.sort(key=lambda c: c.get('due_date', float('inf')) - c.get('ready_time', 0))
            else:
                # 策略3: 随机打乱
                random.shuffle(remaining)
            
            route_count = 0
            while remaining:
                route = [0]
                load = 0
                pos_id = 0
                curr_time = 0.0
                
                while remaining:
                    best_customer = None
                    best_score = float('inf')
                    
                    for c in remaining:
                        cid = c['id']
                        demand = c.get('demand', 0)
                        
                        if load + demand > capacity:
                            continue
                        
                        dist = self.dist_matrix[id_to_idx[pos_id]][id_to_idx[cid]]
                        travel_time = dist * (60.0 / self.calculator.v)
                        arrival_time = curr_time + travel_time
                        
                        Ei = c.get('E_i', 0)
                        Li = c.get('L_i', float('inf'))
                        
                        if arrival_time > Li:
                            continue
                        
                        time_urgency = max(0, arrival_time - c.get('due_date', Li))
                        score = dist + time_urgency * 0.5
                        
                        if score < best_score:
                            best_score = score
                            best_customer = c
                    
                    if best_customer:
                        route.append(best_customer['id'])
                        load += best_customer.get('demand', 0)
                        
                        dist = self.dist_matrix[id_to_idx[pos_id]][id_to_idx[best_customer['id']]]
                        curr_time += dist * (60.0 / self.calculator.v)
                        curr_time = max(curr_time, best_customer.get('ready_time', 0))
                        curr_time += best_customer.get('service_time', 0)
                        
                        pos_id = best_customer['id']
                        remaining.remove(best_customer)
                    else:
                        break
                
                route.append(0)
                
                if len(route) > 2:
                    current_solution.append(route)
                    route_count += 1
            
            # 处理剩余客户
            for c in remaining:
                current_solution.append([0, c['id'], 0])
            
            # 评估此初始解
            init_cost = self.cost(current_solution)
            print(f"  [初始解尝试 {init_attempt+1}/{num_initial_solutions}] 成本: {init_cost:.2f}, 路径数: {len(current_solution)}")
            
            if init_cost < best_initial_cost:
                best_initial_cost = init_cost
                best_initial_solution = copy.deepcopy(current_solution)
        
        # 使用最优初始解
        current_solution = best_initial_solution
        print(f"  [选择最优初始解] 成本: {best_initial_cost:.2f}")
        
        #  使用修复后的 cost 方法
        current_cost = self.cost(current_solution)
        is_feasible = self.validate(current_solution)
        
        best_solution = copy.deepcopy(current_solution)
        best_cost = current_cost
        
        cost_str = f"{current_cost:.2f}" if current_cost < float('inf') else "INF"
        print(f"\n[初始解] 成本: {cost_str}")
        print(f"[初始解] 路径数: {len(current_solution)}")
        print(f"[初始解] 可行性: {'✓' if is_feasible else '✗'}\n")
        
        if not is_feasible or current_cost == float('inf'):
            print("[警告] 初始解不可行，尝试每客户一车...")
            current_solution = [[0, c['id'], 0] for c in non_depot]
            random.shuffle(current_solution)
            current_cost = self.cost(current_solution)
            best_solution = copy.deepcopy(current_solution)
            best_cost = current_cost
        
        print("[ALNS] 开始迭代优化...\n")
        
        # ALNS参数 - 平衡探索与利用
        # 使用适中的初始温度，避免前期成本飙升
        T = max(current_cost * 0.10, 100) if current_cost < float('inf') else 100  # 初始温度（能接受差10%的解）
        
        # 自适应冷却系数：线性冷却策略
        # 目标：平滑地从探索过渡到利用
        target_ratio = 0.02  # 最终温度为初始温度的2%
        alpha = target_ratio ** (1.0 / max_iters)  # 在max_iters次迭代后达到目标温度
        
        print(f"[参数] 初始温度: {T:.2f}, 冷却系数: {alpha:.6f}, 迭代次数: {max_iters}")
        
        improvement_count = 0
        no_improve_count = 0
        last_improve_iter = 0
        
        # 评估周期和重启阈值与迭代次数成比例
        segment_size = max(50, int(max_iters * 0.25))  # 每25%迭代调整一次权重
        restart_threshold = max(60, int(max_iters * 0.6))  # 60%迭代无改善时重启（更宽容）
        
        for iteration in range(max_iters):
            self.current_iteration = iteration  # 更新当前迭代次数（用于history_removal）
            temp_solution = copy.deepcopy(current_solution)
            
            # 自适应破坏程度 - 使用保守策略避免成本暴涨
            # 阈值与总迭代次数成比例
            threshold_medium = max_iters * 0.27  # ~27%无改善
            threshold_high = max_iters * 0.53    # ~53%无改善
            
            if no_improve_count > threshold_high:
                # 长时间无改善，适度增大扰动到30%
                base_remove = max(3, int(len(non_depot) * 0.20))
                max_remove = max(5, int(len(non_depot) * 0.30))
            elif no_improve_count > threshold_medium:
                # 中等扰动20%
                base_remove = max(2, int(len(non_depot) * 0.15))
                max_remove = max(4, int(len(non_depot) * 0.20))
            else:
                # 小步优化（默认12%-18%）
                base_remove = max(2, int(len(non_depot) * 0.12))
                max_remove = max(3, int(len(non_depot) * 0.18))
            num_remove = random.randint(base_remove, max_remove)
            
            try:
                partial, removed_ids = self.destroy(temp_solution, num_remove)
                routes_after_destroy = len(partial)
                
                if removed_ids:
                    new_solution = self.insert(partial, removed_ids)
                else:
                    new_solution = partial
                
                routes_after_insert = len(new_solution)
                
                # ===== 关键修复：验证客户覆盖完整性 =====
                if not self.validate(new_solution):
                    # 如果修复后的解不可行（缺少客户），拒绝此解
                    raise ValueError("修复后的解缺少客户节点")
                
                # 禁用迭代中的2-opt（可能破坏时间窗）
                # 让ALNS自然优化
                
                new_cost = self.cost(new_solution)
            except Exception as e:
                # print(f"[错误] 迭代 {iteration} 出错: {e}")
                new_cost = float('inf')
                new_solution = temp_solution
            
            delta = new_cost - current_cost
            
            if new_cost < current_cost:
                current_solution = new_solution
                current_cost = new_cost
                no_improve_count = 0  # 重置计数
                
                if new_cost < best_cost:
                    best_solution = copy.deepcopy(new_solution)
                    best_cost = new_cost
                    improvement_count += 1
                    last_improve_iter = iteration
                    print(f"[迭代 {iteration:3d}] ✓ 新最优解! 成本: {best_cost:.2f}, 路径数: {len(best_solution)}")
                    
                    # 更新被修改节点的历史记录
                    for node in removed_ids:
                        self.node_history[node] = iteration
                    
                    self.update_weights(reward='global_best')  # σ₁=33
                else:
                    self.update_weights(reward='improved')  # σ₂=9
            
            elif T > 0.01 and delta < float('inf'):  # 降低温度下限，让后期也能接受差解
                # 添加成本上限保护：不接受比初始解差太多的解
                cost_limit = max(current_cost * 1.5, best_cost * 2.0)  # 最多接受50%差的解
                
                if new_cost <= cost_limit:
                    accept_prob = math.exp(-delta / T)
                    if random.random() < accept_prob:
                        current_solution = new_solution
                        current_cost = new_cost
                        self.update_weights(reward='accepted')  # σ₃=3
                        no_improve_count += 1
                    else:
                        self.update_weights(reward='rejected')  # σ₄=0
                        no_improve_count += 1
                else:
                    # 超出成本上限，直接拒绝
                    self.update_weights(reward='rejected')  # σ₄=0
                    no_improve_count += 1
            else:
                self.update_weights(reward='rejected')  # σ₄=0
                no_improve_count += 1
            
            T *= alpha
            
            # ===== 技巧5：记录和可视化 =====
            # 记录成本变化
            self.cost_history.append(current_cost)
            self.best_cost_history.append(best_cost)
            
            # 每 segment_size 轮记录一次权重
            if (iteration + 1) % segment_size == 0:
                self.weight_history['destroy'].append(list(self.d_weights))
                self.weight_history['insert'].append(list(self.i_weights))
            
            # 重启机制：连续一定次数无改善时重启到最优解（带再加热）
            if no_improve_count > restart_threshold:
                current_solution = copy.deepcopy(best_solution)
                current_cost = best_cost
                no_improve_count = 0
                # 再加热：恢复到初始温度的50%，增强探索
                T = max(best_cost * 0.10, 100) * 0.5  
                print(f"[迭代 {iteration:3d}] 重启到最优解（再加热T={T:.1f}，阈值={restart_threshold}）")
            
            # 周期性权重归一化（每 segment_size 轮）
            if (iteration + 1) % segment_size == 0:
                # 归一化破坏算子权重
                d_sum = sum(self.d_weights)
                if d_sum > 0:
                    self.d_weights = [w / d_sum * len(self.d_weights) for w in self.d_weights]
                
                # 归一化修复算子权重
                i_sum = sum(self.i_weights)
                if i_sum > 0:
                    self.i_weights = [w / i_sum * len(self.i_weights) for w in self.i_weights]
            
            if (iteration + 1) % 5 == 0:
                # 输出格式：Iter N: Current=XXX, Best=YYY (方便提取用于可视化)
                print(f"Iter {iteration+1:3d}: Current={current_cost:.2f}, Best={best_cost:.2f}, Temp={T:.2f}, Routes={len(current_solution)}")
        
        # 尝试2-opt优化，但只在成本改善时才保留
        optimized_solution = self.local_search(copy.deepcopy(best_solution))
        optimized_cost = self.cost(optimized_solution)
        if optimized_cost < best_cost:
            best_solution = optimized_solution
            best_cost = optimized_cost
            print(f"[后处理] 2-opt优化成功: {best_cost:.2f}")
        
        print(f"\n{'='*60}")
        print(f"【最终结果】最优成本: {best_cost:.2f}" if best_cost < float('inf') else "【最终结果】最优成本: INF")
        print(f"【最终结果】路径数量: {len(best_solution)}")
        print(f"【最终结果】改进次数: {improvement_count}")
        print(f"【最终结果】最后改进: 第 {last_improve_iter} 次迭代")
        print(f"{'='*60}\n")
        
        return best_solution, best_cost
    
    def solve_multi_run(self, max_iters=300, num_runs=3, base_seed=42):
        '''多次运行取最优解
        
        Args:
            max_iters: 每次运行的迭代次数
            num_runs: 运行次数
            base_seed: 基础随机种子（确保可重复）
        '''
        best_overall = float('inf')
        best_sol_overall = None
        all_costs = []
        
        for run in range(num_runs):
            print(f"\n{'#'*60}")
            print(f"# 第 {run+1}/{num_runs} 次运行")
            print(f"{'#'*60}")
            
            # 每次使用不同但可控的种子
            sol, cost = self.solve(max_iters=max_iters, seed=base_seed + run)
            all_costs.append(cost)
            
            if cost < best_overall:
                best_overall = cost
                best_sol_overall = copy.deepcopy(sol)
                print(f"*** 发现更优解: {cost:.2f} ***")
        
        avg_cost = sum(all_costs) / len(all_costs)
        
        print(f"\n{'='*60}")
        print(f"【多次运行汇总】")
        print(f"  最优成本: {best_overall:.2f}")
        print(f"  平均成本: {avg_cost:.2f}")
        print(f"  路径数量: {len(best_sol_overall)}")
        print(f"  各次成本: {[f'{c:.2f}' for c in all_costs]}")
        print(f"{'='*60}\n")
        
        return best_sol_overall, best_overall
"""