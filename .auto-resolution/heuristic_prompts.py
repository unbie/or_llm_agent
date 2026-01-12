HEURISTIC_PLUGIN_TEMPLATE = r"""

# -*- coding: utf-8 -*-
import random
import math
import copy

class HeuristicPlugin:
    def __init__(self, data):
        self.data = data
        self.capacity = data['vehicle_capacity']
        self.customers = data['customers']
        self.dist_matrix = []  # 由 Solver 注入

        # ==========================
        # 车辆与成本参数 (基础参数设置)
        # ==========================
        self.vehicle_cost = data.get('vehicle_cost', 240)        # 固定成本
        self.drive_cost = data.get('drive_cost', 3)             # 距离成本
        self.cold_cost = data.get('cold_cost', 15)             # 制冷成本
        self.average_speed = data.get('average_speed', 40)     # 速度 (km/h)
        self.product_price = data.get('product_price', 5000)   # 生鲜单价
        self.penalty_early = data.get('penalty_early', 20)     # 早到惩罚
        self.penalty_late = data.get('penalty_late', 40)       # 迟到惩罚

        # ==========================
        # 生鲜损耗参数
        # ==========================
        self.freshness_decay_transport = data.get('theta1', 0.002)
        self.freshness_decay_service = data.get('theta2', 0.005)

        # ==========================
        # ALNS 算子与权重更新
        # ==========================
        self.destroy_ops = [self.random_removal, self.worst_removal]
        self.insert_ops = [self.greedy_insert]

        self.d_weights = [1.0] * len(self.destroy_ops)
        self.i_weights = [1.0] * len(self.insert_ops)
        self.last_d_idx = 0
        self.last_i_idx = 0
        self.rho = 0.1

    # ==========================
    # 成本计算 (修复路径遍历逻辑)
    # ==========================
    def cost(self, solution):
        total_cost = 0.0
        for route in solution:
            if len(route) < 3: # 路径必须包含 [0, 客户, 0]
                continue
            route_dist = 0.0
            route_time = 0.0
            damage_cost_total = 0.0
            penalty_cost_total = 0.0
            prev_node = 0
            
            # 【修复】只遍历中间的客户节点，排除末尾的仓库节点 0
            for node in route[1:-1]:
                cust = self.customers[node]
                dist = self.dist_matrix[prev_node][node]
                route_dist += dist
                route_time += dist / self.average_speed

                # 损耗计算: 基于运输时间与服务时间
                t_transport = route_time
                t_service = cust.get('service_time', 0)
                freshness_init = cust.get('freshness_init', 0.98)
                
                # 生鲜变质率计算公式
                r_i = 1 - freshness_init * math.exp(
                    -self.freshness_decay_transport * t_transport
                    - self.freshness_decay_service * t_service
                )
                r_i = max(r_i, 0.0)

                delta1 = cust.get('customer_delta', 0.02)
                delta2 = cust.get('supplier_delta', 0.05)
                D_i = cust.get('demand', 0)
                P_i = cust.get('return_qty', 0)
                
                # 货损成本
                damage_cost_total += self.product_price * (
                    max(r_i - delta1, 0) * D_i + max(r_i - delta2, 0) * P_i
                )

                # 时间窗惩罚计算
                ready = cust.get('ready_time', 0)
                due = cust.get('due_date', float('inf'))
                if route_time < ready:
                    penalty_cost_total += self.penalty_early * (ready - route_time)
                    route_time = ready
                elif route_time > due:
                    penalty_cost_total += self.penalty_late * (route_time - due)
                
                route_time += t_service
                prev_node = node

            # 返回仓库的额外成本
            dist_to_depot = self.dist_matrix[prev_node][0]
            route_dist += dist_to_depot
            route_time += dist_to_depot / self.average_speed

            total_cost += (self.vehicle_cost + route_dist * self.drive_cost + 
                          route_time * self.cold_cost + damage_cost_total + penalty_cost_total)
        return total_cost

    # ==========================
    # ALNS 核心接口
    # ==========================
    def destroy(self, solution, remove_ratio=0.2):
        
        # 破坏算子：移除部分节点。
        # 返回: (partial_solution, removed_nodes)
        # 注意:
        # 1.不要移除节点
        # 2. 如果某路径移除节点后只剩[0, 0]，应将其从解中删除。
        
        solution = copy.deepcopy(solution)
        self.last_d_idx = random.choices(range(len(self.destroy_ops)), weights=self.d_weights)[0]
        op = self.destroy_ops[self.last_d_idx]

        # 执行具体算子逻辑
        return op(solution, remove_ratio)

    def insert(self, solution, removed_nodes):
        solution = copy.deepcopy(solution)
        self.last_i_idx = random.choices(range(len(self.insert_ops)), weights=self.i_weights)[0]
        op = self.insert_ops[self.last_i_idx]
        return op(solution, removed_nodes)
    
    def update_weights(self, reward):
        self.d_weights[self.last_d_idx] = self.d_weights[self.last_d_idx]*(1-self.rho) + reward*self.rho
        self.i_weights[self.last_i_idx] = self.i_weights[self.last_i_idx]*(1-self.rho) + reward*self.rho

    # ==========================
    # Destroy operators
    # ==========================
    def random_removal(self, solution, ratio):
        # TODO: Implement optimized random removal strategy
        removed = []
        all_customers = [c for route in solution for c in route[1:-1]]
        num_remove = max(1, int(len(all_customers) * ratio))
        removed = random.sample(all_customers, num_remove)
        for c in removed:
            for route in solution:
                if c in route:
                    route.remove(c)
        solution = [r for r in solution if len(r) > 2]
        return solution, removed
    
    def worst_removal(self, solution, ratio):
        # TODO: Implement worst removal based on time window urgency
        removal_costs = []
        for route in solution:
            if len(route) < 3:
                continue
            for i in range(1, len(route)-1):
                node = route[i]
                cust = self.customers[node]
                time_window_width = cust.get('due_date', float('inf')) - cust.get('ready_time', 0)
                time_remaining = cust.get('due_date', float('inf')) - self.arrival_time(route, i)
                cost = (1.0 / max(time_window_width, 0.1)) + (1.0 / max(time_remaining, 0.1))
                removal_costs.append((cost, node))
        removal_costs.sort(key=lambda x: x[0], reverse=True)
        num_remove = max(1, int(len(removal_costs) * ratio))
        removed = [node for _, node in removal_costs[:num_remove]]
        for c in removed:
            for route in solution:
                if c in route:
                    route.remove(c)
        solution = [r for r in solution if len(r) > 2]
        return solution, removed

    # ==========================
    # Insert operator
    # ==========================
    def greedy_insert(self, solution, removed_nodes):
        # TODO: Implement greedy insertion strategy
        removed_nodes_sorted = sorted(
            removed_nodes,
            key=lambda node: self.customers[node].get('due_date', float('inf'))
        )
        for node in removed_nodes_sorted:
            best_cost = float('inf')
            best_route = None
            best_pos = None
            for r_idx, route in enumerate(solution):
                for pos in range(1, len(route)):
                    trial_route = route[:pos] + [node] + route[pos:]
                    if self.check_feasible(trial_route):
                        cost_trial = self.cost([trial_route])
                        if cost_trial < best_cost:
                            best_cost = cost_trial
                            best_route = r_idx
                            best_pos = pos
            if best_route is not None:
                solution[best_route].insert(best_pos, node)
            else:
                solution.append([0, node, 0])
        return solution

    # ==========================
    # Helper functions
    # ==========================
    def arrival_time(self, route, pos):
        time = 0.0
        prev_node = 0
        for i in range(1, pos+1):
            node = route[i]
            cust = self.customers[node]
            time += self.dist_matrix[prev_node][node] / self.average_speed
            ready = cust.get('ready_time', 0)
            if time < ready:
                time = ready
            if i < pos: # 只有到达之前的节点才需要加服务时间
                time += cust.get('service_time', 0)
            prev_node = node
        return time

    def check_feasible(self, route):
        if not route or len(route) < 3:
            return True
        load = 0
        time = 0.0
        prev_node = 0
        # 【修复】只检查中间客户节点的约束
        for node in route[1:-1]:
            cust = self.customers[node]
            travel_time = self.dist_matrix[prev_node][node] / self.average_speed
            time += travel_time
            if time > cust.get('due_date', float('inf')):
                return False
            time = max(time, cust.get('ready_time', 0))
            load += cust.get('demand', 0)
            if load > self.capacity:
                return False
            time += cust.get('service_time', 0)
            prev_node = node
        
        # 检查返回仓库的可行性
        time += self.dist_matrix[prev_node][0] / self.average_speed
        if time > self.customers[0].get('due_date', float('inf')):
            return False
        return True
"""