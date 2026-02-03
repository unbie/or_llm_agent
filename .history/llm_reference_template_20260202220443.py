# LLM生成算子的完整参考模板
# 包含所有改进：容量约束、需求因子、安全归一化

import random
import math

class HeuristicPlugin:
    def __init__(self, **kwargs):  
        self.data = kwargs
        self.capacity = kwargs.get('vehicle_capacity', 200)
        self.customers = kwargs.get('customers', [])
        self.dist_matrix = None
        self.customer_lookup = {c['id']: c for c in self.customers}
        self.node_history = {}

    # ==================== 破坏算子 ====================
    
    def random_removal(self, solution, ratio):
        """随机移除节点"""
        all_nodes = []
        for route_idx, route in enumerate(solution):
            for pos_idx, node in enumerate(route):
                if node != 0:
                    all_nodes.append((route_idx, pos_idx, node))
        
        if not all_nodes:
            return solution, []
        
        total_customers = len(all_nodes)
        if ratio <= 1.0:
            n = max(1, math.ceil(total_customers * ratio))
        else:
            n = int(ratio)
        n = min(n, total_customers)
        
        selected = random.sample(all_nodes, n)
        selected.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        new_solution = [route[:] for route in solution]
        removed_nodes = []
        for route_idx, pos_idx, node in selected:
            del new_solution[route_idx][pos_idx]
            removed_nodes.append(node)
        
        new_solution = [route for route in new_solution if len(route) > 2]
        return new_solution, removed_nodes

    def worst_removal(self, solution, ratio):
        """移除边际贡献最大的节点（考虑距离+需求）"""
        node_contributions = []
        for route_idx, route in enumerate(solution):
            if len(route) < 3:
                continue
            for pos_idx in range(1, len(route)-1):
                node = route[pos_idx]
                prev_node = route[pos_idx-1]
                next_node = route[pos_idx+1]
                
                dist_contrib = (self.dist_matrix[prev_node][node] + 
                               self.dist_matrix[node][next_node] - 
                               self.dist_matrix[prev_node][next_node])
                
                customer = self.customer_lookup.get(node, {})
                demand = customer.get('demand', 0)
                contrib = dist_contrib + demand * 0.01
                
                node_contributions.append((route_idx, pos_idx, node, contrib))
        
        if not node_contributions:
            return solution, []
        
        total_customers = len(node_contributions)
        if ratio <= 1.0:
            n = max(1, math.ceil(total_customers * ratio))
        else:
            n = int(ratio)
        n = min(n, total_customers)
        
        node_contributions.sort(key=lambda x: x[3], reverse=True)
        selected = node_contributions[:n]
        selected.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        new_solution = [route[:] for route in solution]
        removed_nodes = []
        for route_idx, pos_idx, node, _ in selected:
            del new_solution[route_idx][pos_idx]
            removed_nodes.append(node)
        
        new_solution = [route for route in new_solution if len(route) > 2]
        return new_solution, removed_nodes

    def related_removal(self, solution, ratio):
        """移除地理位置相近的节点"""
        all_nodes = []
        for route_idx, route in enumerate(solution):
            for pos_idx, node in enumerate(route):
                if node != 0:
                    all_nodes.append((route_idx, pos_idx, node))
        
        if not all_nodes:
            return solution, []
        
        seed = random.choice(all_nodes)
        seed_node = seed[2]
        
        node_distances = []
        for route_idx, pos_idx, node in all_nodes:
            dist = 0.0 if node == seed_node else self.dist_matrix[seed_node][node]
            node_distances.append((route_idx, pos_idx, node, dist))
        
        node_distances.sort(key=lambda x: x[3])
        
        total_customers = len(all_nodes)
        if ratio <= 1.0:
            n = max(1, math.ceil(total_customers * ratio))
        else:
            n = int(ratio)
        n = min(n, total_customers)
        
        selected = node_distances[:n]
        selected.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        new_solution = [route[:] for route in solution]
        removed_nodes = []
        for route_idx, pos_idx, node, _ in selected:
            del new_solution[route_idx][pos_idx]
            removed_nodes.append(node)
        
        new_solution = [route for route in new_solution if len(route) > 2]
        return new_solution, removed_nodes

    def shaw_removal(self, solution, ratio):
        """基于距离、时间窗、需求综合相似性移除节点"""
        all_nodes = []
        for route_idx, route in enumerate(solution):
            for pos_idx, node in enumerate(route):
                if node != 0:
                    all_nodes.append((route_idx, pos_idx, node))
        
        if not all_nodes:
            return solution, []
        
        seed = random.choice(all_nodes)
        seed_node = seed[2]
        seed_customer = self.customer_lookup[seed_node]
        
        # 安全归一化
        distances = [self.dist_matrix[seed_node][x[2]] for x in all_nodes]
        max_distance = max(distances) if distances else 1.0
        if max_distance < 0.001:
            max_distance = 1.0
        
        time_diffs = [abs(seed_customer.get('ready_time', 0) - 
                         self.customer_lookup[x[2]].get('ready_time', 0)) 
                     for x in all_nodes]
        max_time_diff = max(time_diffs) if time_diffs else 1.0
        if max_time_diff < 0.001:
            max_time_diff = 1.0
        
        demand_diffs = [abs(seed_customer.get('demand', 0) - 
                           self.customer_lookup[x[2]].get('demand', 0)) 
                       for x in all_nodes]
        max_demand = max(demand_diffs) if demand_diffs else 1.0
        if max_demand < 0.001:
            max_demand = 1.0
        
        shaw_scores = []
        for route_idx, pos_idx, node in all_nodes:
            if node == seed_node:
                shaw_scores.append((route_idx, pos_idx, node, 0.0))
            else:
                customer = self.customer_lookup[node]
                normalized_dist = self.dist_matrix[seed_node][node] / max_distance
                time_diff = abs(seed_customer.get('ready_time', 0) - 
                               customer.get('ready_time', 0)) / max_time_diff
                demand_diff = abs(seed_customer.get('demand', 0) - 
                                 customer.get('demand', 0)) / max_demand
                shaw_score = 9 * normalized_dist + 3 * time_diff + 2 * demand_diff
                shaw_scores.append((route_idx, pos_idx, node, shaw_score))
        
        shaw_scores.sort(key=lambda x: x[3])
        
        total_customers = len(all_nodes)
        if ratio <= 1.0:
            n = max(1, math.ceil(total_customers * ratio))
        else:
            n = int(ratio)
        n = min(n, total_customers)
        
        selected = shaw_scores[:n]
        selected.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        new_solution = [route[:] for route in solution]
        removed_nodes = []
        for route_idx, pos_idx, node, _ in selected:
            del new_solution[route_idx][pos_idx]
            removed_nodes.append(node)
        
        new_solution = [route for route in new_solution if len(route) > 2]
        return new_solution, removed_nodes

    def history_removal(self, solution, ratio):
        """移除长期未被修改的节点"""
        all_nodes = []
        for route_idx, route in enumerate(solution):
            for pos_idx, node in enumerate(route):
                if node != 0:
                    all_nodes.append((route_idx, pos_idx, node))
        
        if not all_nodes:
            return solution, []
        
        if not self.node_history:
            return self.random_removal(solution, ratio)
        
        node_history_values = []
        for route_idx, pos_idx, node in all_nodes:
            history = self.node_history.get(node, 0)
            node_history_values.append((route_idx, pos_idx, node, history))
        
        node_history_values.sort(key=lambda x: x[3])
        
        total_customers = len(all_nodes)
        if ratio <= 1.0:
            n = max(1, math.ceil(total_customers * ratio))
        else:
            n = int(ratio)
        n = min(n, total_customers)
        
        selected = node_history_values[:n]
        selected.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        new_solution = [route[:] for route in solution]
        removed_nodes = []
        for route_idx, pos_idx, node, _ in selected:
            del new_solution[route_idx][pos_idx]
            removed_nodes.append(node)
        
        new_solution = [route for route in new_solution if len(route) > 2]
        return new_solution, removed_nodes

    def cluster_removal(self, solution, ratio):
        """基于网格聚类移除节点"""
        all_nodes = []
        node_coords = {}
        for route_idx, route in enumerate(solution):
            for pos_idx, node in enumerate(route):
                if node != 0:
                    customer = self.customer_lookup[node]
                    all_nodes.append((route_idx, pos_idx, node))
                    node_coords[node] = (customer.get('x', 0), customer.get('y', 0))
        
        if not all_nodes:
            return solution, []
        
        xs = [coord[0] for coord in node_coords.values()]
        ys = [coord[1] for coord in node_coords.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        if max_x == min_x: max_x = min_x + 1
        if max_y == min_y: max_y = min_y + 1
        
        grid_size = 5
        grid = {}
        for node, (x, y) in node_coords.items():
            gx = int((x - min_x) / (max_x - min_x) * grid_size)
            gy = int((y - min_y) / (max_y - min_y) * grid_size)
            gx = min(gx, grid_size - 1)
            gy = min(gy, grid_size - 1)
            grid.setdefault((gx, gy), []).append(node)
        
        non_empty_grids = [nodes for nodes in grid.values() if nodes]
        if not non_empty_grids:
            return solution, []
        
        selected_cell_nodes = random.choice(non_empty_grids)
        cluster_nodes = set(selected_cell_nodes)
        
        total_customers = len(all_nodes)
        if ratio <= 1.0:
            n = max(1, math.ceil(total_customers * ratio))
        else:
            n = int(ratio)
        n = min(n, total_customers)
        
        if len(cluster_nodes) < n:
            for cell_coord, nodes in grid.items():
                if set(nodes) & cluster_nodes:
                    gx, gy = cell_coord
                    break
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    neighbor_coord = (gx + dx, gy + dy)
                    if neighbor_coord in grid:
                        cluster_nodes.update(grid[neighbor_coord])
                        if len(cluster_nodes) >= n:
                            break
                if len(cluster_nodes) >= n:
                    break
        
        nodes_to_remove = list(cluster_nodes)[:n]
        
        selected = []
        for route_idx, pos_idx, node in all_nodes:
            if node in nodes_to_remove:
                selected.append((route_idx, pos_idx, node))
        
        selected.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        new_solution = [route[:] for route in solution]
        removed_nodes = []
        for route_idx, pos_idx, node in selected:
            del new_solution[route_idx][pos_idx]
            removed_nodes.append(node)
        
        new_solution = [route for route in new_solution if len(route) > 2]
        return new_solution, removed_nodes

    # ==================== 修复算子（关键：添加容量约束）====================
    
    def greedy_insert(self, solution, removed_nodes):
        """贪心插入 - 带容量约束检查"""
        if not removed_nodes:
            return solution
        
        new_solution = [route[:] for route in solution]
        
        for node in removed_nodes:
            node_demand = self.customer_lookup[node].get('demand', 0)
            best_cost = float('inf')
            best_route_idx = None
            best_position = None
            
            for route_idx, route in enumerate(new_solution):
                if len(route) < 2:
                    continue
                
                # 【关键】检查容量约束
                current_load = sum(self.customer_lookup[n].get('demand', 0) 
                                  for n in route if n != 0)
                if current_load + node_demand > self.capacity:
                    continue
                
                for pos in range(1, len(route)):
                    prev = route[pos-1]
                    next_node = route[pos]
                    
                    cost_increase = (self.dist_matrix[prev][node] + 
                                    self.dist_matrix[node][next_node] - 
                                    self.dist_matrix[prev][next_node])
                    
                    if cost_increase < best_cost:
                        best_cost = cost_increase
                        best_route_idx = route_idx
                        best_position = pos
            
            if best_route_idx is not None:
                new_solution[best_route_idx].insert(best_position, node)
            else:
                new_solution.append([0, node, 0])
        
        return new_solution

    def regret_insert(self, solution, removed_nodes):
        """后悔插入 - 带容量约束检查"""
        if not removed_nodes:
            return solution
        
        new_solution = [route[:] for route in solution]
        remaining_nodes = list(removed_nodes)
        
        while remaining_nodes:
            best_regret = -float('inf')
            best_node = None
            best_route_idx = None
            best_position = None
            
            for node in remaining_nodes:
                node_demand = self.customer_lookup[node].get('demand', 0)
                costs = []
                
                for route_idx, route in enumerate(new_solution):
                    if len(route) < 2:
                        continue
                    
                    # 【关键】检查容量约束
                    current_load = sum(self.customer_lookup[n].get('demand', 0) 
                                      for n in route if n != 0)
                    if current_load + node_demand > self.capacity:
                        continue
                    
                    for pos in range(1, len(route)):
                        prev, next_n = route[pos-1], route[pos]
                        cost_inc = (self.dist_matrix[prev][node] + 
                                   self.dist_matrix[node][next_n] - 
                                   self.dist_matrix[prev][next_n])
                        costs.append((cost_inc, route_idx, pos))
                
                new_cost = self.dist_matrix[0][node] + self.dist_matrix[node][0]
                costs.append((new_cost, None, None))
                
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
            
            if best_route_idx is not None:
                new_solution[best_route_idx].insert(best_position, best_node)
            else:
                new_solution.append([0, best_node, 0])
            
            remaining_nodes.remove(best_node)
        
        return new_solution

    def random_insert(self, solution, removed_nodes):
        """随机插入 - 带容量约束检查"""
        if not removed_nodes:
            return solution
        
        new_solution = [route[:] for route in solution]
        
        for node in removed_nodes:
            node_demand = self.customer_lookup[node].get('demand', 0)
            positions = []
            
            for route_idx, route in enumerate(new_solution):
                if len(route) < 2:
                    continue
                
                # 【关键】检查容量约束
                current_load = sum(self.customer_lookup[n].get('demand', 0) 
                                  for n in route if n != 0)
                if current_load + node_demand > self.capacity:
                    continue
                
                for pos in range(1, len(route)):
                    positions.append((route_idx, pos))
            
            positions.append((None, None))
            
            choice = random.choice(positions)
            
            if choice[0] is not None:
                new_solution[choice[0]].insert(choice[1], node)
            else:
                new_solution.append([0, node, 0])
        
        return new_solution
