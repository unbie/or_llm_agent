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
        
        # ALNS 算子权重管理（从 plugin 移到 solver）
        self.destroy_ops = [self.plugin.random_removal, self.plugin.worst_removal]
        self.insert_ops = [self.plugin.greedy_insert]
        self.d_weights = [1.0] * len(self.destroy_ops)
        self.i_weights = [1.0] * len(self.insert_ops)
        self.last_d_idx = 0
        self.last_i_idx = 0
        self.rho = 0.1
        
        print("Solver 初始化成功")
    
    def destroy(self, solution, remove_ratio=0.2):
        '''调用 Destroy 算子移除解中的客户节点'''
        solution = copy.deepcopy(solution)
        self.last_d_idx = random.choices(range(len(self.destroy_ops)), weights=self.d_weights)[0]
        op = self.destroy_ops[self.last_d_idx]
        return op(solution, remove_ratio)
    
    def insert(self, partial_solution, removed_nodes):
        '''调用 Insert 算子将 removed_nodes 插回解中'''
        partial_solution = copy.deepcopy(partial_solution)
        self.last_i_idx = random.choices(range(len(self.insert_ops)), weights=self.i_weights)[0]
        op = self.insert_ops[self.last_i_idx]
        return op(partial_solution, removed_nodes)
    
    def update_weights(self, reward):
        '''更新算子权重'''
        self.d_weights[self.last_d_idx] = (
            self.d_weights[self.last_d_idx] * (1 - self.rho) + reward * self.rho
        )
        self.i_weights[self.last_i_idx] = (
            self.i_weights[self.last_i_idx] * (1 - self.rho) + reward * self.rho
        )
    
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
    
    def solve(self, max_iters=100):
        '''ALNS 求解器'''
        print("[初始化] 开始生成初始解...")
        
        non_depot = [c for c in self.customers if c['id'] != 0]
        capacity = self.data.get('vehicle_capacity', 200)
        
        id_to_idx = {c['id']: i for i, c in enumerate(self.customers)}
        
        current_solution = []
        remaining = list(non_depot)
        remaining.sort(key=lambda c: c.get('demand', 0), reverse=True)
        
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
                print(f"  [路径 {route_count}] 客户数: {len(route)-2}, 负载: {load}/{capacity}")
        
        if remaining:
            print(f"  [兜底] {len(remaining)} 个客户单独成路径")
            for c in remaining:
                current_solution.append([0, c['id'], 0])
                route_count += 1
        
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
        
        # 优化后的ALNS参数
        T = max(current_cost * 0.05, 50) if current_cost < float('inf') else 500  # 降低初始温度
        alpha = 0.995  # 更慢的冷却速度
        improvement_count = 0
        no_improve_count = 0  # 连续无改善计数
        
        for iteration in range(max_iters):
            temp_solution = copy.deepcopy(current_solution)
            
            # 自适应移除比例：长时间无改善时增加移除量
            if no_improve_count > 20:
                base_remove = max(2, int(len(non_depot) * 0.15))
                max_remove = max(4, int(len(non_depot) * 0.4))
            else:
                base_remove = max(1, int(len(non_depot) * 0.05))  # 减少移除量
                max_remove = max(2, int(len(non_depot) * 0.2))
            num_remove = random.randint(base_remove, max_remove)
            
            try:
                partial, removed_ids = self.destroy(temp_solution, num_remove)
                
                if removed_ids:
                    new_solution = self.insert(partial, removed_ids)
                else:
                    new_solution = partial
                
                new_cost = self.cost(new_solution)
            except Exception as e:
                print(f"[错误] 迭代 {iteration} 出错: {e}")
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
                    print(f"[迭代 {iteration:3d}] ✓ 新最优解! 成本: {best_cost:.2f}, 路径数: {len(best_solution)}")
                
                self.update_weights(reward=3.0)
            
            elif T > 0.1 and delta < float('inf'):
                accept_prob = math.exp(-delta / T)
                if random.random() < accept_prob:
                    current_solution = new_solution
                    current_cost = new_cost
                    self.update_weights(reward=1.0)
                    no_improve_count += 1
                else:
                    self.update_weights(reward=0.5)
                    no_improve_count += 1
            else:
                self.update_weights(reward=0.5)
                no_improve_count += 1
            
            T *= alpha
            
            # 早停：连续50次无改善且温度已经很低
            if no_improve_count > 50 and T < 1.0:
                print(f"[迭代 {iteration:3d}] 早停：连续{no_improve_count}次无改善")
                break
            
            if (iteration + 1) % 20 == 0:
                print(f"[迭代 {iteration+1:3d}] 当前: {current_cost:.2f}, 最优: {best_cost:.2f}, 温度: {T:.2f}, 路径数: {len(current_solution)}")
        
        print(f"\n{'='*60}")
        print("[最终结果] 最优成本: " + (f"{best_cost:.2f}" if best_cost < float('inf') else "INF"))
        print(f"[最终结果] 路径数量: {len(best_solution)}")
        print(f"[最终结果] 改善次数: {improvement_count}")
        print(f"{'='*60}\n")
        
        return best_solution, best_cost
"""