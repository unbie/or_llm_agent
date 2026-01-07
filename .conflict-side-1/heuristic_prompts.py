HEURISTIC_PLUGIN_TEMPLATE = r"""
import random
import math
import copy

# =======================================================
# Heuristic Plugin Template (for LLM completion)
#
# IMPORTANT — Route Semantics (MANDATORY):
#
# - Each route MUST start with depot 0 and MUST end with depot 0.
# - A valid route is represented as:
#       [0, customer_1, ..., customer_k, 0]
#
# - Depot node (0) is INCLUDED at both the beginning and the end.
#
# - Cost, time, cooling, and feasibility checks MUST iterate over
#   the FULL route using:
#
#       for i in range(1, len(route)):
#           prev_node = route[i-1]
#           curr_node = route[i]
#
# - This loop ALREADY covers:
#       depot -> first customer
#       customer -> customer
#       last customer -> depot
#
# - DO NOT separately compute return-to-depot cost or time
#   outside the loop.
#
# - Depot properties:
#       demand = 0
#       service_time = 0
#       time-window MUST still be enforced (due_date constraint).
#
# Distance matrix self.dist_matrix is injected by Solver.
# =======================================================

class HeuristicPlugin:
    def __init__(self, data):
        self.data = data
        self.capacity = data['vehicle_capacity']
        self.customers = data['customers']
        self.dist_matrix = []  # injected by Solver

        # Perishable defaults (use when missing in data)
        self.vehicle_cost = data.get('vehicle_cost', 240)
        self.drive_cost = data.get('drive_cost', 3)
        self.cold_cost = data.get('cold_cost', 15)
        self.average_speed = data.get('average_speed', 40)
        self.product_price = data.get('product_price', 5000)
        self.penalty_early = data.get('penalty_early', 20)
        self.penalty_late = data.get('penalty_late', 40)

        # Freshness decay parameters
        self.freshness_decay_transport = data.get('theta1', 0.002)
        self.freshness_decay_service = data.get('theta2', 0.005)
        self.default_delta_customer = data.get('customer_delta', 0.02)
        self.default_delta_supplier = data.get('supplier_delta', 0.05)
        self.default_freshness_init = data.get('freshness_init', 0.98)

        # ALNS operators
        self.destroy_ops = [self.random_removal, self.worst_removal]
        self.insert_ops = [self.greedy_insert]

        # Operator weights
        self.d_weights = [1.0] * len(self.destroy_ops)
        self.i_weights = [1.0] * len(self.insert_ops)
        self.last_d_idx = 0
        self.last_i_idx = 0
        self.rho = 0.1  # learning rate

    # ===================================================
    # Cost Evaluation
    # ===================================================
    def cost(self, solution):
        
        # Cost evaluation MUST follow the route semantics defined above.
        # 
        # Rules:
        # - Traverse every route using:
        #       for i in range(1, len(route))
        # - Transportation cost and cooling cost are computed INSIDE the loop.
        # - Freshness decay happens during transport and service.
        # - Time-window penalties:
        #       early arrival  -> penalty_early
        #       late arrival   -> penalty_late
        # - Depot node (0):
        #       - demand = 0, so no loss cost
        #       - service_time = 0
        #       - transportation to depot IS counted
        #       - time-window constraint IS checked
        # - DO NOT compute return-to-depot cost after the loop.
        # 

        # TODO:
        # - Traverse all routes
        # - Accumulate:
        #     vehicle_cost
        #     drive_cost * distance
        #     cold_cost * travel_time
        #     freshness-based damage cost
        #     time-window penalties

        total_cost = 0.0
        return total_cost

    # ===================================================
    # Feasibility Check
    # ===================================================
    def validate(self, solution):
        
        # Feasibility rules:
        # 
        # - Capacity and time-window constraints MUST be checked.
        # - Validation MUST traverse the FULL route including final depot 0.
        # - Use the SAME loop structure as cost():
        # 
        #       current_time = 0
        #       current_load = 0
        #       prev_node = route[0]  # depot
        # 
        #       for i in range(1, len(route)):
        #           curr_node = route[i]
        # 
        # - Capacity:
        #       cumulative demand <= vehicle_capacity
        # - Time window:
        #       if arrival < ready_time -> wait
        #       if arrival > due_date   -> infeasible
        # - For final depot (curr_node == 0):
        #       demand = 0
        #       service_time = 0
        #       due_date MUST still be enforced
        # 

        # TODO:
        # - Check cumulative demand
        # - Check arrival time against ready_time / due_date
        # - Enforce due_date constraint for final depot

        return True

    # ===================================================
    # ALNS Interface
    # ===================================================
    def destroy(self, solution, remove_ratio=0.2):
        solution = copy.deepcopy(solution)
        self.last_d_idx = random.choices(
            range(len(self.destroy_ops)),
            weights=self.d_weights
        )[0]
        op = self.destroy_ops[self.last_d_idx]
        return op(solution, remove_ratio)

    def insert(self, partial_solution, removed_nodes):
        partial_solution = copy.deepcopy(partial_solution)
        self.last_i_idx = random.choices(
            range(len(self.insert_ops)),
            weights=self.i_weights
        )[0]
        op = self.insert_ops[self.last_i_idx]
        return op(partial_solution, removed_nodes)

    def update_weights(self, reward):
        self.d_weights[self.last_d_idx] = (
            self.d_weights[self.last_d_idx] * (1 - self.rho)
            + reward * self.rho
        )
        self.i_weights[self.last_i_idx] = (
            self.i_weights[self.last_i_idx] * (1 - self.rho)
            + reward * self.rho
        )

    # ===================================================
    # Destroy Operators
    # ===================================================
    def random_removal(self, solution, ratio):
        
        # Random removal operator.
        # 
        # Rules:
        # - Only CUSTOMER nodes may be removed (exclude depot 0).
        # - Removal count >= 1.
        # - Return:
        #       new_solution, removed_nodes
        # 

        # TODO:
        # - Collect all customer nodes
        # - Randomly remove by ratio

        removed = []
        return solution, removed

    def worst_removal(self, solution, ratio):
        
        # Worst removal operator.
        # 
        # Rules:
        # - Score CUSTOMER nodes by cost contribution or time-window tightness.
        # - Remove highest-score nodes.
        # - Removal count >= 1.
        # 

        # TODO:
        # - Compute node scores
        # - Remove worst nodes

        removed = []
        return solution, removed

    # ===================================================
    # Insert Operator
    # ===================================================
    def greedy_insert(self, solution, removed_nodes):
         
        # Greedy insertion operator.
        # 
        # Rules:
        # - Try all feasible insertion positions.
        # - Only accept positions passing check_feasible().
        # - Choose position with minimal incremental cost.
        # 

        # TODO:
        # - Try all insertion positions
        # - Use check_feasible to filter
        # - Choose minimal cost increase

        return solution

    # ===================================================
    # Helper
    # ===================================================
    def check_feasible(self, route):
        
        # Route feasibility helper.
        # 
        # Rules:
        # - Route format MUST be [0, ..., 0].
        # - Traverse FULL route including final depot.
        # - Capacity and time-window rules identical to validate().
        # 

        # TODO:
        # - Capacity check
        # - Time-window check
        # - Enforce due_date for final depot

        return True
"""
