import json, os, glob

files = glob.glob('experiments_llm_optimize/c1_c105/*/result.json')
print("Files:", files)
for f in files:
    with open(f, 'r', encoding='utf-8') as file:
        data = json.load(file)
        print(f'--- {f} ---')
        print(f'Vehicles: {data.get("num_vehicles")}')
        cost = data.get('cost', {})
        print(f'Fixed: {cost.get("fixed_cost")}')
        print(f'Distance: {cost.get("distance_cost")}')
        print(f'Cooling: {cost.get("cooling_cost")}')
        print(f'Freshness: {cost.get("freshness_cost")}')
        print(f'Time Penalty: {cost.get("time_window_penalty")}')
        best = data.get("best_cost")
        print(f'Total: {best}')

