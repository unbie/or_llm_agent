"""测试单个实验运行，查看详细错误"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_batch_no_llm import run_single_experiment

# 测试单个数据集
dataset_path = r"data\1 Solomon Benchmark\c1\c101.txt"
print(f"测试数据集: {dataset_path}")
print("=" * 60)

result = run_single_experiment(dataset_path, max_iters=100, seed=42)

print("\n结果:")
print(f"成功: {result['success']}")
print(f"最佳成本: {result['best_cost']}")
print(f"路线数: {result['num_routes']}")
print(f"耗时: {result['elapsed_time']:.2f}秒")
print("\n输出:")
print(result['output'])
