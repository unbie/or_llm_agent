"""
批量实验结果分析器 - 针对省Token实验
Result Analyzer for Zero-Token Batch Experiments
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime


# Solomon Benchmark最优解（文献已知最佳解）
BEST_KNOWN_SOLUTIONS = {
    'c101': 828.94, 'c102': 828.94, 'c103': 828.06, 'c104': 824.78,
    'c105': 828.94, 'c106': 828.94, 'c107': 828.94, 'c108': 828.94, 'c109': 828.94,
    'c201': 591.56, 'c202': 591.56, 'c203': 591.17, 'c204': 590.60,
    'c205': 588.88, 'c206': 588.49, 'c207': 588.29, 'c208': 588.32,
    'r101': 1650.80, 'r102': 1486.12, 'r103': 1292.67, 'r104': 1007.31,
    'r105': 1377.11, 'r106': 1252.03, 'r107': 1104.66, 'r108': 960.88,
    'r109': 1194.73, 'r110': 1118.84, 'r111': 1096.72, 'r112': 982.14,
    'r201': 1252.37, 'r202': 1191.70, 'r203': 939.50, 'r204': 825.52,
    'r205': 994.43, 'r206': 906.14, 'r207': 890.61, 'r208': 726.75,
    'r209': 909.16, 'r210': 939.37, 'r211': 885.71,
    'rc101': 1696.95, 'rc102': 1554.75, 'rc103': 1261.67, 'rc104': 1135.48,
    'rc105': 1629.44, 'rc106': 1424.73, 'rc107': 1230.48, 'rc108': 1139.82,
    'rc201': 1406.94, 'rc202': 1365.65, 'rc203': 1049.62, 'rc204': 798.46,
    'rc205': 1297.65, 'rc206': 1146.32, 'rc207': 1061.14, 'rc208': 828.14
}


def load_results(results_dir="experiments_batch/results"):
    """加载所有实验结果"""
    results = []
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"❌ 结果目录不存在: {results_dir}")
        return []
    
    for result_file in results_path.glob("*.json"):
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 计算与BKS的差距
                instance = data.get('instance', '')
                if instance in BEST_KNOWN_SOLUTIONS and data.get('best_cost'):
                    bks = BEST_KNOWN_SOLUTIONS[instance]
                    gap = ((data['best_cost'] - bks) / bks) * 100
                    data['bks'] = bks
                    data['gap_to_bks'] = gap
                else:
                    data['bks'] = None
                    data['gap_to_bks'] = None
                
                results.append(data)
        except Exception as e:
            print(f"Warning: 无法加载 {result_file}: {e}")
    
    return results


def analyze_results(results):
    """分析实验结果"""
    if not results:
        print("❌ 没有结果数据")
        return
    
    df = pd.DataFrame(results)
    
    print("=" * 70)
    print("实验结果分析")
    print("=" * 70)
    print()
    
    # 1. 总体统计
    print("【总体统计】")
    print(f"  总实验数: {len(df)}")
    print(f"  成功率: {df['success'].sum()}/{len(df)} ({df['success'].mean()*100:.1f}%)")
    print()
    
    # 2. 按数据集类型分组
    print("【按数据集类型分组】")
    print("-" * 70)
    
    success_df = df[df['success'] == True].copy()
    
    if len(success_df) > 0:
        type_stats = success_df.groupby('dataset_type').agg({
            'best_cost': ['mean', 'std', 'min'],
            'num_routes': 'mean',
            'elapsed_time': 'mean',
            'gap_to_bks': 'mean'
        }).round(2)
        
        print(type_stats.to_string())
        print()
    
    # 3. 按实例分组（稳定性分析）
    print("【按实例分组 - 稳定性分析】")
    print("-" * 70)
    
    if len(success_df) > 0:
        instance_stats = success_df.groupby('instance').agg({
            'best_cost': ['mean', 'std', 'min', 'max', 'count'],
            'gap_to_bks': 'mean'
        }).round(2)
        
        # 计算变异系数
        instance_stats[('best_cost', 'cv%')] = (
            instance_stats[('best_cost', 'std')] / instance_stats[('best_cost', 'mean')] * 100
        ).round(2)
        
        print(instance_stats.to_string())
        print()
    
    return df


def generate_tables(df, output_dir="experiments_batch"):
    """生成论文表格"""
    output_path = Path(output_dir)
    tables_dir = output_path / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    success_df = df[df['success'] == True].copy()
    
    if len(success_df) == 0:
        print("❌ 没有成功的实验结果")
        return
    
    # 表1: 各数据集类型的平均性能
    print("生成表格...")
    
    table1 = success_df.groupby('dataset_type').agg({
        'best_cost': 'mean',
        'num_routes': 'mean',
        'gap_to_bks': 'mean',
        'elapsed_time': 'mean'
    }).round(2)
    
    table1.columns = ['Avg Cost', 'Avg Vehicles', 'Gap to BKS (%)', 'Avg Time (s)']
    table1.to_csv(tables_dir / "table1_dataset_comparison.csv")
    print(f"✓ 保存: {tables_dir / 'table1_dataset_comparison.csv'}")
    
    # 表2: 各实例的详细结果
    table2 = success_df.groupby(['dataset_type', 'instance']).agg({
        'best_cost': ['mean', 'std', 'min'],
        'bks': 'first',
        'gap_to_bks': 'mean'
    }).round(2)
    
    table2.to_csv(tables_dir / "table2_instance_details.csv")
    print(f"✓ 保存: {tables_dir / 'table2_instance_details.csv'}")
    
    # 表3: 稳定性分析
    table3 = success_df.groupby('instance').agg({
        'best_cost': ['mean', 'std', 'min', 'max']
    }).round(2)
    
    table3[('best_cost', 'cv%')] = (table3[('best_cost', 'std')] / table3[('best_cost', 'mean')] * 100).round(2)
    table3.to_csv(tables_dir / "table3_stability_analysis.csv")
    print(f"✓ 保存: {tables_dir / 'table3_stability_analysis.csv'}")
    
    # 生成LaTeX格式
    latex_dir = output_path / "latex"
    latex_dir.mkdir(parents=True, exist_ok=True)
    
    table1.to_latex(latex_dir / "table1.tex", caption="Performance by Dataset Type", label="tab:dataset_comparison")
    print(f"✓ 保存: {latex_dir / 'table1.tex'}")


def generate_figures(df, output_dir="experiments_batch"):
    """生成论文图表"""
    output_path = Path(output_dir)
    figures_dir = output_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    success_df = df[df['success'] == True].copy()
    
    if len(success_df) == 0:
        print("❌ 没有成功的实验结果")
        return
    
    # 设置学术风格
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'figure.dpi': 150,
        'savefig.dpi': 600
    })
    
    print("生成图表...")
    
    # 图1: 各数据集类型的Gap对比
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图: Gap to BKS
    type_gap = success_df.groupby('dataset_type')['gap_to_bks'].mean().sort_values()
    colors = ['#4393C3' if g < 10 else '#D6604D' for g in type_gap.values]
    type_gap.plot(kind='barh', ax=axes[0], color=colors)
    axes[0].set_xlabel('Gap to Best Known Solution (%)')
    axes[0].set_ylabel('Dataset Type')
    axes[0].set_title('(a) Solution Quality by Dataset Type', fontweight='bold')
    axes[0].axvline(x=0, color='green', linestyle='--', linewidth=1)
    axes[0].grid(axis='x', alpha=0.3)
    
    # 右图: 平均成本
    type_cost = success_df.groupby('dataset_type')['best_cost'].mean().sort_values()
    type_cost.plot(kind='bar', ax=axes[1], color='steelblue')
    axes[1].set_xlabel('Dataset Type')
    axes[1].set_ylabel('Average Cost')
    axes[1].set_title('(b) Average Cost by Dataset Type', fontweight='bold')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'fig1_dataset_comparison.png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存: {figures_dir / 'fig1_dataset_comparison.png'}")
    
    # 图2: 稳定性分析（箱线图）
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 按实例分组的箱线图
    instances = success_df['instance'].unique()
    data_for_box = [success_df[success_df['instance'] == inst]['best_cost'].values for inst in instances]
    
    bp = ax.boxplot(data_for_box, labels=instances, patch_artist=True)
    
    # 着色
    colors_map = {'c': '#4393C3', 'r': '#D6604D', 'rc': '#5AAE61'}
    for i, (patch, inst) in enumerate(zip(bp['boxes'], instances)):
        color = colors_map.get(inst[0], '#888888')
        if inst.startswith('rc'):
            color = colors_map['rc']
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Instance')
    ax.set_ylabel('Cost')
    ax.set_title('Solution Stability Across Multiple Runs', fontweight='bold')
    ax.set_xticklabels(instances, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'fig2_stability_boxplot.png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存: {figures_dir / 'fig2_stability_boxplot.png'}")
    
    # 图3: 时间分析
    fig, ax = plt.subplots(figsize=(10, 6))
    
    time_by_type = success_df.groupby('dataset_type')['elapsed_time'].mean().sort_values()
    time_by_type.plot(kind='bar', ax=ax, color='coral')
    ax.set_xlabel('Dataset Type')
    ax.set_ylabel('Average Computation Time (s)')
    ax.set_title('Computational Efficiency by Dataset Type', fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'fig3_computation_time.png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存: {figures_dir / 'fig3_computation_time.png'}")


def main():
    print("=" * 70)
    print("批量实验结果分析器")
    print("=" * 70)
    print()
    
    # 加载结果
    print("加载实验结果...")
    results = load_results()
    
    if not results:
        print("❌ 未找到实验结果。请先运行: python run_batch_no_llm.py")
        return
    
    print(f"✓ 加载了 {len(results)} 个实验结果")
    print()
    
    # 分析结果
    df = analyze_results(results)
    
    if df is None or len(df) == 0:
        return
    
    # 生成表格
    print()
    generate_tables(df)
    
    # 生成图表
    print()
    generate_figures(df)
    
    print()
    print("=" * 70)
    print("分析完成！")
    print()
    print("输出文件:")
    print("  - 表格: experiments_batch/tables/")
    print("  - 图表: experiments_batch/figures/")
    print("  - LaTeX: experiments_batch/latex/")
    print("=" * 70)


if __name__ == "__main__":
    main()
