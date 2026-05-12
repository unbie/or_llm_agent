"""
客户数/订单数调参实验 - 结果分析与可视化
Customer Scale Experiment - Result Analyzer & Visualizer

生成论文级别的图表:
1. 折线图: 客户数 vs 总成本
2. 折线图: 客户数 vs 计算时间
3. 折线图: 客户数 vs 车辆数
4. 折线图: 客户数 vs 单客户平均成本
5. 箱线图: 不同客户数的成本分布（稳定性）
6. 汇总表格: CSV + LaTeX
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ============================================================
# 学术风格全局设置
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'SimSun'],
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
    'axes.linewidth': 0.8,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'lines.linewidth': 1.5,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'grid.linestyle': '--',
})

# 数据集配色（学术配色方案）
DATASET_COLORS = {
    'c1': '#2166AC',   # 蓝
    'c2': '#67A9CF',   # 浅蓝
    'r1': '#B2182B',   # 红
    'r2': '#EF8A62',   # 橙红
    'rc1': '#4DAF4A',  # 绿
    'rc2': '#984EA3',  # 紫
}

DATASET_MARKERS = {
    'c1': 'o',
    'c2': 'D',
    'r1': 's',
    'r2': 'v',
    'rc1': '^',
    'rc2': 'P',
}

DATASET_LABELS = {
    'c1': 'C1 (Clustered)',
    'c2': 'C2 (Clustered)',
    'r1': 'R1 (Random)',
    'r2': 'R2 (Random)',
    'rc1': 'RC1 (Mixed)',
    'rc2': 'RC2 (Mixed)',
}

DATASET_ORDER = ['c1', 'c2', 'r1', 'r2', 'rc1', 'rc2']


def ordered_dataset_types(values):
    """按 Solomon 类型顺序返回已出现的数据集类型。"""
    present = list(dict.fromkeys(values))
    ordered = [ds for ds in DATASET_ORDER if ds in present]
    ordered.extend(ds for ds in present if ds not in ordered)
    return ordered


def load_results(results_dir="experiments_customer_scale", include_result_files=False):
    """加载所有实验结果"""
    results_dir = Path(results_dir)
    summary_file = results_dir / "all_results.json"

    results = []
    if summary_file.exists():
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary_results = json.load(f)
        results.extend(summary_results)
        print(f"✓ 从 all_results.json 加载 {len(summary_results)} 条结果")

    if (not include_result_files) and results:
        return pd.DataFrame(results)

    single_dir = results_dir / "results"
    if single_dir.exists():
        for f in single_dir.glob("*.json"):
            with open(f, 'r', encoding='utf-8') as fh:
                results.append(json.load(fh))

    if results:
        df = pd.DataFrame(results)
        if 'exp_name' in df.columns:
            df = df.drop_duplicates(subset=['exp_name'], keep='first')
        print(f"✓ 加载 {len(df)} 条结果")
        return df

    print("✗ 未找到实验结果。请先运行: python run_customer_scale_experiment.py")
    return None


def compute_statistics(df):
    """计算按客户数和数据集分组的统计量"""
    # 只保留成功的实验
    df_success = df[df['success'] == True].copy()
    
    if df_success.empty:
        print("✗ 没有成功的实验结果")
        return None
    
    # 按 (数据集类型, 客户数) 分组统计
    stats = df_success.groupby(['dataset_type', 'n_customers']).agg(
        cost_mean=('best_cost', 'mean'),
        cost_std=('best_cost', 'std'),
        cost_min=('best_cost', 'min'),
        cost_max=('best_cost', 'max'),
        routes_mean=('num_routes', 'mean'),
        routes_std=('num_routes', 'std'),
        time_mean=('elapsed_time', 'mean'),
        time_std=('elapsed_time', 'std'),
        cpc_mean=('cost_per_customer', 'mean'),
        cpc_std=('cost_per_customer', 'std'),
        last_improve_mean=('last_improve_iter', 'mean'),
        count=('success', 'count'),
    ).reset_index()
    
    # 填充 NaN 的 std（只有1次运行时）
    stats = stats.fillna(0)
    
    # 计算变异系数 CV (%)
    stats['cost_cv'] = np.where(
        stats['cost_mean'] > 0,
        stats['cost_std'] / stats['cost_mean'] * 100,
        0
    )

    stats['dataset_type'] = pd.Categorical(
        stats['dataset_type'],
        categories=DATASET_ORDER,
        ordered=True
    )
    stats = stats.sort_values(['dataset_type', 'n_customers']).reset_index(drop=True)
    
    return stats


def plot_cost_vs_customers(stats, output_dir):
    """图1: 客户数 vs 总成本（折线图 + 误差带）"""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    for ds_type in ordered_dataset_types(stats['dataset_type'].astype(str).unique()):
        ds_data = stats[stats['dataset_type'] == ds_type].sort_values('n_customers')
        
        ax.plot(ds_data['n_customers'], ds_data['cost_mean'],
                marker=DATASET_MARKERS.get(ds_type, 'o'),
                color=DATASET_COLORS.get(ds_type, '#333'),
                label=DATASET_LABELS.get(ds_type, ds_type),
                linewidth=1.5, markersize=7, zorder=3)
        
        # 误差带
        ax.fill_between(
            ds_data['n_customers'],
            ds_data['cost_mean'] - ds_data['cost_std'],
            ds_data['cost_mean'] + ds_data['cost_std'],
            alpha=0.15,
            color=DATASET_COLORS.get(ds_type, '#333')
        )
    
    ax.set_xlabel('Number of Customers ($n$)')
    ax.set_ylabel('Total Cost')
    ax.set_title('(a) Total Cost vs. Number of Customers', fontweight='bold')
    ax.legend(loc='upper left', ncol=2, frameon=True, edgecolor='gray', fancybox=False, framealpha=0.9)
    ax.grid(True)
    ax.tick_params(direction='in', top=True, right=True)
    
    # 确保 x 轴显示所有刻度
    customer_counts = sorted(stats['n_customers'].unique())
    ax.set_xticks(customer_counts)
    
    plt.tight_layout()
    path = output_dir / 'fig_cost_vs_customers.png'
    plt.savefig(path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'fig_cost_vs_customers.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ {path}")


def plot_time_vs_customers(stats, output_dir):
    """图2: 客户数 vs 计算时间"""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    for ds_type in ordered_dataset_types(stats['dataset_type'].astype(str).unique()):
        ds_data = stats[stats['dataset_type'] == ds_type].sort_values('n_customers')
        
        ax.plot(ds_data['n_customers'], ds_data['time_mean'],
                marker=DATASET_MARKERS.get(ds_type, 'o'),
                color=DATASET_COLORS.get(ds_type, '#333'),
                label=DATASET_LABELS.get(ds_type, ds_type),
                linewidth=1.5, markersize=7, zorder=3)
        
        # 误差带
        ax.fill_between(
            ds_data['n_customers'],
            ds_data['time_mean'] - ds_data['time_std'],
            ds_data['time_mean'] + ds_data['time_std'],
            alpha=0.15,
            color=DATASET_COLORS.get(ds_type, '#333')
        )
    
    ax.set_xlabel('Number of Customers ($n$)')
    ax.set_ylabel('Computation Time (s)')
    ax.set_title('(b) Computation Time vs. Number of Customers', fontweight='bold')
    ax.legend(loc='upper left', ncol=2, frameon=True, edgecolor='gray', fancybox=False, framealpha=0.9)
    ax.grid(True)
    ax.tick_params(direction='in', top=True, right=True)
    
    customer_counts = sorted(stats['n_customers'].unique())
    ax.set_xticks(customer_counts)
    
    plt.tight_layout()
    path = output_dir / 'fig_time_vs_customers.png'
    plt.savefig(path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'fig_time_vs_customers.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ {path}")


def plot_cv_vs_customers(stats, output_dir):
    """图3: 客户数 vs 成本变异系数CV（稳定性）"""
    fig, ax = plt.subplots(figsize=(7, 5))

    for ds_type in ordered_dataset_types(stats['dataset_type'].astype(str).unique()):
        ds_data = stats[stats['dataset_type'] == ds_type].sort_values('n_customers')

        ax.plot(ds_data['n_customers'], ds_data['cost_cv'],
                marker=DATASET_MARKERS.get(ds_type, 'o'),
                color=DATASET_COLORS.get(ds_type, '#333'),
                label=DATASET_LABELS.get(ds_type, ds_type),
                linewidth=1.5, markersize=7, zorder=3)

        # 标注可能异常波动点（例如 n=75）
        if 75 in ds_data['n_customers'].values:
            row_75 = ds_data[ds_data['n_customers'] == 75].iloc[0]
            ax.scatter([75], [row_75['cost_cv']], color=DATASET_COLORS.get(ds_type, '#333'), s=60, zorder=4)

    ax.set_xlabel('Number of Customers ($n$)')
    ax.set_ylabel('Coefficient of Variation (%)')
    ax.set_title('(c) Stability (CV) vs. Number of Customers', fontweight='bold')
    ax.legend(loc='upper left', ncol=2, frameon=True, edgecolor='gray', fancybox=False, framealpha=0.9)
    ax.grid(True)
    ax.tick_params(direction='in', top=True, right=True)

    customer_counts = sorted(stats['n_customers'].unique())
    ax.set_xticks(customer_counts)

    plt.tight_layout()
    path = output_dir / 'fig_cv_vs_customers.png'
    plt.savefig(path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'fig_cv_vs_customers.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ {path}")


def plot_routes_vs_customers(stats, output_dir):
    """图3: 客户数 vs 车辆数"""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    for ds_type in ordered_dataset_types(stats['dataset_type'].astype(str).unique()):
        ds_data = stats[stats['dataset_type'] == ds_type].sort_values('n_customers')
        
        ax.plot(ds_data['n_customers'], ds_data['routes_mean'],
                marker=DATASET_MARKERS.get(ds_type, 'o'),
                color=DATASET_COLORS.get(ds_type, '#333'),
                label=DATASET_LABELS.get(ds_type, ds_type),
                linewidth=1.5, markersize=7, zorder=3)
        
        # 误差带
        ax.fill_between(
            ds_data['n_customers'],
            ds_data['routes_mean'] - ds_data['routes_std'],
            ds_data['routes_mean'] + ds_data['routes_std'],
            alpha=0.15,
            color=DATASET_COLORS.get(ds_type, '#333')
        )
    
    ax.set_xlabel('Number of Customers ($n$)')
    ax.set_ylabel('Number of Vehicles ($K$)')
    ax.set_title('(c) Number of Vehicles vs. Number of Customers', fontweight='bold')
    ax.legend(loc='upper left', ncol=2, frameon=True, edgecolor='gray', fancybox=False, framealpha=0.9)
    ax.grid(True)
    ax.tick_params(direction='in', top=True, right=True)
    
    customer_counts = sorted(stats['n_customers'].unique())
    ax.set_xticks(customer_counts)
    
    plt.tight_layout()
    path = output_dir / 'fig_routes_vs_customers.png'
    plt.savefig(path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'fig_routes_vs_customers.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ {path}")


def plot_cost_per_customer(stats, output_dir):
    """图4: 客户数 vs 单客户平均成本"""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    for ds_type in ordered_dataset_types(stats['dataset_type'].astype(str).unique()):
        ds_data = stats[stats['dataset_type'] == ds_type].sort_values('n_customers')
        
        ax.plot(ds_data['n_customers'], ds_data['cpc_mean'],
                marker=DATASET_MARKERS.get(ds_type, 'o'),
                color=DATASET_COLORS.get(ds_type, '#333'),
                label=DATASET_LABELS.get(ds_type, ds_type),
                linewidth=1.5, markersize=7, zorder=3)
        
        ax.fill_between(
            ds_data['n_customers'],
            ds_data['cpc_mean'] - ds_data['cpc_std'],
            ds_data['cpc_mean'] + ds_data['cpc_std'],
            alpha=0.15,
            color=DATASET_COLORS.get(ds_type, '#333')
        )
    
    ax.set_xlabel('Number of Customers ($n$)')
    ax.set_ylabel('Cost per Customer')
    ax.set_title('(d) Average Cost per Customer vs. Scale', fontweight='bold')
    ax.legend(loc='upper right', ncol=2, frameon=True, edgecolor='gray', fancybox=False, framealpha=0.9)
    ax.grid(True)
    ax.tick_params(direction='in', top=True, right=True)
    
    customer_counts = sorted(stats['n_customers'].unique())
    ax.set_xticks(customer_counts)
    
    plt.tight_layout()
    path = output_dir / 'fig_cost_per_customer.png'
    plt.savefig(path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'fig_cost_per_customer.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ {path}")


def plot_cost_boxplot(df, output_dir):
    """图5: 不同客户数的成本箱线图（展示稳定性）"""
    df_success = df[df['success'] == True].copy()
    
    if df_success.empty:
        return
    
    dataset_types = ordered_dataset_types(df_success['dataset_type'].astype(str).unique())
    n_datasets = len(dataset_types)
    
    n_cols = min(3, n_datasets)
    n_rows = int(np.ceil(n_datasets / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    if n_datasets == 1:
        axes = [axes]
    else:
        axes = np.ravel(axes)
    
    for ax, ds_type in zip(axes, dataset_types):
        ds_data = df_success[df_success['dataset_type'] == ds_type]
        customer_counts = sorted(ds_data['n_customers'].unique())
        
        # 准备箱线图数据
        box_data = []
        positions = []
        for nc in customer_counts:
            costs = ds_data[ds_data['n_customers'] == nc]['best_cost'].values
            box_data.append(costs)
            positions.append(nc)
        
        bp = ax.boxplot(
            box_data, positions=positions,
            widths=[max(3, (max(positions) - min(positions)) * 0.06)] * len(positions),
            patch_artist=True,
            boxprops=dict(facecolor=DATASET_COLORS.get(ds_type, '#ccc'), alpha=0.4,
                         edgecolor='black', linewidth=0.8),
            whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8),
            medianprops=dict(color='#B2182B', linewidth=1.5),
            flierprops=dict(markersize=4),
        )
        
        # 叠加散点
        for nc in customer_counts:
            costs = ds_data[ds_data['n_customers'] == nc]['best_cost'].values
            jitter = np.random.uniform(-1.5, 1.5, size=len(costs))
            ax.scatter(
                [nc + j for j in jitter], costs,
                c=DATASET_COLORS.get(ds_type, '#333'),
                s=25, alpha=0.7, zorder=3, edgecolors='black', linewidths=0.3
            )
        
        ax.set_xlabel('Number of Customers ($n$)')
        ax.set_ylabel('Total Cost')
        ax.set_title(f'{DATASET_LABELS.get(ds_type, ds_type)}', fontweight='bold')
        ax.set_xticks(customer_counts)
        ax.grid(True, axis='y')
        ax.tick_params(direction='in', top=True, right=True)

    for ax in axes[n_datasets:]:
        ax.axis('off')
    
    plt.suptitle('Cost Distribution across Different Scales', fontweight='bold', y=1.02)
    plt.tight_layout()
    
    path = output_dir / 'fig_cost_boxplot.png'
    plt.savefig(path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'fig_cost_boxplot.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ {path}")


def plot_combined_4panel(stats, output_dir):
    """组合图: 2×2 四图合一（方便论文使用）"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    for ds_type in ordered_dataset_types(stats['dataset_type'].astype(str).unique()):
        ds_data = stats[stats['dataset_type'] == ds_type].sort_values('n_customers')
        color = DATASET_COLORS.get(ds_type, '#333')
        marker = DATASET_MARKERS.get(ds_type, 'o')
        label = DATASET_LABELS.get(ds_type, ds_type)
        
        # (a) Cost
        ax = axes[0, 0]
        ax.plot(ds_data['n_customers'], ds_data['cost_mean'],
                marker=marker, color=color, label=label, linewidth=1.5, markersize=6)
        ax.fill_between(ds_data['n_customers'],
                        ds_data['cost_mean'] - ds_data['cost_std'],
                        ds_data['cost_mean'] + ds_data['cost_std'],
                        alpha=0.12, color=color)
        
        # (b) Time
        ax = axes[0, 1]
        ax.plot(ds_data['n_customers'], ds_data['time_mean'],
                marker=marker, color=color, label=label, linewidth=1.5, markersize=6)
        ax.fill_between(ds_data['n_customers'],
                        ds_data['time_mean'] - ds_data['time_std'],
                        ds_data['time_mean'] + ds_data['time_std'],
                        alpha=0.12, color=color)
        
        # (c) Routes
        ax = axes[1, 0]
        ax.plot(ds_data['n_customers'], ds_data['routes_mean'],
                marker=marker, color=color, label=label, linewidth=1.5, markersize=6)
        ax.fill_between(ds_data['n_customers'],
                        ds_data['routes_mean'] - ds_data['routes_std'],
                        ds_data['routes_mean'] + ds_data['routes_std'],
                        alpha=0.12, color=color)
        
        # (d) Cost per Customer
        ax = axes[1, 1]
        ax.plot(ds_data['n_customers'], ds_data['cpc_mean'],
                marker=marker, color=color, label=label, linewidth=1.5, markersize=6)
        ax.fill_between(ds_data['n_customers'],
                        ds_data['cpc_mean'] - ds_data['cpc_std'],
                        ds_data['cpc_mean'] + ds_data['cpc_std'],
                        alpha=0.12, color=color)
    
    # 设置标签
    customer_counts = sorted(stats['n_customers'].unique())
    
    titles = [
        ('(a) Total Cost vs. Scale', 'Total Cost'),
        ('(b) Computation Time vs. Scale', 'Computation Time (s)'),
        ('(c) Number of Vehicles vs. Scale', 'Number of Vehicles ($K$)'),
        ('(d) Cost per Customer vs. Scale', 'Cost per Customer'),
    ]
    
    for idx, (ax, (title, ylabel)) in enumerate(zip(axes.flat, titles)):
        ax.set_xlabel('Number of Customers ($n$)')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='upper left' if idx != 3 else 'upper right',
                  ncol=2, frameon=True, edgecolor='gray', fancybox=False, framealpha=0.9, fontsize=8)
        ax.grid(True)
        ax.tick_params(direction='in', top=True, right=True)
        ax.set_xticks(customer_counts)
    
    plt.tight_layout(h_pad=2.5, w_pad=2.5)
    
    path = output_dir / 'fig_combined_4panel.png'
    plt.savefig(path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'fig_combined_4panel.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ {path}")


def generate_summary_table(stats, output_dir):
    """生成汇总表格 (CSV + LaTeX)"""
    
    # === CSV 表格 ===
    table = stats[[
        'dataset_type', 'n_customers',
        'cost_mean', 'cost_std', 'cost_cv',
        'routes_mean', 'time_mean', 'cpc_mean',
        'last_improve_mean', 'count'
    ]].copy()
    
    table.columns = [
        'Dataset', 'Customers',
        'Avg Cost', 'Std Cost', 'CV (%)',
        'Avg Vehicles', 'Avg Time (s)', 'Avg Cost/Customer',
        'Avg Last Improve Iter', 'Runs'
    ]
    
    csv_path = output_dir / 'summary_table.csv'
    table.to_csv(csv_path, index=False, float_format='%.2f')
    print(f"  ✓ {csv_path}")
    
    # === LaTeX 表格 ===
    latex_lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Impact of Customer Scale on Algorithm Performance}',
        r'\label{tab:customer_scale}',
        r'\begin{tabular}{llrrrrrr}',
        r'\toprule',
        r'Dataset & $n$ & Avg Cost & Std & CV(\%) & Vehicles & Time(s) & Cost/$n$ \\',
        r'\midrule',
    ]
    
    for _, row in stats.iterrows():
        ds_label = DATASET_LABELS.get(row['dataset_type'], row['dataset_type'])
        latex_lines.append(
            f"  {ds_label} & {int(row['n_customers'])} & "
            f"{row['cost_mean']:.2f} & {row['cost_std']:.2f} & {row['cost_cv']:.1f} & "
            f"{row['routes_mean']:.1f} & {row['time_mean']:.1f} & {row['cpc_mean']:.2f} \\\\"
        )
    
    latex_lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])
    
    latex_path = output_dir / 'summary_table.tex'
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_lines))
    print(f"  ✓ {latex_path}")
    
    # === 打印到控制台 ===
    print("\n" + "=" * 100)
    print("汇总统计表")
    print("=" * 100)
    print(table.to_string(index=False))
    print("=" * 100)


def main():
    """主函数"""
    print("=" * 70)
    print("客户数/订单数调参实验 - 结果分析与可视化")
    print("=" * 70)
    print()
    
    # 加载数据
    df = load_results()
    if df is None or df.empty:
        return
    
    # 创建输出目录
    output_dir = Path("experiments_customer_scale/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 计算统计量
    stats = compute_statistics(df)
    if stats is None:
        return
    
    print(f"\n共 {len(stats)} 个分组统计")
    print()
    
    # 生成图表
    print("生成图表:")
    
    plot_cost_vs_customers(stats, output_dir)
    plot_time_vs_customers(stats, output_dir)
    plot_cv_vs_customers(stats, output_dir)
    plot_routes_vs_customers(stats, output_dir)
    plot_cost_per_customer(stats, output_dir)
    plot_cost_boxplot(df, output_dir)
    plot_combined_4panel(stats, output_dir)
    
    print()
    
    # 生成表格
    print("生成表格:")
    generate_summary_table(stats, output_dir)
    
    print()
    print("=" * 70)
    print("分析完成!")
    print(f"图表保存到: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
