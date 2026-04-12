"""
启发式算法结果分析工具 - 针对Solomon Benchmark实验
Result Analyzer for Heuristic Algorithm Experiments
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import List, Dict, Any
import re


class HeuristicResultAnalyzer:
    """启发式算法结果分析器"""
    
    def __init__(self, results_dir: str = "experiments_heuristic/results"):
        self.results_dir = Path(results_dir)
        self.figures_dir = Path("experiments_heuristic/figures")
        self.tables_dir = Path("experiments_heuristic/tables")
        self.latex_dir = Path("experiments_heuristic/latex")
        
        for d in [self.figures_dir, self.tables_dir, self.latex_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # 设置学术绘图风格
        sns.set_style("whitegrid")
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.dpi': 150,
            'savefig.dpi': 600
        })
        
        # Solomon Benchmark最优解（文献中的最佳已知解）
        self.best_known_solutions = {
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
    
    def load_all_results(self) -> List[Dict[str, Any]]:
        """加载所有实验结果"""
        results = []
        
        for result_file in self.results_dir.glob("*.json"):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data['exp_name'] = result_file.stem
                    
                    # 解析实验名称
                    parsed = self.parse_experiment_name(result_file.stem)
                    data.update(parsed)
                    
                    # 计算与BKS的差距
                    instance = data.get('instance', '')
                    if instance in self.best_known_solutions:
                        bks = self.best_known_solutions[instance]
                        best_cost = data.get('best_cost', float('inf'))
                        if best_cost < float('inf'):
                            gap = ((best_cost - bks) / bks) * 100
                            data['gap_to_bks'] = gap
                        else:
                            data['gap_to_bks'] = None
                    
                    results.append(data)
            except Exception as e:
                print(f"Warning: Could not load {result_file}: {e}")
        
        return results
    
    def parse_experiment_name(self, exp_name: str) -> Dict[str, str]:
        """解析实验名称"""
        info = {
            'type': '',
            'model': '',
            'dataset_type': '',
            'instance': '',
            'config': ''
        }
        
        parts = exp_name.split('_')
        if len(parts) > 0:
            info['type'] = parts[0]
        
        # 提取模型名
        for model in ['o3-mini', 'o3mini', 'gpt-4o-mini', 'gpt-4o', 'claude', 'deepseek-r1', 'deepseek-v3']:
            if model in exp_name.lower():
                info['model'] = model
                break
        
        # 提取数据集类型
        for ds_type in ['c1', 'c2', 'r1', 'r2', 'rc1', 'rc2']:
            if ds_type in exp_name:
                info['dataset_type'] = ds_type
                break
        
        # 提取具体实例
        instance_match = re.search(r'([cr]c?\d{3})', exp_name)
        if instance_match:
            info['instance'] = instance_match.group(1)
        
        # 提取配置
        if 'temp_' in exp_name:
            match = re.search(r'temp_(\d+_?\d*)', exp_name)
            if match:
                temp_str = match.group(1).replace('_', '.')
                info['config'] = f"temp_{temp_str}"
        elif 'iter_' in exp_name:
            match = re.search(r'iter_(\d+)', exp_name)
            if match:
                info['config'] = f"iter_{match.group(1)}"
        elif 'destroy_' in exp_name:
            match = re.search(r'destroy_(\d+_\d+)', exp_name)
            if match:
                ratio_str = match.group(1).replace('_', '.')
                info['config'] = f"destroy_{ratio_str}"
        elif 'llm_generated' in exp_name:
            info['config'] = 'llm_generated'
        elif 'baseline_default' in exp_name:
            info['config'] = 'baseline_default'
        
        return info
    
    def create_main_comparison_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建主要对比表格"""
        main_df = df[df['type'] == 'main'].copy()
        
        if main_df.empty:
            print("Warning: No main experiment results found.")
            return pd.DataFrame()
        
        # 按模型和数据集类型分组
        summary = main_df.groupby(['model', 'dataset_type']).agg({
            'best_cost': 'mean',
            'gap_to_bks': 'mean',
            'convergence_iterations': 'mean',
            'num_vehicles': 'mean'
        }).round(2)
        
        return summary
    
    def create_ablation_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建消融实验表格"""
        ablation_df = df[df['type'] == 'ablation'].copy()
        
        if ablation_df.empty:
            print("Warning: No ablation results found.")
            return pd.DataFrame()
        
        pivot = ablation_df.pivot_table(
            values='best_cost',
            index='config',
            columns='model',
            aggfunc='mean'
        ).round(2)
        
        return pivot
    
    def create_hyperparameter_tables(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """创建超参数调优表格"""
        tables = {}
        
        # Temperature调优
        temp_df = df[df['type'] == 'temp'].copy()
        if not temp_df.empty:
            temp_df['temperature'] = temp_df['config'].str.extract(r'temp_(\d+\.\d+)')[0].astype(float)
            temp_summary = temp_df.groupby('temperature').agg({
                'best_cost': ['mean', 'std'],
                'gap_to_bks': ['mean', 'std']
            }).round(2)
            tables['temperature'] = temp_summary
        
        # Iteration调优
        iter_df = df[df['type'] == 'iter'].copy()
        if not iter_df.empty:
            iter_df['max_iterations'] = iter_df['config'].str.extract(r'iter_(\d+)')[0].astype(int)
            iter_summary = iter_df.groupby('max_iterations').agg({
                'best_cost': ['mean', 'std'],
                'convergence_iterations': 'mean'
            }).round(2)
            tables['iterations'] = iter_summary
        
        # Destruction ratio调优
        destroy_df = df[df['type'] == 'destroy'].copy()
        if not destroy_df.empty:
            destroy_df['destruction_ratio'] = destroy_df['config'].str.extract(r'destroy_(\d+\.\d+)')[0].astype(float)
            destroy_summary = destroy_df.groupby('destruction_ratio').agg({
                'best_cost': ['mean', 'std'],
                'gap_to_bks': 'mean'
            }).round(2)
            tables['destruction_ratio'] = destroy_summary
        
        return tables
    
    def create_stability_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建稳定性分析表格"""
        stability_df = df[df['type'] == 'stability'].copy()
        
        if stability_df.empty:
            print("Warning: No stability results found.")
            return pd.DataFrame()
        
        # 提取基础配置（去掉run编号）
        stability_df['base_config'] = stability_df['exp_name'].str.replace(r'_run\d+$', '', regex=True)
        
        summary = stability_df.groupby('base_config').agg({
            'best_cost': ['mean', 'std', 'min', 'max'],
            'gap_to_bks': ['mean', 'std']
        }).round(2)
        
        # 计算变异系数 (CV)
        summary[('best_cost', 'cv')] = (
            summary[('best_cost', 'std')] / summary[('best_cost', 'mean')] * 100
        ).round(2)
        
        return summary
    
    def plot_model_comparison(self, df: pd.DataFrame):
        """绘制模型对比图"""
        main_df = df[df['type'] == 'main'].copy()
        
        if main_df.empty:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 图1: 各模型的平均gap
        model_gap = main_df.groupby('model')['gap_to_bks'].mean().sort_values()
        model_gap.plot(kind='barh', ax=axes[0], color='steelblue')
        axes[0].set_title('Average Gap to Best Known Solution by Model', fontweight='bold')
        axes[0].set_xlabel('Gap to BKS (%)')
        axes[0].set_ylabel('Model')
        axes[0].grid(axis='x', alpha=0.3)
        axes[0].axvline(x=0, color='red', linestyle='--', linewidth=1)
        
        # 图2: 各数据集类型的平均gap
        dataset_gap = main_df.groupby('dataset_type')['gap_to_bks'].mean().sort_values()
        dataset_gap.plot(kind='bar', ax=axes[1], color='coral')
        axes[1].set_title('Average Gap by Solomon Dataset Type', fontweight='bold')
        axes[1].set_xlabel('Dataset Type')
        axes[1].set_ylabel('Gap to BKS (%)')
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'model_comparison.png', dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved figure: {self.figures_dir / 'model_comparison.png'}")
    
    def plot_convergence_comparison(self, df: pd.DataFrame):
        """绘制收敛性对比图"""
        # 选择几个代表性实验绘制收敛曲线
        selected_exps = df[
            (df['type'] == 'main') & 
            (df['instance'] == 'c101')
        ].head(5)
        
        if selected_exps.empty:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for idx, (_, exp) in enumerate(selected_exps.iterrows()):
            # 这里需要从结果中提取收敛历史
            # 假设结果中包含cost_history字段
            if 'cost_history' in exp and exp['cost_history']:
                iterations = range(len(exp['cost_history']))
                ax.plot(iterations, exp['cost_history'], 
                       label=exp['model'], color=colors[idx % len(colors)],
                       linewidth=1.5, alpha=0.8)
        
        ax.set_title('Convergence Comparison on c101 Instance', fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective Value')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'convergence_comparison.png', dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved figure: {self.figures_dir / 'convergence_comparison.png'}")
    
    def plot_hyperparameter_tuning(self, df: pd.DataFrame):
        """绘制超参数调优图"""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Temperature
        temp_df = df[df['type'] == 'temp'].copy()
        if not temp_df.empty:
            temp_df['temperature'] = temp_df['config'].str.extract(r'temp_(\d+\.\d+)')[0].astype(float)
            temp_summary = temp_df.groupby('temperature')['best_cost'].mean()
            axes[0].plot(temp_summary.index, temp_summary.values, 
                        marker='o', linewidth=2, markersize=8, color='#1f77b4')
            axes[0].set_title('Temperature Tuning', fontweight='bold')
            axes[0].set_xlabel('Temperature')
            axes[0].set_ylabel('Average Best Cost')
            axes[0].grid(alpha=0.3)
        
        # Iterations
        iter_df = df[df['type'] == 'iter'].copy()
        if not iter_df.empty:
            iter_df['max_iterations'] = iter_df['config'].str.extract(r'iter_(\d+)')[0].astype(int)
            iter_summary = iter_df.groupby('max_iterations')['best_cost'].mean()
            axes[1].plot(iter_summary.index, iter_summary.values,
                        marker='s', linewidth=2, markersize=8, color='#ff7f0e')
            axes[1].set_title('Iteration Count Tuning', fontweight='bold')
            axes[1].set_xlabel('Max Iterations')
            axes[1].set_ylabel('Average Best Cost')
            axes[1].grid(alpha=0.3)
        
        # Destruction ratio
        destroy_df = df[df['type'] == 'destroy'].copy()
        if not destroy_df.empty:
            destroy_df['destruction_ratio'] = destroy_df['config'].str.extract(r'destroy_(\d+\.\d+)')[0].astype(float)
            destroy_summary = destroy_df.groupby('destruction_ratio')['best_cost'].mean()
            axes[2].plot(destroy_summary.index, destroy_summary.values,
                        marker='^', linewidth=2, markersize=8, color='#2ca02c')
            axes[2].set_title('Destruction Ratio Tuning', fontweight='bold')
            axes[2].set_xlabel('Destruction Ratio')
            axes[2].set_ylabel('Average Best Cost')
            axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'hyperparameter_tuning.png', dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved figure: {self.figures_dir / 'hyperparameter_tuning.png'}")
    
    def plot_stability_analysis(self, df: pd.DataFrame):
        """绘制稳定性分析图"""
        stability_df = df[df['type'] == 'stability'].copy()
        
        if stability_df.empty:
            return
        
        stability_df['base_config'] = stability_df['exp_name'].str.replace(r'_run\d+$', '', regex=True)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 箱线图显示各配置的稳定性
        stability_df.boxplot(column='best_cost', by='base_config', ax=ax)
        ax.set_title('Solution Stability Across Multiple Runs', fontweight='bold')
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Best Cost')
        plt.suptitle('')  # 移除默认标题
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'stability_analysis.png', dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved figure: {self.figures_dir / 'stability_analysis.png'}")
    
    def generate_latex_tables(self, tables: Dict[str, pd.DataFrame]):
        """生成LaTeX表格"""
        for table_name, table_df in tables.items():
            if table_df.empty:
                continue
            
            latex_code = table_df.to_latex(
                float_format="%.2f",
                caption=f"{table_name.replace('_', ' ').title()} Results",
                label=f"tab:{table_name}",
                escape=False
            )
            
            latex_file = self.latex_dir / f"{table_name}.tex"
            with open(latex_file, 'w', encoding='utf-8') as f:
                f.write(latex_code)
            
            print(f"✓ Saved LaTeX table: {latex_file}")
    
    def analyze(self):
        """执行完整分析"""
        print("=" * 70)
        print("Heuristic Algorithm Result Analyzer")
        print("=" * 70)
        print()
        
        # 1. 加载结果
        print("Loading experiment results...")
        results = self.load_all_results()
        
        if not results:
            print("❌ No results found. Please run experiments first.")
            return
        
        print(f"✓ Loaded {len(results)} experiment results")
        print()
        
        df = pd.DataFrame(results)
        
        # 2. 创建表格
        print("Generating tables...")
        tables = {}
        
        main_table = self.create_main_comparison_table(df)
        if not main_table.empty:
            tables['main_comparison'] = main_table
            print(f"✓ Main comparison table")
        
        ablation_table = self.create_ablation_table(df)
        if not ablation_table.empty:
            tables['ablation_study'] = ablation_table
            print(f"✓ Ablation study table")
        
        hyperparam_tables = self.create_hyperparameter_tables(df)
        tables.update(hyperparam_tables)
        for name in hyperparam_tables.keys():
            print(f"✓ {name.title()} tuning table")
        
        stability_table = self.create_stability_analysis(df)
        if not stability_table.empty:
            tables['stability_analysis'] = stability_table
            print(f"✓ Stability analysis table")
        
        print()
        
        # 3. 绘制图表
        print("Generating figures...")
        self.plot_model_comparison(df)
        self.plot_convergence_comparison(df)
        self.plot_hyperparameter_tuning(df)
        self.plot_stability_analysis(df)
        print()
        
        # 4. 生成LaTeX表格
        print("Generating LaTeX tables...")
        self.generate_latex_tables(tables)
        print()
        
        # 5. 保存CSV表格
        print("Saving CSV tables...")
        for table_name, table_df in tables.items():
            csv_file = self.tables_dir / f"{table_name}.csv"
            table_df.to_csv(csv_file)
            print(f"✓ Saved: {csv_file}")
        print()
        
        print("=" * 70)
        print("Analysis complete!")
        print()
        print("Output files:")
        print(f"  - Figures: {self.figures_dir}/")
        print(f"  - Tables (CSV): {self.tables_dir}/")
        print(f"  - LaTeX tables: {self.latex_dir}/")
        print("=" * 70)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='分析启发式算法实验结果')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='结果目录，默认自动检测 experiments_quick 或 experiments_heuristic')
    args = parser.parse_args()
    
    # 自动检测结果目录
    if args.results_dir:
        results_dir = args.results_dir
    else:
        from pathlib import Path
        quick_dir = Path("experiments_quick/results")
        full_dir = Path("experiments_heuristic/results")
        
        if quick_dir.exists() and any(quick_dir.glob("*.json")):
            results_dir = "experiments_quick/results"
            print(f"[自动检测] 使用精简版实验结果: {results_dir}")
        elif full_dir.exists() and any(full_dir.glob("*.json")):
            results_dir = "experiments_heuristic/results"
            print(f"[自动检测] 使用完整版实验结果: {results_dir}")
        else:
            print("❌ 未找到实验结果目录")
            print("请先运行实验：python experiments_quick/run_experiments.py")
            return
    
    analyzer = HeuristicResultAnalyzer(results_dir=results_dir)
    analyzer.analyze()


if __name__ == "__main__":
    main()
