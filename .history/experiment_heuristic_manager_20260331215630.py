"""
启发式算法实验管理系统 - 针对Solomon Benchmark VRP问题
Heuristic Algorithm Experiment Manager for Fresh Food VRP
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime


class HeuristicExperimentConfig:
    """启发式算法实验配置"""
    def __init__(
        self,
        exp_name: str,
        model: str,
        dataset_type: str,  # c1, c2, r1, r2, rc1, rc2
        instance_file: str,
        temperature: float = 0.2,
        max_iterations: int = 1000,
        destruction_ratio: float = 0.3,
        use_plugin_generation: bool = True,
        notes: str = ""
    ):
        self.exp_name = exp_name
        self.model = model
        self.dataset_type = dataset_type
        self.instance_file = instance_file
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.destruction_ratio = destruction_ratio
        self.use_plugin_generation = use_plugin_generation
        self.notes = notes
        
    def to_dict(self):
        return {
            "exp_name": self.exp_name,
            "model": self.model,
            "dataset_type": self.dataset_type,
            "instance_file": self.instance_file,
            "temperature": self.temperature,
            "max_iterations": self.max_iterations,
            "destruction_ratio": self.destruction_ratio,
            "use_plugin_generation": self.use_plugin_generation,
            "notes": self.notes
        }


class HeuristicExperimentManager:
    """启发式算法实验管理器"""
    
    def __init__(self, output_dir: str = "experiments_heuristic"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results_dir = self.output_dir / "results"
        self.logs_dir = self.output_dir / "logs"
        self.configs_dir = self.output_dir / "configs"
        self.figures_dir = self.output_dir / "figures"
        
        for d in [self.results_dir, self.logs_dir, self.configs_dir, self.figures_dir]:
            d.mkdir(exist_ok=True)
        
        # Solomon Benchmark数据集类型
        self.dataset_types = {
            'c1': 'Clustered customers with narrow time windows',
            'c2': 'Clustered customers with wide time windows',
            'r1': 'Random customers with narrow time windows',
            'r2': 'Random customers with wide time windows',
            'rc1': 'Mixed (random+clustered) with narrow time windows',
            'rc2': 'Mixed (random+clustered) with wide time windows'
        }
    
    def generate_comprehensive_experiments(self) -> List[HeuristicExperimentConfig]:
        """生成全面的实验配置"""
        experiments = []
        
        # 主要测试的模型
        main_models = [
            "o3-mini",
            "gpt-4o",
            "gpt-4o-mini",
            "claude-3.5-sonnet",
            "deepseek-r1",
            "deepseek-v3"
        ]
        
        # Solomon Benchmark实例文件（每类选择2-3个代表性实例）
        solomon_instances = {
            'c1': ['c101.txt', 'c105.txt', 'c108.txt'],
            'c2': ['c201.txt', 'c205.txt', 'c208.txt'],
            'r1': ['r101.txt', 'r105.txt', 'r112.txt'],
            'r2': ['r201.txt', 'r205.txt', 'r211.txt'],
            'rc1': ['rc101.txt', 'rc105.txt', 'rc108.txt'],
            'rc2': ['rc201.txt', 'rc205.txt', 'rc208.txt']
        }
        
        # ========================================
        # 实验1: 主要模型 × 所有数据集类型
        # ========================================
        for model in main_models:
            for dataset_type, instances in solomon_instances.items():
                for instance in instances:
                    exp_name = f"main_{model.replace('/', '_')}_{dataset_type}_{instance.replace('.txt', '')}"
                    
                    experiments.append(HeuristicExperimentConfig(
                        exp_name=exp_name,
                        model=model,
                        dataset_type=dataset_type,
                        instance_file=f"data/1 Solomon Benchmark/{dataset_type}/{instance}",
                        temperature=0.2,
                        max_iterations=1000,
                        destruction_ratio=0.3,
                        use_plugin_generation=True,
                        notes=f"Main experiment: {model} on {dataset_type} - {self.dataset_types[dataset_type]}"
                    ))
        
        # ========================================
        # 实验2: 消融实验 - 算子生成策略对比
        # ========================================
        # 对比策略：
        # 1. LLM生成所有算子（full）
        # 2. 仅生成破坏算子，修复算子用默认（destroy_only）
        # 3. 仅生成修复算子，破坏算子用默认（insert_only）
        # 4. 完全使用默认算子（baseline）
        
        ablation_models = ["o3-mini", "gpt-4o", "deepseek-r1"]
        ablation_instances = {
            'c1': 'c101.txt',
            'r1': 'r101.txt',
            'rc1': 'rc101.txt'
        }
        
        for model in ablation_models:
            for dataset_type, instance in ablation_instances.items():
                for strategy, use_plugin in [
                    ("llm_generated", True),
                    ("baseline_default", False)
                ]:
                    exp_name = f"ablation_{model.replace('/', '_')}_{strategy}_{dataset_type}_{instance.replace('.txt', '')}"
                    
                    experiments.append(HeuristicExperimentConfig(
                        exp_name=exp_name,
                        model=model,
                        dataset_type=dataset_type,
                        instance_file=f"data/1 Solomon Benchmark/{dataset_type}/{instance}",
                        temperature=0.2,
                        max_iterations=1000,
                        destruction_ratio=0.3,
                        use_plugin_generation=use_plugin,
                        notes=f"Ablation: {strategy} strategy"
                    ))
        
        # ========================================
        # 实验3: 温度参数调优
        # ========================================
        temperatures = [0.0, 0.2, 0.5, 0.7, 1.0, 1.5]
        
        for temp in temperatures:
            exp_name = f"temp_{str(temp).replace('.', '_')}_o3mini_c101"
            
            experiments.append(HeuristicExperimentConfig(
                exp_name=exp_name,
                model="o3-mini",
                dataset_type="c1",
                instance_file="data/1 Solomon Benchmark/c1/c101.txt",
                temperature=temp,
                max_iterations=1000,
                destruction_ratio=0.3,
                use_plugin_generation=True,
                notes=f"Temperature tuning: {temp}"
            ))
        
        # ========================================
        # 实验4: 迭代次数影响分析
        # ========================================
        iterations_list = [500, 1000, 2000, 5000]
        
        for max_iter in iterations_list:
            exp_name = f"iter_{max_iter}_o3mini_c101"
            
            experiments.append(HeuristicExperimentConfig(
                exp_name=exp_name,
                model="o3-mini",
                dataset_type="c1",
                instance_file="data/1 Solomon Benchmark/c1/c101.txt",
                temperature=0.2,
                max_iterations=max_iter,
                destruction_ratio=0.3,
                use_plugin_generation=True,
                notes=f"Iteration tuning: {max_iter}"
            ))
        
        # ========================================
        # 实验5: 破坏比例调优
        # ========================================
        destruction_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        for ratio in destruction_ratios:
            exp_name = f"destroy_{str(ratio).replace('.', '_')}_o3mini_c101"
            
            experiments.append(HeuristicExperimentConfig(
                exp_name=exp_name,
                model="o3-mini",
                dataset_type="c1",
                instance_file="data/1 Solomon Benchmark/c1/c101.txt",
                temperature=0.2,
                max_iterations=1000,
                destruction_ratio=ratio,
                use_plugin_generation=True,
                notes=f"Destruction ratio tuning: {ratio}"
            ))
        
        # ========================================
        # 实验6: 多次运行稳定性测试（每个配置运行5次）
        # ========================================
        stability_configs = [
            ("o3-mini", "c1", "c101.txt"),
            ("gpt-4o", "r1", "r101.txt"),
            ("deepseek-r1", "rc1", "rc101.txt")
        ]
        
        for model, dataset_type, instance in stability_configs:
            for run_id in range(1, 6):  # 5次重复实验
                exp_name = f"stability_{model.replace('/', '_')}_{dataset_type}_{instance.replace('.txt', '')}_run{run_id}"
                
                experiments.append(HeuristicExperimentConfig(
                    exp_name=exp_name,
                    model=model,
                    dataset_type=dataset_type,
                    instance_file=f"data/1 Solomon Benchmark/{dataset_type}/{instance}",
                    temperature=0.2,
                    max_iterations=1000,
                    destruction_ratio=0.3,
                    use_plugin_generation=True,
                    notes=f"Stability test run {run_id}"
                ))
        
        return experiments
    
    def save_experiment_configs(self, experiments: List[HeuristicExperimentConfig]):
        """保存实验配置"""
        config_file = self.configs_dir / "all_experiments.json"
        
        configs = [exp.to_dict() for exp in experiments]
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(configs, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved {len(experiments)} experiment configs to {config_file}")
        return config_file
    
    def generate_run_script(self, experiments: List[HeuristicExperimentConfig], 
                           script_name: str = "run_heuristic_experiments.sh"):
        """生成批量运行脚本"""
        script_path = self.output_dir / script_name
        
        lines = [
            "#!/bin/bash",
            "",
            "# Heuristic Algorithm Experiment Script",
            f"# Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Total experiments: {len(experiments)}",
            "",
            "echo '=========================================='",
            f"echo 'Starting {len(experiments)} heuristic experiments...'",
            "echo '=========================================='",
            "",
            "mkdir -p experiments_heuristic/results",
            "mkdir -p experiments_heuristic/logs",
            "mkdir -p experiments_heuristic/figures",
            ""
        ]
        
        for idx, exp in enumerate(experiments, 1):
            # 构建命令（需要根据你的实际运行脚本调整）
            cmd = f"python llm_heuristic.py \\\n"
            cmd += f"    --model {exp.model} \\\n"
            cmd += f"    --dataset {exp.instance_file} \\\n"
            cmd += f"    --temperature {exp.temperature} \\\n"
            cmd += f"    --max_iterations {exp.max_iterations} \\\n"
            cmd += f"    --destruction_ratio {exp.destruction_ratio} \\\n"
            if exp.use_plugin_generation:
                cmd += f"    --use_plugin \\\n"
            cmd += f"    --output experiments_heuristic/results/{exp.exp_name}.json"
            
            log_file = f"experiments_heuristic/logs/{exp.exp_name}.log"
            
            lines.extend([
                f"echo '---------- Experiment {idx}/{len(experiments)}: {exp.exp_name} ----------'",
                f"echo 'Model: {exp.model}'",
                f"echo 'Dataset: {exp.dataset_type} - {exp.instance_file}'",
                f"echo 'Config: temp={exp.temperature}, iter={exp.max_iterations}, destroy={exp.destruction_ratio}'",
                f"{cmd} > {log_file} 2>&1",
                f"echo '✓ Completed {exp.exp_name}'",
                "echo ''",
                ""
            ])
        
        lines.extend([
            "echo '=========================================='",
            "echo 'All experiments completed!'",
            "echo '=========================================='",
            ""
        ])
        
        with open(script_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write('\n'.join(lines))
        
        print(f"✓ Generated run script: {script_path}")
        
        # 同时生成Windows批处理文件
        self.generate_windows_batch(experiments)
        
        return script_path
    
    def generate_windows_batch(self, experiments: List[HeuristicExperimentConfig]):
        """生成Windows批处理文件"""
        batch_path = self.output_dir / "run_heuristic_experiments.bat"
        
        lines = [
            "@echo off",
            "REM Heuristic Algorithm Experiment Script for Windows",
            f"REM Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"REM Total experiments: {len(experiments)}",
            "",
            "echo ==========================================",
            f"echo Starting {len(experiments)} heuristic experiments...",
            "echo ==========================================",
            "",
            "if not exist experiments_heuristic\\results mkdir experiments_heuristic\\results",
            "if not exist experiments_heuristic\\logs mkdir experiments_heuristic\\logs",
            "if not exist experiments_heuristic\\figures mkdir experiments_heuristic\\figures",
            ""
        ]
        
        for idx, exp in enumerate(experiments, 1):
            cmd = f"python llm_heuristic.py "
            cmd += f"--model {exp.model} "
            cmd += f"--dataset \"{exp.instance_file}\" "
            cmd += f"--temperature {exp.temperature} "
            cmd += f"--max_iterations {exp.max_iterations} "
            cmd += f"--destruction_ratio {exp.destruction_ratio} "
            if exp.use_plugin_generation:
                cmd += f"--use_plugin "
            cmd += f"--output experiments_heuristic\\results\\{exp.exp_name}.json"
            
            log_file = f"experiments_heuristic\\logs\\{exp.exp_name}.log"
            
            lines.extend([
                f"echo ---------- Experiment {idx}/{len(experiments)}: {exp.exp_name} ----------",
                f"echo Model: {exp.model}",
                f"echo Dataset: {exp.dataset_type} - {exp.instance_file}",
                f"{cmd} > {log_file} 2>&1",
                f"echo Completed {exp.exp_name}",
                "echo.",
                ""
            ])
        
        lines.extend([
            "echo ==========================================",
            "echo All experiments completed!",
            "echo ==========================================",
            "pause",
            ""
        ])
        
        with open(batch_path, 'w', encoding='utf-8') as f:
            f.write('\r\n'.join(lines))
        
        print(f"✓ Generated Windows batch file: {batch_path}")
    
    def generate_experiment_summary(self, experiments: List[HeuristicExperimentConfig]):
        """生成实验摘要文档"""
        summary_file = self.configs_dir / "experiment_summary.md"
        
        # 按类型统计
        exp_types = {}
        for exp in experiments:
            exp_type = exp.exp_name.split('_')[0]
            exp_types[exp_type] = exp_types.get(exp_type, 0) + 1
        
        # 按数据集类型统计
        dataset_stats = {}
        for exp in experiments:
            dataset_stats[exp.dataset_type] = dataset_stats.get(exp.dataset_type, 0) + 1
        
        # 按模型统计
        model_stats = {}
        for exp in experiments:
            model_stats[exp.model] = model_stats.get(exp.model, 0) + 1
        
        content = [
            "# Heuristic Algorithm Experiment Summary",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Experiments:** {len(experiments)}",
            "",
            "## 1. Experiment Types",
            "",
            "| Type | Count | Description |",
            "|------|-------|-------------|"
        ]
        
        type_descriptions = {
            'main': 'Main comparison across models and datasets',
            'ablation': 'Ablation study on algorithm components',
            'temp': 'Temperature parameter tuning',
            'iter': 'Iteration count analysis',
            'destroy': 'Destruction ratio tuning',
            'stability': 'Stability and reproducibility test'
        }
        
        for exp_type, count in sorted(exp_types.items()):
            desc = type_descriptions.get(exp_type, 'Other experiments')
            content.append(f"| {exp_type} | {count} | {desc} |")
        
        content.extend([
            "",
            "## 2. Dataset Distribution (Solomon Benchmark)",
            "",
            "| Dataset Type | Count | Characteristics |",
            "|--------------|-------|-----------------|"
        ])
        
        for ds_type, count in sorted(dataset_stats.items()):
            desc = self.dataset_types.get(ds_type, 'Unknown')
            content.append(f"| {ds_type} | {count} | {desc} |")
        
        content.extend([
            "",
            "## 3. Model Distribution",
            "",
            "| Model | Count |",
            "|-------|-------|"
        ])
        
        for model, count in sorted(model_stats.items(), key=lambda x: -x[1]):
            content.append(f"| {model} | {count} |")
        
        content.extend([
            "",
            "## 4. Hyperparameter Ranges",
            "",
            "| Parameter | Range | Default |",
            "|-----------|-------|---------|",
            "| Temperature | 0.0 - 1.5 | 0.2 |",
            "| Max Iterations | 500 - 5000 | 1000 |",
            "| Destruction Ratio | 0.1 - 0.5 | 0.3 |",
            "",
            "## 5. Expected Outputs",
            "",
            "- **Results:** JSON files with cost, convergence, and solution details",
            "- **Logs:** Detailed execution logs for debugging",
            "- **Figures:** Convergence curves, route visualizations",
            "",
            "## 6. Evaluation Metrics",
            "",
            "1. **Solution Quality**",
            "   - Total cost (objective value)",
            "   - Gap to best-known solution (BKS)",
            "   - Number of vehicles used",
            "",
            "2. **Algorithm Performance**",
            "   - Convergence speed (iterations to best solution)",
            "   - Solution stability (variance across runs)",
            "   - Computation time",
            "",
            "3. **LLM Code Quality**",
            "   - Syntax correctness rate",
            "   - Algorithm completeness",
            "   - Use of proper cost calculation",
            "",
            "## 7. Running the Experiments",
            "",
            "### Linux/Mac:",
            "```bash",
            "chmod +x experiments_heuristic/run_heuristic_experiments.sh",
            "bash experiments_heuristic/run_heuristic_experiments.sh",
            "```",
            "",
            "### Windows:",
            "```cmd",
            "experiments_heuristic\\run_heuristic_experiments.bat",
            "```",
            "",
            "## 8. Result Analysis",
            "",
            "After experiments complete, run:",
            "```bash",
            "python result_analyzer_heuristic.py",
            "```",
            ""
        ])
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
        
        print(f"✓ Generated experiment summary: {summary_file}")


def main():
    """主函数"""
    print("=" * 60)
    print("Heuristic Algorithm Experiment Manager")
    print("Solomon Benchmark VRP + LLM-Generated ALNS")
    print("=" * 60)
    print()
    
    manager = HeuristicExperimentManager()
    
    # 生成实验配置
    print("Generating comprehensive experiment suite...")
    experiments = manager.generate_comprehensive_experiments()
    print(f"✓ Generated {len(experiments)} experiments")
    print()
    
    # 统计
    exp_types = {}
    for exp in experiments:
        exp_type = exp.exp_name.split('_')[0]
        exp_types[exp_type] = exp_types.get(exp_type, 0) + 1
    
    print("Experiment breakdown:")
    for exp_type, count in sorted(exp_types.items()):
        print(f"  - {exp_type}: {count} experiments")
    print()
    
    # 保存配置
    config_file = manager.save_experiment_configs(experiments)
    print()
    
    # 生成运行脚本
    print("Generating run scripts...")
    script_path = manager.generate_run_script(experiments)
    print()
    
    # 生成摘要文档
    print("Generating experiment summary...")
    manager.generate_experiment_summary(experiments)
    print()
    
    print("=" * 60)
    print("Setup complete!")
    print()
    print("Next steps:")
    print(f"  1. Review configs: {config_file}")
    print(f"  2. Run experiments (Linux/Mac): bash {script_path}")
    print(f"  2. Run experiments (Windows): experiments_heuristic\\run_heuristic_experiments.bat")
    print(f"  3. Results: experiments_heuristic/results/")
    print(f"  4. Analyze: python result_analyzer_heuristic.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
