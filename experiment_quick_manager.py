"""
精简版启发式算法实验管理器 - 用于快速测试
Lightweight Heuristic Experiment Manager for Quick Testing
约40-50个实验，验证流程后再运行完整版
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


class QuickExperimentConfig:
    """实验配置类"""
    def __init__(
        self,
        exp_name: str,
        model: str,
        dataset_type: str,
        instance_file: str,
        temperature: float = 0.2,
        max_iterations: int = 500,  # 精简版：减少迭代次数加速
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


class QuickExperimentManager:
    """精简版实验管理器"""
    
    def __init__(self, output_dir: str = "experiments_quick"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results_dir = self.output_dir / "results"
        self.logs_dir = self.output_dir / "logs"
        self.configs_dir = self.output_dir / "configs"
        self.figures_dir = self.output_dir / "figures"
        
        for d in [self.results_dir, self.logs_dir, self.configs_dir, self.figures_dir]:
            d.mkdir(exist_ok=True)
    
    def generate_quick_test_experiments(self) -> List[QuickExperimentConfig]:
        """生成精简测试实验配置（约40-50个）"""
        experiments = []
        
        # ========================================
        # 实验1: 核心模型对比 (12个)
        # 选择3个最有代表性的模型
        # ========================================
        core_models = [
            "o3-mini",           # OpenAI推理模型
            "gpt-4o",            # OpenAI标准模型
            "deepseek-r1"        # DeepSeek推理模型
        ]
        
        # 选择2个最有代表性的数据集类型，每类选2个实例
        test_instances = {
            'c1': ['c101.txt', 'c105.txt'],  # 聚类 + 窄时间窗（简单）
            'r1': ['r101.txt', 'r105.txt']   # 随机 + 窄时间窗（困难）
        }
        
        for model in core_models:
            for dataset_type, instances in test_instances.items():
                for instance in instances:
                    exp_name = f"main_{model.replace('/', '_')}_{dataset_type}_{instance.replace('.txt', '')}"
                    
                    experiments.append(QuickExperimentConfig(
                        exp_name=exp_name,
                        model=model,
                        dataset_type=dataset_type,
                        instance_file=f"data/1 Solomon Benchmark/{dataset_type}/{instance}",
                        temperature=0.2,
                        max_iterations=500,  # 精简版：500迭代
                        destruction_ratio=0.3,
                        use_plugin_generation=True,
                        notes=f"Core test: {model} on {dataset_type}"
                    ))
        
        print(f"✓ Generated {len(experiments)} main comparison experiments")
        
        # ========================================
        # 实验2: 消融实验 (8个)
        # 2个模型 × 2个数据集 × 2种策略
        # ========================================
        ablation_models = ["o3-mini", "gpt-4o"]
        ablation_instances = {
            'c1': 'c101.txt',
            'r1': 'r101.txt'
        }
        
        for model in ablation_models:
            for dataset_type, instance in ablation_instances.items():
                for strategy, use_plugin in [
                    ("llm_generated", True),
                    ("baseline_default", False)
                ]:
                    exp_name = f"ablation_{model.replace('/', '_')}_{strategy}_{dataset_type}_{instance.replace('.txt', '')}"
                    
                    experiments.append(QuickExperimentConfig(
                        exp_name=exp_name,
                        model=model,
                        dataset_type=dataset_type,
                        instance_file=f"data/1 Solomon Benchmark/{dataset_type}/{instance}",
                        temperature=0.2,
                        max_iterations=500,
                        destruction_ratio=0.3,
                        use_plugin_generation=use_plugin,
                        notes=f"Ablation: {strategy}"
                    ))
        
        print(f"✓ Generated {len(experiments) - 12} ablation experiments")
        
        # ========================================
        # 实验3: 温度调优 (3个)
        # 只测试3个关键温度值
        # ========================================
        temperatures = [0.0, 0.2, 0.7]  # 低、中、高
        
        for temp in temperatures:
            exp_name = f"temp_{str(temp).replace('.', '_')}_o3mini_c101"
            
            experiments.append(QuickExperimentConfig(
                exp_name=exp_name,
                model="o3-mini",
                dataset_type="c1",
                instance_file="data/1 Solomon Benchmark/c1/c101.txt",
                temperature=temp,
                max_iterations=500,
                destruction_ratio=0.3,
                use_plugin_generation=True,
                notes=f"Temperature test: {temp}"
            ))
        
        print(f"✓ Generated 3 temperature tuning experiments")
        
        # ========================================
        # 实验4: 迭代次数测试 (3个)
        # ========================================
        iterations_list = [300, 500, 1000]
        
        for max_iter in iterations_list:
            exp_name = f"iter_{max_iter}_o3mini_c101"
            
            experiments.append(QuickExperimentConfig(
                exp_name=exp_name,
                model="o3-mini",
                dataset_type="c1",
                instance_file="data/1 Solomon Benchmark/c1/c101.txt",
                temperature=0.2,
                max_iterations=max_iter,
                destruction_ratio=0.3,
                use_plugin_generation=True,
                notes=f"Iteration test: {max_iter}"
            ))
        
        print(f"✓ Generated 3 iteration tuning experiments")
        
        # ========================================
        # 实验5: 破坏比例测试 (3个)
        # ========================================
        destruction_ratios = [0.2, 0.3, 0.4]
        
        for ratio in destruction_ratios:
            exp_name = f"destroy_{str(ratio).replace('.', '_')}_o3mini_c101"
            
            experiments.append(QuickExperimentConfig(
                exp_name=exp_name,
                model="o3-mini",
                dataset_type="c1",
                instance_file="data/1 Solomon Benchmark/c1/c101.txt",
                temperature=0.2,
                max_iterations=500,
                destruction_ratio=ratio,
                use_plugin_generation=True,
                notes=f"Destruction ratio test: {ratio}"
            ))
        
        print(f"✓ Generated 3 destruction ratio experiments")
        
        # ========================================
        # 实验6: 稳定性测试 (6个)
        # 2个配置 × 3次重复
        # ========================================
        stability_configs = [
            ("o3-mini", "c1", "c101.txt"),
            ("gpt-4o", "r1", "r101.txt")
        ]
        
        for model, dataset_type, instance in stability_configs:
            for run_id in range(1, 4):  # 3次重复（精简版）
                exp_name = f"stability_{model.replace('/', '_')}_{dataset_type}_{instance.replace('.txt', '')}_run{run_id}"
                
                experiments.append(QuickExperimentConfig(
                    exp_name=exp_name,
                    model=model,
                    dataset_type=dataset_type,
                    instance_file=f"data/1 Solomon Benchmark/{dataset_type}/{instance}",
                    temperature=0.2,
                    max_iterations=500,
                    destruction_ratio=0.3,
                    use_plugin_generation=True,
                    notes=f"Stability test run {run_id}"
                ))
        
        print(f"✓ Generated 6 stability experiments")
        
        return experiments
    
    def save_experiment_configs(self, experiments: List[QuickExperimentConfig]):
        """保存实验配置"""
        config_file = self.configs_dir / "quick_experiments.json"
        
        configs = [exp.to_dict() for exp in experiments]
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(configs, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved {len(experiments)} experiment configs to {config_file}")
        return config_file
    
    def generate_run_script_windows(self, experiments: List[QuickExperimentConfig]):
        """生成Windows批处理文件"""
        batch_path = self.output_dir / "run_quick_experiments.bat"
        
        lines = [
            "@echo off",
            "REM Quick Heuristic Experiment Script for Windows",
            f"REM Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"REM Total experiments: {len(experiments)}",
            f"REM Estimated time: {len(experiments) * 5}-{len(experiments) * 10} minutes",
            "",
            "echo ==========================================",
            f"echo Starting {len(experiments)} quick experiments...",
            "echo ==========================================",
            "",
            "if not exist experiments_quick\\results mkdir experiments_quick\\results",
            "if not exist experiments_quick\\logs mkdir experiments_quick\\logs",
            "if not exist experiments_quick\\figures mkdir experiments_quick\\figures",
            "",
            "set START_TIME=%TIME%",
            "set EXP_COUNT=0",
            ""
        ]
        
        for idx, exp in enumerate(experiments, 1):
            # 构建命令 - 使用llm_heuristic.py实际支持的参数
            cmd = f"python llm_heuristic.py "
            cmd += f"--model \"{exp.model}\" "
            cmd += f"--dataset \"{exp.instance_file}\" "
            cmd += f"--max_attempts 5 "
            cmd += f"--exp_name \"{exp.exp_name}\" "
            cmd += f"--output \"experiments_quick\\results\\{exp.exp_name}.json\" "
            cmd += f"--quiet"
            
            # 输出文件
            result_file = f"experiments_quick\\results\\{exp.exp_name}.json"
            log_file = f"experiments_quick\\logs\\{exp.exp_name}.log"
            
            lines.extend([
                f"echo.",
                f"echo [%TIME%] ---------- Experiment {idx}/{len(experiments)}: {exp.exp_name} ----------",
                f"echo Model: {exp.model}",
                f"echo Dataset: {exp.dataset_type} - {exp.instance_file}",
                f"echo Config: temp={exp.temperature}, iter={exp.max_iterations}, destroy={exp.destruction_ratio}",
                "",
                f"REM Run experiment",
                f"{cmd} > {log_file} 2>&1",
                "",
                f"if %ERRORLEVEL% EQU 0 (",
                f"    echo [SUCCESS] Completed {exp.exp_name}",
                f"    set /A EXP_COUNT+=1",
                f") else (",
                f"    echo [FAILED] Error in {exp.exp_name} - Check log file",
                f")",
                ""
            ])
        
        lines.extend([
            "echo.",
            "echo ==========================================",
            "echo Batch process completed!",
            "echo Successful experiments: %EXP_COUNT% / %TOTAL_EXP%",
            "echo Start time: %START_TIME%",
            "echo End time: %TIME%",
            "echo ==========================================",
            "echo.",
            "echo Next step: Run analysis",
            "echo   python result_analyzer_heuristic.py",
            "",
            "pause",
            ""
        ])
        
        with open(batch_path, 'w', encoding='utf-8') as f:
            f.write('\r\n'.join(lines))
        
        print(f"✓ Generated Windows batch file: {batch_path}")
        return batch_path
    
    def generate_python_runner(self, experiments: List[QuickExperimentConfig]):
        """生成Python运行脚本（更灵活，跨平台）"""
        runner_path = self.output_dir / "run_experiments.py"
        
        code = '''"""
快速实验运行器 - Python版本
支持断点续传、进度显示、错误处理
"""
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime

def run_single_experiment(exp_config, idx, total):
    """运行单个实验"""
    exp_name = exp_config['exp_name']
    
    # 检查是否已完成
    result_file = Path(f"experiments_quick/results/{exp_name}.json")
    if result_file.exists():
        print(f"[{idx}/{total}] SKIP: {exp_name} (already completed)")
        return True
    
    print(f"\\n[{idx}/{total}] RUNNING: {exp_name}")
    print(f"  Model: {exp_config['model']}")
    print(f"  Dataset: {exp_config['instance_file']}")
    
    # 构建命令 - 使用llm_heuristic.py实际支持的参数
    cmd = [
        "python", "llm_heuristic.py",
        "--model", exp_config['model'],
        "--dataset", exp_config['instance_file'],
        "--max_attempts", "5",
        "--exp_name", exp_name,
        "--output", f"experiments_quick/results/{exp_name}.json",
        "--quiet"
    ]
    
    # 日志文件
    log_file = Path(f"experiments_quick/logs/{exp_name}.log")
    
    # 运行实验
    start_time = time.time()
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=600,  # 10分钟超时
                check=True
            )
        
        elapsed = time.time() - start_time
        print(f"  ✓ SUCCESS in {elapsed:.1f}s")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"  ✗ TIMEOUT (>10 minutes)")
        return False
    except subprocess.CalledProcessError as e:
        print(f"  ✗ FAILED (exit code {e.returncode})")
        return False
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return False

def main():
    """主函数"""
    print("=" * 70)
    print("Quick Heuristic Experiment Runner")
    print("=" * 70)
    print()
    
    # 加载实验配置
    config_file = Path("experiments_quick/configs/quick_experiments.json")
    with open(config_file, 'r', encoding='utf-8') as f:
        experiments = json.load(f)
    
    total = len(experiments)
    print(f"Total experiments: {total}")
    print(f"Estimated time: {total * 5}-{total * 10} minutes")
    print()
    
    input("Press Enter to start...")
    print()
    
    # 运行实验
    start_time = time.time()
    success_count = 0
    
    for idx, exp in enumerate(experiments, 1):
        if run_single_experiment(exp, idx, total):
            success_count += 1
    
    # 总结
    elapsed_total = time.time() - start_time
    print()
    print("=" * 70)
    print("Batch process completed!")
    print(f"Successful: {success_count} / {total}")
    print(f"Failed: {total - success_count}")
    print(f"Total time: {elapsed_total / 60:.1f} minutes")
    print("=" * 70)
    print()
    print("Next step: Run analysis")
    print("  python result_analyzer_heuristic.py")

if __name__ == "__main__":
    main()
'''
        
        with open(runner_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        print(f"✓ Generated Python runner: {runner_path}")
        return runner_path
    
    def generate_readme(self, experiments: List[QuickExperimentConfig]):
        """生成README说明文档"""
        readme_path = self.output_dir / "README.md"
        
        # 统计信息
        exp_types = {}
        for exp in experiments:
            exp_type = exp.exp_name.split('_')[0]
            exp_types[exp_type] = exp_types.get(exp_type, 0) + 1
        
        content = f"""# Quick Heuristic Experiments

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**实验总数**: {len(experiments)}  
**预计时间**: {len(experiments) * 5}-{len(experiments) * 10} 分钟

## 📊 实验分布

| 类型 | 数量 | 说明 |
|------|------|------|
| main | {exp_types.get('main', 0)} | 核心模型对比（3模型 × 2数据集 × 2实例） |
| ablation | {exp_types.get('ablation', 0)} | 消融实验（LLM生成 vs Baseline） |
| temp | {exp_types.get('temp', 0)} | 温度参数调优 |
| iter | {exp_types.get('iter', 0)} | 迭代次数影响 |
| destroy | {exp_types.get('destroy', 0)} | 破坏比例调优 |
| stability | {exp_types.get('stability', 0)} | 稳定性测试（3次重复） |

## 🚀 运行方式

### 方式1: Python运行器（推荐，跨平台）

```bash
python experiments_quick/run_experiments.py
```

**优势**：
- ✅ 断点续传（已完成的实验会跳过）
- ✅ 实时进度显示
- ✅ 超时保护（10分钟）
- ✅ 跨平台支持

### 方式2: Windows批处理

```cmd
experiments_quick\\run_quick_experiments.bat
```

### 方式3: 手动运行单个实验（测试用）

```bash
python llm_heuristic.py \\
    --model o3-mini \\
    --dataset "data/1 Solomon Benchmark/c1/c101.txt" \\
    --temperature 0.2 \\
    --max_iterations 500 \\
    --destruction_ratio 0.3 \\
    --use_plugin
```

## 📈 结果分析

实验完成后，运行分析脚本：

```bash
python result_analyzer_heuristic.py
```

这将生成：
- 📊 对比图表（PNG格式，600 DPI）
- 📋 统计表格（CSV + LaTeX）
- 📉 收敛曲线分析

## ⚠️ 注意事项

1. **API配置**：确保 `.env` 文件配置了正确的API密钥
2. **数据集路径**：确认Solomon Benchmark数据在 `data/1 Solomon Benchmark/` 目录
3. **磁盘空间**：需要至少2GB空闲空间
4. **中断恢复**：Python运行器支持中断后继续（已完成的会跳过）

## 📁 输出结构

```
experiments_quick/
├── configs/
│   ├── quick_experiments.json    # 实验配置
│   └── README.md                  # 本文件
├── results/
│   ├── main_o3mini_c1_c101.json  # 实验结果
│   └── ...
├── logs/
│   ├── main_o3mini_c1_c101.log   # 详细日志
│   └── ...
└── figures/
    ├── model_comparison.png       # 分析图表
    └── ...
```

## 🔍 检查进度

```bash
# 查看已完成数量
ls experiments_quick/results/*.json | wc -l

# 查看最新日志
tail -f experiments_quick/logs/main_o3mini_c1_c101.log

# 查看错误
grep -r "Error" experiments_quick/logs/
```

## 🎯 下一步

1. ✅ 运行这个快速测试（~40个实验）
2. 📊 查看结果和图表
3. 🔧 根据结果调整参数
4. 🚀 运行完整版实验（350个）
   ```bash
   python experiment_heuristic_manager.py
   ```

## 💡 提示

- 如果某个实验失败，查看对应的 `.log` 文件
- 可以修改 `max_iterations` 来加速测试（如改为300）
- Python运行器支持 Ctrl+C 中断，下次运行会跳过已完成的

---

**祝实验顺利！** 🎉
"""
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✓ Generated README: {readme_path}")


def main():
    """主函数"""
    print("=" * 70)
    print("Quick Heuristic Experiment Manager")
    print("Generating ~40-50 experiments for rapid testing")
    print("=" * 70)
    print()
    
    manager = QuickExperimentManager()
    
    # 生成实验配置
    print("Generating quick test experiments...")
    print()
    experiments = manager.generate_quick_test_experiments()
    
    total = len(experiments)
    print()
    print(f"✓ Total: {total} experiments")
    print(f"✓ Estimated time: {total * 5}-{total * 10} minutes")
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
    manager.generate_run_script_windows(experiments)
    runner_path = manager.generate_python_runner(experiments)
    print()
    
    # 生成README
    print("Generating documentation...")
    manager.generate_readme(experiments)
    print()
    
    print("=" * 70)
    print("Setup complete!")
    print()
    print("📌 Quick Start:")
    print()
    print("  1. Review configuration:")
    print(f"     {config_file}")
    print()
    print("  2. Run experiments (RECOMMENDED):")
    print(f"     python {runner_path}")
    print()
    print("  3. Or use batch file (Windows):")
    print(f"     experiments_quick\\run_quick_experiments.bat")
    print()
    print("  4. After completion, analyze results:")
    print("     python result_analyzer_heuristic.py")
    print()
    print("=" * 70)
    print()
    print("💡 Tip: Test with 1-2 experiments first to verify your setup:")
    print("   python llm_heuristic.py --model o3-mini --dataset \"data/1 Solomon Benchmark/c1/c101.txt\"")
    print()


if __name__ == "__main__":
    main()
