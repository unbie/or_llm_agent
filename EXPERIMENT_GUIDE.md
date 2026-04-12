# OR-LLM-Agent 实验操作指南

> **最后更新**: 2026-03-31  
> **项目**: 基于LLM的ALNS算法求解车辆路径问题(VRP)

---

## 📋 目录

1. [实验概述](#实验概述)
2. [环境准备](#环境准备)
3. [实验方案选择](#实验方案选择)
4. [方案一：省Token批量实验](#方案一省token批量实验)
5. [方案二：LLM生成实验](#方案二llm生成实验)
6. [结果分析](#结果分析)
7. [论文素材生成](#论文素材生成)
8. [常见问题](#常见问题)

---

## 实验概述

本项目使用**自适应大邻域搜索(ALNS)** 算法求解 **Solomon Benchmark** 数据集上的车辆路径问题。

### 实验目标
- 验证算法在不同类型数据集上的性能
- 分析算法稳定性（多次运行）
- 与文献最优解(BKS)进行对比
- 生成论文所需的图表和数据

### 数据集说明

| 类型 | 特征 | 代表实例 |
|------|------|----------|
| **c1** | 聚类分布 + 窄时间窗 | c101, c102, ... |
| **c2** | 聚类分布 + 宽时间窗 | c201, c202, ... |
| **r1** | 随机分布 + 窄时间窗 | r101, r102, ... |
| **r2** | 随机分布 + 宽时间窗 | r201, r202, ... |
| **rc1** | 混合分布 + 窄时间窗 | rc101, rc102, ... |
| **rc2** | 混合分布 + 宽时间窗 | rc201, rc202, ... |

---

## 环境准备

### 1. 安装依赖

```powershell
cd D:\pythonProject\or_llm_agent
pip install -r requirements.txt
```

或手动安装：

```powershell
pip install pandas matplotlib seaborn numpy
```

### 2. 验证数据集

确保以下目录存在：
```
data/1 Solomon Benchmark/
├── c1/
│   ├── c101.txt
│   ├── c102.txt
│   └── ...
├── c2/
├── r1/
├── r2/
├── rc1/
└── rc2/
```

### 3. 验证环境

```powershell
python -c "import pandas; import matplotlib; print('环境OK')"
```

---

## 实验方案选择

| 方案 | Token消耗 | 实验数量 | 耗时 | 适用场景 |
|------|-----------|----------|------|----------|
| **方案一** | ✅ 0 | 36+ | 10-30分钟 | 快速验证、论文初稿 |
| **方案二** | ⚠️ 高 | 35-350 | 2-10小时 | 完整实验、最终论文 |

**推荐**: 先用方案一快速出结果，再根据需要补充方案二。

---

## 方案一：省Token批量实验

### 原理
使用**预设的高质量ALNS算子代码**，不调用LLM，直接运行实验。

### 步骤

#### 步骤1：运行批量实验

```powershell
cd D:\pythonProject\or_llm_agent
python run_batch_no_llm.py
```

**运行信息**:
- 默认配置: 12个实例 × 3次重复 = 36个实验
- 预计耗时: 10-30分钟
- 输出目录: `experiments_batch/results/`

#### 步骤2：分析结果

```powershell
python result_analyzer_batch.py
```

**输出**:
- `experiments_batch/tables/` - CSV表格
- `experiments_batch/figures/` - PNG图表
- `experiments_batch/latex/` - LaTeX源码

### 自定义配置

编辑 `run_batch_no_llm.py` 中的参数：

```python
# 修改数据集列表
datasets = [
    ("c1", "c101"), ("c1", "c102"), ("c1", "c103"),
    ("c2", "c201"), ("c2", "c202"),
    # 添加更多...
]

# 修改重复次数
num_runs = 5  # 默认3次

# 修改ALNS参数
max_iterations = 500  # 默认200
```

---

## 方案二：LLM生成实验

### 原理
调用LLM生成ALNS算子代码，测试LLM的代码生成能力。

### 步骤

#### 步骤1：生成实验配置

```powershell
python experiment_quick_manager.py
```

输出: `experiments_quick/` 目录

#### 步骤2：运行实验

```powershell
python experiments_quick/run_experiments.py
```

**注意**: 
- ⚠️ 消耗大量Token（约50-100万）
- 预计耗时: 2-3小时
- 需要配置API密钥

#### 步骤3：分析结果

```powershell
python result_analyzer_heuristic.py --results_dir experiments_quick/results
```

### API配置

编辑 `llm_heuristic.py` 中的API设置（约第51-56行）：

```python
API_BASE = "https://your-api-endpoint"
API_KEY = "your-api-key"
MODEL_ID = "your-model-id"
```

---

## 结果分析

### 自动生成的分析内容

| 分析类型 | 文件 | 用途 |
|----------|------|------|
| 数据集对比表 | `table1_dataset_comparison.csv` | 不同类型数据集的性能对比 |
| 实例详情表 | `table2_instance_details.csv` | 每个实例的详细结果 |
| 稳定性分析表 | `table3_stability_analysis.csv` | 多次运行的变异系数 |
| 性能对比图 | `fig1_dataset_comparison.png` | 与BKS的Gap对比 |
| 稳定性箱线图 | `fig2_stability_boxplot.png` | 算法稳定性可视化 |
| 计算时间图 | `fig3_computation_time.png` | 各类型数据集耗时 |

### 手动分析

```python
import pandas as pd

# 加载结果
df = pd.read_csv('experiments_batch/tables/table1_dataset_comparison.csv')
print(df)
```

---

## 论文素材生成

### LaTeX表格示例

生成的 `latex/table1.tex` 可直接插入论文：

```latex
\begin{table}[htbp]
\centering
\caption{Performance by Dataset Type}
\label{tab:dataset_comparison}
\input{tables/table1}
\end{table}
```

### 图表引用

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{figures/fig1_dataset_comparison.png}
\caption{Solution quality comparison across dataset types}
\label{fig:dataset_comparison}
\end{figure}
```

### 关键指标说明

| 指标 | 含义 | 论文表述 |
|------|------|----------|
| Gap to BKS (%) | 与文献最优解的差距 | "The average gap to best-known solutions is X%" |
| CV (%) | 变异系数，衡量稳定性 | "The coefficient of variation is below X%, indicating stable performance" |
| Avg Vehicles | 平均使用车辆数 | "The algorithm uses an average of X vehicles" |

---

## 常见问题

### Q1: 找不到数据集文件

**错误**: `FileNotFoundError: data/1 Solomon Benchmark/...`

**解决**: 
1. 检查数据集目录是否存在
2. 确认路径中的空格和特殊字符

### Q2: 导入模块失败

**错误**: `ModuleNotFoundError: No module named 'pandas'`

**解决**:
```powershell
pip install pandas matplotlib seaborn numpy
```

### Q3: 实验运行中断

**解决**: 
- 已完成的结果保存在 `results/` 目录
- 可直接运行分析脚本，只分析已有结果
- 修改 `run_batch_no_llm.py` 跳过已完成的实例

### Q4: 如何增加实验数量

编辑 `run_batch_no_llm.py`:

```python
# 增加数据集
datasets = [
    ("c1", f"c10{i}") for i in range(1, 10)
] + [
    ("r1", f"r10{i}") for i in range(1, 13)
]  # 等等

# 增加重复次数
num_runs = 10
```

### Q5: 如何修改ALNS参数

编辑 `run_batch_no_llm.py` 中的 `run_single_experiment()` 函数：

```python
solver = HeuristicSolver(
    nodes=nodes,
    vehicles=vehicles,
    depot=depot,
    max_iterations=500,      # 迭代次数
    destruction_ratio=0.3,   # 破坏比例
    # ...
)
```

---

## 文件结构

```
or_llm_agent/
├── 核心文件
│   ├── llm_heuristic.py          # LLM调用主程序
│   ├── heuristic_skeleton.py     # ALNS框架
│   └── heuristic_prompts.py      # 提示词模板
│
├── 实验脚本
│   ├── run_batch_no_llm.py       # ⭐ 省Token批量实验
│   ├── experiment_quick_manager.py   # LLM实验配置生成
│   └── experiment_heuristic_manager.py  # 完整实验管理
│
├── 分析脚本
│   ├── result_analyzer_batch.py  # ⭐ 批量实验分析
│   └── result_analyzer_heuristic.py  # LLM实验分析
│
├── 文档
│   ├── EXPERIMENT_GUIDE.md       # ⭐ 本文档
│   └── README.md
│
└── 数据
    └── data/1 Solomon Benchmark/  # 数据集
```

---

## 快速开始（TL;DR）

```powershell
# 1. 进入项目目录
cd D:\pythonProject\or_llm_agent

# 2. 运行实验（不消耗Token）
python run_batch_no_llm.py

# 3. 分析结果
python result_analyzer_batch.py

# 4. 查看输出
# - 表格: experiments_batch/tables/
# - 图表: experiments_batch/figures/
# - LaTeX: experiments_batch/latex/
```

---

> 💡 **提示**: 如有问题，请检查控制台输出的错误信息，或查看 [常见问题](#常见问题) 部分。
