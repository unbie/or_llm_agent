# 🎯 启发式算法实验完整指南

## LLM生成ALNS算法解决Solomon Benchmark VRP问题

---

## 📋 目录

1. [实验概述](#实验概述)
2. [研究问题](#研究问题)
3. [实验设计](#实验设计)
4. [运行实验](#运行实验)
5. [结果分析](#结果分析)
6. [论文撰写建议](#论文撰写建议)

---

## 🔬 实验概述

### 研究目标

评估大语言模型（LLM）生成启发式算法代码的能力，具体应用于车辆路径问题（VRP）的求解。

### 核心创新点

1. **LLM自动生成ALNS算子**：使用LLM生成自适应大邻域搜索（ALNS）的破坏和修复算子
2. **完整成本计算**：包含生鲜物流特有的新鲜度衰减、时间窗惩罚等多目标成本
3. **Solomon Benchmark评估**：在经典VRP基准数据集上全面测试

### 实验规模

- **总实验数量**：~350个实验
- **测试模型**：6个主流LLM（GPT-4o, Claude, DeepSeek等）
- **数据集**：Solomon Benchmark 6大类（C1, C2, R1, R2, RC1, RC2）
- **超参数调优**：温度、迭代次数、破坏比例

---

## 🤔 研究问题

### RQ1: 模型性能对比

**问题**：不同LLM在生成启发式算法代码时的性能差异？

**实验**：主要对比实验（180个）

- 6个模型 × 6个数据集类型 × 5个实例
- **评估指标**：
  - 目标函数值（Total Cost）
  - 与最优解差距（Gap to BKS, %）
  - 收敛速度（Iterations to Best）
  - 代码正确率（Syntax Success Rate）

**预期发现**：

- 推理模型（o3, DeepSeek-R1）优于非推理模型
- 代码生成专用模型表现更好
- 不同数据集难度对模型影响

---

### RQ2: 算子生成策略有效性

**问题**：LLM生成的算子相比默认算子的优劣？

**实验**：消融实验（18个）

- 3个模型 × 3个数据集 × 2种策略
  - LLM生成所有算子
  - 使用默认baseline算子

**评估指标**：

- 解质量对比
- 算子使用频率分析
- 成本计算准确性

**预期发现**：

- LLM生成的算子在特定数据集上更优
- 默认算子更稳定但缺乏适应性
- 算子质量与代码复杂度关系

---

### RQ3: 超参数敏感性分析

**问题**：关键超参数如何影响算法性能？

#### 3.1 Temperature（温度）

**实验**：6个温度 × 1个实例 = 6个实验

- 范围：0.0, 0.2, 0.5, 0.7, 1.0, 1.5
- **观测**：代码多样性 vs 稳定性权衡

#### 3.2 迭代次数

**实验**：4个迭代数 × 1个实例 = 4个实验

- 范围：500, 1000, 2000, 5000
- **观测**：收敛速度 vs 计算时间权衡

#### 3.3 破坏比例

**实验**：5个比例 × 1个实例 = 5个实验

- 范围：0.1, 0.2, 0.3, 0.4, 0.5
- **观测**：搜索强度 vs 解质量关系

---

### RQ4: 算法稳定性与可重复性

**问题**：LLM生成的算法在多次运行中的稳定性如何？

**实验**：稳定性测试（15个）

- 3个配置 × 5次重复运行
- **评估指标**：
  - 标准差（Standard Deviation）
  - 变异系数（Coefficient of Variation, CV）
  - 最优/最差解差距

**预期发现**：

- 温度对稳定性的影响
- 不同模型的一致性表现
- 随机种子控制的必要性

---

## 🧪 实验设计

### 实验类型总览

| 实验类型               | 数量            | 目的             | 关键变量          |
| ---------------------- | --------------- | ---------------- | ----------------- |
| **主要对比**     | ~180            | 模型性能横向比较 | 模型、数据集      |
| **消融实验**     | ~18             | 算子生成策略验证 | 生成策略          |
| **温度调优**     | 6               | 温度参数影响     | Temperature       |
| **迭代调优**     | 4               | 迭代次数影响     | Max Iterations    |
| **破坏比例调优** | 5               | 破坏比例影响     | Destruction Ratio |
| **稳定性测试**   | ~15             | 多次运行稳定性   | 随机种子          |
| **总计**         | **~228+** | -                | -                 |

### Solomon Benchmark数据集

| 类型          | 特点                | 实例数 | 难度       |
| ------------- | ------------------- | ------ | ---------- |
| **C1**  | 聚类客户 + 窄时间窗 | 9      | ⭐⭐       |
| **C2**  | 聚类客户 + 宽时间窗 | 8      | ⭐         |
| **R1**  | 随机客户 + 窄时间窗 | 12     | ⭐⭐⭐⭐   |
| **R2**  | 随机客户 + 宽时间窗 | 11     | ⭐⭐⭐     |
| **RC1** | 混合分布 + 窄时间窗 | 8      | ⭐⭐⭐⭐⭐ |
| **RC2** | 混合分布 + 宽时间窗 | 8      | ⭐⭐⭐⭐   |

### 评估指标体系

#### 1. 解质量指标

```
- Total Cost：总成本（越低越好）
- Gap to BKS：与最优解差距 = (Your_Cost - BKS) / BKS × 100%
- Number of Vehicles：使用车辆数（越少越好）
```

#### 2. 算法性能指标

```
- Convergence Speed：收敛速度（迭代次数）
- Computation Time：计算时间（秒）
- Solution Stability：解稳定性（标准差、CV）
```

#### 3. 代码质量指标

```
- Syntax Correctness：语法正确率
- Cost Calculation Accuracy：成本计算准确性
- Algorithm Completeness：算法完整性（5个必需算子）
```

---

## 🚀 运行实验

### 步骤1: 生成实验配置

```bash
# 生成所有实验配置文件
python experiment_heuristic_manager.py
```

**输出**：

- `experiments_heuristic/configs/all_experiments.json` - 所有实验配置
- `experiments_heuristic/configs/experiment_summary.md` - 实验摘要
- `experiments_heuristic/run_heuristic_experiments.sh` - Linux运行脚本
- `experiments_heuristic/run_heuristic_experiments.bat` - Windows运行脚本

### 步骤2: 检查配置

```bash
# 查看实验摘要
cat experiments_heuristic/configs/experiment_summary.md

# 查看具体配置（可选）
head -n 50 experiments_heuristic/configs/all_experiments.json
```

### 步骤3: 运行实验

#### Linux/Mac:

```bash
chmod +x experiments_heuristic/run_heuristic_experiments.sh
bash experiments_heuristic/run_heuristic_experiments.sh
```

#### Windows:

```cmd
experiments_heuristic\run_heuristic_experiments.bat
```

#### 单个实验测试（推荐先测试）:

```bash
python llm_heuristic.py \
    --model o3-mini \
    --dataset "data/1 Solomon Benchmark/c1/c101.txt" \
    --temperature 0.2 \
    --max_iterations 1000 \
    --destruction_ratio 0.3 \
    --use_plugin \
    --output experiments_heuristic/results/test_c101.json
```

### 步骤4: 监控进度

```bash
# 查看日志
tail -f experiments_heuristic/logs/main_o3mini_c1_c101.log

# 统计完成数量
ls experiments_heuristic/results/*.json | wc -l

# 检查错误
grep -r "Error" experiments_heuristic/logs/
```

### ⚠️ 注意事项

1. **API配置**：确保 `.env`文件配置正确
2. **并行运行**：可以使用GNU Parallel加速
3. **中断恢复**：脚本支持断点续传（已完成的会跳过）
4. **资源消耗**：
   - 每个实验约5-15分钟
   - 总计时间：20-50小时（串行）
   - 建议使用GPU服务器并行运行

---

## 📊 结果分析

### 步骤1: 运行分析工具

```bash
python result_analyzer_heuristic.py
```

**生成的输出**：

```
experiments_heuristic/
├── figures/
│   ├── model_comparison.png          # 模型性能对比
│   ├── convergence_comparison.png    # 收敛曲线对比
│   ├── hyperparameter_tuning.png     # 超参数调优结果
│   └── stability_analysis.png        # 稳定性分析
├── tables/
│   ├── main_comparison.csv           # 主要对比表格
│   ├── ablation_study.csv            # 消融实验表格
│   ├── temperature.csv               # 温度调优表格
│   ├── iterations.csv                # 迭代次数表格
│   ├── destruction_ratio.csv         # 破坏比例表格
│   └── stability_analysis.csv        # 稳定性分析表格
└── latex/
    ├── main_comparison.tex           # LaTeX格式表格
    ├── ablation_study.tex
    └── ...
```

### 步骤2: 深度分析

#### 2.1 统计显著性检验

```python
# 使用scipy进行t检验
from scipy import stats

# 比较两个模型的性能
model_a_costs = df[df['model'] == 'o3-mini']['best_cost']
model_b_costs = df[df['model'] == 'gpt-4o']['best_cost']

t_stat, p_value = stats.ttest_ind(model_a_costs, model_b_costs)
print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")
```

#### 2.2 相关性分析

```python
# 超参数与性能的相关性
correlation_matrix = df[['temperature', 'max_iterations', 'destruction_ratio', 'gap_to_bks']].corr()
sns.heatmap(correlation_matrix, annot=True)
```

#### 2.3 排名分析

```python
# 各模型在不同数据集上的排名
ranking_df = df.groupby(['dataset_type', 'model'])['gap_to_bks'].mean().unstack()
ranking_df = ranking_df.rank(axis=1)
```

---

## 📝 论文撰写建议

### 5.1 实验部分结构（Example）

```markdown
## 5. Experiments

### 5.1 Experimental Setup

**Datasets**: We evaluate our approach on the Solomon Benchmark [Solomon, 1987], 
which contains 56 instances across 6 categories (C1, C2, R1, R2, RC1, RC2).

**Models**: We compare 6 state-of-the-art LLMs:
- GPT-4o, GPT-4o-mini (OpenAI)
- Claude-3.5-Sonnet (Anthropic)
- DeepSeek-R1, DeepSeek-V3 (DeepSeek)
- o3-mini (OpenAI)

**Baseline**: Default ALNS implementation with standard operators.

**Hyperparameters**:
- Temperature: {0.0, 0.2, 0.5, 0.7, 1.0, 1.5}
- Max Iterations: {500, 1000, 2000, 5000}
- Destruction Ratio: {0.1, 0.2, 0.3, 0.4, 0.5}

**Evaluation Metrics**:
- Gap to Best Known Solution (BKS): (Cost - BKS) / BKS × 100%
- Convergence Speed: Number of iterations to best solution
- Solution Stability: Coefficient of Variation across 5 runs

### 5.2 Main Results (RQ1)

Table 1 shows the performance comparison across different models and dataset types.

**Key Findings**:
1. DeepSeek-R1 achieves the best average gap of X.X% to BKS
2. Reasoning models (o3, DeepSeek-R1) outperform standard models by Y%
3. C-type instances are easier (avg gap Z%) than R-type (avg gap W%)

### 5.3 Ablation Study (RQ2)

Figure 2 illustrates the contribution of LLM-generated operators vs baseline.

**Observations**:
- LLM-generated operators improve solution quality by A% on average
- Customized insertion operators show B% better performance
- Default operators are more stable (CV: C% vs D%)

### 5.4 Hyperparameter Analysis (RQ3)

Figure 3 shows the sensitivity to different hyperparameters:

**Temperature**: 
- Lower temp (0.0-0.2) → More deterministic, slightly better quality
- Higher temp (0.7-1.5) → More diverse, higher variance

**Iterations**:
- 1000 iterations achieve 95% of final solution quality
- Diminishing returns after 2000 iterations

**Destruction Ratio**:
- Optimal at 0.3 (sweet spot between exploration and exploitation)

### 5.5 Stability Analysis (RQ4)

Table 3 reports the stability across 5 independent runs:

**Findings**:
- Average CV: E.E% (indicating good reproducibility)
- o3-mini shows highest stability (CV: F.F%)
- Temperature > 0.5 significantly increases variance
```

### 5.2 推荐的表格和图表

#### 表格1: 主要性能对比

```
| Model | C1 | C2 | R1 | R2 | RC1 | RC2 | Average |
|-------|----|----|----|----|-----|-----|---------|
| o3-mini | X.X% | X.X% | ... |
| GPT-4o | X.X% | X.X% | ... |
| ... |
```

#### 表格2: 消融实验

```
| Configuration | Cost | Gap (%) | Vehicles |
|---------------|------|---------|----------|
| Full (LLM-generated) | XXX | X.X% | X |
| Baseline (default) | XXX | X.X% | X |
| Improvement | -XX | -X.X% | -X |
```

#### 图1: 收敛曲线对比

- X轴：迭代次数
- Y轴：目标函数值
- 多条曲线代表不同模型
- 突出最优模型

#### 图2: 超参数热力图

- 展示不同参数组合的性能
- 颜色深浅代表gap to BKS

#### 图3: 稳定性箱线图

- 展示5次运行的分布
- 比较不同模型的方差

### 5.3 统计显著性

**务必报告**：

- P-value（p < 0.05认为显著）
- Effect size（Cohen's d）
- 置信区间（95% CI）

**示例描述**：

```
"DeepSeek-R1 significantly outperforms GPT-4o with an average gap 
improvement of 3.2% (p < 0.001, Cohen's d = 0.87, 95% CI: [2.1%, 4.3%])."
```

---

## 🎓 进阶技巧

### 1. 批量实验管理

```bash
# 使用GNU Parallel并行运行（8个任务）
cat experiments_heuristic/configs/all_experiments.json | \
    jq -r '.[] | @json' | \
    parallel -j 8 'python run_single_exp.py {}'
```

### 2. 实时监控Dashboard

```python
# 使用Streamlit创建实时监控
import streamlit as st
st.title("Experiment Monitor")
st.metric("Completed", f"{completed}/{total}")
st.line_chart(convergence_data)
```

### 3. 结果可视化增强

```python
# 交互式图表（Plotly）
import plotly.express as px
fig = px.scatter(df, x='temperature', y='gap_to_bks', 
                 color='model', size='num_vehicles',
                 hover_data=['instance'])
fig.show()
```

### 4. 自动化报告生成

```bash
# 生成PDF报告
pandoc experiments_heuristic/configs/experiment_summary.md \
    -o experiment_report.pdf \
    --pdf-engine=xelatex
```

---

## ✅ 检查清单

实验运行前：

- [ ] 所有数据集文件存在（Solomon Benchmark）
- [ ] API密钥配置正确
- [ ] Python环境依赖安装完整
- [ ] 有足够的磁盘空间（至少10GB）

实验运行中：

- [ ] 定期检查日志文件
- [ ] 监控API使用量
- [ ] 备份中间结果

实验完成后：

- [ ] 验证结果文件完整性
- [ ] 运行结果分析脚本
- [ ] 生成所有图表和表格
- [ ] 计算统计显著性
- [ ] 备份所有实验数据

---

## 📚 参考文献

1. Solomon, M. M. (1987). "Algorithms for the vehicle routing and scheduling problems with time window constraints." Operations research, 35(2), 254-265.
2. Ropke, S., & Pisinger, D. (2006). "An adaptive large neighborhood search heuristic for the pickup and delivery problem with time windows." Transportation science, 40(4), 455-472.
3. Pisinger, D., & Ropke, S. (2007). "A general heuristic for vehicle routing problems." Computers & operations research, 34(8), 2403-2435.

---

## 💡 常见问题

**Q: 实验太多，如何快速测试？**
A: 先运行一个子集（如只测试C1类型），验证流程正确后再全量运行。

**Q: 某些实验失败怎么办？**
A: 检查日志文件，可能是API限流或代码生成错误。可以单独重跑失败的实验。

**Q: 如何加速实验？**
A: 使用并行运行（GNU Parallel）或云GPU服务器。也可以减少每个配置的重复次数。

**Q: 结果不稳定怎么办？**
A: 增加重复运行次数（5次→10次），降低温度参数，固定随机种子。

---

## 📧 联系与支持

如有问题，请查看：

- GitHub Issues: [项目链接]
- 论文: [arXiv链接]
- 邮箱: [你的邮箱]

---

**祝实验顺利！🎉**
