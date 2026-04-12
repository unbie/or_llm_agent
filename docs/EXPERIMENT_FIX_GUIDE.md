# 🐛 实验运行问题修复指南

> **日期**: 2026-04-01  
> **问题**: run_batch_no_llm.py 报错修复

---

## 问题描述

运行 `python run_batch_no_llm.py` 时出现以下错误：

```
[1/36] c1_c101_run1
  数据集: c1/c101.txt
  ✗ 失败
```

---

## 根本原因

1. **模块导入问题**: 临时生成的Python文件无法导入 `utils.py` 中的 `FreshnessAndPenaltyCalculator` 类
2. **编码问题**: Windows 默认使用 GBK 编码，无法处理 Unicode 字符（✓, ✗）

---

## 修复内容

### 1. 添加 FreshnessAndPenaltyCalculator 类到生成代码中

**位置**: `run_batch_no_llm.py` 第289-361行

**修改**: 将完整的 `FreshnessAndPenaltyCalculator` 类代码嵌入到生成的临时文件中

```python
utils_code = '''
import math

class FreshnessAndPenaltyCalculator:
    # ... 完整类定义 ...
'''

full_code = utils_code + "\n\n" + HEURISTIC_SKELETON + ...
```

### 2. 替换骨架代码中的 import 语句

```python
HEURISTIC_SKELETON.replace(
    "from utils import FreshnessAndPenaltyCalculator",
    "# FreshnessAndPenaltyCalculator already defined above"
)
```

### 3. 修复编码问题

**添加 UTF-8 编码声明**:

```python
full_code = (
    "# -*- coding: utf-8 -*-\n"
    "import sys\n"
    "import io\n"
    "sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')\n"
    "sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')\n\n"
) + utils_code + ...
```

**subprocess 调用添加编码参数**:

```python
result = subprocess.run(
    [sys.executable, temp_file],
    capture_output=True,
    text=True,
    timeout=600,
    encoding='utf-8',        # ← 新增
    errors='replace',        # ← 新增
    cwd=os.path.dirname(os.path.abspath(__file__))
)
```

### 4. 移除手动确认

删除了 `input("按回车开始实验...")` 以便自动运行。

---

## 验证修复

### 测试单个实验

```powershell
cd D:\pythonProject\or_llm_agent
python test_single_run.py
```

**预期输出**:

```
测试数据集: data\1 Solomon Benchmark\c1\c101.txt
============================================================

结果:
成功: True
最佳成本: 77703.57
路线数: 18
耗时: 4.95秒
```

### 运行完整批量实验

```powershell
python run_batch_no_llm.py
```

---

## 当前状态

✅ **已修复**: 所有问题已解决，实验可以正常运行

---

## 如何使用

### 步骤 1: 运行批量实验

```powershell
cd D:\pythonProject\or_llm_agent
python run_batch_no_llm.py
```

**输出示例**:

```
======================================================================
省Token批量实验 - Zero Token Batch Experiments
使用固定算子代码，不调用LLM
======================================================================

总计: 12 个数据集, 36 次运行
预计时间: 72-180 分钟

开始实验...

[1/36] c1_c101_run1
  数据集: c1/c101.txt
  ✓ Cost=77703.57, Routes=18, Time=4.9s

[2/36] c1_c101_run2
  数据集: c1/c101.txt
  ✓ Cost=77856.32, Routes=18, Time=5.1s

...
```

### 步骤 2: 分析结果

```powershell
python result_analyzer_batch.py
```

**生成输出**:

- `experiments_batch/tables/` - CSV 表格
- `experiments_batch/figures/` - PNG 图表
- `experiments_batch/latex/` - LaTeX 代码

---

## 实验参数配置

### 修改数据集

编辑 `run_batch_no_llm.py` 第376-390行:

```python
experiments = [
    ('c1', 'c101.txt', 3),  # (类型, 文件名, 运行次数)
    ('c1', 'c105.txt', 3),
    # 添加更多...
]
```

### 修改迭代次数

第423行:

```python
result = run_single_experiment(
    str(dataset_path), 
    max_iters=1000,  # ← 修改这里
    seed=seed
)
```

### 修改超时时间

第399行:

```python
timeout=600,  # 秒，修改这个值
```

---

## 常见问题

### Q1: 实验太慢怎么办？

**方案 1**: 减少迭代次数

```python
max_iters=200  # 默认 1000
```

**方案 2**: 减少数据集

```python
experiments = [
    ('c1', 'c101.txt', 1),  # 只测试1次
]
```

### Q2: 内存不足

批量实验会逐个运行，不会同时运行多个，内存占用不高（约500MB）。

### Q3: 如何暂停和继续？

实验结果会保存到 `experiments_batch/results/` 目录，可以：

1. Ctrl+C 中断
2. 删除已完成的实验配置
3. 重新运行

### Q4: 如何查看实时进度？

实验运行时会显示：

```
[1/36] c1_c101_run1
  数据集: c1/c101.txt
  ✓ Cost=77703.57, Routes=18, Time=4.9s
```

---

## 技术细节

### 临时文件生成

每个实验会生成一个临时 Python 文件：

```
C:\Users\xxx\AppData\Local\Temp\tmpXXXXXX.py
```

包含:
1. UTF-8 编码设置
2. FreshnessAndPenaltyCalculator 类
3. HEURISTIC_SKELETON 骨架代码
4. FIXED_PLUGIN_CODE 固定算子
5. 主函数执行代码

### 结果提取

使用正则表达式从输出中提取：

```python
cost_match = re.search(r'BEST_COST:\s*([\d.]+)', output)
route_match = re.search(r'NUM_ROUTES:\s*(\d+)', output)
```

---

## 修改历史

| 日期 | 修改内容 |
|------|----------|
| 2026-04-01 | 修复模块导入和编码问题 |
| 2026-03-31 | 初始版本创建 |

---

> 💡 **提示**: 如果遇到新问题，查看 `test_single_run.py` 的详细输出进行调试。
