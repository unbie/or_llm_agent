const pptxgen = require('pptxgenjs');

async function createPresentation() {
  const pptx = new pptxgen();
  pptx.layout = 'LAYOUT_16x9';
  pptx.author = 'Zhang & Luo | OR-LLM-Agent';
  pptx.title = '基于LLM的ALNS算法求解VRP问题';
  pptx.subject = 'ALNS Algorithm with LLM-Generated Operators';

  console.log('Creating presentation...\n');

  // Slide 1: Title
  console.log('[1/15] Adding title slide...');
  const slide1 = pptx.addSlide();
  slide1.addText('基于大语言模型自动生成启发式算子的\nALNS算法求解生鲜物流VRP问题', {
    x: 0.5, y: 1.5, w: 9, h: 2,
    fontSize: 32, bold: true, color: '181B24', align: 'center', valign: 'middle'
  });
  slide1.addShape(pptx.shapes.RECTANGLE, {
    x: 2.5, y: 3.5, w: 5, h: 0.05, fill: { color: 'B165FB' }, line: { type: 'none' }
  });
  slide1.addText('OR-LLM-Agent ALNS模块开发报告', {
    x: 0.5, y: 3.8, w: 9, h: 0.5,
    fontSize: 18, color: '40695B', align: 'center'
  });
  slide1.addText([
    { text: '基于 OR-LLM-Agent (arXiv:2503.10009)\n', options: { fontSize: 14, color: '666666' } },
    { text: 'Zhang & Luo | 上海交通大学 & 南洋理工大学\n', options: { fontSize: 14, color: '666666' } },
    { text: 'DeepSeek R1 | Solomon C105 基准测试', options: { fontSize: 12, color: '999999' } }
  ], {
    x: 0.5, y: 4.5, w: 9, h: 1, align: 'center', valign: 'top'
  });

  // Slide 2: TOC
  console.log('[2/15] Adding table of contents...');
  const slide2 = pptx.addSlide();
  slide2.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: 0.11, h: 5.63, fill: { color: 'B165FB' }, line: { type: 'none' } });
  slide2.addText('目录 Contents', { x: 0.5, y: 0.5, w: 9, h: 0.8, fontSize: 28, bold: true, color: '181B24' });
  
  const tocItems = [
    '01  项目概述 Project Overview',
    '02  系统架构 System Architecture',
    '03  算法设计 Algorithm Design',
    '04  成本模型 Cost Model',
    '05  开发历程 Development Journey',
    '06  实验结果 Experimental Results',
    '07  总结展望 Conclusion & Future Work'
  ];
  
  let yPos = 1.5;
  tocItems.forEach((item, idx) => {
    slide2.addText(item, {
      x: 0.7, y: yPos, w: 8.5, h: 0.4,
      fontSize: 16, color: '181B24'
    });
    yPos += 0.5;
  });

  // Slide 3: Overview
  console.log('[3/15] Adding overview...');
  const slide3 = pptx.addSlide();
  slide3.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: 0.11, h: 5.63, fill: { color: 'B165FB' }, line: { type: 'none' } });
  slide3.addText('01 项目概述 Project Overview', {
    x: 0.5, y: 0.5, w: 9, h: 0.7, fontSize: 24, bold: true, color: '181B24'
  });
  slide3.addText('研究背景 Background', {
    x: 0.5, y: 1.4, w: 9, h: 0.5, fontSize: 18, bold: true, color: '40695B'
  });
  slide3.addText([
    { text: '• ', options: { bullet: true } },
    { text: '车辆路径问题 (VRP) 是经典的 NP-hard 问题\n' },
    { text: '• ', options: { bullet: true } },
    { text: '生鲜物流场景：新鲜度衰减 + 冷链成本 + 时间窗约束\n' },
    { text: '• ', options: { bullet: true } },
    { text: '传统 ALNS 依赖人工设计算子，开发周期长' }
  ], {
    x: 0.7, y: 2, w: 8.8, h: 1.2, fontSize: 14, color: '333333'
  });
  slide3.addText('研究目标 Objectives', {
    x: 0.5, y: 3.3, w: 9, h: 0.5, fontSize: 18, bold: true, color: '40695B'
  });
  slide3.addText([
    { text: '• ', options: { bullet: true } },
    { text: '利用 LLM (DeepSeek R1) 自动生成 ALNS 算子代码\n' },
    { text: '• ', options: { bullet: true } },
    { text: '实现"LLM 生成算子 + 框架执行求解"的混合架构\n' },
    { text: '• ', options: { bullet: true } },
    { text: '在 Solomon 基准上验证算法性能' }
  ], {
    x: 0.7, y: 3.9, w: 8.8, h: 1.2, fontSize: 14, color: '333333'
  });

  // Slide 4: Core Innovations
  console.log('[4/15] Adding core innovations...');
  const slide4 = pptx.addSlide();
  slide4.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: 0.11, h: 5.63, fill: { color: 'B165FB' }, line: { type: 'none' } });
  slide4.addText('核心创新 Core Innovations', {
    x: 0.5, y: 0.5, w: 9, h: 0.7, fontSize: 24, bold: true, color: '181B24'
  });

  const innovations = [
    { num: '01', title: '插件式架构', desc: '将 ALNS 框架与算子实现解耦，LLM 代码作为插件注入' },
    { num: '02', title: '自动调试机制', desc: '多轮对话反馈执行错误，LLM 自主修正代码直至成功' },
    { num: '03', title: '完整成本模型', desc: '融合固定、距离、制冷、货损、时间窗惩罚的五维成本' },
    { num: '04', title: '自适应权重管理', desc: '引入权重下限保护机制，防止算子饥饿' }
  ];

  innovations.forEach((item, idx) => {
    const row = Math.floor(idx / 2);
    const col = idx % 2;
    const x = 0.6 + col * 4.7;
    const y = 1.5 + row * 1.8;

    slide4.addShape(pptx.shapes.RECTANGLE, {
      x, y, w: 4.3, h: 1.4, fill: { color: 'F8F9FA' },
      line: { width: 0.5, color: '40695B', pt: 3 }
    });
    slide4.addText(item.num, {
      x: x + 0.2, y: y + 0.1, w: 1, h: 0.4,
      fontSize: 20, bold: true, color: 'B165FB'
    });
    slide4.addText(item.title, {
      x: x + 0.2, y: y + 0.5, w: 3.9, h: 0.3,
      fontSize: 14, bold: true, color: '181B24'
    });
    slide4.addText(item.desc, {
      x: x + 0.2, y: y + 0.85, w: 3.9, h: 0.5,
      fontSize: 11, color: '555555'
    });
  });

  // Slide 5: System Architecture
  console.log('[5/15] Adding system architecture...');
  const slide5 = pptx.addSlide();
  slide5.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: 0.11, h: 5.63, fill: { color: 'B165FB' }, line: { type: 'none' } });
  slide5.addText('02 系统架构 System Architecture', {
    x: 0.5, y: 0.5, w: 9, h: 0.7, fontSize: 24, bold: true, color: '181B24'
  });

  // Layer 1: LLM Interaction
  slide5.addShape(pptx.shapes.RECTANGLE, {
    x: 1, y: 1.5, w: 8, h: 1, fill: { color: 'F8F9FA' },
    line: { width: 0.5, color: '40695B', pt: 3 }
  });
  slide5.addText('LLM 交互层 (Interaction Layer)', {
    x: 1.2, y: 1.6, w: 7.6, h: 0.3,
    fontSize: 14, bold: true, color: '181B24'
  });
  slide5.addText('llm_heuristic.py + heuristic_prompts.py\n• LLM API 调用 • 代码提取与合并 • 多轮错误反馈', {
    x: 1.2, y: 1.95, w: 7.6, h: 0.5,
    fontSize: 11, color: '555555'
  });

  slide5.addText('↓ 插件注入', { x: 4.5, y: 2.6, w: 1, h: 0.3, fontSize: 14, color: 'B165FB', bold: true, align: 'center' });

  // Layer 2: ALNS Framework
  slide5.addShape(pptx.shapes.RECTANGLE, {
    x: 1, y: 3, w: 8, h: 1, fill: { color: 'F8F9FA' },
    line: { width: 0.5, color: '40695B', pt: 3 }
  });
  slide5.addText('ALNS 框架层 (Framework Layer)', {
    x: 1.2, y: 3.1, w: 7.6, h: 0.3,
    fontSize: 14, bold: true, color: '181B24'
  });
  slide5.addText('heuristic_skeleton.py\n• 破坏/修复阶段 • 轮盘赌选择 + Fallback\n• 模拟退火接受准则 • 自适应权重更新', {
    x: 1.2, y: 3.45, w: 7.6, h: 0.5,
    fontSize: 11, color: '555555'
  });

  slide5.addText('↓ 成本评估', { x: 4.5, y: 4.1, w: 1, h: 0.3, fontSize: 14, color: 'B165FB', bold: true, align: 'center' });

  // Layer 3: Cost Calculation
  slide5.addShape(pptx.shapes.RECTANGLE, {
    x: 1, y: 4.5, w: 8, h: 0.9, fill: { color: 'F8F9FA' },
    line: { width: 0.5, color: '40695B', pt: 3 }
  });
  slide5.addText('成本计算层 (Cost Layer)', {
    x: 1.2, y: 4.6, w: 7.6, h: 0.3,
    fontSize: 14, bold: true, color: '181B24'
  });
  slide5.addText('utils.py - FreshnessAndPenaltyCalculator\n• C₁₁ 固定成本 • C₁₂ 距离成本 • C₁₃ 制冷成本 • C₂ 货损成本 • C₃ 时间窗惩罚', {
    x: 1.2, y: 4.95, w: 7.6, h: 0.4,
    fontSize: 11, color: '555555'
  });

  // Slide 6: Operators
  console.log('[6/15] Adding operators...');
  const slide6 = pptx.addSlide();
  slide6.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: 0.11, h: 5.63, fill: { color: 'B165FB' }, line: { type: 'none' } });
  slide6.addText('03 算法设计 Algorithm Design', {
    x: 0.5, y: 0.5, w: 9, h: 0.7, fontSize: 24, bold: true, color: '181B24'
  });
  slide6.addText('ALNS 算子设计 Operator Design', {
    x: 0.5, y: 1.3, w: 9, h: 0.5, fontSize: 16, bold: true, color: '40695B'
  });

  const tableData = [
    [
      { text: '类型', options: { fill: { color: '181B24' }, color: 'FFFFFF', bold: true } },
      { text: '算子名称', options: { fill: { color: '181B24' }, color: 'FFFFFF', bold: true } },
      { text: '策略描述', options: { fill: { color: '181B24' }, color: 'FFFFFF', bold: true } }
    ],
    ['破坏', 'random_removal', '随机选择 k 个客户节点移除'],
    ['破坏', 'route_removal', '随机选择整条路径移除'],
    ['破坏', 'string_removal', '移除路径中连续的节点段'],
    ['修复', 'greedy_insert', '插入成本增量最小的位置'],
    ['修复', 'regret_insert', '优先插入后悔值最大的节点']
  ];

  slide6.addTable(tableData, {
    x: 0.7, y: 2, w: 8.6, h: 2.8,
    colW: [1.2, 2.4, 5],
    border: { pt: 1, color: 'DDDDDD' },
    fontSize: 12,
    align: 'left',
    valign: 'middle'
  });

  // Slide 7: Cost Model
  console.log('[7/15] Adding cost model...');
  const slide7 = pptx.addSlide();
  slide7.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: 0.11, h: 5.63, fill: { color: 'B165FB' }, line: { type: 'none' } });
  slide7.addText('04 成本模型 Cost Model', {
    x: 0.5, y: 0.5, w: 9, h: 0.7, fontSize: 24, bold: true, color: '181B24'
  });
  
  slide7.addShape(pptx.shapes.RECTANGLE, {
    x: 2, y: 1.3, w: 6, h: 0.7, fill: { color: '181B24' }, line: { type: 'none' }
  });
  slide7.addText('C_total = C₁₁ + C₁₂ + C₁₃ + C₂ + C₃', {
    x: 2, y: 1.3, w: 6, h: 0.7,
    fontSize: 18, bold: true, color: 'FFFFFF', align: 'center', valign: 'middle',
    fontFace: 'Times New Roman'
  });

  const costItems = [
    { label: 'C₁₁ 车辆固定成本', desc: 'K × 240 元/车' },
    { label: 'C₁₂ 运输距离成本', desc: '总距离 × 3 元/km' },
    { label: 'C₁₃ 冷链制冷成本', desc: '行驶时间 × 15 元/h' },
    { label: 'C₂ 货损成本', desc: '新鲜度衰减模型\nr_i = 1 - exp(-θ₁Δt - θ₂t\')' },
    { label: 'C₃ 时间窗惩罚', desc: '早到 20 元/h\n迟到 40 元/h\n硬违反 300 元' }
  ];

  costItems.forEach((item, idx) => {
    const row = Math.floor(idx / 3);
    const col = idx % 3;
    const x = 0.6 + col * 3.1;
    const y = 2.3 + row * 1.3;
    const w = idx === 3 || idx === 4 ? 4.5 : 2.8;

    slide7.addShape(pptx.shapes.RECTANGLE, {
      x, y, w, h: 1.1, fill: { color: 'F8F9FA' },
      line: { width: 0.5, color: '40695B', pt: 3 }
    });
    slide7.addText(item.label, {
      x: x + 0.15, y: y + 0.1, w: w - 0.3, h: 0.3,
      fontSize: 13, bold: true, color: 'B165FB'
    });
    slide7.addText(item.desc, {
      x: x + 0.15, y: y + 0.45, w: w - 0.3, h: 0.6,
      fontSize: 10, color: '555555'
    });
  });

  // Slide 8: Prompt Engineering
  console.log('[8/15] Adding prompt engineering...');
  const slide8 = pptx.addSlide();
  slide8.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: 0.11, h: 5.63, fill: { color: 'B165FB' }, line: { type: 'none' } });
  slide8.addText('提示工程 Prompt Engineering', {
    x: 0.5, y: 0.5, w: 9, h: 0.7, fontSize: 24, bold: true, color: '181B24'
  });
  slide8.addText('六层提示词结构设计', {
    x: 0.5, y: 1.3, w: 9, h: 0.4, fontSize: 16, bold: true, color: '40695B'
  });

  const promptLayers = [
    'Layer 1  角色定义 - ALNS 算法 Python 工程师',
    'Layer 2  成本计算强制要求 - 必须使用 calculate_route_cost()',
    'Layer 3  算子配置 - 恰好 5 个算子（3 破坏 + 2 修复）',
    'Layer 4  数据结构与接口说明 - solution, dist_matrix, capacity',
    'Layer 5  各算子实现指南 - 功能 → 步骤 → 关键代码',
    'Layer 6  检查清单 - 5 个算子 ✓ 完整成本 ✓ 边界处理 ✓'
  ];

  promptLayers.forEach((layer, idx) => {
    const y = 1.85 + idx * 0.45;
    slide8.addShape(pptx.shapes.RECTANGLE, {
      x: 0.7, y, w: 8.6, h: 0.38, fill: { color: 'F8F9FA' },
      line: { width: 0.5, color: 'B165FB', pt: 2 }
    });
    slide8.addText(layer, {
      x: 0.85, y: y + 0.05, w: 8.3, h: 0.28,
      fontSize: 11, color: '333333'
    });
  });

  slide8.addText('关键设计决策', {
    x: 0.5, y: 4.7, w: 9, h: 0.3, fontSize: 14, bold: true, color: '40695B'
  });
  slide8.addText([
    { text: '• ', options: { bullet: true } },
    { text: '正面引导：仅列出需要实现的算子，避免负面提及\n' },
    { text: '• ', options: { bullet: true } },
    { text: '成本计算强制：三次强调必须使用完整成本函数\n' },
    { text: '• ', options: { bullet: true } },
    { text: '统一 __init__：框架层统一注入，消除属性不匹配风险' }
  ], {
    x: 0.7, y: 5.05, w: 8.6, h: 0.5, fontSize: 11, color: '555555'
  });

  // Slide 9: Development Journey
  console.log('[9/15] Adding development journey...');
  const slide9 = pptx.addSlide();
  slide9.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: 0.11, h: 5.63, fill: { color: 'B165FB' }, line: { type: 'none' } });
  slide9.addText('05 开发历程 Development Journey', {
    x: 0.5, y: 0.5, w: 9, h: 0.7, fontSize: 24, bold: true, color: '181B24'
  });

  const phases = [
    { title: '阶段 1: 框架搭建', count: '4', problems: '__init__ 属性不匹配 | LLM 生成错误算子 | 简化距离计算 | 代码执行超时' },
    { title: '阶段 2: 参数调优', count: '3', problems: '随机种子固定 | 迭代次数不足 | 可视化风格不当' },
    { title: '阶段 3: 结构优化', count: '4', problems: '多次运行冗余 | 进度条失效 | 图表空白 | 统计数据未输出' },
    { title: '阶段 4: 性能提升', count: '2', problems: 'route_removal 算子饥饿 | 图表标注重叠' },
    { title: '阶段 5: 工程完善', count: '2', problems: 'Notebook 代码重复 | 文件编辑失败' },
    { title: '阶段 6: 收敛曲线优化', count: '4', problems: '温度参数过激 | 破坏比例过大 | 固定阈值不缩放 | 缺成本上限保护' },
    { title: '阶段 7: 可视化增强', count: '3', problems: '多运行数据混合 | 缺少平滑处理 | Current Cost 语义混淆' }
  ];

  phases.forEach((phase, idx) => {
    const y = 1.4 + idx * 0.57;
    slide9.addShape(pptx.shapes.RECTANGLE, {
      x: 0.85, y, w: 0.05, h: 0.5,
      fill: { color: idx === 0 ? 'B165FB' : 'DDDDDD' }, line: { type: 'none' }
    });
    
    slide9.addShape(pptx.shapes.RECTANGLE, {
      x: 1.05, y: y + 0.05, w: 0.45, h: 0.3,
      fill: { color: 'F0F4FF' }, line: { type: 'none' }
    });
    slide9.addText(phase.count, {
      x: 1.05, y: y + 0.05, w: 0.45, h: 0.3,
      fontSize: 11, bold: true, color: 'B165FB', align: 'center', valign: 'middle'
    });

    slide9.addText(phase.title, {
      x: 1.6, y: y + 0.05, w: 7.7, h: 0.25,
      fontSize: 13, bold: true, color: '181B24'
    });
    slide9.addText(phase.problems, {
      x: 1.6, y: y + 0.32, w: 7.7, h: 0.2,
      fontSize: 9, color: '555555'
    });
  });

  // Slide 10: Problem Classification
  console.log('[10/15] Adding problem classification...');
  const slide10 = pptx.addSlide();
  slide10.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: 0.11, h: 5.63, fill: { color: 'B165FB' }, line: { type: 'none' } });
  slide10.addText('问题分类总览 Problem Classification', {
    x: 0.5, y: 0.5, w: 9, h: 0.7, fontSize: 24, bold: true, color: '181B24'
  });

  slide10.addShape(pptx.shapes.RECTANGLE, {
    x: 2.5, y: 1.4, w: 5, h: 0.9, fill: { color: '181B24' }, line: { type: 'none' }
  });
  slide10.addText('24', {
    x: 2.5, y: 1.5, w: 5, h: 0.5,
    fontSize: 42, bold: true, color: 'B165FB', align: 'center'
  });
  slide10.addText('Total Problems Identified & Solved', {
    x: 2.5, y: 2.05, w: 5, h: 0.25,
    fontSize: 12, color: 'FFFFFF', align: 'center'
  });

  const categories = [
    { count: '4', title: 'A. LLM 代码生成', desc: 'LLM 理解偏差 vs 框架严格要求' },
    { count: '9', title: 'B. 算法参数与机制', desc: '算法配置 vs 求解性能' },
    { count: '8', title: 'C. 可视化与交互', desc: '数据展示 vs 学术规范' },
    { count: '2', title: 'D. 工程与维护', desc: '代码可维护性 vs 开发效率' },
    { count: '1', title: 'E. 性能诊断', desc: '局部最优 vs 全局搜索能力' }
  ];

  categories.forEach((cat, idx) => {
    const col = idx % 3;
    const row = Math.floor(idx / 3);
    const x = 0.7 + col * 3;
    const y = 2.6 + row * 1.5;
    const w = idx >= 3 ? 4 : 2.7;

    slide10.addShape(pptx.shapes.RECTANGLE, {
      x, y, w, h: 1.2, fill: { color: 'F8F9FA' },
      line: { width: 0.5, color: '40695B', pt: 3, dashType: 'dash' }
    });
    slide10.addText(cat.count, {
      x: x + 0.15, y: y + 0.1, w: w - 0.3, h: 0.4,
      fontSize: 28, bold: true, color: 'B165FB'
    });
    slide10.addText(cat.title, {
      x: x + 0.15, y: y + 0.53, w: w - 0.3, h: 0.25,
      fontSize: 13, bold: true, color: '181B24'
    });
    slide10.addText(cat.desc, {
      x: x + 0.15, y: y + 0.82, w: w - 0.3, h: 0.3,
      fontSize: 10, color: '555555'
    });
  });

  // Slide 11: Results
  console.log('[11/15] Adding experimental results...');
  const slide11 = pptx.addSlide();
  slide11.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: 0.11, h: 5.63, fill: { color: 'B165FB' }, line: { type: 'none' } });
  slide11.addText('06 实验结果 Experimental Results', {
    x: 0.5, y: 0.5, w: 9, h: 0.7, fontSize: 24, bold: true, color: '181B24'
  });

  slide11.addShape(pptx.shapes.RECTANGLE, {
    x: 2, y: 1.4, w: 6, h: 1.3, fill: { color: '181B24' }, line: { type: 'none' }
  });
  slide11.addText('42,023.80', {
    x: 2, y: 1.55, w: 6, h: 0.7,
    fontSize: 48, bold: true, color: 'B165FB', align: 'center'
  });
  slide11.addText('最优成本 Optimal Cost (元)', {
    x: 2, y: 2.3, w: 6, h: 0.3,
    fontSize: 14, color: 'FFFFFF', align: 'center'
  });

  const results = [
    { value: '100', label: '客户数量', sublabel: 'Solomon C105' },
    { value: '34', label: '路径数量', sublabel: 'Vehicles Used' },
    { value: '1000', label: '总迭代次数', sublabel: 'Total Iterations' },
    { value: '−1.8%', label: '成本改进', sublabel: 'vs V1 (42,775.07)' }
  ];

  results.forEach((item, idx) => {
    const x = 1.2 + idx * 2;
    const y = 3;

    slide11.addShape(pptx.shapes.RECTANGLE, {
      x, y, w: 1.8, h: 1.6, fill: { color: 'F8F9FA' },
      line: { width: 0.5, color: '40695B', pt: 3 }
    });
    slide11.addText(item.value, {
      x, y: y + 0.2, w: 1.8, h: 0.6,
      fontSize: 36, bold: true, color: 'B165FB', align: 'center'
    });
    slide11.addText(item.label, {
      x, y: y + 0.85, w: 1.8, h: 0.3,
      fontSize: 12, bold: true, color: '181B24', align: 'center'
    });
    slide11.addText(item.sublabel, {
      x, y: y + 1.18, w: 1.8, h: 0.25,
      fontSize: 9, color: '666666', align: 'center'
    });
  });

  // Slide 12: Operator Performance (with charts)
  console.log('[12/15] Adding operator performance...');
  const slide12 = pptx.addSlide();
  slide12.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: 0.11, h: 5.63, fill: { color: 'B165FB' }, line: { type: 'none' } });
  slide12.addText('算子性能分析 Operator Performance', {
    x: 0.5, y: 0.4, w: 9, h: 0.6, fontSize: 24, bold: true, color: '181B24'
  });

  slide12.addText('破坏算子 Destroy Operators', {
    x: 0.6, y: 1.1, w: 4.4, h: 0.4, fontSize: 16, bold: true, color: '40695B'
  });

  const destroyData = [
    {
      name: 'Destroy Operators',
      labels: ['random_removal', 'route_removal', 'string_removal'],
      values: [632, 60, 317]
    }
  ];

  slide12.addChart(pptx.charts.BAR, destroyData, {
    x: 0.5, y: 1.6, w: 4.4, h: 3.5,
    barDir: 'col',
    showTitle: false,
    showLegend: false,
    chartColors: ['B165FB', '40695B', '181B24'],
    catAxisLabelFontSize: 10,
    valAxisLabelFontSize: 10,
    showCatAxisTitle: true,
    catAxisTitle: '算子名称',
    showValAxisTitle: true,
    valAxisTitle: '使用次数',
    dataLabelPosition: 'outEnd',
    dataLabelColor: '333333',
    dataLabelFontSize: 11
  });

  slide12.addText('修复算子 Repair Operators', {
    x: 5.5, y: 1.1, w: 4, h: 0.4, fontSize: 16, bold: true, color: '40695B'
  });

  const repairData = [
    {
      name: 'Repair Operators',
      labels: ['greedy_insert', 'regret_insert'],
      values: [237, 763]
    }
  ];

  slide12.addChart(pptx.charts.BAR, repairData, {
    x: 5.5, y: 1.6, w: 4, h: 3.5,
    barDir: 'col',
    showTitle: false,
    showLegend: false,
    chartColors: ['B165FB', '40695B'],
    catAxisLabelFontSize: 10,
    valAxisLabelFontSize: 10,
    showCatAxisTitle: true,
    catAxisTitle: '算子名称',
    showValAxisTitle: true,
    valAxisTitle: '使用次数',
    dataLabelPosition: 'outEnd',
    dataLabelColor: '333333',
    dataLabelFontSize: 11
  });

  // Slide 13: Conclusions
  console.log('[13/15] Adding conclusions...');
  const slide13 = pptx.addSlide();
  slide13.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: 0.11, h: 5.63, fill: { color: 'B165FB' }, line: { type: 'none' } });
  slide13.addText('07 主要结论 Main Conclusions', {
    x: 0.5, y: 0.5, w: 9, h: 0.7, fontSize: 24, bold: true, color: '181B24'
  });

  const conclusions = [
    { num: '1', title: 'LLM 可有效生成 ALNS 算子：', text: 'DeepSeek R1 在提示工程引导下，能够生成功能正确的破坏和修复算子代码，在 Solomon C105 基准上取得 42,023.80 的目标函数值' },
    { num: '2', title: '插件式架构保证鲁棒性：', text: 'Skeleton + Plugin 的解耦设计，配合 Fallback 机制，即使 LLM 生成的部分算子存在缺陷，系统仍能正常运行' },
    { num: '3', title: '自适应机制的重要性：', text: '权重下限保护、自适应破坏比例、重启再加热等机制显著影响算法性能，防止过早收敛和算子饥饿' },
    { num: '4', title: '单次运行即可：', text: '内置的三重随机化机制使单次 solve() 已具备足够的搜索能力，多次运行的边际收益极低' }
  ];

  conclusions.forEach((item, idx) => {
    const y = 1.35 + idx * 1;
    slide13.addShape(pptx.shapes.RECTANGLE, {
      x: 0.6, y, w: 8.8, h: 0.85, fill: { color: 'F0F4FF' },
      line: { width: 0.5, color: 'B165FB', pt: 3 }
    });

    slide13.addShape(pptx.shapes.OVAL, {
      x: 0.8, y: y + 0.08, w: 0.35, h: 0.35, fill: { color: 'B165FB' }, line: { type: 'none' }
    });
    slide13.addText(item.num, {
      x: 0.8, y: y + 0.08, w: 0.35, h: 0.35,
      fontSize: 14, bold: true, color: 'FFFFFF', align: 'center', valign: 'middle'
    });

    slide13.addText([
      { text: item.title, options: { bold: true } },
      { text: item.text }
    ], {
      x: 1.25, y: y + 0.15, w: 7.9, h: 0.65,
      fontSize: 11, color: '333333'
    });
  });

  // Slide 14: Future Work
  console.log('[14/15] Adding future work...');
  const slide14 = pptx.addSlide();
  slide14.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: 0.11, h: 5.63, fill: { color: 'B165FB' }, line: { type: 'none' } });
  slide14.addText('局限性与未来工作', {
    x: 0.5, y: 0.5, w: 9, h: 0.7, fontSize: 24, bold: true, color: '181B24'
  });

  slide14.addText('局限性 Limitations', {
    x: 0.6, y: 1.3, w: 4.3, h: 0.4, fontSize: 16, bold: true, color: '40695B'
  });
  slide14.addText([
    { text: '• ', options: { bullet: true } },
    { text: '问题规模：', options: { bold: true } },
    { text: '当前仅在 100 客户规模上测试\n' },
    { text: '• ', options: { bullet: true } },
    { text: 'LLM 依赖：', options: { bold: true } },
    { text: '算子质量取决于 LLM 能力\n' },
    { text: '• ', options: { bullet: true } },
    { text: '成本函数固定：', options: { bold: true } },
    { text: '扩展到其他 VRP 变体需重新设计\n' },
    { text: '• ', options: { bullet: true } },
    { text: '缺少对比基线：', options: { bold: true } },
    { text: '未与手工 ALNS 系统对比' }
  ], {
    x: 0.8, y: 1.8, w: 4.1, h: 3.5, fontSize: 11, color: '555555'
  });

  slide14.addText('未来工作 Future Work', {
    x: 5.1, y: 1.3, w: 4.3, h: 0.4, fontSize: 16, bold: true, color: '40695B'
  });
  slide14.addText([
    { text: '• ', options: { bullet: true } },
    { text: '多 LLM 对比：', options: { bold: true } },
    { text: 'GPT-4o、Claude、Gemini 等模型\n' },
    { text: '• ', options: { bullet: true } },
    { text: '自适应提示：', options: { bold: true } },
    { text: '根据执行结果调整提示词\n' },
    { text: '• ', options: { bullet: true } },
    { text: '大规模测试：', options: { bold: true } },
    { text: 'Solomon R/RC 类和 G&H 大规模实例\n' },
    { text: '• ', options: { bullet: true } },
    { text: '算子进化：', options: { bold: true } },
    { text: '让 LLM 根据统计数据改进低效算子' }
  ], {
    x: 5.3, y: 1.8, w: 4.1, h: 3.5, fontSize: 11, color: '555555'
  });

  // Slide 15: Thank You
  console.log('[15/15] Adding thank you slide...');
  const slide15 = pptx.addSlide();
  slide15.background = { color: '181B24' };
  
  slide15.addText('Thank You!', {
    x: 1, y: 1.5, w: 8, h: 1,
    fontSize: 48, bold: true, color: 'FFFFFF', align: 'center'
  });
  
  slide15.addShape(pptx.shapes.RECTANGLE, {
    x: 3.5, y: 2.6, w: 3, h: 0.04, fill: { color: 'B165FB' }, line: { type: 'none' }
  });
  
  slide15.addText('OR-LLM-Agent ALNS 模块开发报告', {
    x: 1, y: 2.9, w: 8, h: 0.4,
    fontSize: 16, color: 'E0E0E0', align: 'center'
  });
  slide15.addText('基于 OR-LLM-Agent (arXiv:2503.10009)', {
    x: 1, y: 3.35, w: 8, h: 0.3,
    fontSize: 14, color: 'E0E0E0', align: 'center'
  });
  
  slide15.addText('上海交通大学 & 南洋理工大学\nShanghai Jiao Tong University & Nanyang Technological University', {
    x: 1, y: 4.2, w: 8, h: 0.6,
    fontSize: 11, color: 'BBBBBB', align: 'center'
  });

  // Save presentation
  const outputPath = 'D:/pythonProject/or_llm_agent/ALNS_VRP_项目报告.pptx';
  console.log('\nSaving presentation...');
  await pptx.writeFile({ fileName: outputPath });
  
  console.log(`\n✅ Presentation created successfully!`);
  console.log(`📁 Location: ${outputPath}`);
  console.log(`📊 Total slides: 15`);
}

createPresentation().catch(err => {
  console.error('❌ Error creating presentation:', err);
  process.exit(1);
});