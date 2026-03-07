const pptxgen = require('pptxgenjs');

async function createPresentation() {
  const pptx = new pptxgen();
  pptx.layout = 'LAYOUT_16x9';
  pptx.author = 'Zhang & Luo | OR-LLM-Agent';
  pptx.title = '基于LLM的ALNS算法求解VRP问题';
  pptx.subject = 'ALNS Algorithm with LLM-Generated Operators';

  // 新配色方案
  const colors = {
    mainBlue: '1E3A8A',      // 主蓝色（深蓝）
    accentBlue: '3B82F6',    // 强调蓝色（亮蓝）
    highlightRed: 'DC2626',  // 重点红色
    textBlack: '1F2937',     // 正文黑色
    lightGray: 'F3F4F6',     // 浅灰背景
    white: 'FFFFFF'
  };

  console.log('Creating presentation with updated design...\n');

  // Slide 1: 封面页
  console.log('[1/15] Adding title slide...');
  const slide1 = pptx.addSlide();
  slide1.addText('基于大语言模型自动生成启发式算子的\nALNS算法求解生鲜物流VRP问题', {
    x: 0.5, y: 1.2, w: 9, h: 2,
    fontSize: 28, bold: true, color: colors.textBlack, align: 'center', valign: 'middle',
    fontFace: 'Microsoft YaHei'
  });
  slide1.addShape(pptx.shapes.RECTANGLE, {
    x: 3, y: 3.3, w: 4, h: 0.06, fill: { color: colors.mainBlue }, line: { type: 'none' }
  });
  slide1.addText('OR-LLM-Agent ALNS模块开发报告', {
    x: 0.5, y: 3.6, w: 9, h: 0.5,
    fontSize: 22, color: colors.accentBlue, align: 'center', fontFace: 'Microsoft YaHei'
  });
  slide1.addText([
    { text: '基于 OR-LLM-Agent (arXiv:2503.10009)\n', options: { fontSize: 20, color: '666666', fontFace: 'Arial' } },
    { text: 'Zhang & Luo | 上海交通大学 & 南洋理工大学\n', options: { fontSize: 20, color: '666666', fontFace: 'Microsoft YaHei' } },
    { text: 'DeepSeek R1 | Solomon C105 基准测试', options: { fontSize: 18, color: '999999', fontFace: 'Arial' } }
  ], {
    x: 0.5, y: 4.3, w: 9, h: 1, align: 'center', valign: 'top'
  });

  // Slide 2: 目录
  console.log('[2/15] Adding table of contents...');
  const slide2 = pptx.addSlide();
  slide2.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: 0.15, h: 5.63, fill: { color: colors.mainBlue }, line: { type: 'none' } });
  slide2.addText('目录 Contents', { 
    x: 0.5, y: 0.5, w: 9, h: 0.8, 
    fontSize: 28, bold: true, color: colors.textBlack, fontFace: 'Microsoft YaHei' 
  });
  
  const tocItems = [
    { num: '01', text: '项目概述 Project Overview' },
    { num: '02', text: '系统架构 System Architecture' },
    { num: '03', text: '算法设计 Algorithm Design' },
    { num: '04', text: '成本模型 Cost Model' },
    { num: '05', text: '开发历程 Development Journey' },
    { num: '06', text: '实验结果 Experimental Results' },
    { num: '07', text: '总结展望 Conclusion & Future Work' }
  ];
  
  let yPos = 1.6;
  tocItems.forEach((item) => {
    slide2.addText([
      { text: item.num, options: { fontSize: 22, bold: true, color: colors.mainBlue, fontFace: 'Arial' } },
      { text: '  ' + item.text, options: { fontSize: 20, color: colors.textBlack, fontFace: 'Microsoft YaHei' } }
    ], {
      x: 0.8, y: yPos, w: 8.5, h: 0.5, valign: 'middle'
    });
    yPos += 0.55;
  });

  // Slide 3: 项目概述
  console.log('[3/15] Adding overview...');
  const slide3 = pptx.addSlide();
  slide3.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: 0.15, h: 5.63, fill: { color: colors.mainBlue }, line: { type: 'none' } });
  slide3.addText('01 项目概述 Project Overview', {
    x: 0.5, y: 0.5, w: 9, h: 0.7, 
    fontSize: 28, bold: true, color: colors.textBlack, fontFace: 'Microsoft YaHei'
  });
  
  slide3.addText('研究背景 Background', {
    x: 0.5, y: 1.4, w: 9, h: 0.5, 
    fontSize: 22, bold: true, color: colors.accentBlue, fontFace: 'Microsoft YaHei'
  });
  slide3.addText([
    { text: '• 车辆路径问题 (', options: { fontSize: 20, fontFace: 'Microsoft YaHei' } },
    { text: 'VRP', options: { fontSize: 20, bold: true, color: colors.highlightRed, fontFace: 'Arial' } },
    { text: ') 是经典的 ', options: { fontSize: 20, fontFace: 'Microsoft YaHei' } },
    { text: 'NP-hard', options: { fontSize: 20, bold: true, color: colors.highlightRed, fontFace: 'Arial' } },
    { text: ' 问题\n', options: { fontSize: 20, fontFace: 'Microsoft YaHei' } },
    { text: '• ', options: { fontSize: 20 } },
    { text: '生鲜物流', options: { fontSize: 20, bold: true, color: colors.accentBlue, fontFace: 'Microsoft YaHei' } },
    { text: '场景：新鲜度衰减 + 冷链成本 + 时间窗约束\n', options: { fontSize: 20, fontFace: 'Microsoft YaHei' } },
    { text: '• 传统 ', options: { fontSize: 20, fontFace: 'Microsoft YaHei' } },
    { text: 'ALNS', options: { fontSize: 20, bold: true, color: colors.highlightRed, fontFace: 'Arial' } },
    { text: ' 依赖人工设计算子，开发周期长', options: { fontSize: 20, fontFace: 'Microsoft YaHei' } }
  ], {
    x: 0.8, y: 2, w: 8.8, h: 1.3, color: colors.textBlack
  });
  
  slide3.addText('研究目标 Objectives', {
    x: 0.5, y: 3.4, w: 9, h: 0.5, 
    fontSize: 22, bold: true, color: colors.accentBlue, fontFace: 'Microsoft YaHei'
  });
  slide3.addText([
    { text: '• 利用 ', options: { fontSize: 20, fontFace: 'Microsoft YaHei' } },
    { text: 'LLM (DeepSeek R1)', options: { fontSize: 20, bold: true, color: colors.accentBlue, fontFace: 'Arial' } },
    { text: ' 自动生成 ALNS 算子代码\n', options: { fontSize: 20, fontFace: 'Microsoft YaHei' } },
    { text: '• 实现"', options: { fontSize: 20, fontFace: 'Microsoft YaHei' } },
    { text: 'LLM 生成算子 + 框架执行求解', options: { fontSize: 20, bold: true, color: colors.highlightRed, fontFace: 'Microsoft YaHei' } },
    { text: '"的混合架构\n', options: { fontSize: 20, fontFace: 'Microsoft YaHei' } },
    { text: '• 在 Solomon 基准上验证算法性能', options: { fontSize: 20, fontFace: 'Microsoft YaHei' } }
  ], {
    x: 0.8, y: 4, w: 8.8, h: 1.3, color: colors.textBlack
  });

  // Slide 4: 核心创新
  console.log('[4/15] Adding core innovations...');
  const slide4 = pptx.addSlide();
  slide4.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: 0.15, h: 5.63, fill: { color: colors.mainBlue }, line: { type: 'none' } });
  slide4.addText('核心创新 Core Innovations', {
    x: 0.5, y: 0.5, w: 9, h: 0.7, 
    fontSize: 28, bold: true, color: colors.textBlack, fontFace: 'Microsoft YaHei'
  });

  const innovations = [
    { num: '01', title: '插件式架构', eng: 'Plugin Architecture', desc: '将 ALNS 框架与算子实现解耦，LLM 代码作为插件注入' },
    { num: '02', title: '自动调试机制', eng: 'Auto-Debugging', desc: '多轮对话反馈执行错误，LLM 自主修正代码直至成功' },
    { num: '03', title: '完整成本模型', eng: 'Cost Model', desc: '融合固定、距离、制冷、货损、时间窗惩罚的五维成本' },
    { num: '04', title: '自适应权重', eng: 'Adaptive Weights', desc: '引入权重下限保护机制，防止算子饥饿' }
  ];

  innovations.forEach((item, idx) => {
    const row = Math.floor(idx / 2);
    const col = idx % 2;
    const x = 0.6 + col * 4.7;
    const y = 1.5 + row * 2;

    slide4.addShape(pptx.shapes.RECTANGLE, {
      x, y, w: 4.3, h: 1.6, fill: { color: colors.lightGray },
      line: { width: 1, color: colors.mainBlue }
    });
    slide4.addText(item.num, {
      x: x + 0.2, y: y + 0.15, w: 1, h: 0.45,
      fontSize: 28, bold: true, color: colors.highlightRed, fontFace: 'Arial'
    });
    slide4.addText([
      { text: item.title, options: { fontSize: 22, bold: true, color: colors.textBlack, fontFace: 'Microsoft YaHei' } },
      { text: ' ' + item.eng, options: { fontSize: 18, color: colors.accentBlue, fontFace: 'Arial' } }
    ], {
      x: x + 0.2, y: y + 0.65, w: 3.9, h: 0.4
    });
    slide4.addText(item.desc, {
      x: x + 0.2, y: y + 1.1, w: 3.9, h: 0.4,
      fontSize: 18, color: colors.textBlack, fontFace: 'Microsoft YaHei'
    });
  });

  // Slide 5: 系统架构
  console.log('[5/15] Adding system architecture...');
  const slide5 = pptx.addSlide();
  slide5.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: 0.15, h: 5.63, fill: { color: colors.mainBlue }, line: { type: 'none' } });
  slide5.addText('02 系统架构 System Architecture', {
    x: 0.5, y: 0.5, w: 9, h: 0.7, 
    fontSize: 28, bold: true, color: colors.textBlack, fontFace: 'Microsoft YaHei'
  });

  // Layer 1: LLM 交互层
  slide5.addShape(pptx.shapes.RECTANGLE, {
    x: 1, y: 1.4, w: 8, h: 1.1, fill: { color: colors.lightGray },
    line: { width: 1, color: colors.mainBlue }
  });
  slide5.addText([
    { text: 'LLM 交互层 ', options: { fontSize: 22, bold: true, color: colors.highlightRed, fontFace: 'Microsoft YaHei' } },
    { text: '(Interaction Layer)', options: { fontSize: 20, color: colors.accentBlue, fontFace: 'Arial' } }
  ], {
    x: 1.2, y: 1.55, w: 7.6, h: 0.35
  });
  slide5.addText('llm_heuristic.py + heuristic_prompts.py\n• LLM API 调用 • 代码提取与合并 • 多轮错误反馈', {
    x: 1.2, y: 1.95, w: 7.6, h: 0.5,
    fontSize: 18, color: colors.textBlack, fontFace: 'Microsoft YaHei'
  });

  slide5.addText('↓ 插件注入 Plugin Pattern', { 
    x: 3.5, y: 2.6, w: 3, h: 0.3, 
    fontSize: 20, color: colors.mainBlue, bold: true, align: 'center', fontFace: 'Arial' 
  });

  // Layer 2: ALNS 框架层
  slide5.addShape(pptx.shapes.RECTANGLE, {
    x: 1, y: 3, w: 8, h: 1.1, fill: { color: colors.lightGray },
    line: { width: 1, color: colors.mainBlue }
  });
  slide5.addText([
    { text: 'ALNS 框架层 ', options: { fontSize: 22, bold: true, color: colors.highlightRed, fontFace: 'Arial' } },
    { text: '(Framework Layer)', options: { fontSize: 20, color: colors.accentBlue, fontFace: 'Arial' } }
  ], {
    x: 1.2, y: 3.15, w: 7.6, h: 0.35
  });
  slide5.addText('heuristic_skeleton.py\n• 破坏/修复阶段 • 轮盘赌选择+Fallback • 模拟退火 • 自适应权重', {
    x: 1.2, y: 3.55, w: 7.6, h: 0.5,
    fontSize: 18, color: colors.textBlack, fontFace: 'Microsoft YaHei'
  });

  slide5.addText('↓ 成本评估 Cost Evaluation', { 
    x: 3.5, y: 4.2, w: 3, h: 0.3, 
    fontSize: 20, color: colors.mainBlue, bold: true, align: 'center', fontFace: 'Arial' 
  });

  // Layer 3: 成本计算层
  slide5.addShape(pptx.shapes.RECTANGLE, {
    x: 1, y: 4.6, w: 8, h: 0.95, fill: { color: colors.lightGray },
    line: { width: 1, color: colors.mainBlue }
  });
  slide5.addText([
    { text: '成本计算层 ', options: { fontSize: 22, bold: true, color: colors.highlightRed, fontFace: 'Microsoft YaHei' } },
    { text: '(Cost Layer)', options: { fontSize: 20, color: colors.accentBlue, fontFace: 'Arial' } }
  ], {
    x: 1.2, y: 4.75, w: 7.6, h: 0.35
  });
  slide5.addText('utils.py - FreshnessAndPenaltyCalculator\n• C₁₁ 固定成本 • C₁₂ 距离成本 • C₁₃ 制冷成本 • C₂ 货损成本 • C₃ 时间窗惩罚', {
    x: 1.2, y: 5.15, w: 7.6, h: 0.35,
    fontSize: 18, color: colors.textBlack, fontFace: 'Microsoft YaHei'
  });

  // Slide 6: 算子设计
  console.log('[6/15] Adding operators...');
  const slide6 = pptx.addSlide();
  slide6.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: 0.15, h: 5.63, fill: { color: colors.mainBlue }, line: { type: 'none' } });
  slide6.addText('03 算法设计 Algorithm Design', {
    x: 0.5, y: 0.5, w: 9, h: 0.7, 
    fontSize: 28, bold: true, color: colors.textBlack, fontFace: 'Microsoft YaHei'
  });
  slide6.addText('ALNS 算子设计 Operator Design', {
    x: 0.5, y: 1.3, w: 9, h: 0.5, 
    fontSize: 22, bold: true, color: colors.accentBlue, fontFace: 'Microsoft YaHei'
  });

  const tableData = [
    [
      { text: '类型', options: { fill: { color: colors.mainBlue }, color: colors.white, bold: true, fontSize: 20, fontFace: 'Microsoft YaHei' } },
      { text: '算子名称', options: { fill: { color: colors.mainBlue }, color: colors.white, bold: true, fontSize: 20, fontFace: 'Microsoft YaHei' } },
      { text: '策略描述', options: { fill: { color: colors.mainBlue }, color: colors.white, bold: true, fontSize: 20, fontFace: 'Microsoft YaHei' } }
    ],
    [
      { text: '破坏', options: { fontSize: 20, bold: true, color: colors.highlightRed, fontFace: 'Microsoft YaHei' } },
      { text: 'random_removal', options: { fontSize: 18, fontFace: 'Arial' } },
      { text: '随机选择 k 个客户节点移除', options: { fontSize: 18, fontFace: 'Microsoft YaHei' } }
    ],
    [
      { text: '破坏', options: { fontSize: 20, bold: true, color: colors.highlightRed, fontFace: 'Microsoft YaHei' } },
      { text: 'route_removal', options: { fontSize: 18, fontFace: 'Arial' } },
      { text: '随机选择整条路径移除', options: { fontSize: 18, fontFace: 'Microsoft YaHei' } }
    ],
    [
      { text: '破坏', options: { fontSize: 20, bold: true, color: colors.highlightRed, fontFace: 'Microsoft YaHei' } },
      { text: 'string_removal', options: { fontSize: 18, fontFace: 'Arial' } },
      { text: '移除路径中连续的节点段', options: { fontSize: 18, fontFace: 'Microsoft YaHei' } }
    ],
    [
      { text: '修复', options: { fontSize: 20, bold: true, color: colors.accentBlue, fontFace: 'Microsoft YaHei' } },
      { text: 'greedy_insert', options: { fontSize: 18, fontFace: 'Arial' } },
      { text: '插入成本增量最小的位置', options: { fontSize: 18, fontFace: 'Microsoft YaHei' } }
    ],
    [
      { text: '修复', options: { fontSize: 20, bold: true, color: colors.accentBlue, fontFace: 'Microsoft YaHei' } },
      { text: 'regret_insert', options: { fontSize: 18, fontFace: 'Arial' } },
      { text: '优先插入后悔值最大的节点', options: { fontSize: 18, fontFace: 'Microsoft YaHei' } }
    ]
  ];

  slide6.addTable(tableData, {
    x: 0.7, y: 2, w: 8.6, h: 3,
    colW: [1.4, 2.6, 4.6],
    border: { pt: 1, color: colors.mainBlue },
    align: 'left',
    valign: 'middle'
  });

  // Slide 7: 成本模型
  console.log('[7/15] Adding cost model...');
  const slide7 = pptx.addSlide();
  slide7.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: 0.15, h: 5.63, fill: { color: colors.mainBlue }, line: { type: 'none' } });
  slide7.addText('04 成本模型 Cost Model', {
    x: 0.5, y: 0.5, w: 9, h: 0.7, 
    fontSize: 28, bold: true, color: colors.textBlack, fontFace: 'Microsoft YaHei'
  });
  
  slide7.addShape(pptx.shapes.RECTANGLE, {
    x: 1.5, y: 1.3, w: 7, h: 0.8, fill: { color: colors.mainBlue }, line: { type: 'none' }
  });
  slide7.addText([
    { text: 'C', options: { fontSize: 24, fontFace: 'Times New Roman', italic: true } },
    { text: 'total', options: { fontSize: 18, fontFace: 'Times New Roman', italic: true } },
    { text: ' = ', options: { fontSize: 24, fontFace: 'Times New Roman' } },
    { text: 'C₁₁ + C₁₂ + C₁₃ + C₂ + C₃', options: { fontSize: 24, bold: true, color: colors.highlightRed, fontFace: 'Times New Roman' } }
  ], {
    x: 1.5, y: 1.3, w: 7, h: 0.8,
    color: colors.white, align: 'center', valign: 'middle'
  });

  const costItems = [
    { label: 'C₁₁', title: '车辆固定成本', desc: 'K × 240 元/车' },
    { label: 'C₁₂', title: '运输距离成本', desc: '总距离 × 3 元/km' },
    { label: 'C₁₃', title: '冷链制冷成本', desc: '行驶时间 × 15 元/h' },
    { label: 'C₂', title: '货损成本', desc: '新鲜度衰减模型\nr = 1 - exp(-θ₁Δt - θ₂t\')' },
    { label: 'C₃', title: '时间窗惩罚', desc: '早到 20 元/h\n迟到 40 元/h\n硬违反 300 元' }
  ];

  costItems.forEach((item, idx) => {
    const row = Math.floor(idx / 3);
    const col = idx % 3;
    const x = 0.6 + col * 3.1;
    const y = 2.5 + row * 1.4;
    const w = idx === 3 || idx === 4 ? 4.5 : 2.8;

    slide7.addShape(pptx.shapes.RECTANGLE, {
      x, y, w, h: 1.2, fill: { color: colors.lightGray },
      line: { width: 1, color: colors.accentBlue }
    });
    slide7.addText([
      { text: item.label, options: { fontSize: 22, bold: true, color: colors.highlightRed, fontFace: 'Times New Roman' } },
      { text: ' ' + item.title, options: { fontSize: 20, bold: true, color: colors.textBlack, fontFace: 'Microsoft YaHei' } }
    ], {
      x: x + 0.15, y: y + 0.12, w: w - 0.3, h: 0.35
    });
    slide7.addText(item.desc, {
      x: x + 0.15, y: y + 0.52, w: w - 0.3, h: 0.6,
      fontSize: 18, color: colors.textBlack, fontFace: 'Microsoft YaHei'
    });
  });

  // Slide 8: 提示工程
  console.log('[8/15] Adding prompt engineering...');
  const slide8 = pptx.addSlide();
  slide8.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: 0.15, h: 5.63, fill: { color: colors.mainBlue }, line: { type: 'none' } });
  slide8.addText('提示工程 Prompt Engineering', {
    x: 0.5, y: 0.5, w: 9, h: 0.7, 
    fontSize: 28, bold: true, color: colors.textBlack, fontFace: 'Microsoft YaHei'
  });
  slide8.addText('六层提示词结构设计', {
    x: 0.5, y: 1.3, w: 9, h: 0.4, 
    fontSize: 22, bold: true, color: colors.accentBlue, fontFace: 'Microsoft YaHei'
  });

  const promptLayers = [
    { layer: 'Layer 1', text: '角色定义 - ALNS 算法 Python 工程师' },
    { layer: 'Layer 2', text: '成本计算强制要求 - 必须使用 calculate_route_cost()' },
    { layer: 'Layer 3', text: '算子配置 - 恰好 5 个算子（3 破坏 + 2 修复）' },
    { layer: 'Layer 4', text: '数据结构与接口说明 - solution, dist_matrix, capacity' },
    { layer: 'Layer 5', text: '各算子实现指南 - 功能 → 步骤 → 关键代码' },
    { layer: 'Layer 6', text: '检查清单 - 5 个算子 ✓ 完整成本 ✓ 边界处理 ✓' }
  ];

  promptLayers.forEach((item, idx) => {
    const y = 1.85 + idx * 0.5;
    slide8.addShape(pptx.shapes.RECTANGLE, {
      x: 0.7, y, w: 8.6, h: 0.42, fill: { color: colors.lightGray },
      line: { width: 1, color: colors.accentBlue }
    });
    slide8.addText([
      { text: item.layer, options: { fontSize: 20, bold: true, color: colors.mainBlue, fontFace: 'Arial' } },
      { text: '  ' + item.text, options: { fontSize: 20, color: colors.textBlack, fontFace: 'Microsoft YaHei' } }
    ], {
      x: 0.85, y: y + 0.06, w: 8.3, h: 0.3
    });
  });

  slide8.addText('关键设计决策', {
    x: 0.5, y: 4.95, w: 9, h: 0.35, 
    fontSize: 22, bold: true, color: colors.accentBlue, fontFace: 'Microsoft YaHei'
  });
  slide8.addText([
    { text: '• ', options: { bullet: true, fontSize: 20 } },
    { text: '正面引导：', options: { fontSize: 20, bold: true, color: colors.highlightRed, fontFace: 'Microsoft YaHei' } },
    { text: '仅列出需要实现的算子，避免负面提及\n', options: { fontSize: 20, fontFace: 'Microsoft YaHei' } },
    { text: '• ', options: { bullet: true, fontSize: 20 } },
    { text: '成本计算强制：', options: { fontSize: 20, bold: true, color: colors.highlightRed, fontFace: 'Microsoft YaHei' } },
    { text: '三次强调必须使用完整成本函数\n', options: { fontSize: 20, fontFace: 'Microsoft YaHei' } },
    { text: '• ', options: { bullet: true, fontSize: 20 } },
    { text: '统一 __init__：', options: { fontSize: 20, bold: true, color: colors.highlightRed, fontFace: 'Microsoft YaHei' } },
    { text: '框架层统一注入，消除属性不匹配风险', options: { fontSize: 20, fontFace: 'Microsoft YaHei' } }
  ], {
    x: 0.7, y: 5.35, w: 8.6, h: 0.2, color: colors.textBlack
  });

  // Slide 9: 开发历程
  console.log('[9/15] Adding development journey...');
  const slide9 = pptx.addSlide();
  slide9.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: 0.15, h: 5.63, fill: { color: colors.mainBlue }, line: { type: 'none' } });
  slide9.addText('05 开发历程 Development Journey', {
    x: 0.5, y: 0.5, w: 9, h: 0.7, 
    fontSize: 28, bold: true, color: colors.textBlack, fontFace: 'Microsoft YaHei'
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
    const y = 1.4 + idx * 0.6;
    
    slide9.addShape(pptx.shapes.RECTANGLE, {
      x: 0.85, y: y + 0.02, w: 0.06, h: 0.54,
      fill: { color: idx === 0 ? colors.mainBlue : 'DDDDDD' }, line: { type: 'none' }
    });
    
    slide9.addShape(pptx.shapes.RECTANGLE, {
      x: 1.05, y: y + 0.08, w: 0.5, h: 0.35,
      fill: { color: colors.lightGray }, line: { width: 1, color: colors.accentBlue }
    });
    slide9.addText(phase.count, {
      x: 1.05, y: y + 0.08, w: 0.5, h: 0.35,
      fontSize: 20, bold: true, color: colors.highlightRed, align: 'center', valign: 'middle', fontFace: 'Arial'
    });

    slide9.addText(phase.title, {
      x: 1.7, y: y + 0.08, w: 7.6, h: 0.3,
      fontSize: 20, bold: true, color: colors.textBlack, fontFace: 'Microsoft YaHei'
    });
    slide9.addText(phase.problems, {
      x: 1.7, y: y + 0.4, w: 7.6, h: 0.18,
      fontSize: 16, color: colors.textBlack, fontFace: 'Microsoft YaHei'
    });
  });

  // Slide 10: 问题分类
  console.log('[10/15] Adding problem classification...');
  const slide10 = pptx.addSlide();
  slide10.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: 0.15, h: 5.63, fill: { color: colors.mainBlue }, line: { type: 'none' } });
  slide10.addText('问题分类总览 Problem Classification', {
    x: 0.5, y: 0.5, w: 9, h: 0.7, 
    fontSize: 28, bold: true, color: colors.textBlack, fontFace: 'Microsoft YaHei'
  });

  slide10.addShape(pptx.shapes.RECTANGLE, {
    x: 3, y: 1.3, w: 4, h: 1, fill: { color: colors.mainBlue }, line: { type: 'none' }
  });
  slide10.addText('24', {
    x: 3, y: 1.4, w: 4, h: 0.55,
    fontSize: 48, bold: true, color: colors.highlightRed, align: 'center', fontFace: 'Arial'
  });
  slide10.addText('Total Problems Identified & Solved', {
    x: 3, y: 2, w: 4, h: 0.25,
    fontSize: 18, color: colors.white, align: 'center', fontFace: 'Arial'
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
    const y = 2.6 + row * 1.55;
    const w = idx >= 3 ? 4 : 2.7;

    slide10.addShape(pptx.shapes.RECTANGLE, {
      x, y, w, h: 1.3, fill: { color: colors.lightGray },
      line: { width: 1, color: colors.accentBlue }
    });
    slide10.addText(cat.count, {
      x: x + 0.15, y: y + 0.12, w: w - 0.3, h: 0.45,
      fontSize: 36, bold: true, color: colors.highlightRed, fontFace: 'Arial'
    });
    slide10.addText(cat.title, {
      x: x + 0.15, y: y + 0.6, w: w - 0.3, h: 0.3,
      fontSize: 20, bold: true, color: colors.textBlack, fontFace: 'Microsoft YaHei'
    });
    slide10.addText(cat.desc, {
      x: x + 0.15, y: y + 0.93, w: w - 0.3, h: 0.3,
      fontSize: 18, color: colors.textBlack, fontFace: 'Microsoft YaHei'
    });
  });

  // Slide 11: 实验结果
  console.log('[11/15] Adding experimental results...');
  const slide11 = pptx.addSlide();
  slide11.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: 0.15, h: 5.63, fill: { color: colors.mainBlue }, line: { type: 'none' } });
  slide11.addText('06 实验结果 Experimental Results', {
    x: 0.5, y: 0.5, w: 9, h: 0.7, 
    fontSize: 28, bold: true, color: colors.textBlack, fontFace: 'Microsoft YaHei'
  });

  slide11.addShape(pptx.shapes.RECTANGLE, {
    x: 2, y: 1.3, w: 6, h: 1.4, fill: { color: colors.mainBlue }, line: { type: 'none' }
  });
  slide11.addText('42,023.80', {
    x: 2, y: 1.45, w: 6, h: 0.75,
    fontSize: 56, bold: true, color: colors.highlightRed, align: 'center', fontFace: 'Arial'
  });
  slide11.addText('最优成本 Optimal Cost (元)', {
    x: 2, y: 2.25, w: 6, h: 0.35,
    fontSize: 20, color: colors.white, align: 'center', fontFace: 'Microsoft YaHei'
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
      x, y, w: 1.8, h: 1.7, fill: { color: colors.lightGray },
      line: { width: 1, color: colors.accentBlue }
    });
    slide11.addText(item.value, {
      x, y: y + 0.25, w: 1.8, h: 0.65,
      fontSize: item.value === '−1.8%' ? 40 : 44, bold: true, 
      color: item.value === '−1.8%' ? colors.highlightRed : colors.accentBlue, 
      align: 'center', fontFace: 'Arial'
    });
    slide11.addText(item.label, {
      x, y: y + 0.95, w: 1.8, h: 0.35,
      fontSize: 20, bold: true, color: colors.textBlack, align: 'center', fontFace: 'Microsoft YaHei'
    });
    slide11.addText(item.sublabel, {
      x, y: y + 1.32, w: 1.8, h: 0.3,
      fontSize: 16, color: '666666', align: 'center', fontFace: 'Arial'
    });
  });

  // Slide 12: 算子性能分析（放大图表）
  console.log('[12/15] Adding operator performance...');
  const slide12 = pptx.addSlide();
  slide12.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: 0.15, h: 5.63, fill: { color: colors.mainBlue }, line: { type: 'none' } });
  slide12.addText('算子性能分析 Operator Performance', {
    x: 0.5, y: 0.4, w: 9, h: 0.6, 
    fontSize: 28, bold: true, color: colors.textBlack, fontFace: 'Microsoft YaHei'
  });

  // 上半部分：破坏算子
  slide12.addText([
    { text: '破坏算子 ', options: { fontSize: 22, bold: true, color: colors.highlightRed, fontFace: 'Microsoft YaHei' } },
    { text: 'Destroy Operators', options: { fontSize: 20, color: colors.accentBlue, fontFace: 'Arial' } }
  ], {
    x: 0.6, y: 1.1, w: 8.8, h: 0.4
  });

  const destroyData = [
    {
      name: '使用次数',
      labels: ['random_removal', 'route_removal', 'string_removal'],
      values: [632, 60, 317]
    }
  ];

  slide12.addChart(pptx.charts.BAR, destroyData, {
    x: 0.5, y: 1.6, w: 9, h: 1.8,
    barDir: 'col',
    showTitle: false,
    showLegend: false,
    chartColors: [colors.mainBlue, colors.accentBlue, colors.highlightRed],
    catAxisLabelFontSize: 18,
    valAxisLabelFontSize: 18,
    showCatAxisTitle: true,
    catAxisTitle: '算子名称',
    catAxisTitleFontSize: 20,
    showValAxisTitle: true,
    valAxisTitle: '使用次数',
    valAxisTitleFontSize: 20,
    dataLabelPosition: 'outEnd',
    dataLabelColor: colors.textBlack,
    dataLabelFontSize: 18,
    valAxisMaxVal: 700
  });

  // 下半部分：修复算子
  slide12.addText([
    { text: '修复算子 ', options: { fontSize: 22, bold: true, color: colors.highlightRed, fontFace: 'Microsoft YaHei' } },
    { text: 'Repair Operators', options: { fontSize: 20, color: colors.accentBlue, fontFace: 'Arial' } }
  ], {
    x: 0.6, y: 3.6, w: 8.8, h: 0.4
  });

  const repairData = [
    {
      name: '使用次数',
      labels: ['greedy_insert', 'regret_insert'],
      values: [237, 763]
    }
  ];

  slide12.addChart(pptx.charts.BAR, repairData, {
    x: 0.5, y: 4.1, w: 9, h: 1.4,
    barDir: 'col',
    showTitle: false,
    showLegend: false,
    chartColors: [colors.mainBlue, colors.accentBlue],
    catAxisLabelFontSize: 18,
    valAxisLabelFontSize: 18,
    showCatAxisTitle: true,
    catAxisTitle: '算子名称',
    catAxisTitleFontSize: 20,
    showValAxisTitle: true,
    valAxisTitle: '使用次数',
    valAxisTitleFontSize: 20,
    dataLabelPosition: 'outEnd',
    dataLabelColor: colors.textBlack,
    dataLabelFontSize: 18,
    valAxisMaxVal: 800
  });

  // Slide 13: 主要结论
  console.log('[13/15] Adding conclusions...');
  const slide13 = pptx.addSlide();
  slide13.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: 0.15, h: 5.63, fill: { color: colors.mainBlue }, line: { type: 'none' } });
  slide13.addText('07 主要结论 Main Conclusions', {
    x: 0.5, y: 0.5, w: 9, h: 0.7, 
    fontSize: 28, bold: true, color: colors.textBlack, fontFace: 'Microsoft YaHei'
  });

  const conclusions = [
    { num: '1', title: 'LLM 可有效生成 ALNS 算子', text: 'DeepSeek R1 在提示工程引导下，能够生成功能正确的破坏和修复算子代码，在 Solomon C105 基准上取得 42,023.80 的目标函数值' },
    { num: '2', title: '插件式架构保证鲁棒性', text: 'Skeleton + Plugin 的解耦设计，配合 Fallback 机制，即使 LLM 生成的部分算子存在缺陷，系统仍能正常运行' },
    { num: '3', title: '自适应机制的重要性', text: '权重下限保护、自适应破坏比例、重启再加热等机制显著影响算法性能，防止过早收敛和算子饥饿' },
    { num: '4', title: '单次运行即可', text: '内置的三重随机化机制使单次 solve() 已具备足够的搜索能力，多次运行的边际收益极低' }
  ];

  conclusions.forEach((item, idx) => {
    const y = 1.35 + idx * 1.05;
    slide13.addShape(pptx.shapes.RECTANGLE, {
      x: 0.6, y, w: 8.8, h: 0.9, fill: { color: colors.lightGray },
      line: { width: 1, color: colors.accentBlue }
    });

    slide13.addShape(pptx.shapes.OVAL, {
      x: 0.8, y: y + 0.1, w: 0.4, h: 0.4, fill: { color: colors.highlightRed }, line: { type: 'none' }
    });
    slide13.addText(item.num, {
      x: 0.8, y: y + 0.1, w: 0.4, h: 0.4,
      fontSize: 22, bold: true, color: colors.white, align: 'center', valign: 'middle', fontFace: 'Arial'
    });

    slide13.addText([
      { text: item.title + '：', options: { fontSize: 20, bold: true, color: colors.mainBlue, fontFace: 'Microsoft YaHei' } },
      { text: item.text, options: { fontSize: 18, color: colors.textBlack, fontFace: 'Microsoft YaHei' } }
    ], {
      x: 1.35, y: y + 0.18, w: 8, h: 0.7
    });
  });

  // Slide 14: 局限性与未来工作
  console.log('[14/15] Adding future work...');
  const slide14 = pptx.addSlide();
  slide14.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: 0.15, h: 5.63, fill: { color: colors.mainBlue }, line: { type: 'none' } });
  slide14.addText('局限性与未来工作', {
    x: 0.5, y: 0.5, w: 9, h: 0.7, 
    fontSize: 28, bold: true, color: colors.textBlack, fontFace: 'Microsoft YaHei'
  });

  slide14.addText([
    { text: '局限性 ', options: { fontSize: 22, bold: true, color: colors.highlightRed, fontFace: 'Microsoft YaHei' } },
    { text: 'Limitations', options: { fontSize: 20, color: colors.accentBlue, fontFace: 'Arial' } }
  ], {
    x: 0.6, y: 1.3, w: 4.3, h: 0.4
  });
  slide14.addText([
    { text: '• ', options: { bullet: true, fontSize: 20 } },
    { text: '问题规模：', options: { fontSize: 20, bold: true, color: colors.mainBlue, fontFace: 'Microsoft YaHei' } },
    { text: '当前仅在 100 客户规模上测试\n', options: { fontSize: 20, fontFace: 'Microsoft YaHei' } },
    { text: '• ', options: { bullet: true, fontSize: 20 } },
    { text: 'LLM 依赖：', options: { fontSize: 20, bold: true, color: colors.mainBlue, fontFace: 'Arial' } },
    { text: '算子质量取决于 LLM 能力\n', options: { fontSize: 20, fontFace: 'Microsoft YaHei' } },
    { text: '• ', options: { bullet: true, fontSize: 20 } },
    { text: '成本函数固定：', options: { fontSize: 20, bold: true, color: colors.mainBlue, fontFace: 'Microsoft YaHei' } },
    { text: '扩展到其他 VRP 变体需重新设计\n', options: { fontSize: 20, fontFace: 'Microsoft YaHei' } },
    { text: '• ', options: { bullet: true, fontSize: 20 } },
    { text: '缺少对比基线：', options: { fontSize: 20, bold: true, color: colors.mainBlue, fontFace: 'Microsoft YaHei' } },
    { text: '未与手工 ALNS 系统对比', options: { fontSize: 20, fontFace: 'Microsoft YaHei' } }
  ], {
    x: 0.8, y: 1.8, w: 4.1, h: 3.5, color: colors.textBlack
  });

  slide14.addText([
    { text: '未来工作 ', options: { fontSize: 22, bold: true, color: colors.highlightRed, fontFace: 'Microsoft YaHei' } },
    { text: 'Future Work', options: { fontSize: 20, color: colors.accentBlue, fontFace: 'Arial' } }
  ], {
    x: 5.1, y: 1.3, w: 4.3, h: 0.4
  });
  slide14.addText([
    { text: '• ', options: { bullet: true, fontSize: 20 } },
    { text: '多 LLM 对比：', options: { fontSize: 20, bold: true, color: colors.mainBlue, fontFace: 'Arial' } },
    { text: 'GPT-4o、Claude、Gemini 等模型\n', options: { fontSize: 20, fontFace: 'Arial' } },
    { text: '• ', options: { bullet: true, fontSize: 20 } },
    { text: '自适应提示：', options: { fontSize: 20, bold: true, color: colors.mainBlue, fontFace: 'Microsoft YaHei' } },
    { text: '根据执行结果调整提示词\n', options: { fontSize: 20, fontFace: 'Microsoft YaHei' } },
    { text: '• ', options: { bullet: true, fontSize: 20 } },
    { text: '大规模测试：', options: { fontSize: 20, bold: true, color: colors.mainBlue, fontFace: 'Microsoft YaHei' } },
    { text: 'Solomon R/RC 类和 G&H 大规模实例\n', options: { fontSize: 20, fontFace: 'Microsoft YaHei' } },
    { text: '• ', options: { bullet: true, fontSize: 20 } },
    { text: '算子进化：', options: { fontSize: 20, bold: true, color: colors.mainBlue, fontFace: 'Microsoft YaHei' } },
    { text: '让 LLM 根据统计数据改进低效算子', options: { fontSize: 20, fontFace: 'Microsoft YaHei' } }
  ], {
    x: 5.3, y: 1.8, w: 4.1, h: 3.5, color: colors.textBlack
  });

  // Slide 15: Thank You
  console.log('[15/15] Adding thank you slide...');
  const slide15 = pptx.addSlide();
  slide15.background = { color: colors.mainBlue };
  
  slide15.addText('Thank You!', {
    x: 1, y: 1.5, w: 8, h: 1,
    fontSize: 48, bold: true, color: colors.white, align: 'center', fontFace: 'Arial'
  });
  
  slide15.addShape(pptx.shapes.RECTANGLE, {
    x: 3.5, y: 2.6, w: 3, h: 0.05, fill: { color: colors.highlightRed }, line: { type: 'none' }
  });
  
  slide15.addText('OR-LLM-Agent ALNS 模块开发报告', {
    x: 1, y: 2.9, w: 8, h: 0.4,
    fontSize: 22, color: colors.white, align: 'center', fontFace: 'Microsoft YaHei'
  });
  slide15.addText('基于 OR-LLM-Agent (arXiv:2503.10009)', {
    x: 1, y: 3.35, w: 8, h: 0.3,
    fontSize: 20, color: colors.white, align: 'center', fontFace: 'Arial'
  });
  
  slide15.addText('上海交通大学 & 南洋理工大学\nShanghai Jiao Tong University & Nanyang Technological University', {
    x: 1, y: 4.2, w: 8, h: 0.6,
    fontSize: 18, color: 'CCCCCC', align: 'center', fontFace: 'Microsoft YaHei'
  });

  // Save presentation
  const outputPath = 'D:/pythonProject/or_llm_agent/ALNS_VRP_项目报告.pptx';
  console.log('\nSaving updated presentation...');
  await pptx.writeFile({ fileName: outputPath });
  
  console.log(`\n✅ Presentation updated successfully!`);
  console.log(`📁 Location: ${outputPath}`);
  console.log(`📊 Total slides: 15`);
  console.log(`\n🎨 Design Updates:`);
  console.log(`   • Color scheme: Blue (${colors.mainBlue}) + Red (${colors.highlightRed})`);
  console.log(`   • Font size: 20-22pt for body text`);
  console.log(`   • Font family: Microsoft YaHei (CN) + Arial (EN)`);
  console.log(`   • Charts: Enlarged for better visibility`);
  console.log(`   • Highlights: Red for key points, Blue for secondary points`);
}

createPresentation().catch(err => {
  console.error('❌ Error creating presentation:', err);
  process.exit(1);
});