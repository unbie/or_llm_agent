const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType, VerticalAlign, LevelFormat } = require('docx');
const fs = require('fs');

const children = [];

children.push(new Paragraph({
  heading: HeadingLevel.TITLE,
  alignment: AlignmentType.CENTER,
  spacing: { before: 400, after: 400 },
  children: [new TextRun({ text: "运营优化问题的LLM驱动启发式算法实验结果报告", bold: true, size: 44 })]
}));

children.push(new Paragraph({
  heading: HeadingLevel.HEADING_1,
  spacing: { before: 240, after: 120 },
  children: [new TextRun({ text: "摘要", bold: true, size: 28 })]
}));

children.push(new Paragraph({
  spacing: { after: 120 },
  children: [new TextRun({ text: "本报告呈现了基于大型语言模型（LLM）的启发式算法在车辆路径优化问题（VRPTW）上的实验结果。实验针对Solomon benchmark数据集的C1和C2类型实例进行了系统性评估，包括算法性能比较、稳定性分析以及参数敏感性研究。结果表明该方法在不同问题规模和特征下展现出差异化的性能表现，同时参数设定对算法效果具有显著影响。", size: 20 })]
}));

children.push(new Paragraph({ children: [new TextRun({ text: "---" })] }));

children.push(new Paragraph({
  heading: HeadingLevel.HEADING_1,
  spacing: { before: 240, after: 120 },
  children: [new TextRun({ text: "1. 数据集性能比较分析", bold: true, size: 28 })]
}));

children.push(new Paragraph({
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 180, after: 120 },
  children: [new TextRun({ text: "1.1 整体性能表现", bold: true, size: 24 })]
}));

children.push(new Paragraph({
  spacing: { after: 80 },
  children: [new TextRun({ text: "表1展示了LLM驱动启发式算法在C1和C2两类数据集上的综合性能指标。", size: 20 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "表1: 不同数据集类型的算法性能对比", bold: true, size: 20 })]
}));

const tableBorder = { style: BorderStyle.SINGLE, size: 8, color: "000000" };
const cellBorders = { top: tableBorder, bottom: tableBorder, left: tableBorder, right: tableBorder };

children.push(new Table({
  columnWidths: [2500, 2500, 2500, 2500],
  rows: [
    new TableRow({
      tableHeader: true,
      children: [
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, shading: { fill: "D5E8F0", type: ShadingType.CLEAR }, verticalAlign: VerticalAlign.CENTER, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Dataset Type", bold: true })] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, shading: { fill: "D5E8F0", type: ShadingType.CLEAR }, verticalAlign: VerticalAlign.CENTER, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Avg Cost", bold: true })] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, shading: { fill: "D5E8F0", type: ShadingType.CLEAR }, verticalAlign: VerticalAlign.CENTER, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Avg Vehicles", bold: true })] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, shading: { fill: "D5E8F0", type: ShadingType.CLEAR }, verticalAlign: VerticalAlign.CENTER, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Avg Time (s)", bold: true })] })] })
      ]
    }),
    new TableRow({
      children: [
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("C1")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("75,510.65")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("18.5")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("130.98")] })] })
      ]
    }),
    new TableRow({
      children: [
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("C2")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("395,346.65")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("9.5")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("235.19")] })] })
      ]
    })
  ]
}));

children.push(new Paragraph({
  spacing: { before: 120, after: 80 },
  children: [new TextRun({ text: "关键发现：", bold: true, size: 20 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "1. 成本差异显著: C2类型实例的平均成本（395,346.65）显著高于C1类型（75,510.65），约为其5.24倍，反映了C2类型问题的复杂性和大时间窗特征。", size: 20 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "2. 车辆使用策略: C1类型实例平均使用18.5辆车，而C2类型仅需9.5辆。这表明C1类型的聚类特征导致需要更多车辆服务分散的客户群，而C2类型的宽时间窗允许更高效的路径合并。", size: 20 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "3. 计算时间: C2类型实例的平均求解时间（235.19秒）比C1类型（130.98秒）增加79.5%，符合问题复杂度增加的预期。", size: 20 })]
}));

children.push(new Paragraph({
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 180, after: 120 },
  children: [new TextRun({ text: "1.2 实例级详细分析", bold: true, size: 24 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "表2提供了四个典型实例的详细性能分析，揭示了算法在不同实例上的稳定性和有效性。", size: 20 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "表2: 典型实例性能详情", bold: true, size: 20 })]
}));

children.push(new Table({
  columnWidths: [2000, 2000, 2500, 2500, 2500],
  rows: [
    new TableRow({
      tableHeader: true,
      children: [
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, shading: { fill: "D5E8F0", type: ShadingType.CLEAR }, verticalAlign: VerticalAlign.CENTER, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Dataset", bold: true })] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, shading: { fill: "D5E8F0", type: ShadingType.CLEAR }, verticalAlign: VerticalAlign.CENTER, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Instance", bold: true })] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, shading: { fill: "D5E8F0", type: ShadingType.CLEAR }, verticalAlign: VerticalAlign.CENTER, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Best Cost (Mean)", bold: true })] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, shading: { fill: "D5E8F0", type: ShadingType.CLEAR }, verticalAlign: VerticalAlign.CENTER, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Best Cost (Std)", bold: true })] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, shading: { fill: "D5E8F0", type: ShadingType.CLEAR }, verticalAlign: VerticalAlign.CENTER, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Best Cost (Min)", bold: true })] })] })
      ]
    }),
    new TableRow({
      children: [
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("C1")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("c101")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("69,972.19")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("3,605.40")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("66,644.78")] })] })
      ]
    }),
    new TableRow({
      children: [
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("C1")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("c105")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("81,049.12")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("7,246.46")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("73,061.88")] })] })
      ]
    }),
    new TableRow({
      children: [
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("C2")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("c201")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("397,226.47")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("26,503.57")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("368,679.80")] })] })
      ]
    }),
    new TableRow({
      children: [
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("C2")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("c205")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("393,466.83")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("57,723.99")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("330,981.73")] })] })
      ]
    })
  ]
}));

children.push(new Paragraph({
  spacing: { before: 120, after: 80 },
  children: [new TextRun({ text: "深层分析：", bold: true, size: 20 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "1. 方差特征: C1类型实例的标准差（3,605.40-7,246.46）相对较小，而C2类型（26,503.57-57,723.99）显著更高，表明算法在C2类型问题上的解质量波动性更大。", size: 20 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "2. 实例间差异: 在C1类型中，c105实例的标准差（7,246.46）是c101（3,605.40）的2.01倍，显示即使在同一类型内，实例特征也会影响算法稳定性。", size: 20 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "3. 最优解质量: 所有实例的最小成本值均显著优于平均值，例如c205的最小值（330,981.73）比平均值（393,466.83）低15.9%，说明算法具有找到高质量解的潜力，但一致性需要改进。", size: 20 })]
}));

children.push(new Paragraph({ children: [new TextRun({ text: "---" })] }));

children.push(new Paragraph({
  heading: HeadingLevel.HEADING_1,
  spacing: { before: 240, after: 120 },
  children: [new TextRun({ text: "2. 算法稳定性分析", bold: true, size: 28 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "稳定性是评价启发式算法可靠性的关键指标。表3通过统计分析量化了算法在多次运行中的表现一致性。", size: 20 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "表3: 算法稳定性统计分析", bold: true, size: 20 })]
}));

children.push(new Table({
  columnWidths: [1500, 2500, 2000, 2000, 2000, 2000],
  rows: [
    new TableRow({
      tableHeader: true,
      children: [
        new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA }, shading: { fill: "D5E8F0", type: ShadingType.CLEAR }, verticalAlign: VerticalAlign.CENTER, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Instance", bold: true })] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, shading: { fill: "D5E8F0", type: ShadingType.CLEAR }, verticalAlign: VerticalAlign.CENTER, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Mean Cost", bold: true })] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, shading: { fill: "D5E8F0", type: ShadingType.CLEAR }, verticalAlign: VerticalAlign.CENTER, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Std Cost", bold: true })] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, shading: { fill: "D5E8F0", type: ShadingType.CLEAR }, verticalAlign: VerticalAlign.CENTER, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Min Cost", bold: true })] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, shading: { fill: "D5E8F0", type: ShadingType.CLEAR }, verticalAlign: VerticalAlign.CENTER, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Max Cost", bold: true })] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, shading: { fill: "D5E8F0", type: ShadingType.CLEAR }, verticalAlign: VerticalAlign.CENTER, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "CV (%)", bold: true })] })] })
      ]
    }),
    new TableRow({
      children: [
        new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("c101")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("69,972.19")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("3,605.40")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("66,644.78")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("73,802.73")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("5.15")] })] })
      ]
    }),
    new TableRow({
      children: [
        new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("c105")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("81,049.12")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("7,246.46")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("73,061.88")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("87,202.41")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("8.94")] })] })
      ]
    }),
    new TableRow({
      children: [
        new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("c201")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("397,226.47")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("26,503.57")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("368,679.80")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("421,052.56")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("6.67")] })] })
      ]
    }),
    new TableRow({
      children: [
        new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("c205")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("393,466.83")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("57,723.99")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("330,981.73")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("444,803.37")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("14.67")] })] })
      ]
    })
  ]
}));

children.push(new Paragraph({
  spacing: { before: 80, after: 40 },
  children: [new TextRun({ text: "注: CV (Coefficient of Variation) = (Std / Mean) × 100%", italics: true, size: 18 })]
}));

children.push(new Paragraph({
  spacing: { before: 120, after: 80 },
  children: [new TextRun({ text: "稳定性评估：", bold: true, size: 20 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "1. 变异系数分析: 变异系数（CV）范围从5.15%（c101）到14.67%（c205），表明算法稳定性在不同实例上存在显著差异。c205的CV值最高，表明该实例对算法的随机性更为敏感。", size: 20 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "2. 解空间跨度: 最大成本与最小成本的差值在c101中为7,157.95（10.7%范围），而在c205中达到113,821.64（34.4%范围），进一步证实了算法在C2类型实例上的不稳定性。", size: 20 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "3. 性能一致性: C1类型实例的CV值（5.15%-8.94%）总体低于C2类型（6.67%-14.67%），表明算法在聚类型、紧时间窗问题上更具可预测性。", size: 20 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "4. 实用意义: c101和c201的CV值分别为5.15%和6.67%，属于可接受范围，表明对于这些实例类型，算法能够提供相对可靠的解决方案。", size: 20 })]
}));

children.push(new Paragraph({ children: [new TextRun({ text: "---" })] }));

children.push(new Paragraph({
  heading: HeadingLevel.HEADING_1,
  spacing: { before: 240, after: 120 },
  children: [new TextRun({ text: "3. 参数敏感性实验", bold: true, size: 28 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "为了理解模型参数对算法性能的影响，本研究针对新鲜度损失参数（theta1）和惩罚成本参数（theta2）进行了敏感性分析。", size: 20 })]
}));

children.push(new Paragraph({
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 180, after: 120 },
  children: [new TextRun({ text: "3.1 参数配置与成本分解", bold: true, size: 24 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "表4: 不同参数配置下的成本分解", bold: true, size: 20 })]
}));

children.push(new Table({
  columnWidths: [1500, 1500, 1500, 2500, 2000, 2000, 1500, 1500, 1500],
  rows: [
    new TableRow({
      tableHeader: true,
      children: [
        new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA }, shading: { fill: "D5E8F0", type: ShadingType.CLEAR }, verticalAlign: VerticalAlign.CENTER, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "参数组", bold: true })] })] }),
        new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA }, shading: { fill: "D5E8F0", type: ShadingType.CLEAR }, verticalAlign: VerticalAlign.CENTER, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "θ₁", bold: true })] })] }),
        new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA }, shading: { fill: "D5E8F0", type: ShadingType.CLEAR }, verticalAlign: VerticalAlign.CENTER, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "θ₂", bold: true })] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, shading: { fill: "D5E8F0", type: ShadingType.CLEAR }, verticalAlign: VerticalAlign.CENTER, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Best Cost (Mean)", bold: true })] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, shading: { fill: "D5E8F0", type: ShadingType.CLEAR }, verticalAlign: VerticalAlign.CENTER, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Best Cost (Std)", bold: true })] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, shading: { fill: "D5E8F0", type: ShadingType.CLEAR }, verticalAlign: VerticalAlign.CENTER, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Freshness Cost", bold: true })] })] }),
        new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA }, shading: { fill: "D5E8F0", type: ShadingType.CLEAR }, verticalAlign: VerticalAlign.CENTER, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Penalty Cost", bold: true })] })] }),
        new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA }, shading: { fill: "D5E8F0", type: ShadingType.CLEAR }, verticalAlign: VerticalAlign.CENTER, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Routes", bold: true })] })] }),
        new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA }, shading: { fill: "D5E8F0", type: ShadingType.CLEAR }, verticalAlign: VerticalAlign.CENTER, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Time (s)", bold: true })] })] })
      ]
    }),
    new TableRow({
      children: [
        new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("低腐损")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("0.001")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("0.003")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("61,692.31")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("1,339.43")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("49,353.84")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("2,467.69")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("19.0")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("136.67")] })] })
      ]
    }),
    new TableRow({
      children: [
        new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("默认")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("0.002")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("0.005")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("69,972.19")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("3,605.40")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("55,977.75")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("2,798.89")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("19.0")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("131.93")] })] })
      ]
    }),
    new TableRow({
      children: [
        new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("高腐损")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("0.004")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("0.008")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("76,950.36")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("1,499.15")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("61,560.29")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("3,078.01")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("19.0")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("125.33")] })] })
      ]
    }),
    new TableRow({
      children: [
        new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("极高腐损")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("0.006")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("0.012")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("87,103.15")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("1,266.93")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 2000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("69,682.52")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("3,484.13")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("19.0")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 1500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("129.33")] })] })
      ]
    })
  ]
}));

children.push(new Paragraph({
  spacing: { before: 120, after: 80 },
  children: [new TextRun({ text: "参数影响观察：", bold: true, size: 20 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "1. 成本单调性: 总成本随参数值增加而单调递增，从低腐损配置的61,692.31增至极高腐损的87,103.15，增幅达41.2%，验证了参数对目标函数的直接影响。", size: 20 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "2. 成本结构: 新鲜度成本占据总成本的主导地位（约80%），而惩罚成本仅占4%左右。这一结构表明优化重心应放在新鲜度保持策略上。", size: 20 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "3. 稳定性悖论: 标准差呈现非单调变化，极高腐损配置（1,266.93）反而展现出最佳稳定性，低于默认配置（3,605.40）的稳定性。这可能源于较高的参数值减少了解空间的有效区域，从而降低了搜索的随机性。", size: 20 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "4. 计算效率: 求解时间在125.33-136.67秒之间波动，波动率仅为8.3%，表明参数变化对计算复杂度的影响有限。", size: 20 })]
}));

children.push(new Paragraph({
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 180, after: 120 },
  children: [new TextRun({ text: "3.2 相对变化率分析", bold: true, size: 24 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "表5: 参数配置的相对成本变化", bold: true, size: 20 })]
}));

children.push(new Table({
  columnWidths: [2500, 3000, 3000, 3000],
  rows: [
    new TableRow({
      tableHeader: true,
      children: [
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, shading: { fill: "D5E8F0", type: ShadingType.CLEAR }, verticalAlign: VerticalAlign.CENTER, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "参数组", bold: true })] })] }),
        new TableCell({ borders: cellBorders, width: { size: 3000, type: WidthType.DXA }, shading: { fill: "D5E8F0", type: ShadingType.CLEAR }, verticalAlign: VerticalAlign.CENTER, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Best Cost", bold: true })] })] }),
        new TableCell({ borders: cellBorders, width: { size: 3000, type: WidthType.DXA }, shading: { fill: "D5E8F0", type: ShadingType.CLEAR }, verticalAlign: VerticalAlign.CENTER, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Freshness Cost", bold: true })] })] }),
        new TableCell({ borders: cellBorders, width: { size: 3000, type: WidthType.DXA }, shading: { fill: "D5E8F0", type: ShadingType.CLEAR }, verticalAlign: VerticalAlign.CENTER, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Cost Change (%)", bold: true })] })] })
      ]
    }),
    new TableRow({
      children: [
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("低腐损")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("61,692.31")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("49,353.84")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("-11.83")] })] })
      ]
    }),
    new TableRow({
      children: [
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("默认")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("69,972.19")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("55,977.75")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("0.00")] })] })
      ]
    }),
    new TableRow({
      children: [
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("高腐损")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("76,950.36")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("61,560.29")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("+9.97")] })] })
      ]
    }),
    new TableRow({
      children: [
        new TableCell({ borders: cellBorders, width: { size: 2500, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("极高腐损")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("87,103.15")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("69,682.52")] })] }),
        new TableCell({ borders: cellBorders, width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun("+24.48")] })] })
      ]
    })
  ]
}));

children.push(new Paragraph({
  spacing: { before: 120, after: 80 },
  children: [new TextRun({ text: "敏感性量化：", bold: true, size: 20 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "1. 参数灵敏度: 相对于默认配置，参数调整导致成本变化范围为-11.83%至+24.48%，表明算法对参数设定具有中等到高度敏感性。", size: 20 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "2. 非对称响应: 参数增加的影响（+24.48%）大于参数减少的影响（-11.83%），呈现非对称响应特征。这暗示在高参数区域，成本函数的梯度更陡峭。", size: 20 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "3. 优化空间: 低腐损配置实现了11.83%的成本降低，为实际应用中的参数调优提供了改进空间。", size: 20 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "4. 鲁棒性考量: 从低腐损到极高腐损的参数跨度（θ₁: 0.001→0.006，6倍变化）导致总成本变化36.31%，表明参数选择对最终方案的经济性具有实质性影响。", size: 20 })]
}));

children.push(new Paragraph({ children: [new TextRun({ text: "---" })] }));

children.push(new Paragraph({
  heading: HeadingLevel.HEADING_1,
  spacing: { before: 240, after: 120 },
  children: [new TextRun({ text: "4. 可视化结果说明", bold: true, size: 28 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "本研究生成了以下可视化图表以直观呈现实验结果：", size: 20 })]
}));

children.push(new Paragraph({
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 180, after: 120 },
  children: [new TextRun({ text: "4.1 主要性能图表", bold: true, size: 24 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "• fig1_dataset_comparison.png: 展示C1与C2数据集在平均成本、车辆数和计算时间三个维度的对比柱状图，直观体现两类问题的性能差异。", size: 20 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "• fig2_stability_boxplot.png: 采用箱线图展示各实例的成本分布，可视化中位数、四分位数及异常值，为稳定性分析提供直观依据。", size: 20 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "• fig3_computation_time.png: 呈现不同实例的平均计算时间，分析算法的计算效率特征。", size: 20 })]
}));

children.push(new Paragraph({
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 180, after: 120 },
  children: [new TextRun({ text: "4.2 敏感性分析图表", bold: true, size: 24 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "• fig_sensitivity_cost_comparison.png: 对比不同参数配置下的总成本及成本分解（新鲜度成本、惩罚成本），揭示参数对成本结构的影响。", size: 20 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "• fig_sensitivity_cost_breakdown.png: 堆叠柱状图展示各参数配置下的成本组成比例，量化不同成本项的贡献。", size: 20 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "• fig_sensitivity_scatter.png: 散点图展示参数值与总成本的关系，识别潜在的非线性响应模式。", size: 20 })]
}));

children.push(new Paragraph({ children: [new TextRun({ text: "---" })] }));

children.push(new Paragraph({
  heading: HeadingLevel.HEADING_1,
  spacing: { before: 240, after: 120 },
  children: [new TextRun({ text: "5. 结论与讨论", bold: true, size: 28 })]
}));

children.push(new Paragraph({
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 180, after: 120 },
  children: [new TextRun({ text: "5.1 主要发现", bold: true, size: 24 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "1. 问题依赖性: 算法在C1类型（聚类、紧时间窗）实例上表现出不同于C2类型（随机分布、宽时间窗）的特征，表明算法性能与问题结构的匹配度密切相关。", size: 20 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "2. 稳定性权衡: 算法展现出可接受的稳定性（CV: 5.15%-14.67%），但在某些实例（如c205）上波动较大，需要通过多次运行或集成策略增强鲁棒性。", size: 20 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "3. 参数敏感性: 新鲜度和惩罚参数对最终成本具有显著影响（±11.83%至+24.48%），强调了参数调优在实际应用中的重要性。", size: 20 })]
}));

children.push(new Paragraph({
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 180, after: 120 },
  children: [new TextRun({ text: "5.2 方法局限性", bold: true, size: 24 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "1. 可扩展性: 计算时间（130.98-235.19秒）对于实时决策场景可能过长，限制了工业应用潜力。", size: 20 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "2. 参数依赖: 性能对参数选择敏感，要求针对具体问题场景进行细致的参数调校。", size: 20 })]
}));

children.push(new Paragraph({
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 180, after: 120 },
  children: [new TextRun({ text: "5.3 未来研究方向", bold: true, size: 24 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "1. 混合策略: 结合传统启发式算法（如遗传算法、模拟退火）与LLM，利用LLM的模式识别能力指导传统算法的搜索方向。", size: 20 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "2. 自适应参数: 开发参数自适应机制，根据问题特征和求解过程动态调整θ₁和θ₂值。", size: 20 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "3. 多目标优化: 扩展框架以同时优化成本、车辆数和服务质量等多个目标，提供帕累托前沿解集。", size: 20 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "4. 大规模实例测试: 在更大规模的benchmark（如100+客户节点）上验证算法的可扩展性和鲁棒性。", size: 20 })]
}));

children.push(new Paragraph({ children: [new TextRun({ text: "---" })] }));

children.push(new Paragraph({
  heading: HeadingLevel.HEADING_1,
  spacing: { before: 240, after: 120 },
  children: [new TextRun({ text: "6. 附录：数据表详细说明", bold: true, size: 28 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "本节详细解释实验数据表的结构、字段含义及其分析价值，为研究者复现和扩展本研究提供完整的数据文档。", size: 20 })]
}));

children.push(new Paragraph({
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 180, after: 120 },
  children: [new TextRun({ text: "6.1 表格1：数据集汇总对比 (table1_dataset_comparison.csv)", bold: true, size: 24 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "用途: 提供不同数据集类型的宏观性能对比", size: 20 })]
}));

children.push(new Paragraph({
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 180, after: 120 },
  children: [new TextRun({ text: "6.2 表格2：实例详细统计 (table2_instance_details.csv)", bold: true, size: 24 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "用途: 提供具体实例的多次运行统计结果", size: 20 })]
}));

children.push(new Paragraph({
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 180, after: 120 },
  children: [new TextRun({ text: "6.3 表格3：稳定性深度分析 (table3_stability_analysis.csv)", bold: true, size: 24 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "用途: 通过统计学指标量化算法可靠性", size: 20 })]
}));

children.push(new Paragraph({
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 180, after: 120 },
  children: [new TextRun({ text: "6.4 表格4：参数敏感性完整分析 (table_sensitivity_analysis.csv)", bold: true, size: 24 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "用途: 系统评估模型参数对算法性能的多维影响", size: 20 })]
}));

children.push(new Paragraph({
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 180, after: 120 },
  children: [new TextRun({ text: "6.5 表格5：参数影响量化 (table_sensitivity_relative_change.csv)", bold: true, size: 24 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "用途: 量化参数变化对成本的相对影响", size: 20 })]
}));

children.push(new Paragraph({
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 180, after: 120 },
  children: [new TextRun({ text: "6.6 数据表使用指南", bold: true, size: 24 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "复现研究:", size: 20, bold: true })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "1. 使用Table 1和Table 2验证算法在标准benchmark上的基准性能", size: 20 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "2. 使用Table 3评估算法稳定性是否满足应用要求（建议CV<15%）", size: 20 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "3. 使用Table 4-5进行参数调优", size: 20 })]
}));

children.push(new Paragraph({ children: [new TextRun({ text: "---" })] }));

children.push(new Paragraph({
  heading: HeadingLevel.HEADING_1,
  spacing: { before: 240, after: 120 },
  children: [new TextRun({ text: "7. 数据可用性", bold: true, size: 28 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "所有实验数据、图表和详细结果已整理在以下目录：", size: 20 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "• 图表目录: experiments_batch/figures/", size: 20 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "• 数据表目录: experiments_batch/tables/", size: 20 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "完整的数值结果可用于进一步的统计分析和方法比较研究。", size: 20 })]
}));

children.push(new Paragraph({ children: [new TextRun({ text: "---" })] }));

children.push(new Paragraph({
  spacing: { before: 240, after: 40 },
  children: [new TextRun({ text: "报告生成日期: 2026-04-03", size: 18 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "实验框架版本: or_llm_agent v1.0", size: 18 })]
}));

children.push(new Paragraph({
  spacing: { after: 40 },
  children: [new TextRun({ text: "评估基准: Solomon VRPTW Benchmark (C1, C2类型实例)", size: 18 })]
}));

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 20 } } }
  },
  sections: [{
    properties: { page: { margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    children: children
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("D:\\course information\\毕设\\实验结果报告.docx", buffer);
  console.log("Word document created successfully!");
});
