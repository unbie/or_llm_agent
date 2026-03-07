const fs = require('fs');
const { Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, 
        ShadingType, BorderStyle, Header, Footer, PageNumber, TableOfContents } = require('docx');

console.log('🚀 开始快速转换...\n');

// 读取Markdown
const content = fs.readFileSync('ALNS_VRP_项目报告.md', 'utf-8');
const lines = content.split('\n');

console.log(`📄 文件总行数: ${lines.length}`);

// 简化的段落解析
const children = [];
let inCodeBlock = false;
let skipNext = false;

// 封面页
children.push(
    new Paragraph({ spacing: { before: 1440 }, children: [] }),
    new Paragraph({
        heading: HeadingLevel.TITLE,
        children: [new TextRun('基于大语言模型自动生成启发式算子的ALNS算法求解生鲜物流VRP问题')]
    }),
    new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 240 },
        children: [new TextRun({ text: '——OR-LLM-Agent ALNS模块开发报告', size: 28, bold: true })]
    }),
    new Paragraph({ spacing: { before: 720 }, children: [] }),
    new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun('作者: Zhang & Luo')]
    }),
    new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun('上海交通大学 & 南洋理工大学')]
    }),
    new Paragraph({ pageBreakBefore: true, children: [] })
);

// 目录
children.push(
    new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun('目录')]
    }),
    new TableOfContents('目录', { hyperlink: true, headingStyleRange: '1-4' }),
    new Paragraph({ pageBreakBefore: true, children: [] })
);

console.log('📝 解析内容...');
let processed = 0;

for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    
    if (i % 100 === 0) {
        process.stdout.write(`\r进度: ${Math.floor(i/lines.length*100)}%`);
    }
    
    // 代码块
    if (line.trim().startsWith('```')) {
        inCodeBlock = !inCodeBlock;
        continue;
    }
    
    if (inCodeBlock) {
        children.push(new Paragraph({
            style: 'CodeBlock',
            children: [new TextRun({ text: line || ' ', font: 'Consolas', size: 20 })]
        }));
        continue;
    }
    
    // 跳过分隔线和空行
    if (line.trim() === '' || line.trim() === '---') {
        continue;
    }
    
    // 标题
    if (line.startsWith('# ')) {
        children.push(new Paragraph({
            heading: HeadingLevel.HEADING_1,
            children: [new TextRun(line.slice(2))]
        }));
    } else if (line.startsWith('## ')) {
        children.push(new Paragraph({
            heading: HeadingLevel.HEADING_2,
            children: [new TextRun(line.slice(3))]
        }));
    } else if (line.startsWith('### ')) {
        children.push(new Paragraph({
            heading: HeadingLevel.HEADING_3,
            children: [new TextRun(line.slice(4))]
        }));
    } else if (line.startsWith('#### ')) {
        children.push(new Paragraph({
            heading: HeadingLevel.HEADING_4,
            children: [new TextRun(line.slice(5))]
        }));
    }
    // 引用块
    else if (line.trim().startsWith('>')) {
        children.push(new Paragraph({
            children: [new TextRun({ 
                text: line.trim().slice(1).trim(),
                color: '666666'
            })],
            indent: { left: 720 }
        }));
    }
    // 列表
    else if (line.match(/^\s*[-*]\s+/) || line.match(/^\s*\d+\.\s+/)) {
        const text = line.replace(/^\s*[-*\d.]+\s+/, '');
        children.push(new Paragraph({
            children: [new TextRun('• ' + text)],
            indent: { left: 360 }
        }));
    }
    // 跳过表格分隔线
    else if (line.match(/^\s*\|[\s-:|]+\|\s*$/)) {
        continue;
    }
    // 表格行（简化处理为普通文本）
    else if (line.includes('|') && line.trim().startsWith('|')) {
        const cells = line.split('|').filter(c => c.trim()).map(c => c.trim()).join(' | ');
        children.push(new Paragraph({
            children: [new TextRun({ text: cells, size: 20 })],
            spacing: { before: 60, after: 60 }
        }));
    }
    // 数学公式（保留为文本）
    else if (line.trim().startsWith('$$')) {
        skipNext = !skipNext;
        continue;
    }
    else if (skipNext) {
        children.push(new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [new TextRun({ text: '[公式] ' + line.trim(), italics: true })],
            spacing: { before: 120, after: 120 }
        }));
    }
    // 普通段落
    else if (line.trim().length > 0) {
        // 清理Markdown格式
        let text = line;
        text = text.replace(/\*\*(.+?)\*\*/g, '$1'); // 粗体
        text = text.replace(/\*(.+?)\*/g, '$1'); // 斜体
        text = text.replace(/`(.+?)`/g, '$1'); // 代码
        text = text.replace(/\[(.+?)\]\(.+?\)/g, '$1'); // 链接
        
        if (text.trim()) {
            children.push(new Paragraph({
                children: [new TextRun(text)],
                spacing: { line: 360 }
            }));
        }
    }
    
    processed++;
}

console.log(`\n✅ 解析完成,共${processed}行`);
console.log(`📦 生成了${children.length}个Word段落`);

// 创建文档
console.log('📝 创建Word文档...');
const doc = new Document({
    styles: {
        default: { document: { run: { font: 'SimSun', size: 24 } } },
        paragraphStyles: [
            {
                id: 'Title', name: 'Title', basedOn: 'Normal',
                run: { font: 'SimHei', size: 32, bold: true },
                paragraph: { spacing: { before: 240, after: 120 }, alignment: AlignmentType.CENTER }
            },
            {
                id: 'Heading1', name: 'Heading 1', basedOn: 'Normal',
                run: { font: 'SimHei', size: 32, bold: true },
                paragraph: { spacing: { before: 360, after: 240 }, outlineLevel: 0 }
            },
            {
                id: 'Heading2', name: 'Heading 2', basedOn: 'Normal',
                run: { font: 'SimHei', size: 30, bold: true },
                paragraph: { spacing: { before: 240, after: 180 }, outlineLevel: 1 }
            },
            {
                id: 'Heading3', name: 'Heading 3', basedOn: 'Normal',
                run: { font: 'SimHei', size: 28, bold: true },
                paragraph: { spacing: { before: 180, after: 120 }, outlineLevel: 2 }
            },
            {
                id: 'Heading4', name: 'Heading 4', basedOn: 'Normal',
                run: { font: 'SimHei', size: 24, bold: true },
                paragraph: { spacing: { before: 120, after: 60 }, outlineLevel: 3 }
            },
            {
                id: 'CodeBlock', name: 'Code Block', basedOn: 'Normal',
                run: { font: 'Consolas', size: 20 },
                paragraph: { 
                    spacing: { before: 60, after: 60 },
                    shading: { fill: 'F5F5F5', type: ShadingType.CLEAR }
                }
            }
        ]
    },
    sections: [{
        properties: {
            page: {
                margin: { top: 1440, bottom: 1440, left: 1814, right: 1814 }
            }
        },
        headers: {
            default: new Header({
                children: [new Paragraph({
                    alignment: AlignmentType.CENTER,
                    children: [new TextRun({ text: 'OR-LLM-Agent ALNS模块开发报告', size: 20 })]
                })]
            })
        },
        footers: {
            default: new Footer({
                children: [new Paragraph({
                    alignment: AlignmentType.CENTER,
                    children: [
                        new TextRun('第 '),
                        new TextRun({ children: [PageNumber.CURRENT] }),
                        new TextRun(' 页')
                    ]
                })]
            })
        },
        children: children
    }]
});

// 保存
console.log('💾 保存文件...');
Packer.toBuffer(doc).then(buffer => {
    fs.writeFileSync('ALNS_VRP_项目报告_学术版.docx', buffer);
    console.log(`\n✅ 转换完成！`);
    console.log(`📁 文件: ALNS_VRP_项目报告_学术版.docx`);
    console.log(`📊 大小: ${(buffer.length / 1024).toFixed(2)} KB\n`);
}).catch(err => {
    console.error('❌ 保存失败:', err);
    process.exit(1);
});
