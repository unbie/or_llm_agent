const fs = require('fs');
const path = require('path');
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, 
        Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType, 
        ShadingType, VerticalAlign, PageNumber, TableOfContents, LevelFormat,
        PageBreak } = require('docx');

// ========== 配置参数 ==========
const INPUT_FILE = 'ALNS_VRP_项目报告.md';
const OUTPUT_FILE = 'ALNS_VRP_项目报告_学术版.docx';

// ========== 辅助函数 ==========

// 读取Markdown文件
function readMarkdown(filePath) {
    return fs.readFileSync(filePath, 'utf-8');
}

// 解析Markdown行
function parseMarkdownLines(content) {
    const lines = content.split('\n');
    const elements = [];
    let i = 0;
    
    while (i < lines.length) {
        const line = lines[i];
        
        // 跳过空行（但记录用于段落间距）
        if (line.trim() === '') {
            i++;
            continue;
        }
        
        // 代码块
        if (line.trim().startsWith('```')) {
            const codeBlock = { type: 'code', lang: line.trim().slice(3), lines: [] };
            i++;
            while (i < lines.length && !lines[i].trim().startsWith('```')) {
                codeBlock.lines.push(lines[i]);
                i++;
            }
            elements.push(codeBlock);
            i++; // 跳过结束的 ```
            continue;
        }
        
        // 标题
        const headingMatch = line.match(/^(#{1,4})\s+(.+)$/);
        if (headingMatch) {
            const level = headingMatch[1].length;
            const text = headingMatch[2];
            elements.push({ type: 'heading', level, text });
            i++;
            continue;
        }
        
        // 引用块
        if (line.trim().startsWith('>')) {
            const quoteLines = [];
            while (i < lines.length && lines[i].trim().startsWith('>')) {
                quoteLines.push(lines[i].trim().slice(1).trim());
                i++;
            }
            elements.push({ type: 'quote', lines: quoteLines });
            continue;
        }
        
        // 表格（简化检测：包含 | 的行）
        if (line.includes('|') && line.trim().startsWith('|')) {
            const tableLines = [line];
            i++;
            while (i < lines.length && lines[i].includes('|') && lines[i].trim().startsWith('|')) {
                tableLines.push(lines[i]);
                i++;
            }
            elements.push({ type: 'table', lines: tableLines });
            continue;
        }
        
        // 数学公式块
        if (line.trim().startsWith('$$')) {
            const formulaLines = [];
            i++;
            while (i < lines.length && !lines[i].trim().startsWith('$$')) {
                formulaLines.push(lines[i]);
                i++;
            }
            elements.push({ type: 'formula', content: formulaLines.join('\n') });
            i++; // 跳过结束的 $$
            continue;
        }
        
        // 无序列表
        if (line.match(/^\s*[-*]\s+/)) {
            const listItems = [];
            while (i < lines.length && lines[i].match(/^\s*[-*]\s+/)) {
                const indent = lines[i].match(/^(\s*)[-*]\s+/)[1].length;
                const text = lines[i].replace(/^\s*[-*]\s+/, '');
                listItems.push({ text, indent });
                i++;
            }
            elements.push({ type: 'list', ordered: false, items: listItems });
            continue;
        }
        
        // 有序列表
        if (line.match(/^\s*\d+\.\s+/)) {
            const listItems = [];
            while (i < lines.length && lines[i].match(/^\s*\d+\.\s+/)) {
                const text = lines[i].replace(/^\s*\d+\.\s+/, '');
                listItems.push({ text, indent: 0 });
                i++;
            }
            elements.push({ type: 'list', ordered: true, items: listItems });
            continue;
        }
        
        // 普通段落
        const paraLines = [];
        while (i < lines.length && lines[i].trim() !== '' && 
               !lines[i].match(/^#{1,4}\s/) && 
               !lines[i].trim().startsWith('```') &&
               !lines[i].trim().startsWith('>') &&
               !(lines[i].includes('|') && lines[i].trim().startsWith('|')) &&
               !lines[i].trim().startsWith('$$') &&
               !lines[i].match(/^\s*[-*]\s+/) &&
               !lines[i].match(/^\s*\d+\.\s+/)) {
            paraLines.push(lines[i]);
            i++;
        }
        if (paraLines.length > 0) {
            elements.push({ type: 'paragraph', text: paraLines.join(' ') });
        }
    }
    
    return elements;
}

// 解析表格
function parseTable(tableLines) {
    // 移除分隔线（第二行通常是 |---|---|）
    const rows = tableLines.filter(line => !line.match(/^\s*\|[\s-:|]+\|\s*$/));
    
    return rows.map(row => {
        // 分割单元格
        const cells = row.split('|')
            .map(cell => cell.trim())
            .filter(cell => cell !== '');
        return cells;
    });
}

// 清理文本中的Markdown格式标记（粗体、斜体、行内代码等）
function cleanMarkdownText(text) {
    if (!text) return '';
    
    // 移除粗体标记 **text** 或 __text__
    text = text.replace(/\*\*(.+?)\*\*/g, '$1');
    text = text.replace(/__(.+?)__/g, '$1');
    
    // 移除斜体标记 *text* 或 _text_
    text = text.replace(/\*(.+?)\*/g, '$1');
    text = text.replace(/_(.+?)_/g, '$1');
    
    // 移除行内代码 `code`
    text = text.replace(/`(.+?)`/g, '$1');
    
    // 移除链接 [text](url)
    text = text.replace(/\[(.+?)\]\(.+?\)/g, '$1');
    
    // 移除行内数学公式 $...$
    text = text.replace(/\$(.+?)\$/g, '[$1]');
    
    return text;
}

// 创建Word文档
function createWordDocument(elements) {
    // 定义样式
    const styles = {
        default: {
            document: {
                run: { font: 'SimSun', size: 24 } // 宋体 12pt (24半pt)
            }
        },
        paragraphStyles: [
            {
                id: 'Title',
                name: 'Title',
                basedOn: 'Normal',
                run: { font: 'SimHei', size: 32, bold: true, color: '000000' }, // 黑体 16pt
                paragraph: { 
                    spacing: { before: 240, after: 120 }, 
                    alignment: AlignmentType.CENTER 
                }
            },
            {
                id: 'Heading1',
                name: 'Heading 1',
                basedOn: 'Normal',
                next: 'Normal',
                quickFormat: true,
                run: { font: 'SimHei', size: 32, bold: true, color: '000000' }, // 黑体 三号 16pt
                paragraph: { 
                    spacing: { before: 360, after: 240 }, 
                    outlineLevel: 0 
                }
            },
            {
                id: 'Heading2',
                name: 'Heading 2',
                basedOn: 'Normal',
                next: 'Normal',
                quickFormat: true,
                run: { font: 'SimHei', size: 30, bold: true, color: '000000' }, // 黑体 小三 15pt
                paragraph: { 
                    spacing: { before: 240, after: 180 }, 
                    outlineLevel: 1 
                }
            },
            {
                id: 'Heading3',
                name: 'Heading 3',
                basedOn: 'Normal',
                next: 'Normal',
                quickFormat: true,
                run: { font: 'SimHei', size: 28, bold: true, color: '000000' }, // 黑体 四号 14pt
                paragraph: { 
                    spacing: { before: 180, after: 120 }, 
                    outlineLevel: 2 
                }
            },
            {
                id: 'Heading4',
                name: 'Heading 4',
                basedOn: 'Normal',
                next: 'Normal',
                quickFormat: true,
                run: { font: 'SimHei', size: 24, bold: true, color: '000000' }, // 黑体 小四加粗 12pt
                paragraph: { 
                    spacing: { before: 120, after: 60 }, 
                    outlineLevel: 3 
                }
            },
            {
                id: 'NormalText',
                name: 'Normal Text',
                basedOn: 'Normal',
                run: { font: 'SimSun', size: 24 }, // 宋体 小四 12pt
                paragraph: { 
                    spacing: { line: 360, lineRule: 'auto' }, // 1.5倍行距
                    alignment: AlignmentType.JUSTIFIED
                }
            },
            {
                id: 'CodeBlock',
                name: 'Code Block',
                basedOn: 'Normal',
                run: { font: 'Consolas', size: 20 }, // Consolas 10pt
                paragraph: { 
                    spacing: { before: 120, after: 120, line: 240 },
                    shading: { fill: 'F5F5F5', type: ShadingType.CLEAR }
                }
            },
            {
                id: 'Quote',
                name: 'Quote',
                basedOn: 'Normal',
                run: { font: 'SimSun', size: 22, color: '666666' }, // 宋体 11pt 灰色
                paragraph: { 
                    spacing: { before: 120, after: 120 },
                    indent: { left: 720 }, // 左缩进
                    border: {
                        left: { style: BorderStyle.SINGLE, size: 6, color: 'CCCCCC' }
                    }
                }
            }
        ]
    };

    // 定义页面边距（上下2.54cm = 1440 DXA, 左右3.18cm = 1814 DXA）
    const pageMargins = {
        top: 1440,    // 2.54cm
        bottom: 1440, // 2.54cm
        left: 1814,   // 3.18cm
        right: 1814   // 3.18cm
    };

    // 创建封面页内容
    const coverPageChildren = [
        new Paragraph({
            spacing: { before: 1440 }, // 顶部留白
            children: []
        }),
        new Paragraph({
            heading: HeadingLevel.TITLE,
            children: [new TextRun('基于大语言模型自动生成启发式算子的ALNS算法求解生鲜物流VRP问题')]
        }),
        new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { before: 240, after: 120 },
            children: [new TextRun({ 
                text: '——OR-LLM-Agent ALNS模块开发报告',
                font: 'SimHei',
                size: 28,
                bold: true
            })]
        }),
        new Paragraph({
            spacing: { before: 1440 }, // 间隔
            children: []
        }),
        new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { after: 120 },
            children: [new TextRun({ 
                text: '项目基础: OR-LLM-Agent (arXiv:2503.10009)',
                font: 'SimSun',
                size: 24
            })]
        }),
        new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { after: 120 },
            children: [new TextRun({ 
                text: '作者: Zhang & Luo',
                font: 'SimSun',
                size: 24
            })]
        }),
        new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { after: 120 },
            children: [new TextRun({ 
                text: '上海交通大学 & 南洋理工大学',
                font: 'SimSun',
                size: 24
            })]
        }),
        new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { after: 120 },
            children: [new TextRun({ 
                text: 'LLM模型: DeepSeek R1 (火山引擎 API)',
                font: 'SimSun',
                size: 22
            })]
        }),
        new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [new TextRun({ 
                text: '测试基准: Solomon C105 (100 customers)',
                font: 'SimSun',
                size: 22
            })]
        }),
        new Paragraph({
            pageBreakBefore: true, // 封面后分页
            children: []
        })
    ];

    // 目录页
    const tocChildren = [
        new Paragraph({
            heading: HeadingLevel.HEADING_1,
            children: [new TextRun('目录')]
        }),
        new TableOfContents('目录', {
            hyperlink: true,
            headingStyleRange: '1-4'
        }),
        new Paragraph({
            pageBreakBefore: true, // 目录后分页
            children: []
        })
    ];

    // 列表编号配置
    const numberingConfig = [
        {
            reference: 'bullet-list',
            levels: [
                {
                    level: 0,
                    format: LevelFormat.BULLET,
                    text: '•',
                    alignment: AlignmentType.LEFT,
                    style: {
                        paragraph: {
                            indent: { left: 720, hanging: 360 }
                        }
                    }
                }
            ]
        },
        {
            reference: 'numbered-list',
            levels: [
                {
                    level: 0,
                    format: LevelFormat.DECIMAL,
                    text: '%1.',
                    alignment: AlignmentType.LEFT,
                    style: {
                        paragraph: {
                            indent: { left: 720, hanging: 360 }
                        }
                    }
                }
            ]
        }
    ];

    // 转换元素为Word段落
    const contentChildren = [];
    
    for (const elem of elements) {
        switch (elem.type) {
            case 'heading':
                const headingLevels = {
                    1: HeadingLevel.HEADING_1,
                    2: HeadingLevel.HEADING_2,
                    3: HeadingLevel.HEADING_3,
                    4: HeadingLevel.HEADING_4
                };
                
                contentChildren.push(new Paragraph({
                    heading: headingLevels[elem.level] || HeadingLevel.HEADING_1,
                    children: [new TextRun(cleanMarkdownText(elem.text))]
                }));
                break;
            
            case 'paragraph':
                if (elem.text.trim()) {
                    contentChildren.push(new Paragraph({
                        style: 'NormalText',
                        children: [new TextRun(cleanMarkdownText(elem.text))]
                    }));
                }
                break;
            
            case 'code':
                // 代码块：每行作为单独的段落
                for (const codeLine of elem.lines) {
                    contentChildren.push(new Paragraph({
                        style: 'CodeBlock',
                        children: [new TextRun(codeLine || ' ')] // 空行也保留
                    }));
                }
                // 代码块后添加空段落
                contentChildren.push(new Paragraph({ children: [] }));
                break;
            
            case 'quote':
                for (const quoteLine of elem.lines) {
                    contentChildren.push(new Paragraph({
                        style: 'Quote',
                        children: [new TextRun(cleanMarkdownText(quoteLine))]
                    }));
                }
                break;
            
            case 'table':
                const tableData = parseTable(elem.lines);
                if (tableData.length > 0) {
                    const numCols = tableData[0].length;
                    const colWidth = Math.floor(7732 / numCols); // 7732 = 页面宽度 - 左右边距
                    
                    const tableRows = tableData.map((rowData, rowIndex) => {
                        const isHeader = rowIndex === 0;
                        return new TableRow({
                            tableHeader: isHeader,
                            children: rowData.map(cellText => new TableCell({
                                width: { size: colWidth, type: WidthType.DXA },
                                shading: isHeader ? 
                                    { fill: 'E6E6E6', type: ShadingType.CLEAR } : 
                                    { fill: 'FFFFFF', type: ShadingType.CLEAR },
                                borders: {
                                    top: { style: BorderStyle.SINGLE, size: 1, color: '000000' },
                                    bottom: { style: BorderStyle.SINGLE, size: 1, color: '000000' },
                                    left: { style: BorderStyle.SINGLE, size: 1, color: 'CCCCCC' },
                                    right: { style: BorderStyle.SINGLE, size: 1, color: 'CCCCCC' }
                                },
                                verticalAlign: VerticalAlign.CENTER,
                                children: [new Paragraph({
                                    alignment: isHeader ? AlignmentType.CENTER : AlignmentType.LEFT,
                                    children: [new TextRun({
                                        text: cleanMarkdownText(cellText),
                                        bold: isHeader,
                                        size: 22,
                                        font: 'SimSun'
                                    })]
                                })]
                            }))
                        });
                    });
                    
                    contentChildren.push(new Table({
                        columnWidths: Array(numCols).fill(colWidth),
                        margins: { top: 60, bottom: 60, left: 120, right: 120 },
                        rows: tableRows
                    }));
                    
                    // 表格后添加空段落
                    contentChildren.push(new Paragraph({ children: [] }));
                }
                break;
            
            case 'formula':
                // 数学公式保留为纯文本（Word公式转换较复杂，需要专门的库）
                contentChildren.push(new Paragraph({
                    style: 'CodeBlock',
                    alignment: AlignmentType.CENTER,
                    children: [new TextRun({
                        text: '[公式] ' + elem.content,
                        font: 'Times New Roman',
                        italics: true
                    })]
                }));
                contentChildren.push(new Paragraph({ children: [] }));
                break;
            
            case 'list':
                const listRef = elem.ordered ? 'numbered-list' : 'bullet-list';
                for (const item of elem.items) {
                    contentChildren.push(new Paragraph({
                        numbering: { reference: listRef, level: 0 },
                        children: [new TextRun(cleanMarkdownText(item.text))]
                    }));
                }
                break;
        }
    }

    // 创建文档
    const doc = new Document({
        styles: styles,
        numbering: {
            config: numberingConfig
        },
        sections: [
            {
                properties: {
                    page: {
                        margin: pageMargins
                    }
                },
                headers: {
                    default: new Header({
                        children: [new Paragraph({
                            alignment: AlignmentType.CENTER,
                            children: [new TextRun({
                                text: 'OR-LLM-Agent ALNS模块开发报告',
                                font: 'SimSun',
                                size: 20
                            })]
                        })]
                    })
                },
                footers: {
                    default: new Footer({
                        children: [new Paragraph({
                            alignment: AlignmentType.CENTER,
                            children: [
                                new TextRun({ text: '第 ', font: 'SimSun', size: 20 }),
                                new TextRun({ children: [PageNumber.CURRENT] }),
                                new TextRun({ text: ' 页', font: 'SimSun', size: 20 })
                            ]
                        })]
                    })
                },
                children: [
                    ...coverPageChildren,
                    ...tocChildren,
                    ...contentChildren
                ]
            }
        ]
    });

    return doc;
}

// ========== 主函数 ==========
async function main() {
    try {
        console.log('🚀 开始转换 Markdown 到 Word 文档...\n');
        
        // 读取Markdown文件
        console.log('📖 读取Markdown文件:', INPUT_FILE);
        const markdownContent = readMarkdown(INPUT_FILE);
        
        // 解析Markdown
        console.log('🔍 解析Markdown结构...');
        const elements = parseMarkdownLines(markdownContent);
        console.log(`   - 解析了 ${elements.length} 个元素`);
        
        // 统计元素类型
        const stats = {};
        elements.forEach(elem => {
            stats[elem.type] = (stats[elem.type] || 0) + 1;
        });
        console.log('   - 元素统计:', stats);
        
        // 创建Word文档
        console.log('\n📝 创建Word文档...');
        const doc = createWordDocument(elements);
        
        // 保存文件
        console.log('💾 保存文件:', OUTPUT_FILE);
        const buffer = await Packer.toBuffer(doc);
        fs.writeFileSync(OUTPUT_FILE, buffer);
        
        console.log('\n✅ 转换完成!');
        console.log(`   文件已保存到: ${path.resolve(OUTPUT_FILE)}`);
        console.log(`   文件大小: ${(buffer.length / 1024).toFixed(2)} KB\n`);
        
    } catch (error) {
        console.error('\n❌ 转换失败:', error.message);
        console.error(error.stack);
        process.exit(1);
    }
}

// 执行
main();
