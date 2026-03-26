"""
生成开题报告PPT：
  - 基于 开题报告.pptx 模板
  - 内容来自 2022216431_黄纤纤_合肥工业大学毕业设计(论文)开题报告.docx
  - 纯 zipfile + lxml，无需 python-pptx
"""

import zipfile
import copy
import io
import re
from lxml import etree

# ─── 命名空间 ────────────────────────────────────────────────
P = "http://schemas.openxmlformats.org/presentationml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PR = "http://schemas.openxmlformats.org/package/2006/relationships"

SRC = r"D:\pythonProject\or_llm_agent\开题报告.pptx"
DEST = r"D:\pythonProject\or_llm_agent\开题报告_黄纤纤.pptx"

# ─── 把整个 zip 读入内存，避免反复随机访问出错 ──────────────
src_zip = zipfile.ZipFile(SRC, "r")
all_files = {}
for item in src_zip.infolist():
    all_files[item.filename] = src_zip.read(item.filename)
src_zip.close()


def read_xml(path):
    return etree.fromstring(all_files[path])


def xml_bytes(root):
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


# ─── 辅助：查找 sp ─────────────────────────────────────────
def find_sp(root, name):
    for sp in root.iter(f"{{{P}}}sp"):
        cnv = sp.find(f"{{{P}}}nvSpPr/{{{P}}}cNvPr")
        if cnv is not None and cnv.get("name") == name:
            return sp
    return None


# ─── 辅助：写入文本到 sp ───────────────────────────────────
def set_text(sp, new_text, bold=None, sz=None, color=None):
    """用 new_text 替换 sp 的 txBody（多行用 \\n）"""
    txBody = sp.find(f"{{{P}}}txBody")
    if txBody is None:
        return
    # 取原始第一个 <a:p> 作为格式模板
    first_p = txBody.find(f"{{{A}}}p")
    base_pPr = None
    base_rPr = None
    if first_p is not None:
        pp = first_p.find(f"{{{A}}}pPr")
        if pp is not None:
            base_pPr = copy.deepcopy(pp)
        r = first_p.find(f"{{{A}}}r")
        if r is not None:
            rp = r.find(f"{{{A}}}rPr")
            if rp is not None:
                base_rPr = copy.deepcopy(rp)
    # 删除所有旧 <a:p>
    for p in txBody.findall(f"{{{A}}}p"):
        txBody.remove(p)

    for line in new_text.split("\n"):
        new_p = etree.SubElement(txBody, f"{{{A}}}p")
        if base_pPr is not None:
            new_p.append(copy.deepcopy(base_pPr))
        new_r = etree.SubElement(new_p, f"{{{A}}}r")
        if base_rPr is not None:
            rPr = copy.deepcopy(base_rPr)
        else:
            rPr = etree.Element(f"{{{A}}}rPr", lang="zh-CN")
        if bold is not None:
            rPr.set("b", "1" if bold else "0")
        if sz is not None:
            rPr.set("sz", str(sz))
        if color is not None:
            fill = rPr.find(f"{{{A}}}solidFill")
            if fill is None:
                fill = etree.SubElement(rPr, f"{{{A}}}solidFill")
            clr = fill.find(f"{{{A}}}srgbClr")
            if clr is None:
                clr = etree.SubElement(fill, f"{{{A}}}srgbClr")
            clr.set("val", color)
        new_r.append(rPr)
        t = etree.SubElement(new_r, f"{{{A}}}t")
        t.text = line


def get_slide(n):
    return copy.deepcopy(etree.fromstring(all_files[f"ppt/slides/slide{n}.xml"]))


# ════════════════════════════════════════════════════════════════
# 幻灯片规划
# ════════════════════════════════════════════════════════════════
slides_plan = [
    ("cover", 1),  # 封面
    ("toc", 2),  # 目录
    ("sec01", 3),  # 分节01
    ("bg", 5),  # 研究背景
    ("status", 11),  # 国内外研究现状
    ("problem", 5),  # 尚待研究的问题
    ("sec02", 8),  # 分节02
    ("method1", 11),  # 需求解析与建模
    ("method2", 11),  # ALNS启发式求解
    ("method3", 7),  # 闭环自修正
    ("sec03", 12),  # 分节03
    ("exp", 15),  # 实验方案
    ("sec04", 16),  # 分节04
    ("equip", 17),  # 所需设备
    ("thanks", 19),  # 致谢
]

slides = {key: get_slide(n) for key, n in slides_plan}

# ════════════════════════════════════════════════════════════════
# 填充各幻灯片内容
# ════════════════════════════════════════════════════════════════

# ── 1. 封面 ─────────────────────────────────────────────────
s = slides["cover"]
sp = find_sp(s, "TextBox 21")
if sp is not None:
    set_text(sp, "大模型驱动的生鲜物流配送路径自动优化研究")
sp = find_sp(s, "TextBox 22")
if sp is not None:
    set_text(sp, "学生姓名：黄纤纤    学号：2022216431")
sp = find_sp(s, "TextBox 23")
if sp is not None:
    set_text(sp, "指导教师：（指导教师姓名）")
sp = find_sp(s, "TextBox 24")
if sp is not None:
    set_text(sp, "合肥工业大学    2026年3月")
sp = find_sp(s, "TextBox 25")
if sp is not None:
    set_text(sp, "THESIS PROPOSAL PRESENTATION")

# ── 2. 目录 ─────────────────────────────────────────────────
s = slides["toc"]
for name, text in [
    ("TextBox 21", "研究背景与意义"),
    ("TextBox 22", "国内外研究现状"),
    ("TextBox 23", "研究方案"),
    ("TextBox 24", "实验计划与预期成果"),
    ("TextBox 25", "RESEARCH BACKGROUND AND SIGNIFICANCE"),
    ("TextBox 26", "LITERATURE REVIEW"),
    ("TextBox 27", "RESEARCH METHODS AND PLAN"),
    ("TextBox 28", "EXPERIMENT PLAN AND RESULTS"),
]:
    sp = find_sp(s, name)
    if sp is not None:
        set_text(sp, text)

# ── 3. Section 01 ────────────────────────────────────────────
s = slides["sec01"]
sp = find_sp(s, "TextBox 12")
if sp is not None:
    set_text(sp, "研究背景与意义")
sp = find_sp(s, "TextBox 13")
if sp is not None:
    set_text(sp, "RESEARCH BACKGROUND AND SIGNIFICANCE")
sp = find_sp(s, "TextBox 14")
if sp is not None:
    set_text(sp, "01")

# ── 4. 研究背景（slide5布局）────────────────────────────────
s = slides["bg"]
sp = find_sp(s, "TextBox 14")
if sp is not None:
    set_text(sp, "研究背景与意义")
sp = find_sp(s, "TextBox 26")
if sp is not None:
    set_text(sp, "课题背景")
sp = find_sp(s, "TextBox 27")
if sp is not None:
    set_text(
        sp,
        "生鲜冷链物流快速发展，消费者对食品新鲜度与配送时效要求日益提升\n"
        "鲜度随时间衰减，配送路径需严格遵守时间窗并考虑鲜度惩罚成本",
    )
sp = find_sp(s, "TextBox 28")
if sp is not None:
    set_text(
        sp,
        "现有路径规划面临多车型、动态需求、复杂约束等NP-hard难题\n"
        "传统运筹方法暴露出人工建模慢、算法调整难的瓶颈",
    )
sp = find_sp(s, "TextBox 29")
if sp is not None:
    set_text(
        sp,
        "引入大语言模型（LLM）技术，构建自动化建模与求解框架\n"
        "实现结构化数据输入 -> 代码自动生成 -> 闭环自动求解",
    )
sp = find_sp(s, "TextBox 30")
if sp is not None:
    set_text(sp, "研究价值")
sp = find_sp(s, "TextBox 31")
if sp is not None:
    set_text(sp, "应用意义")

# ── 5. 国内外研究现状（slide11布局：4块+3tag+大标题）────────
s = slides["status"]
sp = find_sp(s, "TextBox 6")
if sp is not None:
    set_text(sp, "国内外研究现状")
sp = find_sp(s, "TextBox 8")
if sp is not None:
    set_text(
        sp,
        "传统生鲜优化：葛显龙等研究时间窗约束；马佳等引入鲜度衰减；\n"
        "夏扬坤等探讨客户分级与需求可拆分；王勇等研究动态需求配送",
    )
sp = find_sp(s, "TextBox 9")
if sp is not None:
    set_text(
        sp,
        "LLM作为优化器（阶段一）：Yang等提出OPRO框架，但复杂VRP约束下\n"
        "易产生幻觉，可行性难以保证",
    )
sp = find_sp(s, "TextBox 10")
if sp is not None:
    set_text(
        sp,
        "LLM作为建模/编程助手（阶段二）：Li等提出OptiGuide；\n"
        "Zhang等提出OR-LLM-Agent实现自然语言->数学模型->代码全流程自动化",
    )
sp = find_sp(s, "TextBox 11")
if sp is not None:
    set_text(
        sp,
        "LLM驱动启发式进化（阶段三）：Ye等提出LLM-LNS双层自进化框架；\n"
        "Hottung等提出VRPAgent，在大规模MILP上显著超越传统求解器",
    )
sp = find_sp(s, "TextBox 12")
if sp is not None:
    set_text(sp, "发展趋势：多智能体闭环框架 + 自进化启发式算法")
sp = find_sp(s, "TextBox 13")
if sp is not None:
    set_text(sp, "阶段一")
sp = find_sp(s, "TextBox 14")
if sp is not None:
    set_text(sp, "阶段二")
sp = find_sp(s, "TextBox 15")
if sp is not None:
    set_text(sp, "阶段三")

# ── 6. 尚待研究问题（复用slide5布局）──────────────────────
s = slides["problem"]
sp = find_sp(s, "TextBox 14")
if sp is not None:
    set_text(sp, "尚待研究的问题")
sp = find_sp(s, "TextBox 26")
if sp is not None:
    set_text(sp, "三大核心挑战")
sp = find_sp(s, "TextBox 27")
if sp is not None:
    set_text(
        sp,
        "(1) 复杂专属约束的有效映射：如何将时间窗、多温区、鲜度惩罚\n"
        "等非线性约束高成功率地无损转化为数学模型和可执行代码",
    )
sp = find_sp(s, "TextBox 28")
if sp is not None:
    set_text(
        sp,
        "(2) 大规模网络的启发式求解：如何利用大模型高效编写并优化ALNS\n"
        "等元启发式算法，在计算时间与解质量间取得最佳平衡",
    )
sp = find_sp(s, "TextBox 29")
if sp is not None:
    set_text(
        sp,
        "(3) 闭环修正的鲁棒性：如何设计深度错误分析（Error Analysis）机制，\n"
        "确保代码在遇到不可行解或报错时能高效实现自我修复",
    )
sp = find_sp(s, "TextBox 30")
if sp is not None:
    set_text(sp, "约束建模")
sp = find_sp(s, "TextBox 31")
if sp is not None:
    set_text(sp, "启发式求解")

# ── 7. Section 02 ────────────────────────────────────────────
s = slides["sec02"]
sp = find_sp(s, "TextBox 8")
if sp is not None:
    set_text(sp, "研究方案")
sp = find_sp(s, "TextBox 9")
if sp is not None:
    set_text(sp, "RESEARCH METHODS AND PLAN")
sp = find_sp(s, "TextBox 17")
if sp is not None:
    set_text(sp, "02")

# ── 8. 需求解析与自动建模（slide11）─────────────────────────
s = slides["method1"]
sp = find_sp(s, "TextBox 6")
if sp is not None:
    set_text(sp, "研究方案")
sp = find_sp(s, "TextBox 8")
if sp is not None:
    set_text(
        sp,
        "数据输入与预处理：引入生鲜配送结构化数据集，\n"
        "包含车辆容量、客户坐标、时间窗、鲜度衰减参数等，数据清洗后提供高质量输入",
    )
sp = find_sp(s, "TextBox 9")
if sp is not None:
    set_text(
        sp,
        "Agent架构与Prompt工程：依托OR-LLM-Agent框架，\n"
        "结合DeepSeek-R1推理能力，设计VRP变体数据集专属Prompt模板",
    )
sp = find_sp(s, "TextBox 10")
if sp is not None:
    set_text(
        sp,
        "思维链驱动建模：通过Chain-of-Thought引导LLM逐步解析数据集，\n"
        "生成决策变量、目标函数（最小化成本+鲜度惩罚）及约束的标准化数学表达",
    )
sp = find_sp(s, "TextBox 11")
if sp is not None:
    set_text(sp, "输出：标准化数学模型 + 评估函数代码，为后续ALNS求解提供基础")
sp = find_sp(s, "TextBox 12")
if sp is not None:
    set_text(sp, "需求解析与大模型自动建模")
sp = find_sp(s, "TextBox 13")
if sp is not None:
    set_text(sp, "数据输入")
sp = find_sp(s, "TextBox 14")
if sp is not None:
    set_text(sp, "自动建模")
sp = find_sp(s, "TextBox 15")
if sp is not None:
    set_text(sp, "代码生成")

# ── 9. ALNS启发式求解（slide11复用）───────────────────────
s = slides["method2"]
sp = find_sp(s, "TextBox 6")
if sp is not None:
    set_text(sp, "研究方案")
sp = find_sp(s, "TextBox 8")
if sp is not None:
    set_text(
        sp,
        "算法选型：针对大规模高时效性配送需求，聚焦自适应大邻域搜索（ALNS）\n"
        "元启发式算法，实现高效近似最优解",
    )
sp = find_sp(s, "TextBox 9")
if sp is not None:
    set_text(
        sp,
        "算子代码自动生成：通过定制化Prompt驱动LLM自动生成\n"
        "破坏算子（Destroy Operators）与修复算子（Repair Operators），嵌入ALNS框架",
    )
sp = find_sp(s, "TextBox 10")
if sp is not None:
    set_text(
        sp,
        "智能算子探索：引导LLM自动设计适应鲜度衰减、\n"
        "多温区约束、时间窗紧迫度的新型专属启发式算子",
    )
sp = find_sp(s, "TextBox 11")
if sp is not None:
    set_text(
        sp,
        "动态参数调优：基于算子历史表现动态调整选择概率，\n"
        "高效跳出局部最优，满足分钟级调度需求",
    )
sp = find_sp(s, "TextBox 12")
if sp is not None:
    set_text(sp, "大规模场景下的ALNS启发式求解策略")
sp = find_sp(s, "TextBox 13")
if sp is not None:
    set_text(sp, "ALNS框架")
sp = find_sp(s, "TextBox 14")
if sp is not None:
    set_text(sp, "算子生成")
sp = find_sp(s, "TextBox 15")
if sp is not None:
    set_text(sp, "参数自适应")

# ── 10. 闭环自修正（slide7：双列布局）──────────────────────
s = slides["method3"]
sp = find_sp(s, "TextBox 13")
if sp is not None:
    set_text(sp, "研究方案")
sp = find_sp(s, "TextBox 25")
if sp is not None:
    set_text(
        sp,
        "闭环自修正机制（生成-验证-修复）\n\n"
        "构建多智能体协作闭环：\n"
        "(1) 沙盒环境执行生成的Python求解代码\n"
        "(2) 错误分析引擎捕获语法错误/求解超时/不可行解\n"
        "(3) 将错误日志+思考轨迹反馈给LLM，指令其扮演Debugger角色\n"
        "(4) 重新生成修正代码，多轮迭代直至输出符合约束的可行方案",
    )
sp = find_sp(s, "TextBox 26")
if sp is not None:
    set_text(sp, "生成")
sp = find_sp(s, "TextBox 27")
if sp is not None:
    set_text(sp, "验证")
sp = find_sp(s, "TextBox 28")
if sp is not None:
    set_text(sp, "闭环自修正确保代码鲁棒性，实现大规模生鲜配送路径的自动优化求解")
sp = find_sp(s, "TextBox 29")
if sp is not None:
    set_text(sp, "修复->迭代")

# ── 11. Section 03 ───────────────────────────────────────────
s = slides["sec03"]
sp = find_sp(s, "TextBox 8")
if sp is not None:
    set_text(sp, "实验计划与预期成果")
sp = find_sp(s, "TextBox 9")
if sp is not None:
    set_text(sp, "EXPERIMENT PLAN AND EXPECTED RESULTS")
sp = find_sp(s, "TextBox 11")
if sp is not None:
    set_text(sp, "研究计划与预期成果")
sp = find_sp(s, "TextBox 20")
if sp is not None:
    set_text(sp, "03")

# ── 12. 实验方案（slide15：4个成果块）──────────────────────
s = slides["exp"]
sp = find_sp(s, "TextBox 2")
if sp is not None:
    set_text(sp, "实验计划与预期成果")
sp = find_sp(s, "TextBox 20")
if sp is not None:
    set_text(sp, "数据集准备")
sp = find_sp(s, "TextBox 21")
if sp is not None:
    set_text(sp, "算法实现")
sp = find_sp(s, "TextBox 22")
if sp is not None:
    set_text(sp, "对比实验")
sp = find_sp(s, "TextBox 23")
if sp is not None:
    set_text(sp, "论文撰写")
sp = find_sp(s, "TextBox 24")
if sp is not None:
    set_text(
        sp,
        "基于Solomon VRP Benchmark Dataset，结合生鲜物流特征，\n"
        "构造大规模时间窗与鲜度变体数据集",
    )
sp = find_sp(s, "TextBox 25")
if sp is not None:
    set_text(
        sp, "实现LLM驱动的ALNS算法框架，包含Prompt工程、\n算子自动生成与闭环自修正模块"
    )
sp = find_sp(s, "TextBox 26")
if sp is not None:
    set_text(
        sp,
        "与传统手工编写启发式算法对比：\n"
        "求解质量、计算时间、代码可运行率（Executable Rate）",
    )
sp = find_sp(s, "TextBox 27")
if sp is not None:
    set_text(sp, "完成毕业论文撰写，验证LLM框架在大规模场景下\n的可行性与优越性")

# ── 13. Section 04 ───────────────────────────────────────────
s = slides["sec04"]
sp = find_sp(s, "TextBox 8")
if sp is not None:
    set_text(sp, "所需设备与研究计划")
sp = find_sp(s, "TextBox 9")
if sp is not None:
    set_text(sp, "REQUIRED EQUIPMENT AND RESEARCH PLAN")
sp = find_sp(s, "TextBox 17")
if sp is not None:
    set_text(sp, "04")

# ── 14. 所需设备（slide17）─────────────────────────────────
s = slides["equip"]
sp = find_sp(s, "TextBox 11")
if sp is not None:
    set_text(sp, "所需设备与研究计划")
sp = find_sp(s, "TextBox 13")
if sp is not None:
    set_text(
        sp,
        "软硬件环境：Python 3.10；DeepSeek-R1 API（核心推理大模型）；\n"
        "OR-LLM-Agent开发框架；Visual Studio Code IDE；\n"
        "Solomon VRP Benchmark Dataset及其生鲜场景扩展变体数据集",
    )
sp = find_sp(s, "TextBox 14")
if sp is not None:
    set_text(
        sp,
        "研究计划：\n"
        "2026年3-4月：数据集构建 + 模型框架搭建\n"
        "2026年4-5月：ALNS算子自动生成 + 闭环修正实现\n"
        "2026年5-6月：对比实验 + 论文撰写",
    )
sp = find_sp(s, "TextBox 15")
if sp is not None:
    set_text(sp, "计算资源")
sp = find_sp(s, "TextBox 16")
if sp is not None:
    set_text(sp, "开发工具")

# ── 15. 致谢（slide19）─────────────────────────────────────
s = slides["thanks"]
sp = find_sp(s, "TextBox 3")
if sp is not None:
    set_text(sp, "感谢")
sp = find_sp(s, "TextBox 4")
if sp is not None:
    set_text(
        sp,
        "感谢各位老师在百忙之中出席本次开题报告！\n\n"
        "欢迎各位老师批评指正，给予宝贵意见！",
    )
sp = find_sp(s, "TextBox 5")
if sp is not None:
    set_text(sp, "谢  谢")

# ════════════════════════════════════════════════════════════════
# 构建新 ZIP（所有数据已预读到 all_files，直接操作）
# ════════════════════════════════════════════════════════════════

new_slide_count = len(slides_plan)

# ── 预处理 slide _rels（清除 notesSlide 引用）───────────────
slide_rels_data = {}  # tmpl_n -> processed rels bytes
seen_tmpl = set()
for key, tmpl_n in slides_plan:
    if tmpl_n in seen_tmpl:
        continue
    seen_tmpl.add(tmpl_n)
    rels_key = f"ppt/slides/_rels/slide{tmpl_n}.xml.rels"
    if rels_key in all_files:
        src_rels = etree.fromstring(all_files[rels_key])
        for rel in list(src_rels):
            if "notesSlide" in rel.get("Type", ""):
                src_rels.remove(rel)
        slide_rels_data[tmpl_n] = xml_bytes(src_rels)
    else:
        slide_rels_data[tmpl_n] = None

# ── 更新 presentation.xml.rels ──────────────────────────────
new_pres_rels = etree.fromstring(all_files["ppt/_rels/presentation.xml.rels"])
# 删除旧 slide 关系
for rel in list(new_pres_rels):
    t = rel.get("Type", "")
    if "slide" in t and "slideMaster" not in t and "slideLayout" not in t:
        new_pres_rels.remove(rel)
# 找最大 rId 数字
max_rid = 0
for rel in new_pres_rels:
    rid = rel.get("Id", "rId0")
    m = re.search(r"\d+", rid)
    if m:
        max_rid = max(max_rid, int(m.group()))
# 添加新 slide 关系
slide_rids = []
for i in range(1, new_slide_count + 1):
    max_rid += 1
    rid = f"rId{max_rid}"
    slide_rids.append(rid)
    rel = etree.SubElement(new_pres_rels, f"{{{PR}}}Relationship")
    rel.set("Id", rid)
    rel.set(
        "Type",
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide",
    )
    rel.set("Target", f"slides/slide{i}.xml")
new_pres_rels_bytes = xml_bytes(new_pres_rels)

# ── 更新 presentation.xml ────────────────────────────────────
new_pres_xml = etree.fromstring(all_files["ppt/presentation.xml"])
sldIdLst = new_pres_xml.find(f"{{{P}}}sldIdLst")
if sldIdLst is None:
    sldIdLst = etree.SubElement(new_pres_xml, f"{{{P}}}sldIdLst")
for child in list(sldIdLst):
    sldIdLst.remove(child)
for i, rid in enumerate(slide_rids):
    sldId = etree.SubElement(sldIdLst, f"{{{P}}}sldId")
    sldId.set("id", str(256 + i))
    sldId.set(f"{{{R}}}id", rid)
new_pres_xml_bytes = xml_bytes(new_pres_xml)

# ── 更新 [Content_Types].xml ─────────────────────────────────
ct_xml = etree.fromstring(all_files["[Content_Types].xml"])
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
for ov in list(ct_xml):
    pn = ov.get("PartName", "")
    if re.match(r"^/ppt/slides/slide\d+\.xml$", pn):
        ct_xml.remove(ov)
for i in range(1, new_slide_count + 1):
    ov = etree.SubElement(ct_xml, f"{{{CT_NS}}}Override")
    ov.set("PartName", f"/ppt/slides/slide{i}.xml")
    ov.set(
        "ContentType",
        "application/vnd.openxmlformats-officedocument.presentationml.slide+xml",
    )
new_ct_bytes = xml_bytes(ct_xml)

# ── 写出新 ZIP ────────────────────────────────────────────────
skip_patterns = re.compile(
    r"ppt/slides/slide\d+\.xml$|ppt/slides/_rels/slide\d+\.xml\.rels$"
    r"|ppt/_rels/presentation\.xml\.rels$|ppt/presentation\.xml$|\[Content_Types\]\.xml$"
)

out_buf = io.BytesIO()
out_zip = zipfile.ZipFile(out_buf, "w", zipfile.ZIP_DEFLATED)

# 复制不需修改的原始文件
for fname, data in all_files.items():
    if not skip_patterns.search(fname):
        out_zip.writestr(fname, data)

# 写入修改过的全局文件
out_zip.writestr("ppt/_rels/presentation.xml.rels", new_pres_rels_bytes)
out_zip.writestr("ppt/presentation.xml", new_pres_xml_bytes)
out_zip.writestr("[Content_Types].xml", new_ct_bytes)

# 写入新幻灯片 + 其 _rels
for new_idx, (key, tmpl_n) in enumerate(slides_plan, start=1):
    out_zip.writestr(f"ppt/slides/slide{new_idx}.xml", xml_bytes(slides[key]))
    rels_bytes = slide_rels_data.get(tmpl_n)
    if rels_bytes:
        out_zip.writestr(f"ppt/slides/_rels/slide{new_idx}.xml.rels", rels_bytes)

out_zip.close()

with open(DEST, "wb") as f:
    f.write(out_buf.getvalue())

print(f"Done! Generated: {DEST}")
print(f"Total slides: {new_slide_count}")
