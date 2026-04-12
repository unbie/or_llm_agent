# 🚀 AI 学习路线图：大模型 + AI Agent + Vibe Coding

> **创建日期**: 2026-03-31  
> **适用对象**: 有编程基础，想系统学习 AI 应用开发的开发者  
> **预计学习周期**: 3-6 个月

---

## 📋 目录

1. [学习路线总览](#学习路线总览)
2. [第一阶段：大模型基础](#第一阶段大模型基础2-4周)
3. [第二阶段：Prompt Engineering](#第二阶段prompt-engineering1-2周)
4. [第三阶段：RAG 技术](#第三阶段rag-技术2-3周)
5. [第四阶段：AI Agent 开发](#第四阶段ai-agent-开发3-4周)
6. [第五阶段：MCP 协议](#第五阶段mcp-协议2周)
7. [第六阶段：Vibe Coding](#第六阶段vibe-coding持续实践)
8. [推荐学习资源汇总](#推荐学习资源汇总)
9. [实战项目清单](#实战项目清单)
10. [学习建议](#学习建议)

---

## 学习路线总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AI 应用开发学习路线                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [阶段1] 大模型基础 ──▶ [阶段2] Prompt工程 ──▶ [阶段3] RAG技术          │
│      2-4周                  1-2周                 2-3周                 │
│        │                      │                     │                   │
│        ▼                      ▼                     ▼                   │
│   API调用、模型原理      提示词设计、优化      检索增强、向量数据库       │
│                                                                         │
│  [阶段4] AI Agent ────▶ [阶段5] MCP协议 ────▶ [阶段6] Vibe Coding       │
│      3-4周                  2周                   持续实践               │
│        │                      │                     │                   │
│        ▼                      ▼                     ▼                   │
│   工具调用、多Agent      协议标准、工具集成     AI辅助编程、自动化        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 第一阶段：大模型基础（2-4周）

### 🎯 学习目标
- 理解大模型的基本原理（Transformer、注意力机制）
- 掌握主流大模型 API 的调用方法
- 能够本地部署开源模型

### 📚 核心学习资源

| 资源 | 类型 | 链接 | 说明 |
|------|------|------|------|
| **llm-universe** | 教程 | [datawhalechina/llm-universe](https://github.com/datawhalechina/llm-universe) ⭐12.4k | 🔥 中文入门首选，面向小白 |
| **base-llm** | 教程 | [datawhalechina/base-llm](https://github.com/datawhalechina/base-llm) | 从 NLP 到 LLM 的算法全栈 |
| **handy-ollama** | 教程 | [datawhalechina/handy-ollama](https://github.com/datawhalechina/handy-ollama) ⭐2.3k | CPU 玩转本地大模型部署 |
| 吴恩达 LLM 课程 | 视频 | [DeepLearning.AI](https://www.deeplearning.ai/short-courses/) | 免费短课程，英文 |
| 李宏毅 ML 课程 | 视频 | [YouTube/B站](https://www.youtube.com/@HungyiLeeNTU) | 中文讲解，深入原理 |

### 📝 学习任务

- [ ] 完成 llm-universe 前4章
- [ ] 使用 Ollama 本地部署 Qwen/Llama 模型
- [ ] 调用 OpenAI / DeepSeek / 通义千问 API
- [ ] 理解 Token、Temperature、Top-p 等参数含义

### 💻 实践代码

```python
# OpenAI API 调用示例
from openai import OpenAI

client = OpenAI(api_key="your-key")
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

---

## 第二阶段：Prompt Engineering（1-2周）

### 🎯 学习目标
- 掌握 Prompt 设计的核心技巧
- 理解 Few-shot、CoT、ReAct 等方法
- 能够针对具体任务优化 Prompt

### 📚 核心学习资源

| 资源 | 类型 | 链接 | 说明 |
|------|------|------|------|
| **Prompt Engineering Guide** | 指南 | [dair-ai/Prompt-Engineering-Guide](https://github.com/dair-ai/Prompt-Engineering-Guide) ⭐72.5k | 🔥 最全面的 Prompt 工程指南 |
| OpenAI 官方指南 | 文档 | [platform.openai.com](https://platform.openai.com/docs/guides/prompt-engineering) | 官方最佳实践 |
| 鱼皮 AI 指南 | 教程 | [liyupi/ai-guide](https://github.com/liyupi/ai-guide) ⭐10.8k | 中文 Prompt 大全 |

### 📝 核心技巧

```
1. Zero-shot: 直接提问
2. Few-shot: 给出示例
3. Chain-of-Thought (CoT): 让模型一步步思考
4. ReAct: 思考-行动-观察循环
5. Self-Consistency: 多次采样取最优
```

### 💡 Prompt 模板示例

```markdown
# 角色设定
你是一个专业的Python程序员。

# 任务描述
请帮我实现一个函数，要求：
1. 输入：一个整数列表
2. 输出：去重后的排序列表
3. 时间复杂度要求 O(n log n)

# 输出格式
请提供：
1. 完整代码
2. 时间复杂度分析
3. 测试用例

# 示例
输入: [3, 1, 2, 3, 1]
输出: [1, 2, 3]
```

---

## 第三阶段：RAG 技术（2-3周）

### 🎯 学习目标
- 理解 RAG（检索增强生成）的原理
- 掌握向量数据库的使用
- 能够构建知识库问答系统

### 📚 核心学习资源

| 资源 | 类型 | 链接 | 说明 |
|------|------|------|------|
| **all-in-rag** | 教程 | [datawhalechina/all-in-rag](https://github.com/datawhalechina/all-in-rag) ⭐5.5k | 🔥 RAG 技术全栈指南 |
| **AgentGuide** | 教程 | [adongwanai/AgentGuide](https://github.com/adongwanai/AgentGuide) ⭐3.1k | AI Agent + 高级 RAG |
| LangChain 文档 | 文档 | [python.langchain.com](https://python.langchain.com/docs/) | 官方文档 |

### 🔧 技术栈

```
文档处理: LangChain, LlamaIndex
向量数据库: Milvus, Chroma, Faiss, Pinecone
Embedding: OpenAI, BGE, M3E
```

### 📝 学习任务

- [ ] 理解 Embedding 和向量检索原理
- [ ] 使用 Chroma/Milvus 搭建向量数据库
- [ ] 实现一个文档问答系统
- [ ] 学习高级 RAG 技术（HyDE、Rerank、GraphRAG）

### 💻 实践代码

```python
# 简单 RAG 示例
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 创建向量存储
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings()
)

# 检索相关文档
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
relevant_docs = retriever.invoke("你的问题")
```

---

## 第四阶段：AI Agent 开发（3-4周）

### 🎯 学习目标
- 理解 AI Agent 的架构和原理
- 掌握主流 Agent 框架的使用
- 能够构建具有工具调用能力的 Agent

### 📚 核心学习资源

| 资源 | ⭐ Stars | 链接 | 说明 |
|------|---------|------|------|
| **LangChain** | 132k | [langchain-ai/langchain](https://github.com/langchain-ai/langchain) | 🔥 Agent 开发首选框架 |
| **LangGraph** | 28k | [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) | Agent 工作流编排 |
| **MetaGPT** | 66k | [FoundationAgents/MetaGPT](https://github.com/FoundationAgents/MetaGPT) | 多 Agent 协作框架 |
| **AutoGen** | 57k | [microsoft/autogen](https://github.com/microsoft/autogen) | 微软出品 |
| **CrewAI** | 48k | [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI) | 角色扮演 Agent |
| **OpenAI Agents** | 20k | [openai/openai-agents-python](https://github.com/openai/openai-agents-python) | OpenAI 官方框架 |
| **12-factor-agents** | 19k | [humanlayer/12-factor-agents](https://github.com/humanlayer/12-factor-agents) | Agent 设计原则 |

### 🧠 Agent 核心概念

```
┌─────────────────────────────────────────────────┐
│                   AI Agent                       │
├─────────────────────────────────────────────────┤
│                                                 │
│  [感知] ──▶ [规划] ──▶ [决策] ──▶ [执行]        │
│     │         │          │          │           │
│     ▼         ▼          ▼          ▼           │
│   输入      思考链     工具选择   行动执行       │
│   理解      分解任务   API调用    结果反馈       │
│                                                 │
│  核心组件:                                       │
│  • LLM (大脑)                                   │
│  • Tools (工具集)                               │
│  • Memory (记忆)                                │
│  • Planning (规划)                              │
│                                                 │
└─────────────────────────────────────────────────┘
```

### 📝 学习任务

- [ ] 理解 ReAct 模式和 Function Calling
- [ ] 使用 LangChain 创建简单 Agent
- [ ] 学习 LangGraph 构建复杂工作流
- [ ] 实现一个多 Agent 协作系统

### 💻 实践代码

```python
# LangChain Agent 示例
from langchain.agents import create_react_agent, Tool
from langchain_openai import ChatOpenAI

# 定义工具
tools = [
    Tool(
        name="Search",
        func=search_function,
        description="搜索互联网信息"
    ),
    Tool(
        name="Calculator",
        func=calculator_function,
        description="数学计算"
    )
]

# 创建 Agent
llm = ChatOpenAI(model="gpt-4")
agent = create_react_agent(llm, tools, prompt)
```

---

## 第五阶段：MCP 协议（2周）

### 🎯 学习目标
- 理解 MCP（Model Context Protocol）标准
- 学会使用和开发 MCP Server
- 将工具能力扩展到 AI 助手

### 📚 核心学习资源

| 资源 | ⭐ Stars | 链接 | 说明 |
|------|---------|------|------|
| **mcp-for-beginners** | 15.7k | [microsoft/mcp-for-beginners](https://github.com/microsoft/mcp-for-beginners) | 🔥 微软官方 MCP 入门教程 |
| **awesome-mcp-servers** | 5.3k | [appcypher/awesome-mcp-servers](https://github.com/appcypher/awesome-mcp-servers) | MCP Server 列表 |
| **mcp-agent** | 8.2k | [lastmile-ai/mcp-agent](https://github.com/lastmile-ai/mcp-agent) | MCP + Agent 结合 |
| **fastapi_mcp** | 11.7k | [tadata-org/fastapi_mcp](https://github.com/tadata-org/fastapi_mcp) | FastAPI 转 MCP Server |
| MCP 官方规范 | 文档 | [modelcontextprotocol.io](https://modelcontextprotocol.io/) | 协议规范 |

### 🔧 MCP 架构

```
┌─────────────┐     MCP协议      ┌─────────────┐
│  AI 助手     │ ◀───────────▶  │ MCP Server  │
│ (Claude等)   │                │ (工具提供者) │
└─────────────┘                 └─────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
              ┌─────────┐      ┌─────────┐      ┌─────────┐
              │ GitHub  │      │ Database│      │ Browser │
              │ 操作    │      │ 查询    │      │ 自动化  │
              └─────────┘      └─────────┘      └─────────┘
```

### 📝 学习任务

- [ ] 完成 mcp-for-beginners 教程
- [ ] 在 Claude Desktop 配置 MCP Server
- [ ] 开发一个自定义 MCP Server
- [ ] 理解 MCP 与 Function Calling 的区别

---

## 第六阶段：Vibe Coding（持续实践）

### 🎯 学习目标
- 掌握 AI 辅助编程的最佳实践
- 熟练使用 Cursor / Copilot / Claude Code
- 能够高效地与 AI 协作完成项目

### 📚 核心学习资源

| 资源 | ⭐ Stars | 链接 | 说明 |
|------|---------|------|------|
| **ai-guide** | 10.8k | [liyupi/ai-guide](https://github.com/liyupi/ai-guide) | 🔥 鱼皮 AI + Vibe Coding 教程 |
| **easy-vibe** | 4.8k | [datawhalechina/easy-vibe](https://github.com/datawhalechina/easy-vibe) | 🔥 Vibe Coding 系统教程 |
| **awesome-vibe-coding** | 3.8k | [filipecalegario/awesome-vibe-coding](https://github.com/filipecalegario/awesome-vibe-coding) | Vibe Coding 资源汇总 |
| **rulebook-ai** | 585 | [botingw/rulebook-ai](https://github.com/botingw/rulebook-ai) | AI 编程规则模板 |

### 🛠️ 主流 AI 编程工具

| 工具 | 特点 | 适用场景 |
|------|------|----------|
| **Cursor** | IDE 集成、Agent 模式 | 日常开发首选 |
| **GitHub Copilot** | VS Code 深度集成 | 代码补全 |
| **Claude Code** | 终端 Agent | 复杂任务 |
| **Windsurf** | 免费、功能全面 | 预算有限 |
| **Lovable/v0** | UI 生成 | 前端原型 |

### 💡 Vibe Coding 核心原则

```
1. 🎯 明确目标: 清晰描述你想要什么
2. 📝 提供上下文: 给 AI 足够的背景信息
3. 🔄 迭代优化: 逐步完善，不求一步到位
4. ✅ 及时验证: 检查 AI 生成的代码
5. 📚 学习理解: 理解 AI 写的代码，而非盲目使用
```

### 📝 学习任务

- [ ] 安装并配置 Cursor
- [ ] 完成 easy-vibe 教程
- [ ] 使用 AI 完成一个完整项目
- [ ] 学习编写 .cursorrules / CLAUDE.md

### 💻 工作流示例

```markdown
# 使用 Cursor 开发项目的工作流

1. 创建 .cursorrules 文件定义项目规范
2. 用自然语言描述需求
3. AI 生成代码框架
4. Review 并调整
5. 迭代完善功能
6. AI 辅助写测试
7. AI 辅助写文档
```

---

## 推荐学习资源汇总

### 📖 中文教程（推荐）

| 教程 | Stars | 内容 |
|------|-------|------|
| [llm-universe](https://github.com/datawhalechina/llm-universe) | 12.4k | LLM 应用开发入门 |
| [all-in-rag](https://github.com/datawhalechina/all-in-rag) | 5.5k | RAG 技术全栈 |
| [easy-vibe](https://github.com/datawhalechina/easy-vibe) | 4.8k | Vibe Coding 教程 |
| [ai-guide](https://github.com/liyupi/ai-guide) | 10.8k | AI 资源大全 |
| [AgentGuide](https://github.com/adongwanai/AgentGuide) | 3.1k | Agent 开发指南 |
| [handy-ollama](https://github.com/datawhalechina/handy-ollama) | 2.3k | 本地大模型部署 |

### 📖 英文教程

| 教程 | Stars | 内容 |
|------|-------|------|
| [Prompt-Engineering-Guide](https://github.com/dair-ai/Prompt-Engineering-Guide) | 72.5k | Prompt 工程指南 |
| [mcp-for-beginners](https://github.com/microsoft/mcp-for-beginners) | 15.7k | MCP 入门 |
| [12-factor-agents](https://github.com/humanlayer/12-factor-agents) | 19k | Agent 设计原则 |

### 🔧 框架和工具

| 框架 | Stars | 用途 |
|------|-------|------|
| [LangChain](https://github.com/langchain-ai/langchain) | 132k | Agent 开发 |
| [LangGraph](https://github.com/langchain-ai/langgraph) | 28k | 工作流编排 |
| [MetaGPT](https://github.com/FoundationAgents/MetaGPT) | 66k | 多 Agent |
| [AutoGen](https://github.com/microsoft/autogen) | 57k | Agent 框架 |
| [CrewAI](https://github.com/crewAIInc/crewAI) | 48k | 角色 Agent |

---

## 实战项目清单

### 🌟 入门项目

| 项目 | 难度 | 技术点 | 预计时间 |
|------|------|--------|----------|
| 命令行 AI 助手 | ⭐ | API 调用 | 1天 |
| Markdown 翻译器 | ⭐ | Prompt 设计 | 1天 |
| 代码解释器 | ⭐⭐ | Prompt + 解析 | 2天 |

### 🌟 进阶项目

| 项目 | 难度 | 技术点 | 预计时间 |
|------|------|--------|----------|
| 文档问答机器人 | ⭐⭐ | RAG | 1周 |
| SQL 生成器 | ⭐⭐ | Few-shot | 3天 |
| 自动代码审查 | ⭐⭐⭐ | Agent + Git | 1周 |

### 🌟 高级项目

| 项目 | 难度 | 技术点 | 预计时间 |
|------|------|--------|----------|
| 多 Agent 协作系统 | ⭐⭐⭐ | Multi-Agent | 2周 |
| 自定义 MCP Server | ⭐⭐⭐ | MCP 协议 | 1周 |
| AI 驱动的自动化工作流 | ⭐⭐⭐⭐ | LangGraph | 2周 |
| 本地知识库 + Agent | ⭐⭐⭐⭐ | RAG + Agent | 3周 |

---

## 学习建议

### ⏰ 时间分配建议

```
每周学习时间: 10-15 小时

分配:
├── 理论学习: 30% (看教程、读文档)
├── 动手实践: 50% (写代码、做项目)
└── 复盘总结: 20% (整理笔记、写博客)
```

### 💡 学习心得

1. **边学边做**: 不要只看不练，每学一个概念就写代码验证
2. **项目驱动**: 带着具体目标学习效率更高
3. **社区交流**: 加入 Discord/微信群，和他人讨论
4. **持续更新**: AI 领域变化快，保持关注最新动态
5. **理解原理**: 不要只会调 API，理解背后原理才能走得更远

### 📅 建议学习计划

| 周次 | 内容 | 产出 |
|------|------|------|
| 1-2 | 大模型基础 | 能调用各种 API |
| 3 | Prompt Engineering | 完成一个 Prompt 项目 |
| 4-5 | RAG 技术 | 搭建文档问答系统 |
| 6-8 | AI Agent | 完成一个 Agent 项目 |
| 9-10 | MCP + Vibe Coding | 日常使用 AI 辅助编程 |
| 11+ | 综合项目 | 持续实践和深入 |

---

## 🔗 快速链接

### 必读资源
- 📚 [llm-universe](https://github.com/datawhalechina/llm-universe) - 中文 LLM 入门
- 📚 [Prompt-Engineering-Guide](https://github.com/dair-ai/Prompt-Engineering-Guide) - Prompt 工程
- 📚 [easy-vibe](https://github.com/datawhalechina/easy-vibe) - Vibe Coding

### 官方文档
- 📖 [LangChain Docs](https://python.langchain.com/docs/)
- 📖 [OpenAI Docs](https://platform.openai.com/docs/)
- 📖 [MCP Spec](https://modelcontextprotocol.io/)

### 社区
- 💬 [Hugging Face](https://huggingface.co/)
- 💬 [Reddit r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)
- 💬 [DataWhale](https://datawhale.club/)

---

> 💪 **记住**: 最好的学习方式是动手实践！选择一个感兴趣的项目，边做边学。
> 
> 📧 有问题随时问 AI 助手帮你解答！
