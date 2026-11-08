# agentflow · 自研智能体框架 v1.0

统一承载三种 Agent 范式（ReAct / Reflection / Plan-and-Solve）、低代码工作流、可替换 Model Layer，
并提供**工具热插拔、Context/Memory 管理、全链路 Trace 回调**三大新能力。

需求规划见仓库父目录 `自研框架v1.0-需求规划.md`；本仓库为落地实现。

## 分层

| 包 | 职责 |
|---|---|
| `core` | Message / Config / Agent 基类 / exceptions |
| `llm` | 可替换 Model Layer（公司中转站，OpenAI 兼容，换 model 即切换） |
| `tools` | Tool / ToolRegistry（热插拔）/ ToolChain / 异步执行器 / builtin |
| `agents` | ReActAgent / ReflectionAgent / PlanAndSolveAgent（均支持自动注入记忆上下文） |
| `memory` | 工作记忆（窗口+TTL）/ 长期记忆（SQLite 向量库）/ MemoryManager / Context/Memory 管理 |
| `retrieval` | 文档加载（txt/md/pdf/docx/URL）/ 标题感知分块 / Embedder / BM25 / SQLite·Qdrant / RRF 混合检索 / RAGPipeline |
| `trace` | 全链路 Trace 回调（TraceEvent / Tracer / 订阅） |
| `workflow` | 低代码工作流（自研轻量 StateGraph，不依赖 LangGraph） |
| `utils` | helpers / logging / serialization |

## 快速开始

```powershell
# 运行全部测试（无需安装，离线）
py -3.14 -m pytest

# 端到端演示：四形态 + 工具热插拔 + Memory 持久化 + Trace 落盘（离线，不联网）
py -3.14 -m examples.demo_e2e

# M8 Demo：三范式记忆注入 + PDF/DOCX/URL 入库 + 带/不带 RAG 对照（离线）
py -3.14 -m examples.demo_travel_memory_rag --out-dir demo_output

# M8 评估：召回命中率 + 回答层指标，生成 docs/m8-memory-rag-eval.md
py -3.14 scripts/eval_rag.py --out docs/m8-memory-rag-eval.md

# Web 控制台：浏览器里操作全部能力（文档入库 / 检索 / 记忆 / 三范式 / Demo / 评估 / 测试）
py -3.14 -m examples.web_app            # 默认 http://127.0.0.1:8000
py -3.14 -m examples.web_app --port 8080 --no-browser

# 三范式轨迹对比（生成 JSONL + 对比表，驱动 D6 报告）
py -3.14 scripts/gen_trajectory_report.py

# 架构图 PNG（交付物 D2）
py -3.14 scripts/render_architecture.py

# 可选：开发模式安装，供任意脚本 import agentflow
py -3.14 -m pip install -e .
# 可选 extras：embed（FastEmbed 语义向量）/ rag（PDF & Word 解析）/ qdrant（远程向量库）
py -3.14 -m pip install -e ".[embed,rag]"
```

## Web 控制台

`examples/web_app.py` 用标准库 `ThreadingHTTPServer` 提供零依赖的本地控制台（仅监听 `127.0.0.1`），
所有数据和产物保存在 `.data/webui/`，重启后文档索引、长期记忆与历史 Demo 运行都会保留。

| 页签 | 能做什么 |
|---|---|
| 概览 | 索引 / 记忆统计、运行环境、快捷入口 |
| 知识库 / RAG | 上传 MD/TXT/PDF/DOCX、公网 URL 入库、删除文档、混合 / BM25 / 向量检索、带引用问答 |
| 记忆 | 写入 semantic/episodic/procedural 记忆、记录对话轮次、检索、构建注入上下文、清空 |
| 三范式 | ReAct / Reflection / Plan-and-Solve 无记忆 vs 有记忆对照（使用隔离的临时记忆） |
| 完整 Demo | 调用 `demo_travel_memory_rag`，查看报告与产物下载 |
| RAG 评估 | 运行 `scripts/eval_rag.py` 的后端评估，展示 recall@k / MRR / 事实命中 / 引用覆盖 |
| 测试 | 列出并运行 `tests/test_*.py`（子进程 + 白名单校验） |

接口为同源 JSON API（`/api/status`、`/api/ingest`、`/api/search`、`/api/memory/*`、`/api/paradigms/run`、
`/api/demo/run`、`/api/eval/run`、`/api/tests/run` 等），也可直接用 `curl` 调用。

## 交付物清单

| # | 交付物 | 位置 |
|---|---|---|
| D1 | 代码仓库（零外部框架依赖） | 本仓库 |
| D2 | 架构图（Mermaid + PNG） | [`docs/architecture.md`](docs/architecture.md) + [`docs/architecture.png`](docs/architecture.png) |
| D3 | 单元测试（248 个通过 / 2 跳过，全绿） | [`tests/`](tests/) |
| D4 | 端到端演示 | [`examples/demo_e2e.py`](examples/demo_e2e.py) |
| D5 | API 设计文档 | [`docs/api-design.md`](docs/api-design.md) |
| D6 | 三范式轨迹对比报告 | [`docs/trajectory-comparison.md`](docs/trajectory-comparison.md) |
| D7 | M8 记忆 + RAG 演示 | [`examples/demo_travel_memory_rag.py`](examples/demo_travel_memory_rag.py) |
| D8 | M8 召回与回答评估报告 | [`docs/m8-memory-rag-eval.md`](docs/m8-memory-rag-eval.md) |

## 里程碑

- [x] M0 脚手架：包结构 + pyproject + pytest 空跑
- [x] M1 core + llm：Message/Config/Agent 基类 + BaseLLM/公司中转站/模型注册表
- [x] M2 tools：热插拔 ToolRegistry + ToolChain + 异步执行器 + 安全计算器/降级搜索
- [x] M3 agents：三范式（ReAct / Reflection / PlanAndSolve）
- [x] M4 memory + trace：MessageHistory + 可插拔后端 + TraceEvent/Tracer/回调
- [x] M5 workflow：自研 StateGraph + 旅行助手 verify 矩阵
- [x] M6 端到端演示：四形态 + 热插拔 + Memory + Trace
- [x] M7 文档：API 设计 + 架构图 + 轨迹对比报告
- [x] M8.1 检索基础层：Embedder / BM25 / SQLite 向量库 / RRF 混合检索
- [x] M8.2 文档加载 / 标题感知分块 / RAG 管线 / Qdrant 适配
- [x] M8.3 记忆：工作记忆 / 长期记忆 / MemoryManager / MemoryTool
- [x] M8.4 三范式自动注入记忆上下文
- [x] M8.5 记忆 + RAG 演示、上传 PDF/DOCX/URL、召回与回答评估报告
- [x] M8.6 回归与交付检查（[`docs/m8-delivery-check.md`](docs/m8-delivery-check.md)）
