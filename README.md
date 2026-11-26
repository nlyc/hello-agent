# agentflow · 自研智能体框架 v2.0

统一承载三种 Agent 范式（ReAct / Reflection / Plan-and-Solve）、低代码工作流、可替换 Model Layer，
并提供**工具热插拔、Context/Memory 管理、全链路 Trace 回调**三大新能力。

v2.0 在 v1.0 与 M8 记忆/RAG 之上新增：
**上下文管线**（ContextBuilder + NoteTool + TerminalTool）、**MCP / A2A 协议接入**、**评估基准与对比报告**。
需求见 [docs/v2.0-requirements.md](docs/v2.0-requirements.md)，评估见 [docs/v2.0-eval-report.md](docs/v2.0-eval-report.md)。

需求规划见仓库父目录 `自研框架v1.0-需求规划.md`；本仓库为落地实现。

## 分层

| 包 | 职责 |
|---|---|
| `core` | Message / Config / Agent 基类 / exceptions |
| `llm` | 可替换 Model Layer（公司中转站，OpenAI 兼容，换 model 即切换） |
| `tools` | Tool / ToolRegistry（热插拔）/ ToolChain / 异步执行器 / builtin |
| `agents` | ReActAgent / ReflectionAgent / PlanAndSolveAgent / FunctionCallAgent（均支持自动注入记忆上下文） |
| `memory` | 工作记忆（窗口+TTL）/ 长期记忆（SQLite 向量库）/ MemoryManager / Context/Memory 管理 |
| `retrieval` | 文档加载（txt/md/pdf/docx/URL）/ 标题感知分块 / Embedder / BM25 / SQLite·Qdrant / RRF 混合检索 / RAGPipeline |
| `context` | 上下文管线：GSSC / 六分区 / ContextPacket / Memory·RAG·History Source / 预算压缩 |
| `notes` | 项目式笔记：Markdown + YAML front matter / notes_index.json / 五类型 / NoteContextSource |
| `protocols` | MCP（JSON-RPC over stdio；Tools / Resources / Prompts + MCPTool 自动展开）/ A2A（AgentCard/Task/Artifact + A2ATool + SSE 任务事件） |
| `evaluation` | 数据集 / AST 工具调用 / QEM / LLM Judge / Win Rate / 报告与 EvaluationTool |
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

# v2.0：上下文工程 Demo（Terminal → Note → ContextBuilder，离线，48% token 节省）
py -3.14 -m examples.demo_context_engineering --out .cache/context_demo

# v2.0：MCP 发现 → 展开 → 注册 → Agent 调用（离线）
py -3.14 -m examples.demo_mcp_travel

# v2.0：A2A 三 Agent 协作（离线）
py -3.14 -m examples.demo_a2a_travel --out .cache/a2a_demo

# v2.0：评估基准（默认诚实负基线；在线系统需 LLM_API_KEY）
py -3.14 scripts/eval_v2.py --suite all --limit 5

# v2.0：在线对比 S0（v1.1）vs S1（ContextBuilder）；默认 limit=3 控制成本
py -3.14 scripts/eval_online_v2.py --suites all --limit 3 --out .cache/eval_online_full

# v2.0：S2–S5 组合验证（Notes / MCP / A2A）
py -3.14 scripts/eval_combo_v2.py --suite all --limit 6 --out .cache/eval_combo_s6

# v2.0：原生 Function Calling 对照（与 ReAct 同 12 题）
py -3.14 scripts/eval_online_v2.py --suites tool_calling --agent fc --limit 12 --out .cache/eval_online_fc

# 架构图 PNG（交付物 D2）
py -3.14 scripts/render_architecture.py

# 可选：开发模式安装，供任意脚本 import agentflow
py -3.14 -m pip install -e .
# 可选 extras：embed（FastEmbed 语义向量）/ rag（PDF & Word 解析）/ qdrant（远程向量库）
py -3.14 -m pip install -e ".[embed,rag]"
```

## 配置（.env）

所有在线能力统一从项目根目录 `.env` 读取（Shell 环境变量优先；改文件后热加载，无需重启）：

```powershell
Copy-Item .env.example .env    # 然后按需填写
```

| 能力 | 变量 |
|---|---|
| 在线 LLM | `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` / `LLM_TEMPERATURE` / `LLM_TIMEOUT` |
| 联网搜索 | `TAVILY_API_KEY` / `SERPAPI_API_KEY` / `SEARCH_MAX_RESULTS`（Tavily 优先，失败回落 SerpAPI）/ `SEARCH_FALLBACK`（知识库未命中时自动联网作答，默认开启） |
| 语义向量 | `EMBED_BACKEND`（`hashing`/`fastembed`）/ `FASTEMBED_MODEL` / `FASTEMBED_CACHE_PATH` / `HF_ENDPOINT`（默认 `https://hf-mirror.com`） |
| 检索与回答 | `RAG_MIN_SCORE`（向量召回相似度下限，0=关闭）/ `RAG_PLACE_GUARD`（地名覆盖校验）/ `RAG_EXTRA_PLACES`（追加地名，逗号/顿号/空格分隔）。地名校验先匹配内置城市与常见景点词表，再用“山/湖/古镇/博物馆/…”后缀启发式识别未收录景点；问题点名的地点在检索结果中不存在时直接拒答并触发联网兜底。静态词表无法穷举全国所有景点，冷门地名请写入 `RAG_EXTRA_PLACES` |
| 向量库 | `VECTOR_STORE`（`sqlite`/`qdrant`）/ `RAG_SQLITE_PATH` / `QDRANT_URL` / `QDRANT_API_KEY` / `QDRANT_COLLECTION` / `QDRANT_TIMEOUT` / `QDRANT_PREFER_GRPC` |
| 记忆 | `MEMORY_ENABLED` / `MEMORY_SQLITE_PATH` / `MEMORY_TOP_K` / `USER_ID` |
| Agent | `AGENT_MAX_STEPS` / `AGENT_MAX_TOOL_ITERATIONS` / `MAX_HISTORY_LENGTH` / `DEBUG` |
| 上下文 | `CONTEXT_MAX_TOKENS` / `CONTEXT_RESERVE_RATIO` / `CONTEXT_RELEVANCE_WEIGHT` / `CONTEXT_RECENCY_WEIGHT` / `CONTEXT_RECENT_HISTORY` / `CONTEXT_RECENCY_HALF_LIFE_HOURS` |
| 笔记 | `NOTES_DIR` / `USER_ID`（笔记按用户隔离，默认 `.data/notes/<user>/`） |
| 终端 | `TERMINAL_WORKSPACE`（默认沙箱由调用方显式传入；`allowed_commands` / 超时 / 截断可构造传参） |
| MCP | server 命令由 `MCPTool(command=[...])` 显式传入；连接/调用超时可构造传参 |
| A2A | 默认 `127.0.0.1:0`；`A2A_TOKEN` / 超时 / 1MB 上限可构造传参 |
| 评估 | 数据集默认 `data/eval/`；`EVAL_JUDGE_MODEL` / `EVAL_JUDGE_BASE_URL` / `EVAL_JUDGE_API_KEY` 可配置独立 Judge；`EVAL_TEMPERATURE` 只影响在线 Judge |

`.env` 已被 `.gitignore` 忽略，不会提交；`.env.example` 只保留占位值。
Web 控制台的“检索与在线能力”卡片可直接切换 Embedding 后端与向量库并自动重建索引，
选择保存在 `.data/webui/settings.json`；不同后端使用独立的 SQLite 文件（`rag-<后端>.sqlite3`）
和 Qdrant 集合（`<collection>_<后端>`），不会混用向量维度。

## Web 控制台

`examples/web_app.py` 用标准库 `ThreadingHTTPServer` 提供零依赖的本地控制台（仅监听 `127.0.0.1`），
所有数据和产物保存在 `.data/webui/`，重启后文档索引、长期记忆与历史 Demo 运行都会保留。

| 页签 | 能做什么 |
|---|---|
| 概览 | 索引 / 记忆统计、运行环境、检索与在线能力（切换 Embedding / 向量库、查看 LLM/搜索/Qdrant 配置与 SQLite 占用）、快捷入口；可关闭“写入内置示例”（4 篇知识文档 + 2 条演示记忆），关闭后立即移除并在重启后保持关闭，也可用 `--no-seed` 启动 |
| 知识库 / RAG | 上传 MD/TXT/PDF/DOCX、公网 URL 入库、删除文档、混合 / BM25 / 向量检索、带引用问答（自动 / 在线 LLM / 仅检索）；问题点名但知识库未收录的城市或景点会明确提示，不用其它地点凑答案；若配置了搜索 Key 且 `SEARCH_FALLBACK=true`，会自动联网检索并由在线 LLM 总结（回答标注“答案来源：联网搜索”） |
| 记忆 | 写入 semantic/episodic/procedural 记忆、记录对话轮次、检索、构建注入上下文、清空；首次启动的 2 条北京演示记忆可通过“概览”开关移除 |
| 上下文 | 用 ContextBuilder 构建六分区上下文预览，展示 tokens、分区、packets 与丢弃统计 |
| 笔记 | 新建 / 列出 / 删除 Markdown 笔记，支持 blocker、action、conclusion 等类型 |
| 协议 | 发现自建旅行 MCP server 的工具 / 资源 / Prompts；读取 A2A 资料员/规划师/审查员 AgentCard |
| v2 评估 | 运行 `scripts/eval_v2.py` 离线负基线（tool_calling / qa / quality / all） |
| 三范式 | ReAct / Reflection / Plan-and-Solve 无记忆 vs 有记忆对照；始终使用在线 LLM，ReAct 的天气工具通过 wttr.in 观测接口取实时气温（不可用时降级为搜索摘要并标注），景点工具直接联网查询，Reflection / Plan-and-Solve 注入同一份实时检索资料；“有记忆”读取“记忆”页签中的真实长期记忆，且只读不写回；运行完成后播放一次提示音；需要配置 `LLM_API_KEY` 与 `TAVILY_API_KEY`（或 SerpAPI） |
| 完整 Demo | 全在线模式：基于你输入的问题与当前知识库检索，由在线 LLM 生成带引用的回答，知识库未命中时自动联网补充；报告只包含本次问题的真实结果，不再输出固定北京模拟章节；可勾选“并入当前知识库”把上传文档加入临时索引；运行完成后播放一次提示音；需要 `LLM_API_KEY` |
| RAG 评估 | 运行 `scripts/eval_rag.py` 的固定 benchmark（`data/eval/travel_qa.jsonl` + 内置四城语料），展示 recall@k / MRR / 事实命中 / 引用覆盖；不评估控制台上传的文档 |
| 测试 | 列出并运行 `tests/test_*.py`（子进程 + 白名单校验） |

接口为同源 JSON API（`/api/status`、`/api/settings`、`/api/settings/embed`、`/api/settings/vector`、
`/api/settings/seed`、`/api/answer`、`/api/ingest`、`/api/search`、`/api/memory/*`、
`/api/paradigms/run`、`/api/demo/run`、`/api/eval/run`、`/api/tests/run` 等），也可直接用 `curl` 调用。
`/api/paradigms/run` 默认 `{"online": true}`（在线 LLM + 实时检索），未配置 `LLM_API_KEY` 时返回 400；
`/api/demo/run` 默认在线模式，必须提供 `query`，并支持 `{"use_console_kb": true}` 并入控制台上传的文档；
`/api/answer` 的 `llm_mode` 支持 `auto` / `online` / `none`，离线回答模式已移除。

## 交付物清单

| # | 交付物 | 位置 |
|---|---|---|
| D1 | 代码仓库（零外部框架依赖） | 本仓库 |
| D2 | 架构图（Mermaid + PNG） | [`docs/architecture.md`](docs/architecture.md) + [`docs/architecture.png`](docs/architecture.png) |
| D3 | 单元测试（451 个通过 / 4 跳过，全绿） | [`tests/`](tests/) |
| D4 | 端到端演示 | [`examples/demo_e2e.py`](examples/demo_e2e.py) |
| D5 | API 设计文档 | [`docs/api-design.md`](docs/api-design.md) |
| D6 | 三范式轨迹对比报告 | [`docs/trajectory-comparison.md`](docs/trajectory-comparison.md) |
| D7 | M8 记忆 + RAG 演示 | [`examples/demo_travel_memory_rag.py`](examples/demo_travel_memory_rag.py) |
| D8 | M8 召回与回答评估报告 | [`docs/m8-memory-rag-eval.md`](docs/m8-memory-rag-eval.md) |
| D9 | 上下文管线（ContextBuilder + NoteTool + TerminalTool） | [`agentflow/context/`](agentflow/context/) · [`agentflow/notes/`](agentflow/notes/) · [`agentflow/tools/terminal_tool.py`](agentflow/tools/terminal_tool.py) · [`examples/demo_context_engineering.py`](examples/demo_context_engineering.py) |
| D10 | MCP 接入与自建旅行 server | [`agentflow/protocols/mcp/`](agentflow/protocols/mcp/) · [`examples/mcp_server_travel.py`](examples/mcp_server_travel.py) · [`examples/demo_mcp_travel.py`](examples/demo_mcp_travel.py) |
| D11 | A2A 协作 Demo | [`agentflow/protocols/a2a/`](agentflow/protocols/a2a/) · [`examples/demo_a2a_travel.py`](examples/demo_a2a_travel.py) |
| D12 | 评估基准集与评估工具 | [`agentflow/evaluation/`](agentflow/evaluation/) · [`scripts/eval_v2.py`](scripts/eval_v2.py) · [`data/eval/`](data/eval/) |
| D13 | v2.0 对比评估报告 | [`docs/v2.0-eval-report.md`](docs/v2.0-eval-report.md) |
| D14 | v2.0 交付检查 | [`docs/v2.0-delivery-check.md`](docs/v2.0-delivery-check.md) |

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
- [x] M8.7 浏览器 Web 控制台与 Demo 完整报告输出
- [x] M8.8 统一 .env 配置、在线 LLM 回答、Embedding / 向量库在线切换、远程 Qdrant 接入
- [x] M8.9 检索相关性阈值与地名覆盖校验
- [x] M8.10 知识库未命中时自动联网检索作答
- [x] M8.11 控制台自定义数据模式：内置示例开关、在线三范式与 Demo 并入当前知识库
- [x] M8.12 在线模式真实数据：三范式实时天气 / 景点工具与 Demo 自定义问题在线作答
- [x] M8.13 全在线演示：移除离线回答与固定 Demo 章节；未收录景点启发式识别并联网兜底
- [x] M8.14 三范式真实数据修正：天气改用观测接口、记忆对照读取真实长期记忆（只读）
- [x] M8.15 三范式 / 完整 Demo 完成提示音（`878253__qvantor__ui-pop-up-open.wav`）
- [x] M9.0 v2.0 需求冻结、任务提示词、测试与 RAG 基线
- [x] M9.1 上下文基础层：ContextPacket/Config、GSSC Builder、四类 Source、分区压缩
- [x] M9.2 笔记系统：NoteStore / NoteTool / NoteContextSource（blocker 优先）
- [x] M9.3 TerminalTool：白名单 / 沙箱 / 超时 / 截断 + shell=False 加固
- [x] M9.4 Agent 集成：`context_builder` 可选注入三范式，默认行为不变
- [x] M9.5 MCP：JSON-RPC over stdio、工具自动展开、自建旅行 server、离线 Demo
- [x] M9.6 A2A：AgentCard/Task/Artifact、server/client/A2ATool、三 Agent Demo
- [x] M9.7 评估：mini-BFCL、旅行 QA、行程质量、LLM Judge / Win Rate、EvaluationTool
- [x] M9.8 对比报告（离线基线；在线 S2–S5 需 LLM_API_KEY）
- [x] M9.9 回归、文档、版本与交付检查
