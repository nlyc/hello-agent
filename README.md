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
| `agents` | ReActAgent / ReflectionAgent / PlanAndSolveAgent |
| `memory` | Context/Memory 管理（历史 + 可插拔持久化后端） |
| `trace` | 全链路 Trace 回调（TraceEvent / Tracer / 订阅） |
| `workflow` | 低代码工作流（自研轻量 StateGraph，不依赖 LangGraph） |
| `utils` | helpers / logging / serialization |

## 快速开始

```powershell
# 运行全部测试（94 个，无需安装）
py -3.14 -m pytest

# 端到端演示：四形态 + 工具热插拔 + Memory 持久化 + Trace 落盘（离线，不联网）
py -3.14 -m examples.demo_e2e

# 三范式轨迹对比（生成 JSONL + 对比表，驱动 D6 报告）
py -3.14 scripts/gen_trajectory_report.py

# 架构图 PNG（交付物 D2）
py -3.14 scripts/render_architecture.py

# 可选：开发模式安装，供任意脚本 import agentflow
py -3.14 -m pip install -e .
```

## 交付物清单

| # | 交付物 | 位置 |
|---|---|---|
| D1 | 代码仓库（零外部框架依赖） | 本仓库 |
| D2 | 架构图（Mermaid + PNG） | [`docs/architecture.md`](docs/architecture.md) + [`docs/architecture.png`](docs/architecture.png) |
| D3 | 单元测试（94 个，全绿） | [`tests/`](tests/) |
| D4 | 端到端演示 | [`examples/demo_e2e.py`](examples/demo_e2e.py) |
| D5 | API 设计文档 | [`docs/api-design.md`](docs/api-design.md) |
| D6 | 三范式轨迹对比报告 | [`docs/trajectory-comparison.md`](docs/trajectory-comparison.md) |

## 里程碑

- [x] M0 脚手架：包结构 + pyproject + pytest 空跑
- [x] M1 core + llm：Message/Config/Agent 基类 + BaseLLM/公司中转站/模型注册表
- [x] M2 tools：热插拔 ToolRegistry + ToolChain + 异步执行器 + 安全计算器/降级搜索
- [x] M3 agents：三范式（ReAct / Reflection / PlanAndSolve）
- [x] M4 memory + trace：MessageHistory + 可插拔后端 + TraceEvent/Tracer/回调
- [x] M5 workflow：自研 StateGraph + 旅行助手 verify 矩阵
- [x] M6 端到端演示：四形态 + 热插拔 + Memory + Trace
- [x] M7 文档：API 设计 + 架构图 + 轨迹对比报告
