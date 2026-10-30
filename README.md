# agentflow · 自研智能体框架 v1.0

统一承载三种 Agent 范式（ReAct / Reflection / Plan-and-Solve）、低代码工作流、可替换 Model Layer，
并提供工具热插拔、Context/Memory 管理与全链路 Trace 回调。

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
| `workflow` | 低代码工作流（自研轻量 StateGraph） |
| `utils` | helpers / logging / serialization |

## 快速开始

```powershell
# 运行测试（无需安装）
py -3.14 -m pytest

# 可选：开发模式安装，供示例脚本 import
py -3.14 -m pip install -e .
```

## 里程碑

- [x] M0 脚手架：包结构 + pyproject + pytest 空跑
- [ ] M1 core + llm
- [ ] M2 tools（热插拔）
- [ ] M3 agents（三范式）
- [ ] M4 memory + trace
- [ ] M5 workflow（迁移 LangGraph）
- [ ] M6 端到端演示
- [ ] M7 文档 + 轨迹对比报告
