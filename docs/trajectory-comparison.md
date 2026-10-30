# 三种范式在同一框架下的轨迹对比报告（交付物 D6）

> 同一旅行助手任务、同一框架（agentflow v1.0）、同一工具集，对比 ReAct / Reflection / Plan-and-Solve 三范式。
> 轨迹由全链路 Trace 回调落盘为 JSONL（见 [`docs/trajectory/`](trajectory/)），数据可复现。

---

## 1. 实验设置

| 项 | 取值 |
|---|---|
| 任务 | 「我想去北京玩一天，天气怎么样？有什么景点推荐？」 |
| 框架 | agentflow v1.0（自研，不依赖 LangGraph/LangChain） |
| 模型 | 离线 `ScriptedLLM`（确定性脚本，用于复现轨迹形态；在线可换公司中转站真实模型） |
| 工具集 | `get_weather` / `get_attraction`（统一 `ToolRegistry`，仅 ReAct 会调用工具） |
| 温度 | 0（离线脚本无随机性） |
| 观测 | `Tracer` + `JSONLFileCallback`（落盘）+ `StatsCallback`（统计） |

> 说明：离线演示不注入 token 计数，故 token=0、耗时可忽略；在线运行时 `StatsCallback` 会统计真实 token 与端到端耗时。

---

## 2. 量化对比

| 范式 | 最大 step | LLM 调用 | 工具调用 | 事件数 | 终止方式 |
|---|---:|---:|---:|---:|---|
| ReAct | 3 | 3 | 2 | 7 | `Finish[答案]` |
| Reflection | 4 | 4 | 0 | 6 | 「无需改进」魔法串 |
| Plan-and-Solve | 3 | 3 | 0 | 5 | 计划执行完毕 |

### 事件序列（轨迹形态）

| 范式 | 事件序列 | 形态 |
|---|---|---|
| ReAct | `start → llm → tool → llm → tool → llm → end` | Thought/Action/Observation **交错**，边走边看 |
| Reflection | `start → llm → llm → llm → llm → end` | 草稿 → 批评 → 修正 → 再批评，**迭代收敛** |
| Plan-and-Solve | `start → llm → llm → llm → end` | 先出计划 → 逐步执行，**前静后动** |

---

## 3. 轨迹形态解读

### 3.1 ReAct —— 边走边看（交错型）

```
start → llm(Thought: 查天气 / Action: get_weather) → tool(get_weather)
      → llm(Thought: 查景点 / Action: get_attraction) → tool(get_attraction)
      → llm(Action: Finish[最终答案]) → end
```

- 每一步都**先思考再行动**，工具调用与推理交错进行。
- `Finish` 检查先于工具查找，答案里出现工具名也不会误触发调用（有回归测试）。
- **适合**：需要实时工具、任务路径不可预知、追求可控/可解释的场景。

### 3.2 Reflection —— 事后自校（迭代型）

```
start → llm(初稿) → llm(评审反馈) → llm(修订稿) → llm(评审：无需改进) → end
```

- 先给出完整结果，再以「评审专家」视角挑错、修订，直到「无需改进」或达到 `max_iterations`。
- 每次迭代都产生一份完整答案，**质量换成本**（LLM 调用次数最多）。
- **适合**：对答案质量敏感、任务可离线迭代、无需实时工具的场景（写作、方案、代码审查）。

### 3.3 Plan-and-Solve —— 三思而后行（先规划型）

```
start → llm(Planner：输出步骤列表) → llm(执行步骤1) → llm(执行步骤2) → end
```

- 先强制产出结构化计划（Python 列表），再严格按计划逐步执行，`history` 携带前序结果做状态管理。
- **计划一次成型，执行稳定**，适合多步、有明确分解逻辑的复杂任务。

---

## 4. 结论与选型建议

| 维度 | ReAct | Reflection | Plan-and-Solve |
|---|---|---|---|
| 可控性 / 可解释性 | ⭐⭐⭐（每步可见工具调用） | ⭐⭐ | ⭐⭐ |
| 效率（LLM 调用少） | ⭐⭐ | ⭐（最多） | ⭐⭐ |
| 复杂多步任务 | ⭐⭐ | ⭐⭐ | ⭐⭐⭐（先规划） |
| 简单 / 需实时信息 | ⭐⭐⭐（工具实时） | ⭐⭐ | ⭐⭐ |
| 质量敏感任务 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

**建议**：

- **需要实时信息/外部工具、且要看清每一步推理** → 选 **ReAct**。
- **答案要反复打磨、任务可离线迭代** → 选 **Reflection**（以多一次 LLM 调用换质量）。
- **任务可拆成清晰步骤、追求执行稳定** → 选 **Plan-and-Solve**。
- 三者共享同一框架入口，切换只改一个类名，`memory` / `tracer` / `registry` 自动复用。

---

## 5. 附录

### 5.1 JSONL 轨迹样本（ReAct）

```jsonl
{"type": "start", "agent_name": "react", "step": 0, "data": {"input": "我想去北京玩一天，天气怎么样？有什么景点推荐？"}, "trace_id": "react"}
{"type": "llm", "agent_name": "react", "step": 1, "data": {"output": "Thought: 先查北京今天天气\nAction: get_weather(city=\"北京\")"}, "trace_id": "react"}
{"type": "tool", "agent_name": "react", "step": 1, "data": {"tool": "get_weather", "params": {"city": "北京"}, "output": "北京今天晴，20~25℃，适合户外"}, "trace_id": "react"}
{"type": "llm", "agent_name": "react", "step": 2, "data": {"output": "Thought: 天气不错，查适合的景点\nAction: get_attraction(city=\"北京\", weather=\"晴\")"}, "trace_id": "react"}
{"type": "tool", "agent_name": "react", "step": 2, "data": {"tool": "get_attraction", "params": {"city": "北京", "weather": "晴"}, "output": "北京天气晴，推荐：故宫、颐和园、长城"}, "trace_id": "react"}
{"type": "llm", "agent_name": "react", "step": 3, "data": {"output": "Thought: 信息齐了\nAction: Finish[北京今天晴，推荐故宫、颐和园、长城，适合户外游玩。]"}, "trace_id": "react"}
{"type": "end", "agent_name": "react", "step": 0, "data": {"result": "北京今天晴，推荐故宫、颐和园、长城，适合户外游玩。"}, "trace_id": "react"}
```

### 5.2 复现命令

```powershell
# 生成三范式 JSONL 轨迹 + 对比表
py -3.14 scripts/gen_trajectory_report.py

# 端到端演示（含四形态 + 热插拔 + Memory + Trace）
py -3.14 -m examples.demo_e2e
```
