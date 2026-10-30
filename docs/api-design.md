# agentflow 框架 API 设计文档（交付物 D5）

> 自研智能体框架 v1.0。本文档覆盖每个公开类/方法的签名、参数、返回值、异常与示例，
> 末尾附「如何扩展自定义组件」指南。分层结构见 [`architecture.md`](architecture.md)。

---

## 1. 总览

### 1.1 分层

| 包 | 职责 |
|---|---|
| `agentflow.core` | Message / Config / Agent 基类 / exceptions |
| `agentflow.llm` | 可替换 Model Layer（BaseLLM 协议 + 公司中转站 + 模型注册表） |
| `agentflow.tools` | Tool / ToolParameter / ToolRegistry（热插拔）/ ToolChain / 异步执行器 / builtin |
| `agentflow.agents` | ReActAgent / ReflectionAgent / PlanAndSolveAgent |
| `agentflow.memory` | MessageHistory / BaseMemory / 可插拔后端 / as_tool 桥接 |
| `agentflow.trace` | TraceEvent / Tracer / 内置回调 |
| `agentflow.workflow` | StateGraph（自研低代码状态图，不依赖 LangGraph） |
| `agentflow.utils` | helpers（`extract_json_object` 等） |

### 1.2 设计理念

1. **万物皆为工具**：Memory 可通过 `as_tool()` 回归 Tool 形态。
2. **模板方法**：所有 Agent 共享 `run()` 公开 / `_execute()` 抽象，入口统一。
3. **依赖注入**：`llm / registry / memory / tracer` 均构造注入，测试可替换为 fake。
4. **开闭原则**：扩展组件用继承拦截 + `super()` 兜底，不改库源码。

### 1.3 命名约定

- 异常分层：`AgentFlowError` 基类 → `ConfigError / LLMError / ToolError / AgentError / WorkflowError`。
- 所有公开类/方法均提供类型注解与中文 docstring。

### 1.4 最小可运行示例（离线，不联网）

```python
from agentflow.agents import ReActAgent
from agentflow.tools import ToolParameter, ToolRegistry
from agentflow.trace import Tracer, ConsoleCallback


class FakeLLM:
    def chat(self, messages, *, temperature=None, json_mode=False, tools=None):
        return 'Thought: 查天气\nAction: get_weather(city="北京")'


registry = ToolRegistry()
registry.register_function(
    "get_weather", "查询城市天气", lambda city: f"{city}今天晴",
    [ToolParameter("city", "string", "城市名", True)],
)
tracer = Tracer(trace_id="demo")
tracer.subscribe(ConsoleCallback())

agent = ReActAgent("travel", FakeLLM(), tools=registry, tracer=tracer)
print(agent.run("北京天气怎么样？"))
```

---

## 2. `agentflow.core`

### 2.1 `Message`

```python
class Message(pydantic.BaseModel):
    role: MessageRole                       # "user" | "assistant" | "system" | "tool"
    content: str
    timestamp: datetime = now()
    metadata: dict = {}
    name: Optional[str] = None
```

| 方法 | 说明 |
|---|---|
| `to_dict() -> dict` | 转成 OpenAI API 消息字典（`{"role","content"}`，含 `name` 时带上） |
| `str(msg)` | `"[role] content"` |

### 2.2 `Config` / `LLMConfig` / `AgentConfig`

```python
class LLMConfig(BaseModel):
    base_url: str = ""          # 公司中转站（OpenAI 兼容网关）
    api_key: str = ""
    model: str = "deepseek-chat"
    temperature: float = 0.2
    timeout: float = 60.0

class AgentConfig(BaseModel):
    max_steps: int = 6
    max_tool_iterations: int = 3

class Config(BaseModel):
    llm: LLMConfig
    agent: AgentConfig
    max_history_length: int = 100
    debug: bool = False

    @classmethod
    def from_env(cls) -> "Config": ...
```

- 零配置即可实例化；`from_env()` 从环境变量覆盖（`LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` / `LLM_TEMPERATURE` / `LLM_TIMEOUT` / `AGENT_MAX_STEPS` / `AGENT_MAX_TOOL_ITERATIONS` / `MAX_HISTORY_LENGTH` / `DEBUG`）。
- 配置优先级：显式传参 > 环境变量 > 默认值。

### 2.3 `Agent`（基类，抽象）

```python
class Agent(ABC):
    def __init__(self, name, llm, system_prompt=None, config=None, memory=None, tracer=None): ...

    def run(self, input_text: str, **kwargs) -> str: ...       # 模板方法
    @abstractmethod
    def _execute(self, input_text: str, **kwargs) -> str: ...  # 子类实现范式循环
    def add_message(self, message: Message) -> None
    def clear_history(self) -> None
    def get_history(self) -> list[Message]                      # 返回副本，防篡改
```

- `run()` 顺序：`emit("start")` → 追加 user 消息 → `_execute()` → 追加 assistant 消息 → `emit("end")`。
- `_execute()` 抛异常时 `emit("error")` 后重新抛出。

### 2.4 exceptions

| 异常 | 继承 | 用途 |
|---|---|---|
| `AgentFlowError` | `Exception` | 框架异常基类 |
| `ConfigError` | `AgentFlowError` | 配置缺失/非法 |
| `LLMError` | `AgentFlowError` | 模型调用失败 |
| `ToolError` | `AgentFlowError` | 工具执行失败 |
| `AgentError` | `AgentFlowError` | Agent 执行失败（解析/超步数） |
| `WorkflowError` | `AgentFlowError` | 工作流执行失败 |

---

## 3. `agentflow.llm`（可替换 Model Layer）

### 3.1 `BaseLLM`（协议，抽象）

```python
class BaseLLM(ABC):
    @abstractmethod
    def chat(
        self,
        messages: Sequence[Any],
        *,
        temperature: Optional[float] = None,
        json_mode: bool = False,
        tools: Optional[list[dict]] = None,
    ) -> str: ...
```

- `messages` 每项可为 `Message` 或 OpenAI 风格 dict。
- `json_mode=True` 要求服务端返回 JSON 对象；`tools` 为原生 function calling 预留。
- 所有范式/工作流只依赖此协议，替换实现即可（测试注入 fake 靠它）。

### 3.2 `OpenAICompatibleLLM(BaseLLM)` — 公司中转站

```python
class OpenAICompatibleLLM(BaseLLM):
    def __init__(self, base_url=None, api_key=None, model=None, temperature=0.2,
                 timeout=60.0, registry=None, client=None): ...
    def chat(self, messages, *, temperature=None, json_mode=False, tools=None) -> str
    def use_model(self, alias: str) -> None
```

- `base_url / api_key / model` 优先取构造参数，其次环境变量 `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`，默认 `deepseek-chat`。
- ⭐ **三家云端 = 同一公司中转站**：DeepSeek / GLM / OpenAI 都是不同 `model` 名，`base_url` 不变。
- `chat` 对不支持 `response_format` 的网关自动去掉 `json_mode` 重试一次；失败抛 `LLMError`。
- `use_model(alias)`：别名经注册表解析（可换 `model` + 可选独立 `api_key/base_url`），否则当作裸模型名。

### 3.3 `ModelRegistry` / `ModelSpec`

```python
@dataclass(frozen=True)
class ModelSpec:
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None

class ModelRegistry:
    def __init__(self, specs: Optional[dict[str, ModelSpec]] = None): ...
    def register(self, alias, model, api_key=None, base_url=None) -> None
    def register_many(self, specs: dict[str, ModelSpec]) -> None
    def get(self, alias) -> ModelSpec
    def list_aliases(self) -> list[str]
    def __contains__(self, alias) -> bool
```

### 3.4 换模型示例

```python
from agentflow.llm import ModelRegistry, OpenAICompatibleLLM

reg = ModelRegistry()
reg.register("deepseek", "deepseek-chat")
reg.register("glm", "glm-4-flash")
reg.register("openai", "gpt-4o-mini")

llm = OpenAICompatibleLLM(registry=reg)   # 读取 LLM_* 环境变量
llm.use_model("glm")                       # 一行切换，base_url 不变
```

---

## 4. `agentflow.tools`

### 4.1 `ToolParameter`

```python
@dataclass
class ToolParameter:
    name: str
    type: str = "string"
    description: str = ""
    required: bool = False
    default: Any = None
```

### 4.2 `Tool`（抽象）

```python
class Tool(ABC):
    def __init__(self, name: str, description: str = ""): ...
    @abstractmethod
    def run(self, parameters: dict) -> str: ...
    def get_parameters(self) -> list[ToolParameter]     # 默认 []
    def validate_parameters(self, parameters) -> list[str]   # 缺失必填项列表
    def to_dict(self) -> dict
    def to_openai_schema(self) -> dict                  # OpenAI function calling schema
```

### 4.3 `FunctionTool(Tool)`

```python
class FunctionTool(Tool):
    def __init__(self, name, description, func, parameters=None): ...
```

- `func` 的形参名须与 `parameters` 的 `name` 一致；缺省 `parameters` 时使用单个 `input` 参数。

### 4.4 `ToolRegistry`（热插拔，线程安全）

```python
class ToolRegistry:
    def register_tool(self, tool: Tool) -> None
    def register_function(self, name, description, func, parameters=None) -> None
    def unregister(self, name) -> bool            # 返回是否确实存在
    def get(self, name) -> Tool                    # 未知抛 ToolError
    def execute(self, name, parameters) -> str     # 非 ToolError 异常包装为 ToolError
    def list_tools(self) -> list[Tool]
    def names(self) -> list[str]
    def get_tools_description(self) -> str
    def to_openai_schema(self) -> list[dict]
    def clear(self) -> None
    def __contains__(self, name) -> bool
    def __len__(self) -> int
```

- ⭐ **热插拔**：运行中增删替换，内部 `RLock` 加锁；Agent 每步实时读最新工具集。

### 4.5 `ToolChain` / `ToolChainManager`

```python
class ToolChain:
    def add_step(self, tool, input_template="{input}", output_key=None) -> "ToolChain"
    def run(self, input_text, context=None) -> dict      # 返回完整 context
    @property
    def steps(self) -> int

class ToolChainManager:
    def register(self, chain: ToolChain) -> None
    def execute(self, name, input_text, context=None) -> dict
    def list_chains(self) -> list[str]
```

- `input_template` 用 context 变量替换后作为工具 `input` 参数；结果按 `output_key` 写回 context。

### 4.6 `AsyncToolExecutor`

```python
class AsyncToolExecutor:
    def __init__(self, max_workers: int = 4): ...
    async def run_parallel(self, calls: Sequence[tuple[Tool, dict]]) -> list[str]
    def run_parallel_sync(self, calls: Sequence[tuple[Tool, dict]]) -> list[str]
    def close(self) -> None
```

- 线程池 + `asyncio.gather`，按输入顺序返回结果；支持 `with` 上下文。

### 4.7 builtin

| 工具 | 参数 | 说明 |
|---|---|---|
| `CalculatorTool` (name=`calculator`) | `expression` | `ast.parse` + 运算符/函数白名单，**绝不 eval**；支持 `+ - * / ** sqrt abs round max min sum sin cos tan log exp` |
| `SearchTool` (name=`search`) | `input` | 多源降级链：Tavily → SerpApi → 明确报错（不静默失败），需 `TAVILY_API_KEY` 或 `SERPAPI_API_KEY` |

---

## 5. `agentflow.agents`（三范式）

三者构造签名一致地接受 `name / llm / system_prompt / config / memory / tracer`，共享 `run()` 入口与 trace 事件。

### 5.1 `ReActAgent`

```python
class ReActAgent(Agent):
    def __init__(self, name, llm, tools=None, system_prompt=None, config=None,
                 custom_prompt=None, memory=None, tracer=None): ...
```

- 循环：Thought → Action → Observation；**Finish 检查先于工具查找**。
- `Action` 格式：`function_name(arg="value")` 或 `Finish[最终答案]`；支持全角冒号。
- `max_steps` 来自 `config.agent.max_steps`，超限抛 `AgentError`。
- `custom_prompt` 可整体替换系统提示词（覆盖 `{role}` / `{tools}` 模板）。

### 5.2 `ReflectionAgent`

```python
class ReflectionAgent(Agent):
    def __init__(self, name, llm, system_prompt=None, config=None, max_iterations=3,
                 role="资深专家", reviewer_role="极其严格的评审专家",
                 custom_prompts=None, memory=None, tracer=None): ...
    def get_trajectory(self) -> str
```

- 三段：initial → reflect → refine；「无需改进」魔法串 + `max_iterations` 双终止。
- `custom_prompts` 覆盖 `initial / reflect / refine` 模板（`{task}` `{result}` `{feedback}` `{last_result}` `{role}` `{reviewer_role}`）。

### 5.3 `PlanAndSolveAgent`

```python
class PlanAndSolveAgent(Agent):
    def __init__(self, name, llm, system_prompt=None, config=None,
                 custom_prompts=None, memory=None, tracer=None): ...
```

- Planner 强制输出 Python 列表（`ast.literal_eval` 解析），Executor 五要素 `{question}{plan}{history}{current_step}`。
- `custom_prompts` 覆盖 `planner / executor` 模板。

---

## 6. `agentflow.memory`（Context/Memory 管理）

### 6.1 `MessageHistory` / `estimate_tokens`

```python
def estimate_tokens(text: str) -> int          # CJK 每字≈1 token，其余按词近似

class MessageHistory:
    def __init__(self, max_length: int = 100): ...
    def add(self, message: Message) -> None    # 超长自动裁尾
    def get(self) -> list[Message]              # 返回副本
    def get_recent(self, n: int) -> list[Message]
    def clear(self) -> None
    def truncate(self, max_tokens: int) -> list[Message]   # 从尾部保留，至少留最近一条
    def to_dicts(self) -> list[dict]
    def __len__ / __iter__ / __contains__
```

### 6.2 `BaseMemory`（抽象）

```python
class BaseMemory(ABC):
    @abstractmethod
    def remember(self, key: str, value: str) -> None: ...
    @abstractmethod
    def recall(self, query: str) -> str: ...
    @abstractmethod
    def clear(self) -> None: ...
    def as_tool(self, name: str = "memory") -> Tool   # 回归「万物皆为工具」
```

### 6.3 可插拔后端

| 后端 | 持久化 | 语义 |
|---|---|---|
| `ShortTermMemory(max_items=50)` | 否（进程内） | 滑动窗口 |
| `FileMemory(path)` | 是（JSON 文件） | 追加式 key/value |
| `SQLiteMemory(path)` | 是（SQLite） | key 唯一，重复 remember 覆盖 |

### 6.4 验收路径

- 进程重启后历史仍在：新建 `FileMemory(path)` / `SQLiteMemory(path)` 实例读取同一文件/库。
- 超长对话裁剪：`MessageHistory.truncate(max_tokens)` 或 `ShortTermMemory` 滑动窗口。

---

## 7. `agentflow.trace`（全链路 Trace 回调）

### 7.1 `TraceEvent`

```python
@dataclass
class TraceEvent:
    type: str                  # start / end / error / llm / tool / node / memory ...
    agent_name: str = ""
    step: int = 0
    data: dict = {}
    timestamp: datetime = now()
    trace_id: str = ""
    def to_dict(self) -> dict
```

### 7.2 `Tracer`

```python
class Tracer:
    def __init__(self, trace_id: str = ""): ...          # 空则随机 12 位 hex
    def subscribe(self, callback, *, event_type=None, agent_name=None, filter_fn=None) -> handle
    def unsubscribe(self, callback) -> None
    def emit(self, event_type, agent_name="", step=0, data=None) -> None
    def __len__(self) -> int
```

- 过滤条件（`event_type` / `agent_name` / `filter_fn`）全部满足才触发；回调异常不炸业务。

### 7.3 内置回调

| 回调 | 说明 |
|---|---|
| `ConsoleCallback(out=None)` | 打印 `[trace] type agent=... step=...` |
| `JSONLFileCallback(path)` | 逐事件落盘 JSONL（`close()` 关闭） |
| `StatsCallback()` | `summary()` → `{counts, llm_calls, tool_calls, tokens, duration_seconds}` |

### 7.4 回调编写

```python
def my_callback(event: TraceEvent) -> None:
    print(event.type, event.agent_name, event.step, event.data)

tracer.subscribe(my_callback, event_type="tool")   # 只订阅工具事件
```

---

## 8. `agentflow.workflow`（低代码工作流）

### 8.1 `StateGraph` / `START` / `END`

```python
from agentflow.workflow import START, END, StateGraph

class StateGraph:
    def __init__(self, state_schema: Optional[type] = None): ...   # TypedDict，仅文档化
    def add_node(self, name: str, fn: Callable[[dict], dict]) -> None
    def add_edge(self, start, end) -> None
    def add_conditional_edges(self, source, router, mapping: dict[str, str]) -> None
    def compile(self, max_steps: int = 64) -> CompiledGraph
```

- `router(state) -> str` 返回 key，`mapping[key]` 指向下一节点名。
- `max_steps` 安全阀：超步数抛 `WorkflowError`（防死循环）。

### 8.2 `CompiledGraph`

```python
def invoke(self, inputs: dict, max_steps=None) -> dict      # 执行整图，返回最终 state
def stream(self, inputs: dict, max_steps=None)               # 逐步 yield (node_name, update)
def to_mermaid(self) -> str
```

- 边界：v1.0 只做「无环主链路 + 显式重试/降级边」，不做任意递归/多角色/checkpointer。

---

## 9. 扩展指南

### 9.1 自定义 Tool

```python
from agentflow.tools import Tool, ToolParameter

class GreetTool(Tool):
    def __init__(self):
        super().__init__(name="greet", description="打招呼")
    def run(self, parameters):
        return f"你好，{parameters['name']}！"
    def get_parameters(self):
        return [ToolParameter("name", "string", "名字", True)]
```

或一行注册现成函数：`registry.register_function("greet", "打招呼", lambda name: f"你好，{name}", [ToolParameter("name","string","名字",True)])`。

### 9.2 自定义 LLM

继承 `BaseLLM` 实现 `chat()` 即可，所有范式/工作流自动可用：

```python
class MyLLM(BaseLLM):
    def chat(self, messages, *, temperature=None, json_mode=False, tools=None) -> str:
        ...
```

### 9.3 自定义 Memory

继承 `BaseMemory` 实现 `remember / recall / clear`，自动获得 `as_tool()` 桥接。

### 9.4 自定义 Agent

继承 `Agent` 实现 `_execute(input_text)`；`run()` 模板方法自动处理历史/trace，无需重写。

### 9.5 自定义 Callback

实现 `__call__(self, event: TraceEvent) -> None`，经 `tracer.subscribe(...)` 注册，可加 `event_type/agent_name/filter_fn` 过滤。
