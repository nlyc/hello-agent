"""自研轻量状态图引擎：StateGraph + START/END + 条件边，替代 LangGraph。

v1.0 边界：只做「无环主链路 + 显式重试/降级边」，不做任意递归、多角色、checkpointer。
"""

from typing import Any, Callable, Optional

from agentflow.core.exceptions import WorkflowError


class _Sentinel:
    def __init__(self, name: str):
        self._name = name

    def __repr__(self) -> str:
        return self._name


START = _Sentinel("START")
END = _Sentinel("END")


class StateGraph:
    """低代码状态图：所有节点读写同一 state 字典，边定义执行顺序，条件边实现循环/降级。"""

    def __init__(self, state_schema: Optional[type] = None):
        self._state_schema = state_schema  # 仅文档化，运行时不强制校验
        self._nodes: dict[str, Callable[[dict], dict]] = {}
        self._edges: dict[Any, Any] = {}
        self._conditional: dict[str, tuple[Callable[[dict], str], dict[str, str]]] = {}

    def add_node(self, name: str, fn: Callable[[dict], dict]) -> None:
        self._nodes[name] = fn

    def add_edge(self, start: Any, end: Any) -> None:
        self._edges[start] = end

    def add_conditional_edges(
        self, source: str, router: Callable[[dict], str], mapping: dict[str, str]
    ) -> None:
        """router(state) 返回一个 key，mapping[key] 指向下一节点名。"""
        self._conditional[source] = (router, mapping)

    def compile(self, max_steps: int = 64) -> "CompiledGraph":
        return CompiledGraph(self._nodes, self._edges, self._conditional, max_steps)


class CompiledGraph:
    def __init__(self, nodes, edges, conditional, max_steps):
        self._nodes = nodes
        self._edges = edges
        self._conditional = conditional
        self._max_steps = max_steps

    def _route(self, node: Any, state: dict) -> Any:
        if node in self._conditional:
            router, mapping = self._conditional[node]
            key = router(state)
            if key not in mapping:
                raise WorkflowError(f"条件边路由 '{key}' 不在映射 {sorted(mapping)} 中")
            return mapping[key]
        return self._edges.get(node, END)

    def _walk(self, inputs: dict, max_steps: Optional[int]):
        state = dict(inputs)
        limit = max_steps or self._max_steps
        node = self._route(START, state)
        steps = 0
        while node is not END:
            steps += 1
            if steps > limit:
                raise WorkflowError(f"状态图超过最大步数 {limit}（可能存在死循环）")
            fn = self._nodes.get(node)
            if fn is None:
                raise WorkflowError(f"未知节点 '{node}'")
            update = fn(state) or {}
            state.update(update)
            yield node, update
            node = self._route(node, state)

    def invoke(self, inputs: dict, max_steps: Optional[int] = None) -> dict:
        """执行整张图，返回最终 state。"""
        state = dict(inputs)
        for _, update in self._walk(inputs, max_steps):
            state.update(update)
        return state

    def stream(self, inputs: dict, max_steps: Optional[int] = None):
        """逐步执行，逐个 yield (node_name, update)。"""
        for node, update in self._walk(inputs, max_steps):
            yield node, update

    def to_mermaid(self) -> str:
        lines = ["flowchart TD"]
        for src, dst in self._edges.items():
            lines.append(f"    {self._m(src)} --> {self._m(dst)}")
        for src, (_, mapping) in self._conditional.items():
            for key, dst in mapping.items():
                lines.append(f"    {self._m(src)} -->|\"{key}\"| {self._m(dst)}")
        return "\n".join(lines)

    @staticmethod
    def _m(node: Any) -> str:
        if node is START:
            return "START"
        if node is END:
            return "END"
        return str(node)
