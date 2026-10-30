"""旅行助手·低代码工作流案例：把 LangGraph 显式状态图迁到 agentflow 自研引擎。

    understand → weather ─┬─ok────────────→ attraction ─┬─ok──────→ answer
                          ├─失败且未达上限→ weather       ├─失败且未达上限→ attraction
                          └─失败且达上限→ degrade_weather └─失败且达上限→ fallback_attraction → answer

依赖注入（weather_tool / attraction_tool / llm），离线可用假函数跑通 verify 矩阵。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypedDict

from agentflow.utils import extract_json_object
from agentflow.workflow import END, START, StateGraph


class TravelState(TypedDict, total=False):
    province: str
    city: str
    user_query: str

    intent: str
    weather_query: str
    attraction_query: str

    weather: str
    weather_status: str  # ok | failed | degraded
    weather_attempts: int
    weather_error: str

    attractions: str
    attraction_status: str  # ok | failed | fallback
    attraction_attempts: int
    attraction_error: str

    final_answer: str
    step: str


@dataclass
class TravelDeps:
    """旅行工作流依赖。llm 需实现 chat(messages, *, temperature, json_mode, tools) -> str。"""

    llm: Any = None
    weather_tool: Optional[Callable[[str], str]] = None
    attraction_tool: Optional[Callable[..., str]] = None
    max_retries: int = 2
    max_steps: int = 30


_UNDERSTAND_SYSTEM_PROMPT = """你是旅行规划 Agent 的「理解节点」。
你的任务不是回答用户，而是把用户的自然语言请求整理成后续节点可用的结构化信息。
只输出一个 JSON 对象，字段如下：
{
  "thought": "一句话说明你如何理解这次出行需求",
  "intent": "用户想去哪里、关心什么",
  "weather_query": "查询天气时使用的简短关键词，必须包含城市名",
  "attraction_query": "交给搜索引擎的景点检索词，必须包含城市名，写成关键词而非口语长句"
}
不要输出 Markdown 代码围栏，不要输出 JSON 以外的解释。"""

_ERROR_MARKERS = ("错误:", "错误：", "未配置", "抱歉，没有找到", "无法查询")


def _is_tool_error(observation: str) -> bool:
    text = (observation or "").strip()
    return not text or any(marker in text[:120] for marker in _ERROR_MARKERS)


def _weather_brief(observation: str) -> str:
    """从天气观察里抽出「多云/小雨/晴」这类短描述，避免把整段观察塞进搜索词。"""
    match = re.search(r"当前天气[:：]\s*([^，,\n]+)", observation or "")
    if match:
        return match.group(1).strip()
    return (observation or "").strip()[:24]


def _understand_intent(user_query: str, city: str, llm) -> dict:
    fallback = {
        "thought": "结构化理解暂不可用，改用规则生成检索词，继续执行。",
        "intent": f"用户想了解{city}今天的天气，并获得适合当天游玩的景点建议。",
        "weather_query": f"{city}今天天气",
        "attraction_query": f"{city} 适合游玩的景点 推荐 理由",
    }
    try:
        raw = llm.chat(
            [
                {"role": "system", "content": _UNDERSTAND_SYSTEM_PROMPT},
                {"role": "user", "content": user_query},
            ],
            temperature=0.0,
            json_mode=True,
        )
        data = extract_json_object(raw)
    except Exception:
        return fallback

    result = {
        "thought": data.get("thought", "") or fallback["thought"],
        "intent": data.get("intent", "") or fallback["intent"],
        "weather_query": data.get("weather_query", "") or fallback["weather_query"],
        "attraction_query": data.get("attraction_query", "") or fallback["attraction_query"],
    }
    if city not in result["weather_query"]:
        result["weather_query"] = f"{city} {result['weather_query']}".strip()
    if city not in result["attraction_query"]:
        result["attraction_query"] = f"{city} {result['attraction_query']}".strip()
    return result


def build_travel_graph(deps: Optional[TravelDeps] = None):
    deps = deps or TravelDeps()

    def understand_node(state: TravelState) -> dict:
        intent = _understand_intent(state["user_query"], state["city"], deps.llm)
        return {
            "intent": intent["intent"],
            "weather_query": intent["weather_query"],
            "attraction_query": intent["attraction_query"],
            "step": "understand",
        }

    def weather_node(state: TravelState) -> dict:
        city = state["city"]
        attempts = int(state.get("weather_attempts", 0)) + 1
        try:
            observation = deps.weather_tool(city)
        except Exception as exc:
            observation = f"错误: 天气工具抛出异常 - {exc}"
        ok = not _is_tool_error(observation)
        return {
            "weather": observation if ok else "",
            "weather_status": "ok" if ok else "failed",
            "weather_attempts": attempts,
            "weather_error": "" if ok else observation,
            "step": "weather",
        }

    def degrade_weather_node(state: TravelState) -> dict:
        return {
            "weather": "天气信息暂不可用",
            "weather_status": "degraded",
            "step": "degrade_weather",
        }

    def attraction_node(state: TravelState) -> dict:
        city = state["city"]
        weather_full = state.get("weather") or "天气信息暂不可用"
        weather = _weather_brief(weather_full) if state.get("weather_status") == "ok" else weather_full
        attempts = int(state.get("attraction_attempts", 0)) + 1

        base_query = state.get("attraction_query") or f"{city} 适合游玩的景点 推荐"
        query = base_query
        if state.get("weather_status") == "ok" and weather not in query:
            weather_kw = f"{weather}天" if weather in {"晴", "阴", "多云", "雾"} else f"{weather}天气"
            query = f"{base_query} {weather_kw} 游玩建议"

        try:
            observation = deps.attraction_tool(city=city, weather=weather, optimized_query=query)
        except TypeError:
            try:
                observation = deps.attraction_tool(city=city, weather=weather)
            except Exception as exc:
                observation = f"错误: 景点工具抛出异常 - {exc}"
        except Exception as exc:
            observation = f"错误: 景点工具抛出异常 - {exc}"

        ok = not _is_tool_error(observation)
        return {
            "attractions": observation if ok else "",
            "attraction_status": "ok" if ok else "failed",
            "attraction_attempts": attempts,
            "attraction_error": "" if ok else observation,
            "step": "attraction",
        }

    def fallback_attraction_node(state: TravelState) -> dict:
        city = state["city"]
        weather = state.get("weather") or "天气信息暂不可用"
        prompt = (
            f"城市：{city}\n天气：{weather}\n用户需求：{state['user_query']}\n\n"
            "搜索服务暂时不可用。请基于你已有的知识，推荐 3-4 个该城市适合游玩的景点，"
            "每个景点说明一句推荐理由；若天气信息缺失，请提醒用户出行前核实。"
            "不要编造门票、营业时间等易变信息。"
        )
        try:
            answer = deps.llm.chat(
                [
                    {"role": "system", "content": "你是谨慎的旅行顾问，信息不确定时会明确说明。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
            )
        except Exception as exc:
            answer = (
                "景点搜索服务暂时不可用，且模型知识兜底也未成功。"
                f"（{str(exc)[:120]}）\n建议稍后重试，或先通过地图/景区官网确认开放信息。"
            )
        return {"attractions": answer, "attraction_status": "fallback", "step": "fallback_attraction"}

    def answer_node(state: TravelState) -> dict:
        city = state["city"]
        weather = state.get("weather") or "天气信息暂不可用"
        attractions = state.get("attractions") or "暂未获得景点信息"
        notes = []
        if state.get("weather_status") != "ok":
            notes.append("天气服务本次不可用，天气信息需用户出行前自行核实。")
        if state.get("attraction_status") == "fallback":
            notes.append("景点信息来自模型知识兜底，未经实时搜索验证。")

        prompt = (
            f"用户请求：{state['user_query']}\n"
            f"城市：{city}\n"
            f"天气观察：{weather}\n"
            f"景点观察：{attractions}\n"
            f"状态说明：{'；'.join(notes) if notes else '天气与景点信息均已获取。'}\n\n"
            "请生成最终中文旅行建议：先用一两句话概括当天天气，再分点推荐 2-4 个"
            "适合当天游玩的景点并说明理由。语气友好、有画面感，但不要编造实时数据；"
            "信息缺失时如实说明。只输出最终答案，不要输出 Thought/Action 标签。"
        )
        try:
            final_answer = deps.llm.chat(
                [
                    {"role": "system", "content": "你是「智游·山水行」旅行助手，擅长把天气与景点信息整合成可执行的当日建议。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
            ).strip()
        except Exception:
            final_answer = ""

        if not final_answer:
            final_answer = (
                f"**{city}今日出行建议**\n\n"
                f"天气：{weather}\n\n"
                f"推荐参考：\n{attractions}\n\n"
                + (f"提示：{'；'.join(notes)}\n" if notes else "")
            )
        return {"final_answer": final_answer, "step": "answer"}

    def route_after_weather(state: TravelState) -> str:
        if state.get("weather_status") == "ok":
            return "attraction"
        if int(state.get("weather_attempts", 0)) < deps.max_retries:
            return "weather"  # 原生循环：回到同一节点重试
        return "degrade_weather"

    def route_after_attraction(state: TravelState) -> str:
        if state.get("attraction_status") == "ok":
            return "answer"
        if int(state.get("attraction_attempts", 0)) < deps.max_retries:
            return "attraction"  # 原生循环：回到同一节点重试
        return "fallback_attraction"

    graph = StateGraph(TravelState)
    graph.add_node("understand", understand_node)
    graph.add_node("weather", weather_node)
    graph.add_node("degrade_weather", degrade_weather_node)
    graph.add_node("attraction", attraction_node)
    graph.add_node("fallback_attraction", fallback_attraction_node)
    graph.add_node("answer", answer_node)

    graph.add_edge(START, "understand")
    graph.add_edge("understand", "weather")
    graph.add_conditional_edges(
        "weather",
        route_after_weather,
        {"weather": "weather", "attraction": "attraction", "degrade_weather": "degrade_weather"},
    )
    graph.add_edge("degrade_weather", "attraction")
    graph.add_conditional_edges(
        "attraction",
        route_after_attraction,
        {"attraction": "attraction", "answer": "answer", "fallback_attraction": "fallback_attraction"},
    )
    graph.add_edge("fallback_attraction", "answer")
    graph.add_edge("answer", END)

    return graph.compile(max_steps=deps.max_steps)
