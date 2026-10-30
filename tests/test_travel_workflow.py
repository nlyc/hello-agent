"""旅行助手·低代码工作流 verify 矩阵（离线，不联网）。

覆盖：成功主链路、天气失败重试降级、景点失败重试兜底、理解失败规则兜底、回答失败模板兜底。
与 `travel-agent-langgraph/backend/verify_graph.py` 的验证矩阵对齐，验证迁移后的自研引擎等价。
"""

from __future__ import annotations

import json

from examples.travel_workflow import TravelDeps, build_travel_graph

INPUT = {
    "province": "北京",
    "city": "北京",
    "user_query": "我想去北京玩一天，天气怎么样？有什么景点推荐？",
}


class FakeTravelLLM:
    def __init__(self, *, fail_understand=False, fail_answer=False, fail_fallback=False):
        self.fail_understand = fail_understand
        self.fail_answer = fail_answer
        self.fail_fallback = fail_fallback
        self.calls: list[str] = []

    def chat(self, messages, *, temperature=None, json_mode=False, tools=None):
        if json_mode:
            self.calls.append("understand")
            if self.fail_understand:
                raise RuntimeError("模拟理解节点失败")
            return json.dumps(
                {
                    "thought": "先确认城市与出行意图，再改写检索词。",
                    "intent": "用户想知道北京今天的天气，并获得适合游玩的景点。",
                    "weather_query": "北京今天天气",
                    "attraction_query": "北京 秋季 适合游玩的景点 推荐 理由",
                },
                ensure_ascii=False,
            )

        user_prompt = messages[-1]["content"]
        if "搜索服务暂时不可用" in user_prompt:
            self.calls.append("fallback")
            if self.fail_fallback:
                raise RuntimeError("模拟兜底失败")
            return "1. 故宫：室内外结合，雨天也能安排。\n2. 颐和园：天气晴好时适合散步。"

        self.calls.append("answer")
        if self.fail_answer:
            raise RuntimeError("模拟回答节点失败")
        return "今日北京多云，适合游览故宫与颐和园。出行前请核实开放信息。"


class FakeWeather:
    def __init__(self, fail_times: int = 0):
        self.fail_times = fail_times
        self.calls = 0

    def __call__(self, city: str) -> str:
        self.calls += 1
        if self.calls <= self.fail_times:
            return "错误: 模拟天气网络故障"
        return "北京当前天气:多云，气温20摄氏度，湿度55%，今日15~24摄氏度，降水概率10%"


class FakeAttraction:
    def __init__(self, fail_times: int = 0):
        self.fail_times = fail_times
        self.calls = 0
        self.queries: list[str] = []
        self.weathers: list[str] = []

    def __call__(self, city: str, weather: str, optimized_query: str = "") -> str:
        self.calls += 1
        self.queries.append(optimized_query)
        self.weathers.append(weather)
        if self.calls <= self.fail_times:
            return "错误: 模拟 Tavily 搜索失败"
        return "故宫：适合阴晴皆宜的文化游览。颐和园：多云天气适合漫步。"


def run(deps: TravelDeps) -> dict:
    graph = build_travel_graph(deps)
    return graph.invoke(dict(INPUT))


def test_happy_path():
    weather = FakeWeather()
    attraction = FakeAttraction()
    state = run(TravelDeps(weather_tool=weather, attraction_tool=attraction, llm=FakeTravelLLM()))

    assert state["weather_status"] == "ok"
    assert state["attraction_status"] == "ok"
    assert state["step"] == "answer"
    assert state["final_answer"]
    assert weather.calls == 1 and attraction.calls == 1
    assert "多云天" in attraction.queries[0]


def test_weather_retry_then_degrade():
    weather = FakeWeather(fail_times=99)
    attraction = FakeAttraction()
    state = run(TravelDeps(weather_tool=weather, attraction_tool=attraction, llm=FakeTravelLLM()))

    assert weather.calls == 2  # max_retries = 2
    assert state["weather_status"] == "degraded"
    assert attraction.calls >= 1
    assert attraction.weathers[0] == "天气信息暂不可用"
    assert state["final_answer"]


def test_attraction_retry_then_fallback():
    weather = FakeWeather()
    attraction = FakeAttraction(fail_times=99)
    llm = FakeTravelLLM()
    state = run(TravelDeps(weather_tool=weather, attraction_tool=attraction, llm=llm))

    assert attraction.calls == 2  # max_retries = 2
    assert state["attraction_status"] == "fallback"
    assert "fallback" in llm.calls
    assert state["final_answer"]


def test_understand_failure_uses_rule_fallback():
    weather = FakeWeather()
    attraction = FakeAttraction()
    state = run(
        TravelDeps(weather_tool=weather, attraction_tool=attraction, llm=FakeTravelLLM(fail_understand=True))
    )

    assert weather.calls == 1
    assert state["weather_status"] == "ok"
    assert state["final_answer"]


def test_answer_failure_uses_template():
    weather = FakeWeather()
    attraction = FakeAttraction()
    state = run(
        TravelDeps(weather_tool=weather, attraction_tool=attraction, llm=FakeTravelLLM(fail_answer=True))
    )

    assert "今日出行建议" in state["final_answer"]


def test_mermaid_contains_all_nodes():
    graph = build_travel_graph(
        TravelDeps(weather_tool=FakeWeather(), attraction_tool=FakeAttraction(), llm=FakeTravelLLM())
    )
    mermaid = graph.to_mermaid()
    for node in ("understand", "weather", "degrade_weather", "attraction", "fallback_attraction", "answer"):
        assert node in mermaid
