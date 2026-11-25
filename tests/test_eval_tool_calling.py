"""工具调用 AST 匹配与评估器测试。"""

from __future__ import annotations

import pytest

from agentflow.core.exceptions import EvaluationError
from agentflow.evaluation import (
    EvalCase,
    ToolCall,
    ToolCallingEvaluator,
    calls_equal,
    evaluate_tool_calls,
    match_calls,
    safe_eval,
)


def test_parse_call_and_safe_eval():
    call = ToolCall.from_text('get_weather(city="北京", days=2+3)')
    assert call.name == "get_weather"
    assert call.arguments["city"] == "北京"
    assert call.arguments["days"] == 5
    assert ToolCall.from_text("f(x='hello')").arguments["x"] == "hello"


def test_safe_eval_rejects_calls_and_names():
    import ast

    with pytest.raises(EvaluationError):
        safe_eval(ast.parse("__import__('os')", mode="eval").body)
    with pytest.raises(EvaluationError):
        safe_eval(ast.parse("open('x')", mode="eval").body)
    with pytest.raises(EvaluationError):
        ToolCall.from_text("not a call")


def test_calls_equal_ignores_argument_order_and_evaluates_values():
    left = ToolCall.from_text('f(a=1, b="x")')
    right = ToolCall.from_text('f(b="x", a=2+3-4)')
    assert calls_equal(left, right)

    assert not calls_equal(
        ToolCall.from_text('f(a=1)'), ToolCall.from_text('g(a=1)')
    )
    assert not calls_equal(
        ToolCall.from_text('f(a="1")'), ToolCall.from_text('f(a=1)')
    )
    assert not calls_equal(
        ToolCall.from_text('f(a=True)'), ToolCall.from_text('f(a=1)')
    )


def test_match_calls_multiset():
    expected = [
        ToolCall.from_text('get_weather(city="北京")'),
        ToolCall.from_text('get_weather(city="上海")'),
    ]
    actual = [
        ToolCall.from_text('get_weather(city="上海")'),
        ToolCall.from_text('get_weather(city="北京")'),
    ]
    result = match_calls(expected, actual)
    assert result["exact_match"] is True
    assert result["matched"] == 2
    assert result["extra_calls"] == 0
    assert result["missing_calls"] == 0


def test_match_calls_reports_extra_and_missing():
    expected = [ToolCall.from_text('get_weather(city="北京")')]
    actual = [
        ToolCall.from_text('get_weather(city="北京")'),
        ToolCall.from_text('search_notes(query="北京 景点")'),
    ]
    result = match_calls(expected, actual)
    assert result["exact_match"] is False
    assert result["matched"] == 1
    assert result["extra_calls"] == 1
    assert result["missing_calls"] == 0

    missing = match_calls(
        expected, [ToolCall.from_text('get_weather(city="上海")')]
    )
    assert missing["missing_calls"] == 1
    assert missing["extra_calls"] == 1


def test_evaluate_irrelevance():
    expected = []
    assert evaluate_tool_calls(expected, [], category="irrelevance")["passed"]
    result = evaluate_tool_calls(
        expected,
        [ToolCall.from_text('get_weather(city="北京")')],
        category="irrelevance",
    )
    assert result["passed"] is False
    assert result["false_positive"] is True


def test_loose_text_matching_for_free_text_fields():
    expected = ToolCall.from_text('search_notes(query="故宫闭馆")')
    actual = ToolCall.from_text('search_notes(query="故宫 周一 闭馆")')
    assert not calls_equal(expected, actual)
    assert calls_equal(expected, actual, loose_fields=["query"])
    assert not calls_equal(
        ToolCall.from_text('search_notes(query="上海外滩")'),
        actual,
        loose_fields=["query"],
    )


def test_evaluate_tool_calls_with_loose_fields():
    expected = [ToolCall.from_text('search_notes(query="故宫闭馆")')]
    actual = [ToolCall.from_text('search_notes(query="故宫 周一 闭馆")')]
    assert not evaluate_tool_calls(expected, actual)["passed"]
    assert evaluate_tool_calls(
        expected, actual, loose_fields=["query"]
    )["passed"]


class FakeAgent:
    tracer = None

    def __init__(self, calls_by_question: dict[str, list[str]]) -> None:
        self.calls_by_question = calls_by_question

    def run(self, question: str) -> str:
        for text in self.calls_by_question.get(question, []):
            call = ToolCall.from_text(text)
            self.tracer.emit(
                "tool",
                agent_name="fake",
                data={"tool": call.name, "params": call.arguments},
            )
        return "done"


def test_tool_calling_evaluator_uses_trace_and_aggregates():
    cases = [
        EvalCase(
            id="c1",
            suite="tool_calling",
            category="simple",
            question="北京天气",
            ground_truth=[{"name": "get_weather", "arguments": {"city": "北京"}}],
        ),
        EvalCase(
            id="c2",
            suite="tool_calling",
            category="irrelevance",
            question="你好",
            ground_truth=[],
        ),
        EvalCase(
            id="c3",
            suite="tool_calling",
            category="simple",
            question="上海天气",
            ground_truth=[{"name": "get_weather", "arguments": {"city": "上海"}}],
        ),
        EvalCase(
            id="c4",
            suite="tool_calling",
            category="simple",
            question="故宫闭馆吗",
            ground_truth=[{"name": "search_notes", "arguments": {"query": "故宫闭馆"}}],
            metadata={"loose_fields": ["query"]},
        ),
    ]
    agent = FakeAgent(
        {
            "北京天气": ['get_weather(city="北京")'],
            "上海天气": ['get_weather(city="杭州")'],
            "故宫闭馆吗": ['search_notes(query="故宫 周一 闭馆")'],
        }
    )
    report = ToolCallingEvaluator(agent).run(cases, system="fake")
    assert report.metrics["cases"] == 4
    assert report.metrics["passed"] == 3  # c1 + irrelevance + loose c4
    assert report.metrics["by_category"]["simple"]["passed"] == 2
    assert report.metrics["by_category"]["irrelevance"]["passed"] == 1
    assert report.cases[0].passed
    assert not report.cases[2].passed
    assert report.cases[3].passed
    assert report.cases[3].metrics["loose_fields"] == ["query"]
