"""LLM Judge / Win Rate / 约束测试。"""

from __future__ import annotations

import pytest

from agentflow.evaluation import (
    LLMJudge,
    WinRateJudge,
    check_itinerary,
    parse_judge_output,
)
from tests.fakes import ScriptedLLM


def test_parse_judge_output_with_code_fence():
    result = parse_judge_output(
        '```json\n{"feasibility": 4, "constraint_compliance": 5, '
        '"information_quality": 4, "format": 5, "comment": "不错"}\n```'
    )
    assert result.average == 4.5
    assert result.passed and result.excellent
    assert result.comment == "不错"


def test_parse_judge_output_with_prefix_brace():
    result = parse_judge_output(
        '说明 {不是 JSON} 结果: {"feasibility": 4, "constraint_compliance": 4, '
        '"information_quality": 4, "format": 4}'
    )
    assert result.scores["feasibility"] == 4
    assert result.average == 4.0


def test_parse_judge_output_missing_fields_scores_zero():
    result = parse_judge_output('{"feasibility": 5}')
    assert result.scores["feasibility"] == 5
    assert result.scores["format"] == 0
    assert result.average == 1.25
    assert not result.passed


def test_judge_handles_invalid_output():
    judge = LLMJudge(ScriptedLLM(["不是 JSON"]))
    result = judge.judge("任务", "行程")
    assert result.error
    assert result.average == 0.0


def test_judge_with_scripted_llm():
    llm = ScriptedLLM(
        [
            '{"feasibility": 4, "constraint_compliance": 4, '
            '"information_quality": 4, "format": 4, "comment": "可用"}'
        ]
    )
    result = LLMJudge(llm).judge("北京两日游", "D1 故宫")
    assert result.passed
    assert result.average == 4.0
    assert llm.calls[0]["json_mode"] is True


def test_win_rate_swaps_order():
    # 第一次 A 胜，第二次交换后 B（原 A）仍胜 → A 两票。
    llm = ScriptedLLM(
        [
            '{"winner": "A", "reason": "更好"}',
            '{"winner": "B", "reason": "更好"}',
        ]
    )
    result = WinRateJudge(llm).compare("任务", "行程 A", "行程 B")
    assert result["wins_a"] == 2
    assert result["wins_b"] == 0
    assert result["win_rate"] == 1.0
    assert result["rounds"] == 2


def test_win_rate_disagreement_ties():
    llm = ScriptedLLM(
        [
            '{"winner": "A"}',
            '{"winner": "A"}',
        ]
    )
    result = WinRateJudge(llm).compare("任务", "行程 A", "行程 B")
    # 第二票翻转后是 B，因此 A/B 各一票。
    assert result["wins_a"] == 1
    assert result["wins_b"] == 1
    assert result["win_rate"] == 0.5


def test_constraints_checker():
    text = "D1 故宫（周二）→ 颐和园东宫门；D2 香山。预算 1500 元。带父母。"
    result = check_itinerary(
        text,
        {
            "days": 2,
            "budget": 1500,
            "must_include": ["故宫"],
            "must_avoid": ["周一", "爬坡"],
            "companions": ["父母"],
        },
    )
    assert result["passed"] is True
    assert all(result["checks"].values())

    failed = check_itinerary("D1 故宫，预算 3000 元。", {"days": 2, "budget": 1500})
    assert failed["passed"] is False
    assert failed["checks"]["days"] is False
    assert failed["checks"]["budget"] is False


def test_constraints_distinguish_negation():
    # 「不安排故宫」不能被判为包含故宫。
    assert not check_itinerary(
        "不安排故宫、景山等走路多的项目。D1 天坛；D2 颐和园。预算 1500 元。",
        {"days": 2, "budget": 1500, "must_include": ["故宫"]},
    )["checks"]["must_include"]

    # 「避开周一」「周一闭馆」是合规表达。
    assert check_itinerary(
        "避开周一，故宫周一闭馆。D1 天坛；D2 颐和园。预算 1500 元。",
        {"days": 2, "budget": 1500, "must_avoid": ["周一"]},
    )["checks"]["must_avoid"]

    # 真的安排在周一才违规。
    assert not check_itinerary(
        "D1 周一去故宫；D2 颐和园。预算 1500 元。",
        {"days": 2, "budget": 1500, "must_avoid": ["周一"]},
    )["checks"]["must_avoid"]

    assert check_itinerary(
        "东宫门一线平缓，不爬坡。预算 1500 元。",
        {"budget": 1500, "must_avoid": ["爬坡"]},
    )["checks"]["must_avoid"]

    # 条件句「若恰逢周一…确保不在周一」不算违规。
    assert check_itinerary(
        "D1 故宫（避开周一）。若D1恰逢周一，与D2对调，确保故宫不在周一参观。预算 1500 元。",
        {"days": 1, "budget": 1500, "must_avoid": ["周一"]},
    )["checks"]["must_avoid"]

    # 上一句的「闭馆」不应把下一句的「带父母」误判为否定。
    assert check_itinerary(
        "故宫周一闭馆；带父母。预算 1500 元。",
        {"budget": 1500, "companions": ["父母"]},
    )["checks"]["companions"]

    # 「如遇周一」「须避开周一」都是条件/规避表达。
    assert check_itinerary(
        "D1 故宫（须避开周一）。如遇周一，与D2对调。预算 1500 元。",
        {"days": 1, "budget": 1500, "must_avoid": ["周一"]},
    )["checks"]["must_avoid"]

    # 复杂真实输出：多处提及周一但全部是规避/闭馆/条件。
    assert check_itinerary(
        "D1必须避开周一；若你出行D1恰逢周一，则D1/D2对调。"
        "故宫周一闭馆；D1不要排周一。建议选非周一。预算 1500 元。",
        {"days": 1, "budget": 1500, "must_avoid": ["周一"]},
    )["checks"]["must_avoid"]

    # 「少爬坡」是约束表达，不是安排了爬坡。
    assert check_itinerary(
        "尽量少爬坡；不安排登山。预算 1500 元。",
        {"budget": 1500, "must_avoid": ["爬坡"]},
    )["checks"]["must_avoid"]

    # 预算块内取最大金额：「合计 ≤500」应通过，不会被前面的门票价误导。
    assert check_itinerary(
        "预算控制：故宫门票 60 元，交通餐饮预留 440 元，合计 ≤500 元。",
        {"budget": 500},
    )["checks"]["budget"]

    # 调整段落里「多人会涨到 1000–1100」不应覆盖预算明细的实际合计。
    assert check_itinerary(
        "## 预算明细\n住宿 140 元，餐饮 170 元，合计 555 元，800 元内预留 245 元。\n\n"
        "## 同行人调整\n若 2 人总预算仅 800 元会偏紧，建议提高到 1000–1100 元。",
        {"budget": 800},
    )["checks"]["budget"]

    # 否定列表跨越较长距离；引用资料的「需要连续爬坡」不算安排。
    assert check_itinerary(
        "不安排：长城、景山、香山、颐和园万寿山等爬坡/台阶多的景点。"
        "资料明确“登顶需要连续爬坡约一小时”。D2 安排室内博物馆。预算 1500 元。",
        {"budget": 1500, "must_avoid": ["爬坡"]},
    )["checks"]["must_avoid"]

    # 但真实的「安排爬坡」仍应判违规。
    assert not check_itinerary(
        "D1 安排爬坡上山。预算 1500 元。",
        {"budget": 1500, "must_avoid": ["爬坡"]},
    )["checks"]["must_avoid"]

    assert check_itinerary(
        "不建议安排香山登顶：资料明确“登顶需要连续爬坡约一小时”。",
        {"must_avoid": ["爬坡"]},
    )["checks"]["must_avoid"]
