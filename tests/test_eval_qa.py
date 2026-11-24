"""QA 准精确匹配与评估器测试。"""

from __future__ import annotations

from agentflow.evaluation import (
    EvalCase,
    QAEvaluator,
    as_number,
    citation_coverage,
    containment_match,
    evaluate_qa,
    extract_final_answer,
    normalize_list,
    normalize_string,
    quasi_exact_match,
)


def test_as_number_normalizes_currency_units_and_percent():
    assert as_number("$1,234.56") == 1234.56
    assert as_number("60元") == 60.0
    assert as_number("50%") == 50.0
    assert as_number("不是数字") is None
    assert as_number(True) is None


def test_normalize_string_and_list():
    assert normalize_string("The United States.") == "united states"
    assert normalize_list("Paris, London, Berlin") == [
        "berlin",
        "london",
        "paris",
    ]
    assert normalize_list("单个值") is None


def test_quasi_exact_and_containment():
    assert quasi_exact_match("$1,234.56", "1234.56")
    assert quasi_exact_match("Paris, London, Berlin", "Berlin、London、Paris")
    assert quasi_exact_match("The United States.", "united states")
    assert not quasi_exact_match("周一闭馆", "周二闭馆")
    assert containment_match("故宫周一闭馆，需预约", "周一闭馆")


def test_extract_final_answer():
    assert extract_final_answer("思考过程\nFINAL ANSWER: 周一闭馆") == "周一闭馆"
    assert extract_final_answer("答案：西湖") == "西湖"
    assert extract_final_answer("第一行\n最后一行") == "最后一行"


def test_citation_coverage_and_evaluate_qa():
    assert citation_coverage(["guide.md p.1", "guide.md p.2"], ["p.1"]) == 1.0
    assert citation_coverage(["guide.md p.1"], ["p.1", "p.2"]) == 0.5
    assert citation_coverage([], []) == 1.0

    result = evaluate_qa(
        "FINAL ANSWER: 周一闭馆",
        "周一闭馆",
        citations=["故宫参观须知 p.1"],
        expected_citations=["p.1"],
    )
    assert result["exact_match"] is True
    assert result["passed"] is True


def test_qa_evaluator():
    cases = [
        EvalCase(
            id="q1",
            suite="qa",
            question="故宫周几闭馆？",
            ground_truth="周一闭馆",
            metadata={"expected_citations": ["p.1"]},
        ),
        EvalCase(
            id="q2",
            suite="qa",
            question="上海雨天去哪？",
            ground_truth="上海博物馆",
        ),
    ]

    def answer(question: str) -> str:
        if "故宫" in question:
            return "依据：故宫参观须知 p.1\nFINAL ANSWER: 周一闭馆"
        return "FINAL ANSWER: 外滩"

    report = QAEvaluator(answer, system="fake").run(cases)
    assert report.metrics["cases"] == 2
    assert report.metrics["passed"] == 1
    assert report.metrics["qem"] == 0.5
    assert report.cases[0].passed
    assert not report.cases[1].passed


def test_qa_evaluator_records_invalid_callable_as_case_error():
    report = QAEvaluator(object()).run(
        [EvalCase(id="x", suite="qa", question="q", ground_truth="a")]
    )
    assert report.cases[0].error
    assert not report.cases[0].passed
