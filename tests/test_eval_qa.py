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
    assert normalize_string("最好在**早上开园时**前往") == "最好在早上开园时前往"
    assert normalize_string("晚上七点至九点") == "19点-21点"
    assert normalize_string("19:00–21:00") == "19点-21点"
    assert normalize_string("十七点") == "17点"
    assert normalize_string("十月中旬到十一月初") == "10月中旬到11月初"
    assert normalize_string("8:30") == "8点30分"
    assert normalize_string("八点半") == "8点30分"
    assert normalize_string("六十元") == "60元"
    assert normalize_string("一点五公里") == "1.5公里"
    assert normalize_string("1.5 公里") == "1.5公里"
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
    assert quasi_exact_match("晚上7点至9点", "晚上七点至九点")
    assert not quasi_exact_match("周一闭馆", "周二闭馆")
    assert containment_match("故宫周一闭馆，需预约", "周一闭馆")
    assert containment_match("最好在**早上开园时**前往", "早上开园时前往")
    assert containment_match(
        "可以参观中国茶叶博物馆、浙江省博物馆", "中国茶叶博物馆和浙江省博物馆"
    )
    assert not containment_match("呼市", "呼和浩特")


def test_extract_final_answer():
    assert extract_final_answer("思考过程\nFINAL ANSWER: 周一闭馆") == "周一闭馆"
    assert extract_final_answer("答案：西湖") == "西湖"
    assert extract_final_answer("第一行\n最后一行") == "最后一行"
    assert (
        extract_final_answer("故宫周一闭馆，法定节假日除外。\n\n依据：资料1")
        == "故宫周一闭馆，法定节假日除外。"
    )
    assert (
        extract_final_answer("故宫周一闭馆。\n参考来源：guide.md p.1")
        == "故宫周一闭馆。"
    )


def test_citation_coverage_and_evaluate_qa():
    assert citation_coverage(["guide.md p.1", "guide.md p.2"], ["p.1"]) == 1.0
    assert citation_coverage(["guide.md p.1"], ["p.1", "p.2"]) == 0.5
    assert citation_coverage(
        ["北京旅行知识/故宫"], ["北京旅行知识 / 故宫"]
    ) == 1.0
    assert citation_coverage([], []) is None

    result = evaluate_qa(
        "FINAL ANSWER: 周一闭馆",
        "周一闭馆",
        citations=["故宫参观须知 p.1"],
        expected_citations=["p.1"],
    )
    assert result["exact_match"] is True
    assert result["citation_coverage"] == 1.0
    assert result["passed"] is True

    # 没有引用期望时，引用覆盖不参与通过判定，也不应伪装成 1.0。
    no_citation = evaluate_qa("FINAL ANSWER: 周一闭馆", "周一闭馆")
    assert no_citation["passed"] is True
    assert no_citation["citation_coverage"] is None

    # 引用覆盖是独立诊断指标：答案正确但引用为空，仍算通过。
    missing_citation = evaluate_qa(
        "FINAL ANSWER: 周一闭馆",
        "周一闭馆",
        expected_citations=["p.1"],
    )
    assert missing_citation["passed"] is True
    assert missing_citation["citation_coverage"] == 0.0


def test_qa_containment_checks_multiline_body():
    """多行 RAG 回答：事实在正文、结论在末行时不应误判。"""
    text = (
        "颐和园是否适合爬坡？\n"
        "- 东宫门到长廊一线平缓，适合不爬坡的游客。\n"
        "- 登万寿山需要连续爬台阶。\n\n"
        "因此建议走平缓路线。\n\n依据：北京旅行知识 / 颐和园"
    )
    result = evaluate_qa(text, "适合不爬坡的游客")
    assert result["contains"] is True
    assert result["passed"] is True
    assert result["exact_match"] is False


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


def test_qa_evaluator_resolves_gold_chunks_to_citations():
    case = EvalCase(
        id="q1",
        suite="qa",
        question="故宫周几闭馆？",
        ground_truth="周一闭馆",
        metadata={"gold_chunk_ids": ["c1"]},
    )
    report = QAEvaluator(
        lambda question: "FINAL ANSWER: 周一闭馆\n依据：故宫 p.1",
        citation_map={"c1": "故宫 p.1"},
    ).run([case])
    assert report.cases[0].metrics["citation_coverage"] == 1.0
    assert report.metrics["citation_cases"] == 1
