"""地名识别测试：城市检测、排序与自定义地名扩展。"""

from agentflow.retrieval import detect_places


def test_detects_city_in_question():
    assert detect_places("南昌下雨天适合去哪？", extra=[]) == ["南昌"]
    assert detect_places("我想去北京玩一天，有什么景点推荐？", extra=[]) == ["北京"]


def test_no_place_returns_empty():
    assert detect_places("故宫几点开放？", extra=[]) == []
    assert detect_places("下雨天适合去哪？", extra=[]) == []
    assert detect_places("", extra=[]) == []


def test_multiple_places_ordered_by_occurrence():
    assert detect_places("从北京坐高铁去上海，再去杭州。", extra=[]) == [
        "北京",
        "上海",
        "杭州",
    ]


def test_extra_places_are_supported():
    assert detect_places("去某某镇玩两天", extra=["某某镇"]) == ["某某镇"]


def test_extra_places_from_env(monkeypatch):
    monkeypatch.setenv("RAG_EXTRA_PLACES", "某某镇、某某村")
    assert detect_places("去某某村看花") == ["某某村"]
