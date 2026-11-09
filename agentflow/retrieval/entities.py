"""地名识别：检测问题中提到的城市，用于回答阶段的覆盖性校验。

RAG 的常见失效模式：知识库没有某个城市，但检索仍会返回其它城市里
语义相近的片段（例如问“南昌下雨天去哪”，命中成都/上海的“雨天备选”）。
本模块提供轻量的城市词表匹配，不引入分词或 NER 依赖；需要扩展时通过
环境变量 ``RAG_EXTRA_PLACES``（逗号/顿号/空格分隔）追加自定义地名。
"""

from __future__ import annotations

import re
from typing import Iterable

from agentflow.core.env import get_env

# 常见城市与旅行目的地。只收地名本身（不含“市/省”），便于直接子串匹配。
PLACE_NAMES: frozenset[str] = frozenset(
    {
        "北京", "上海", "天津", "重庆",
        "石家庄", "唐山", "秦皇岛", "邯郸", "保定", "承德", "张家口",
        "太原", "大同", "晋中", "呼和浩特", "包头",
        "沈阳", "大连", "鞍山", "抚顺", "丹东", "锦州",
        "长春", "吉林", "哈尔滨", "齐齐哈尔", "牡丹江",
        "南京", "无锡", "徐州", "常州", "苏州", "南通", "连云港", "淮安",
        "盐城", "扬州", "镇江", "泰州", "宿迁",
        "杭州", "宁波", "温州", "嘉兴", "湖州", "绍兴", "金华", "衢州",
        "舟山", "台州", "丽水",
        "合肥", "芜湖", "蚌埠", "黄山",
        "福州", "厦门", "泉州", "漳州",
        "南昌", "景德镇", "九江", "赣州", "吉安", "宜春", "上饶",
        "济南", "青岛", "烟台", "潍坊", "威海", "日照", "临沂",
        "郑州", "洛阳", "开封", "平顶山", "安阳", "新乡", "焦作",
        "武汉", "黄石", "宜昌", "襄阳", "荆州", "恩施",
        "长沙", "株洲", "湘潭", "衡阳", "岳阳", "常德", "张家界",
        "广州", "韶关", "深圳", "珠海", "汕头", "佛山", "江门", "湛江",
        "东莞", "中山", "惠州",
        "南宁", "桂林", "北海", "柳州",
        "海口", "三亚",
        "成都", "自贡", "绵阳", "乐山", "宜宾",
        "贵阳", "遵义", "昆明", "丽江", "大理", "拉萨",
        "西安", "宝鸡", "咸阳", "延安",
        "兰州", "敦煌", "西宁", "银川", "乌鲁木齐", "喀什", "吐鲁番",
        "香港", "澳门", "台北", "高雄", "台中",
        "婺源", "阳朔", "香格里拉", "凤凰",
    }
)

_SPLIT_RE = re.compile(r"[,，、;；\s]+")


def _extra_places() -> set[str]:
    """读取 ``RAG_EXTRA_PLACES`` 追加的自定义地名。"""
    raw = get_env("RAG_EXTRA_PLACES")
    if not raw:
        return set()
    return {item.strip() for item in _SPLIT_RE.split(raw) if item.strip()}


def detect_places(text: str, extra: Iterable[str] | None = None) -> list[str]:
    """返回文本中出现的城市名（按首次出现排序，长名优先去重）。

    ``extra`` 为额外的地名白名单，通常来自 ``RAG_EXTRA_PLACES``；
    调用方不传时自动读取该配置。
    """
    if not text:
        return []
    names = set(PLACE_NAMES)
    names.update(_extra_places() if extra is None else extra)
    if not names:
        return []

    found: list[tuple[int, str]] = []
    for name in names:
        position = text.find(str(name))
        if position >= 0:
            found.append((position, str(name)))
    found.sort(key=lambda item: (item[0], -len(item[1])))

    result: list[str] = []
    for _, name in found:
        # 已被更长地名覆盖（如同时配置“吉林市”和“吉林”）时跳过。
        if any(name in existing for existing in result):
            continue
        result.append(name)
    return result
