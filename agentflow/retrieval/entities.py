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
        # 常见全国性景点：城市词表无法穷举所有景区，这里补充高频目的地。
        "泰山", "华山", "嵩山", "衡山", "恒山", "庐山", "峨眉山", "武夷山",
        "长白山", "天山", "普陀山", "五台山", "九华山", "雁荡山", "武功山",
        "九寨沟", "黄龙", "青海湖", "洱海", "泸沽湖", "千岛湖", "太湖",
        "西湖", "鼓浪屿", "平遥", "宏村", "西递", "乌镇", "西塘", "周庄",
        "同里", "甪直", "南浔", "嘉峪关", "壶口", "神农架",
        "张家界国家森林公园", "亚龙湾", "蜈支洲岛", "涠洲岛", "普者黑",
        "梵净山", "荔波", "镇远", "腾冲", "稻城亚丁", "海螺沟", "色达",
        "茶卡盐湖", "可可西里", "呼伦贝尔", "阿尔山", "漠河", "雪乡",
        "长隆", "西双版纳", "喀纳斯", "禾木", "赛里木湖", "那拉提",
        "林芝", "日喀则", "珠峰大本营", "玉龙雪山",
        "北戴河", "承德避暑山庄", "平遥古城", "丽江古城", "凤凰古城",
        "阆中", "徽州", "歙县", "绩溪", "泾县", "查济", "塔川", "篁岭",
    }
)

# 未收录景区名（如“武功山”“篁岭”“滕王阁”）的轻量启发式识别：
# 先定位常见景区后缀，再向左截取少量汉字并去掉动词/助词。用于地名
# 覆盖校验的兜底，不作为通用 NER 使用；显式地名仍建议写入
# ``RAG_EXTRA_PLACES``，静态词表无法穷举全国所有景点。
_PLACE_SUFFIXES = (
    "景区", "风景区", "度假区", "古镇", "古城", "古村", "公园", "博物馆",
    "纪念馆", "美术馆", "植物园", "动物园", "湿地", "温泉", "瀑布", "峡谷",
    "草原", "雪山", "长城", "遗址", "寺", "庙", "宫", "殿", "塔", "楼",
    "阁", "园", "馆", "山", "湖", "岛", "村", "镇", "峡", "洞", "泉",
    "溪", "岭", "峰",
)
_LEADING_STOP = (
    "想我你他她它们要去到在游逛爬住往从和与的着了看玩请这一那个些"
    "还都也就又再计划准备帮带同跟给让把被对为"
)
# 这些词虽然带景区后缀，但只是泛指，不能当成具体地名。
_GENERIC_PLACES = frozenset(
    {
        "景区", "风景区", "度假区", "古镇", "古城", "古村", "公园", "博物馆",
        "纪念馆", "美术馆", "植物园", "动物园", "湿地", "温泉", "瀑布", "峡谷",
        "草原", "雪山", "长城", "遗址", "爬山", "上山", "下山", "游山", "山顶",
        "山脚", "山里", "山上", "山下", "河边", "湖边", "海边", "寺里", "庙里",
        "座山", "那片湖", "这片湖", "整个山", "这座山", "那座山",
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


_LOCATION_VERBS = "去到在游逛爬住往从看玩"
_QUESTION_MARKERS = (
    "周几", "几点", "什么时候", "开放", "闭馆", "门票", "时间", "怎么", "如何",
    "适合", "天气", "气温", "下雨", "好玩", "值得", "旅游", "旅行", "一日游",
    "两日游", "三日游", "呢", "吗", "？", "?", "，", ",", "。", "！", "!",
)


def _trim_leading(token: str) -> str:
    """截掉景区名前误带入的动词、代词与问句词。"""
    # “周末去爬山”这类前缀：以最后一个位置动词为界，只保留其后的部分。
    cut = max((token.rfind(verb) for verb in _LOCATION_VERBS), default=-1)
    if cut >= 0:
        token = token[cut + 1:]
    # “故宫周几闭馆”这类问句：在地名后的疑问词处截断。
    stops = [token.find(marker) for marker in _QUESTION_MARKERS]
    stops = [position for position in stops if position >= 0]
    if stops:
        token = token[: min(stops)]
    start = 0
    while start < len(token) and token[start] in _LEADING_STOP:
        start += 1
    return token[start:]


def detect_place_candidates(
    text: str, extra: Iterable[str] | None = None
) -> list[str]:
    """识别词表未收录的景点名，作为地名覆盖校验的兜底。

    基于常见景区后缀（山/湖/古镇/博物馆/…）向左截取 6 个汉字并清理
    动词与助词，例如「去武功山玩」→ ``["武功山"]``。仅用于判断
    “问题点名的地点在检索结果里是否真的出现”，不是通用 NER。
    ``extra`` 的含义与 :func:`detect_places` 相同。
    """
    if not text:
        return []
    known = set(PLACE_NAMES)
    known.update(_extra_places() if extra is None else extra)
    found: list[tuple[int, str]] = []
    for suffix in _PLACE_SUFFIXES:
        start = 0
        while True:
            index = text.find(suffix, start)
            if index < 0:
                break
            end = index + len(suffix)
            candidate = _trim_leading(text[max(0, index - 6):end])
            start = index + 1
            if len(candidate) < 2 or candidate in _GENERIC_PLACES:
                continue
            # 已知地名由 detect_places 负责，避免重复与误切。
            if candidate in known or any(name in candidate for name in known):
                continue
            found.append((text.find(candidate), candidate))
    found.sort(key=lambda item: (item[0], -len(item[1])))
    result: list[str] = []
    for _, name in found:
        if any(name in existing for existing in result):
            continue
        result.append(name)
    return result
