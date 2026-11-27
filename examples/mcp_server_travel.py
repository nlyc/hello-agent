"""自建旅行 MCP server（stdio）：天气 / 景点 / 笔记检索。

运行（由 MCP 客户端通过 stdio 启动）：

    py -3.14 -m examples.mcp_server_travel

本文件只依赖标准库与 agentflow，可作为 MCPTool 的 server_command，
也能被 Claude Desktop 等任意 MCP 客户端使用。
"""

from __future__ import annotations

from agentflow.protocols.mcp import MCPServer

_WEATHER = {
    "北京": "晴，18℃，适合出行。",
    "上海": "多云，22℃，傍晚可能有阵雨。",
    "杭州": "小雨，16℃，建议带伞。",
    "成都": "阴，20℃，适合室内活动。",
    "广州": "多云，27℃，午后有雷阵雨。",
    "深圳": "晴，28℃，适合海边活动。",
    "西安": "晴，20℃，早晚温差大。",
    "南京": "阴，21℃，适宜出行。",
    "重庆": "小雨，19℃，路面湿滑。",
    "苏州": "多云，22℃，适合园林游览。",
    "厦门": "晴，26℃，海风较大。",
    "青岛": "多云，18℃，海边风大。",
    "武汉": "小雨，20℃，注意带伞。",
    "长沙": "多云，24℃，适合夜游。",
    "昆明": "晴，17℃，紫外线较强。",
    "三亚": "晴，30℃，注意防晒。",
    "哈尔滨": "晴，8℃，早晚较冷。",
    "丽江": "多云，15℃，昼夜温差大。",
}

_ATTRACTIONS = {
    "北京": "故宫（周一闭馆，需预约）、颐和园（东宫门一线平缓）、香山（秋季红叶）。",
    "上海": "外滩（夜景 19:00–21:00）、上海博物馆（雨天备选）。",
    "杭州": "西湖（环湖骑行约 15 公里）、灵隐寺。",
    "成都": "大熊猫基地（早上开园时最佳）、都江堰（雨天建议改期）。",
    "广州": "广州塔、沙面、陈家祠。",
    "深圳": "世界之窗、深圳湾公园。",
    "西安": "兵马俑、大雁塔、西安城墙。",
    "南京": "中山陵、夫子庙、南京博物院。",
    "重庆": "洪崖洞、解放碑、磁器口。",
    "苏州": "拙政园、平江路、虎丘。",
    "厦门": "鼓浪屿、厦门大学、环岛路。",
    "青岛": "栈桥、八大关、崂山。",
    "武汉": "黄鹤楼、东湖、湖北省博物馆。",
    "长沙": "橘子洲、岳麓山、湖南省博物馆。",
    "昆明": "滇池、石林、翠湖。",
    "三亚": "亚龙湾、天涯海角、蜈支洲岛。",
    "哈尔滨": "中央大街、圣索菲亚教堂、冰雪大世界。",
    "丽江": "玉龙雪山、丽江古城、束河古镇。",
}

_NOTES = (
    ("故宫周一闭馆", "故宫博物院周一闭馆，法定节假日除外。"),
    ("故宫预约", "故宫门票需提前预约，旺季建议提前三天。"),
    ("颐和园路线", "东宫门到长廊一线地势平缓，适合不爬坡的游客。"),
    ("橘子洲", "橘子洲位于长沙湘江中心，适合傍晚散步，周末有烟花表演。"),
    ("玉龙雪山", "玉龙雪山海拔较高，建议提前预约索道并注意高原反应。"),
)


def get_weather(city: str) -> str:
    """查询城市天气；只在用户明确询问天气时使用。"""
    return _WEATHER.get(str(city).strip(), f"暂未收录 {city} 的天气。")


def get_attraction(city: str) -> str:
    """推荐城市景点；只在用户要求景点推荐时使用。"""
    return _ATTRACTIONS.get(str(city).strip(), f"暂未收录 {city} 的景点。")


def search_notes(query: str) -> str:
    """检索旅行笔记中的事实（开放时间、闭馆、预约、最佳时间等）；一次只传一个检索词，不要并行多个变体。"""
    text = str(query).strip()
    if not text:
        return "检索词不能为空。"
    hits = [
        f"- {title}：{content}"
        for title, content in _NOTES
        if text in title or text in content
    ]
    return "\n".join(hits) if hits else f"未找到与「{text}」相关的笔记。"


def build_server() -> MCPServer:
    """构建旅行 MCP server。"""
    server = MCPServer("travel-tools", version="2.0.3")
    server.add_tool(get_weather)
    server.add_tool(get_attraction)
    server.add_tool(search_notes)
    server.add_resource(
        "travel://guides/beijing",
        name="北京旅行攻略",
        description="故宫、颐和园与雨天备选的静态攻略",
        content=(
            "北京：故宫周一闭馆，需提前预约；"
            "颐和园东宫门到长廊一线平缓；"
            "雨天优先中国国家博物馆、首都博物馆。"
        ),
    )
    server.add_prompt(
        "plan_trip",
        description="生成旅行规划提示词",
        arguments=[{"name": "city", "description": "目的地城市", "required": True}],
        template=(
            "请为{city}规划两日行程：写清 D1/D2、预算、同行人约束，"
            "并标注依据。"
        ),
    )
    return server


def main(argv: list[str] | None = None) -> int:
    """默认 stdio；``--http`` 时启动 HTTP/SSE 服务。"""
    import argparse
    import time

    from agentflow.protocols.mcp import MCPHttpServer

    parser = argparse.ArgumentParser(description="旅行 MCP server")
    parser.add_argument("--http", action="store_true", help="使用 HTTP/SSE 传输")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    server = build_server()
    if not args.http:
        server.run_stdio()
        return 0

    http = MCPHttpServer(server, host=args.host, port=args.port)
    url = http.start()
    print(f"MCP HTTP endpoint: {url}", flush=True)
    print(f"MCP SSE endpoint:  {url.rsplit('/mcp', 1)[0]}/mcp/sse", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        http.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
