"""生成架构图 PNG（交付物 D2）。

用法：
    py -3.14 scripts/render_architecture.py

输出：docs/architecture.png（分层 + 数据流标注）。仅依赖 Pillow，离线可跑。
"""

from __future__ import annotations

import math
import os
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "docs", "architecture.png")

FONT = "C:/Windows/Fonts/msyh.ttc"
FONT_BOLD = "C:/Windows/Fonts/msyhbd.ttc"

W, H = 1560, 1000

# 配色
C_TEXT = (38, 50, 56)
C_ARROW = (84, 110, 122)
C_CAPTION = (96, 125, 139)
C_LAYER_LABEL = (69, 90, 100)

C_APP_BG = (227, 242, 253)
C_APP_FILL = (144, 202, 249)
C_ORCH_BG = (232, 245, 233)
C_ORCH_FILL = (165, 214, 167)
C_BASE_BG = (255, 248, 225)
C_BASE_FILL = (255, 224, 130)
C_UTILS_BG = (236, 239, 241)
C_UTILS_FILL = (176, 190, 197)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size, index=0)


def text_w(d: ImageDraw.ImageDraw, text: str, f) -> int:
    return d.textbbox((0, 0), text, font=f)[2]


def center_text(d, cx, cy, text, f, fill=C_TEXT):
    w = text_w(d, text, f)
    d.text((cx - w / 2, cy - f.size / 2), text, font=f, fill=fill)


def draw_box(d, xy, lines, fill):
    """lines: list[(text, font)]；整块垂直居中。"""
    x0, y0, x1, y1 = xy
    d.rounded_rectangle(xy, radius=12, fill=fill, outline=C_ARROW, width=2)
    gap = 10
    total = sum(f.size for _, f in lines) + gap * (len(lines) - 1)
    cy = (y0 + y1) / 2 - total / 2
    for text, f in lines:
        center_text(d, (x0 + x1) / 2, cy + f.size / 2, text, f)
        cy += f.size + gap


def draw_layer(d, label, y, h, bg):
    d.rounded_rectangle([40, y, W - 40, y + h], radius=18, fill=bg, outline=(207, 216, 220), width=1)
    d.text((64, y + 14), label, font=_font(20, bold=True), fill=C_LAYER_LABEL)


def arrow(d, x1, y1, x2, y2, color=C_ARROW, width=3, head=13):
    d.line([x1, y1, x2, y2], fill=color, width=width)
    ang = math.atan2(y2 - y1, x2 - x1)
    for da in (math.radians(150), math.radians(-150)):
        d.line(
            [x2, y2, x2 + head * math.cos(ang + da), y2 + head * math.sin(ang + da)],
            fill=color,
            width=width,
        )


def build() -> Image.Image:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    f_title = _font(34, bold=True)
    f_layer = _font(20, bold=True)
    f_box = _font(22, bold=True)
    f_sub = _font(16)
    f_cap = _font(16)
    f_legend_title = _font(20, bold=True)
    f_legend = _font(16)

    # 标题
    center_text(d, W / 2, 46, "agentflow · 自研智能体框架 v1.0 —— 分层架构", f_title)

    # ---- 应用层 ----
    draw_layer(d, "应用层 · 案例", 90, 170, C_APP_BG)
    draw_box(
        d,
        (100, 140, 520, 232),
        [("旅行助手 · 三范式", f_box), ("ReAct / Reflection / PlanAndSolve", f_sub)],
        C_APP_FILL,
    )
    draw_box(
        d,
        (560, 140, 1050, 232),
        [("旅行助手 · 低代码工作流", f_box), ("understand → weather → attraction → answer", f_sub)],
        C_APP_FILL,
    )

    # ---- 编排层 ----
    draw_layer(d, "编排层", 290, 160, C_ORCH_BG)
    draw_box(
        d,
        (100, 330, 520, 420),
        [("agents/", f_box), ("ReAct · Reflection · PlanAndSolve", f_sub)],
        C_ORCH_FILL,
    )
    draw_box(
        d,
        (560, 330, 1050, 420),
        [("workflow/", f_box), ("StateGraph · 节点 · 条件边", f_sub)],
        C_ORCH_FILL,
    )

    # ---- 基础层 ----
    draw_layer(d, "基础层", 470, 200, C_BASE_BG)
    base = [
        ("core/", "Message · Config\nAgent基类 · exceptions"),
        ("llm/", "BaseLLM · OpenAI兼容\n模型注册表"),
        ("tools/", "Tool · Registry热插拔\nChain · async"),
        ("memory/", "MessageHistory\nBaseMemory · 持久化"),
        ("trace/", "TraceEvent · Tracer\n回调订阅"),
    ]
    bx, bw, gap = 100, 176, 14
    for i, (name, sub) in enumerate(base):
        x = bx + i * (bw + gap)
        sub_lines = [(ln, f_sub) for ln in sub.split("\n")]
        draw_box(d, (x, 520, x + bw, 640), [(name, f_box)] + sub_lines, C_BASE_FILL)

    # ---- utils ----
    draw_layer(d, "通用工具", 690, 110, C_UTILS_BG)
    draw_box(
        d,
        (100, 720, 1050, 780),
        [("utils/", f_box), ("helpers · logging · serialization", f_sub)],
        C_UTILS_FILL,
    )

    # ---- 箭头（数据流） ----
    arrow(d, 575, 260, 575, 330)  # 应用层 → 编排层
    arrow(d, 575, 450, 575, 520)  # 编排层 → 基础层
    arrow(d, 575, 670, 575, 720)  # 基础层 → utils

    # 数据流标注：右侧面板
    px, py, pw, ph = 1140, 90, 380, 500
    d.rounded_rectangle([px, py, px + pw, py + ph], radius=16, fill=(250, 250, 250), outline=(207, 216, 220), width=2)
    d.text((px + 24, py + 20), "数据流标注", font=f_legend_title, fill=C_LAYER_LABEL)

    legends = [
        ("trace 贯穿全链路", "Agent / Workflow 每个关键动作 emit 事件，\n经 Tracer 订阅回调，落盘 JSONL / 统计。"),
        ("memory 读写", "MessageHistory 管理上下文；\nBaseMemory 可插拔持久化，支持 as_tool 桥接。"),
        ("模型切换", "同一公司中转站（OpenAI 兼容），\n换 model 名即切换 DeepSeek / GLM / OpenAI。"),
    ]
    ly = py + 70
    for title, body in legends:
        d.text((px + 24, ly), title, font=f_box, fill=C_TEXT)
        ly += 34
        for ln in body.split("\n"):
            d.text((px + 28, ly), ln, font=f_legend, fill=C_CAPTION)
            ly += 24
        ly += 12

    # 底部依赖说明
    caption = (
        "依赖关系：agents/ 依赖 core · llm · tools · memory · trace；"
        "workflow/ 依赖 core · llm · tools · trace；底层统一依赖 utils/"
    )
    cw = text_w(d, caption, f_cap)
    d.text(((W - cw) / 2, H - 44), caption, font=f_cap, fill=C_CAPTION)

    return img


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    img = build()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    img.save(OUT)
    print(f"已生成 {OUT}  ({img.width}x{img.height})")
