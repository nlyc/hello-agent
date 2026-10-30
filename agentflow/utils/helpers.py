"""通用小工具：JSON 提取等。"""

import json


def extract_json_object(text: str) -> dict:
    """从 LLM 输出里提取 JSON 对象，容忍 Markdown 代码围栏与前后杂文。"""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        for prefix in ("json", "python"):
            if text.lower().startswith(prefix):
                text = text[len(prefix) :]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("未找到 JSON 对象")
    return json.loads(text[start : end + 1])
