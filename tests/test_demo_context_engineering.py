"""上下文工程 Demo 回归测试（离线）。"""

from __future__ import annotations

import json

from examples.demo_context_engineering import main


def test_demo_context_engineering(tmp_path, capsys):
    assert main(tmp_path) == 0
    output = capsys.readouterr().out
    assert "# Role & Policies" in output
    assert "# Evidence" in output
    assert "# State" in output
    assert "周六" not in output  # 不编造不存在的信息
    assert "blocker" in output
    assert "节省=" in output

    payload = json.loads((tmp_path / "context.json").read_text(encoding="utf-8"))
    assert payload["naive_tokens"] > 0
    assert payload["saved_tokens"] > 0
    assert payload["built"]["stats"]["max_tokens"] == 600
    assert "# Task" not in payload["built"]["prompt_text"]
    assert any(note["type"] == "blocker" for note in payload["notes"])
