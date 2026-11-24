"""TerminalTool 测试：白名单、沙箱、元字符、超时、截断与只读命令。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from agentflow.tools.terminal_tool import TerminalTool


def make_tool(tmp_path: Path, **kwargs) -> TerminalTool:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    (workspace / "a.md").write_text("故宫周一闭馆\n需要提前预约\n", encoding="utf-8")
    (workspace / "b.txt").write_text("上海外滩夜景\n", encoding="utf-8")
    sub = workspace / "sub"
    sub.mkdir(exist_ok=True)
    (sub / "c.md").write_text("苏州园林\n", encoding="utf-8")
    return TerminalTool(workspace, **kwargs)


def test_pwd_cd_and_sandbox_escape(tmp_path):
    tool = make_tool(tmp_path)
    assert tool.run({"command": "pwd"}) == str(tool.workspace)
    assert "cwd=" in tool.run({"command": "cd sub"})
    assert tool.run({"command": "pwd"}).endswith("sub")
    assert "cwd=" in tool.run({"command": "cd .."})
    assert tool.run({"command": "pwd"}) == str(tool.workspace)
    assert tool.run({"command": "cd ~"}) == f"cwd={tool.workspace}"
    assert tool.run({"command": "cd ../../.."}).startswith("错误: 路径越界")
    assert tool.run({"command": "cd missing"}).startswith("错误:")


def test_read_only_commands(tmp_path):
    tool = make_tool(tmp_path)
    assert "a.md" in tool.run({"command": "ls"})
    assert "sub/" in tool.run({"command": "ls"})
    assert tool.run({"command": "cat a.md"}).startswith("故宫周一闭馆")
    assert tool.run({"command": "head -n 1 a.md"}) == "故宫周一闭馆"
    assert tool.run({"command": "tail -n 1 a.md"}) == "需要提前预约"
    assert tool.run({"command": "echo 你好 世界"}) == "你好 世界"
    assert "a.md" in tool.run({"command": 'find . -name "*.md"'})
    assert "c.md" in tool.run({"command": 'find sub -name "*.md"'})
    assert "a.md:1" in tool.run({"command": 'grep -n "闭馆" .'})
    assert tool.run({"command": "wc -l a.md"}) == "2 a.md"
    assert "size=" in tool.run({"command": "stat a.md"})
    assert tool.run({"command": "which definitely-not-a-command"}).startswith("未找到")


def test_sort_uniq_cut(tmp_path):
    tool = make_tool(tmp_path)
    (tool.workspace / "nums.txt").write_text("b\na\nb\n", encoding="utf-8")
    (tool.workspace / "csv.txt").write_text("a,1\nb,2\n", encoding="utf-8")
    assert tool.run({"command": "sort nums.txt"}) == "a\nb\nb"
    assert tool.run({"command": "uniq nums.txt"}) == "b\na\nb"
    assert tool.run({"command": "cut -d , -f 2 csv.txt"}) == "1\n2"
    assert tool.run({"command": "cut -d , -f 1-2 csv.txt"}) == "a,1\nb,2"


def test_rejects_non_whitelisted_and_meta_characters(tmp_path):
    tool = make_tool(tmp_path)
    assert tool.run({"command": "rm -rf /"}).startswith("错误: 不允许的命令")
    for command in ("echo hi; rm -rf /", "echo hi && whoami", "echo $(whoami)", "echo `whoami`"):
        assert "元字符" in tool.run({"command": command})


def test_rejects_path_escape_and_absolute(tmp_path):
    tool = make_tool(tmp_path)
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    assert "路径越界" in tool.run({"command": "cat ../secret.txt"})
    assert "路径越界" in tool.run({"command": "cat /etc/passwd"})
    assert "路径越界" in tool.run({"command": "grep secret .."})


def test_pipes_disabled_by_default(tmp_path):
    tool = make_tool(tmp_path)
    result = tool.run({"command": "cat a.md | head -n 1"})
    assert result.startswith("错误: 不允许使用管道符")
    piped = make_tool(tmp_path, allow_pipes=True)
    result = piped.run({"command": "cat a.md | head -n 1"})
    assert "管道暂只支持外部命令" in result


def test_external_command_pipeline_and_exit_code(tmp_path):
    tool = make_tool(
        tmp_path,
        allowed_commands=[Path(sys.executable).name],
        allow_pipes=True,
    )
    workspace = tool.workspace
    (workspace / "echoer.py").write_text(
        "print('b')\nprint('a')\n", encoding="utf-8"
    )
    (workspace / "upper.py").write_text(
        "import sys\nsys.stdout.write(sys.stdin.read().upper())\n",
        encoding="utf-8",
    )
    (workspace / "exit3.py").write_text(
        "raise SystemExit(3)\n", encoding="utf-8"
    )
    python = Path(sys.executable).as_posix()
    pipeline = tool.run(
        {"command": f'"{python}" echoer.py | "{python}" upper.py'}
    )
    assert pipeline == "B\nA"
    failed = tool.run({"command": f'"{python}" exit3.py'})
    assert failed.startswith("[exit 3]")


def test_timeout(tmp_path):
    tool = make_tool(tmp_path, allowed_commands=[Path(sys.executable).name], timeout=0.2)
    (tool.workspace / "sleep.py").write_text(
        "import time\ntime.sleep(2)\n", encoding="utf-8"
    )
    python = Path(sys.executable).as_posix()
    result = tool.run({"command": f'"{python}" sleep.py'})
    assert "超时" in result


def test_output_truncation(tmp_path):
    tool = make_tool(tmp_path, max_output_chars=10, max_output_bytes=10)
    (tool.workspace / "big.txt").write_text("x" * 100, encoding="utf-8")
    result = tool.run({"command": "cat big.txt"})
    assert "已截断" in result
    assert len(result) < 100


def test_binary_and_missing_file(tmp_path):
    tool = make_tool(tmp_path)
    (tool.workspace / "bin.dat").write_bytes(b"\x00\x01\x02")
    assert "疑似二进制" in tool.run({"command": "cat bin.dat"})
    assert "不存在" in tool.run({"command": "cat missing.md"})


def test_symlink_escape_is_blocked(tmp_path):
    tool = make_tool(tmp_path)
    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    link = tool.workspace / "link.md"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("当前平台不允许创建符号链接")
    assert "路径越界" in tool.run({"command": "cat link.md"})


def test_empty_command_and_non_terminal_registry(tmp_path):
    tool = make_tool(tmp_path)
    assert tool.run({}) == "错误: command 不能为空"
    assert tool.validate_parameters({"command": "pwd"}) == []
