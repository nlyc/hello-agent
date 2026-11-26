"""TerminalTool：只读文件系统即时探索（JIT 上下文）。

四层安全：
1. 命令白名单；
2. 工作目录沙箱（禁止 ``..`` 逃逸与符号链接逃逸）；
3. 超时；
4. 输出截断。

相比教学实现额外加固：默认 ``shell=False`` + ``shlex``，拒绝 shell 元字符；
常用命令用进程内只读实现，保证 Windows / Linux 行为一致。
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from agentflow.core.exceptions import ToolError
from agentflow.tools.base import Tool, ToolParameter

# 默认只读命令。刻意不包含 awk / sed：二者可执行系统命令或写文件，
# 与「只读探索」目标冲突。
_DEFAULT_COMMANDS: tuple[str, ...] = (
    "pwd",
    "cd",
    "echo",
    "ls",
    "dir",
    "cat",
    "head",
    "tail",
    "find",
    "grep",
    "wc",
    "stat",
    "which",
    "sort",
    "uniq",
    "cut",
    "basename",
    "dirname",
    "realpath",
    "du",
    "df",
)

_SHELL_META = set(";&><`$()")
_TEXT_SUFFIXES = {
    ".md",
    ".txt",
    ".py",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".csv",
    ".log",
    ".html",
    ".xml",
    ".toml",
    ".ini",
    ".cfg",
}
_MAX_TEXT_FILE_BYTES = 2_000_000


class TerminalTool(Tool):
    """安全的只读终端探索工具。

    参数：
        workspace: 沙箱根目录；不存在时自动创建。
        allowed_commands: 命令白名单；默认只读集合。
        timeout: 外部命令超时（秒）。
        max_output_bytes: 输出字节上限。
        max_output_chars: 输出字符上限。
        allow_pipes: 是否允许 ``|`` 管道（仅外部命令）。
    """

    def __init__(
        self,
        workspace: str | Path,
        *,
        allowed_commands: Sequence[str] | None = None,
        timeout: float = 30.0,
        max_output_bytes: int = 1_000_000,
        max_output_chars: int = 20_000,
        allow_pipes: bool = False,
    ) -> None:
        super().__init__(
            name="terminal",
            description=(
                "在只读沙箱内即时探索文件系统。command 示例："
                "`pwd`、`cd sub`、`ls -a .`、`cat docs/guide.md`、"
                "`head -n 20 docs/guide.md`、`find . -name \"*.md\"`、"
                "`grep -n \"闭馆\" .`、`wc -l docs/guide.md`。"
                "只允许白名单命令，禁止 shell 元字符与越界路径。"
            ),
        )
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        if not self.workspace.is_dir():
            raise ToolError(f"terminal workspace 不是目录：{self.workspace}")
        self._cwd = self.workspace
        entries = [
            str(command).strip()
            for command in (allowed_commands or _DEFAULT_COMMANDS)
            if str(command).strip()
        ]
        path_entries = [
            entry
            for entry in entries
            if any(separator in entry for separator in ("/", "\\", ":"))
        ]
        self.allowed_commands = {
            entry.lower().removesuffix(".exe")
            for entry in entries
            if entry not in path_entries
        }
        self.allowed_paths = {
            str(Path(entry).expanduser().resolve()) for entry in path_entries
        }
        self.timeout = float(timeout)
        self.max_output_bytes = max(1, int(max_output_bytes))
        self.max_output_chars = max(1, int(max_output_chars))
        self.allow_pipes = bool(allow_pipes)
        self._builtins: dict[str, Callable[[list[str], Path], str]] = {
            "pwd": self._builtin_pwd,
            "cd": self._builtin_cd,
            "echo": self._builtin_echo,
            "ls": self._builtin_ls,
            "dir": self._builtin_ls,
            "cat": self._builtin_cat,
            "head": self._builtin_head,
            "tail": self._builtin_tail,
            "find": self._builtin_find,
            "grep": self._builtin_grep,
            "wc": self._builtin_wc,
            "stat": self._builtin_stat,
            "which": self._builtin_which,
            "sort": self._builtin_sort,
            "uniq": self._builtin_uniq,
            "cut": self._builtin_cut,
        }

    def run(self, parameters: dict[str, Any]) -> str:
        """执行只读命令；错误返回字符串而不是抛异常。"""
        command = str(
            parameters.get("command") or parameters.get("input") or ""
        ).strip()
        if not command:
            return "错误: command 不能为空"
        try:
            return self.execute(command)
        except ToolError as exc:
            return f"错误: {exc}"
        except Exception as exc:  # 防御性：工具不拖垮 Agent
            return f"错误: 终端执行失败：{exc}"

    def execute(self, command: str) -> str:
        """校验并执行命令（供测试直接调用）。"""
        text = str(command or "").strip()
        if not text:
            raise ToolError("command 不能为空")
        self._check_meta(text)
        segments = [segment.strip() for segment in text.split("|")]
        if len(segments) > 1:
            if not self.allow_pipes:
                raise ToolError("不允许使用管道符 |")
            if any(not segment for segment in segments):
                raise ToolError("管道命令不能为空")
            return self._run_pipeline(segments)
        return self._run_single(text)

    def get_parameters(self) -> list[ToolParameter]:
        """工具参数自描述。"""
        return [
            ToolParameter(
                "command", "string", "只读命令，如 `grep -n \"闭馆\" .`", True
            )
        ]

    # ------------------------------------------------------------ dispatcher

    def _check_meta(self, text: str) -> None:
        for char in _SHELL_META:
            if char in text:
                raise ToolError(f"不允许的 shell 元字符：{char}")
        for char, label in (("\n", "换行"), ("\r", "回车"), ("\0", "空字符")):
            if char in text:
                raise ToolError(f"不允许的{label}")

    def _is_allowed(self, command: str) -> bool:
        if any(separator in command for separator in ("/", "\\", ":")):
            try:
                resolved = str(Path(command).expanduser().resolve())
            except OSError:
                return False
            return resolved in self.allowed_paths
        name = Path(command).name
        if name.lower().endswith(".exe"):
            name = name[:-4]
        return name.lower() in self.allowed_commands

    def _run_single(self, text: str) -> str:
        argv = _split(text)
        if not argv:
            raise ToolError("命令为空")
        command = argv[0]
        if not self._is_allowed(command):
            raise ToolError(f"不允许的命令：{command}")
        handler = self._builtins.get(Path(command).name.lower().removesuffix(".exe"))
        if handler is not None:
            return self._truncate(handler(argv[1:], self._cwd))
        return self._run_external(argv, self._cwd)

    def _run_pipeline(self, segments: list[str]) -> str:
        """外部命令顺序管道：上一段 stdout 作为下一段 stdin。"""
        parsed: list[list[str]] = []
        for segment in segments:
            argv = _split(segment)
            if not argv:
                raise ToolError("管道命令为空")
            name = Path(argv[0]).name.lower().removesuffix(".exe")
            if name in self._builtins:
                raise ToolError("管道暂只支持外部命令")
            if not self._is_allowed(argv[0]):
                raise ToolError(f"不允许的命令：{argv[0]}")
            parsed.append(argv)
        output = ""
        for index, argv in enumerate(parsed):
            output = self._run_external(
                argv, self._cwd, input_text=output if index > 0 else None
            )
        return output

    # ------------------------------------------------------------- external

    def _run_external(
        self,
        argv: list[str],
        cwd: Path,
        *,
        input_text: str | None = None,
    ) -> str:
        command = argv[0]
        if any(separator in command for separator in ("/", "\\", ":")):
            resolved = Path(command).expanduser().resolve()
            if str(resolved) not in self.allowed_paths:
                raise ToolError(f"路径越界：{command}")
            if not resolved.is_file():
                raise ToolError(f"可执行文件不存在：{command}")
            executable = str(resolved)
        else:
            executable = shutil.which(command)
            if executable is None:
                raise ToolError(f"命令不存在或不在 PATH：{command}")
        self._validate_argv_paths(argv, cwd)
        env = dict(os.environ)
        env.setdefault("PYTHONIOENCODING", "utf-8")
        try:
            completed = subprocess.run(
                [executable, *argv[1:]],
                cwd=str(cwd),
                input=input_text.encode("utf-8") if input_text is not None else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout,
                env=env,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolError(f"命令超时（>{self.timeout:g}s）：{command}") from exc
        except OSError as exc:
            raise ToolError(f"命令启动失败：{exc}") from exc

        output = _merge_output(completed.stdout, completed.stderr)
        if completed.returncode != 0:
            output = f"[exit {completed.returncode}] {output}".strip()
        return self._truncate(output)

    def _validate_argv_paths(self, argv: list[str], cwd: Path) -> None:
        """对外部命令中看起来像路径的参数做沙箱校验。"""
        for argument in argv[1:]:
            if _looks_like_path(argument):
                self._resolve_path(argument, cwd)

    # -------------------------------------------------------------- builtins

    def _builtin_pwd(self, args: list[str], cwd: Path) -> str:
        return str(cwd)

    def _builtin_cd(self, args: list[str], cwd: Path) -> str:
        target = args[0] if args else "~"
        if target in ("~", "$HOME"):
            path = self.workspace
        else:
            path = self._resolve_path(target, cwd)
        if not path.exists():
            raise ToolError(f"目录不存在：{target}")
        if not path.is_dir():
            raise ToolError(f"不是目录：{target}")
        self._cwd = path
        return f"cwd={self._cwd}"

    def _builtin_echo(self, args: list[str], cwd: Path) -> str:
        # shell=False，因此这里不做变量展开，原样输出。
        return " ".join(args)

    def _builtin_ls(self, args: list[str], cwd: Path) -> str:
        show_hidden = False
        target = "."
        for argument in args:
            if argument.startswith("-"):
                if "a" in argument.lower():
                    show_hidden = True
                continue
            target = argument
        path = self._resolve_path(target, cwd)
        if not path.exists():
            raise ToolError(f"路径不存在：{target}")
        if path.is_file():
            return path.name
        entries = sorted(
            path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())
        )
        lines: list[str] = []
        for entry in entries:
            if not show_hidden and entry.name.startswith("."):
                continue
            suffix = "/" if entry.is_dir() else ""
            lines.append(f"{entry.name}{suffix}")
            if len(lines) >= 500:
                lines.append("…（已截断）")
                break
        return "\n".join(lines) if lines else "（空目录）"

    def _builtin_cat(self, args: list[str], cwd: Path) -> str:
        paths = [arg for arg in args if not arg.startswith("-")]
        if not paths:
            raise ToolError("cat 需要文件路径")
        parts: list[str] = []
        for raw in paths:
            path = self._resolve_path(raw, cwd)
            text = self._read_text(path)
            if len(paths) > 1:
                parts.append(f"==> {path.relative_to(self.workspace)} <==")
            parts.append(text)
        return "\n".join(parts).rstrip() or "（空文件）"

    def _builtin_head(self, args: list[str], cwd: Path) -> str:
        count, paths = _parse_line_count(args, default=10)
        return self._head_or_tail(count, paths, cwd, head=True)

    def _builtin_tail(self, args: list[str], cwd: Path) -> str:
        count, paths = _parse_line_count(args, default=10)
        return self._head_or_tail(count, paths, cwd, head=False)

    def _head_or_tail(
        self, count: int, paths: list[str], cwd: Path, *, head: bool
    ) -> str:
        if not paths:
            raise ToolError("head/tail 需要文件路径")
        if count <= 0:
            return "（空文件）"
        path = self._resolve_path(paths[0], cwd)
        lines = self._read_text(path).splitlines()
        selected = lines[:count] if head else lines[-count:]
        return "\n".join(selected) if selected else "（空文件）"

    def _builtin_find(self, args: list[str], cwd: Path) -> str:
        target = "."
        pattern = "*"
        index = 0
        while index < len(args):
            argument = args[index]
            if argument == "-name" and index + 1 < len(args):
                pattern = args[index + 1]
                index += 2
                continue
            if argument.startswith("-"):
                index += 1
                continue
            target = argument
            index += 1
        path = self._resolve_path(target, cwd)
        if not path.exists():
            raise ToolError(f"路径不存在：{target}")
        if path.is_file():
            return str(path.relative_to(self.workspace))
        try:
            matches = sorted(path.rglob(pattern))
        except ValueError as exc:
            raise ToolError(f"无效的文件名模式：{pattern}") from exc
        lines: list[str] = []
        for match in matches:
            resolved = match.resolve()
            try:
                resolved.relative_to(self.workspace)
            except ValueError:
                continue
            if not resolved.is_file():
                continue
            lines.append(str(match.relative_to(self.workspace)))
            if len(lines) >= 500:
                lines.append("…（已截断）")
                break
        return "\n".join(lines) if lines else "（无匹配文件）"

    def _builtin_grep(self, args: list[str], cwd: Path) -> str:
        show_line = False
        ignore_case = False
        rest: list[str] = []
        for argument in args:
            if argument in ("-n", "--line-number"):
                show_line = True
            elif argument in ("-i", "--ignore-case"):
                ignore_case = True
            elif argument.startswith("-") and not rest:
                continue
            else:
                rest.append(argument)
        if not rest:
            raise ToolError("grep 需要 pattern")
        pattern = rest[0]
        targets = rest[1:] or ["."]
        flags = re.IGNORECASE if ignore_case else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as exc:
            raise ToolError(f"无效正则：{exc}") from exc

        files: list[Path] = []
        for raw in targets:
            path = self._resolve_path(raw, cwd)
            if path.is_dir():
                files.extend(
                    candidate
                    for candidate in sorted(path.rglob("*"))
                    if candidate.is_file()
                    and candidate.suffix.lower() in _TEXT_SUFFIXES
                    and candidate.resolve().is_relative_to(self.workspace)
                )
            elif path.is_file():
                files.append(path)
            else:
                raise ToolError(f"路径不存在：{raw}")

        lines: list[str] = []
        for path in files:
            try:
                text = self._read_text(path)
            except ToolError:
                continue
            display = path.relative_to(self.workspace)
            for line_no, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    prefix = f"{display}:{line_no}" if show_line else str(display)
                    lines.append(f"{prefix}:{line}")
                    if len(lines) >= 200:
                        lines.append("…（已截断）")
                        return "\n".join(lines)
        return "\n".join(lines) if lines else "（无匹配）"

    def _builtin_wc(self, args: list[str], cwd: Path) -> str:
        flags = {arg for arg in args if arg.startswith("-")}
        paths = [arg for arg in args if not arg.startswith("-")]
        if not paths:
            raise ToolError("wc 需要文件路径")
        counts_lines = ("-l" in flags) or not flags
        counts_words = ("-w" in flags) or not flags
        counts_chars = ("-c" in flags) or not flags
        outputs: list[str] = []
        for raw in paths:
            path = self._resolve_path(raw, cwd)
            text = self._read_text(path)
            values: list[str] = []
            if counts_lines:
                values.append(str(len(text.splitlines())))
            if counts_words:
                values.append(str(len(text.split())))
            if counts_chars:
                values.append(str(len(text)))
            outputs.append(f"{' '.join(values)} {path.relative_to(self.workspace)}")
        return "\n".join(outputs)

    def _builtin_stat(self, args: list[str], cwd: Path) -> str:
        paths = [arg for arg in args if not arg.startswith("-")]
        if not paths:
            raise ToolError("stat 需要路径")
        lines: list[str] = []
        for raw in paths:
            path = self._resolve_path(raw, cwd)
            if not path.exists():
                raise ToolError(f"路径不存在：{raw}")
            info = path.stat()
            modified = datetime.fromtimestamp(info.st_mtime).isoformat(timespec="seconds")
            kind = "directory" if path.is_dir() else "file"
            lines.append(
                f"{path.relative_to(self.workspace)}: type={kind} "
                f"size={info.st_size} modified={modified}"
            )
        return "\n".join(lines)

    def _builtin_which(self, args: list[str], cwd: Path) -> str:
        if not args:
            raise ToolError("which 需要命令名")
        found = shutil.which(args[0])
        return found or f"未找到：{args[0]}"

    def _builtin_sort(self, args: list[str], cwd: Path) -> str:
        paths = [arg for arg in args if not arg.startswith("-")]
        if not paths:
            raise ToolError("sort 需要文件路径")
        lines = self._read_text(self._resolve_path(paths[0], cwd)).splitlines()
        return "\n".join(sorted(lines))

    def _builtin_uniq(self, args: list[str], cwd: Path) -> str:
        paths = [arg for arg in args if not arg.startswith("-")]
        if not paths:
            raise ToolError("uniq 需要文件路径")
        lines = self._read_text(self._resolve_path(paths[0], cwd)).splitlines()
        output: list[str] = []
        previous: str | None = None
        for line in lines:
            if line != previous:
                output.append(line)
                previous = line
        return "\n".join(output)

    def _builtin_cut(self, args: list[str], cwd: Path) -> str:
        delimiter = "\t"
        fields: list[int] = []
        paths: list[str] = []
        index = 0
        while index < len(args):
            argument = args[index]
            if argument in ("-d", "--delimiter") and index + 1 < len(args):
                delimiter = args[index + 1]
                index += 2
                continue
            if argument in ("-f", "--fields") and index + 1 < len(args):
                fields = _parse_fields(args[index + 1])
                index += 2
                continue
            if not argument.startswith("-"):
                paths.append(argument)
            index += 1
        if not paths:
            raise ToolError("cut 需要文件路径")
        if not fields:
            raise ToolError("cut 需要 -f 字段列表，例如 -f 1,3")
        lines = self._read_text(self._resolve_path(paths[0], cwd)).splitlines()
        output: list[str] = []
        for line in lines:
            parts = line.split(delimiter)
            selected = [
                parts[field - 1] for field in fields if 0 < field <= len(parts)
            ]
            output.append(delimiter.join(selected))
        return "\n".join(output)

    # --------------------------------------------------------------- helpers

    def _resolve_path(self, raw: str, cwd: Path) -> Path:
        text = str(raw or ".")
        if text in ("~", "$HOME"):
            return self.workspace
        if text.startswith("~/"):
            text = text[2:]
            candidate = self.workspace / text
        else:
            candidate = Path(text)
            if not candidate.is_absolute():
                candidate = cwd / candidate
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self.workspace)
        except ValueError as exc:
            raise ToolError(f"路径越界：{raw}") from exc
        return resolved

    def _read_text(self, path: Path) -> str:
        if not path.exists():
            raise ToolError(f"文件不存在：{path}")
        if not path.is_file():
            raise ToolError(f"不是文件：{path}")
        try:
            if path.stat().st_size > _MAX_TEXT_FILE_BYTES:
                raise ToolError(f"文件过大（>{_MAX_TEXT_FILE_BYTES} 字节）：{path}")
            data = path.read_bytes()
        except OSError as exc:
            raise ToolError(f"读取失败：{exc}") from exc
        if b"\x00" in data[:4096]:
            raise ToolError(f"疑似二进制文件：{path}")
        return data.decode("utf-8", errors="replace")

    def _truncate(self, text: str) -> str:
        result = text.rstrip()
        encoded = result.encode("utf-8")
        notices: list[str] = []
        if len(encoded) > self.max_output_bytes:
            truncated = encoded[: self.max_output_bytes].decode("utf-8", errors="ignore")
            result = truncated
            notices.append(f"输出超过 {self.max_output_bytes} 字节")
        if len(result) > self.max_output_chars:
            result = result[: self.max_output_chars]
            notices.append(f"输出超过 {self.max_output_chars} 字符")
        if notices:
            result += "\n…（已截断：" + "；".join(notices) + "）"
        return result


def _split(text: str) -> list[str]:
    """按 shell 规则拆分参数，但绝不交给 shell 执行。"""
    try:
        return shlex.split(text, posix=True)
    except ValueError as exc:
        raise ToolError(f"命令解析失败：{exc}") from exc


def _looks_like_path(value: str) -> bool:
    """保守判断参数是否像路径，用于外部命令的沙箱校验。"""
    if not value or value.startswith("-"):
        return False
    if value in (".", "..", "~") or value.startswith("~"):
        return True
    return "/" in value or "\\" in value or ":" in value


def _merge_output(stdout: bytes, stderr: bytes) -> str:
    """合并 stdout / stderr，统一 UTF-8 与换行。"""
    out = (
        stdout.decode("utf-8", errors="replace")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .rstrip()
    )
    err = (
        stderr.decode("utf-8", errors="replace")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .rstrip()
    )
    if out and err:
        return f"{out}\n[stderr]\n{err}"
    return out or err


def _parse_line_count(
    args: list[str], *, default: int
) -> tuple[int, list[str]]:
    """解析 ``-n N`` / ``-N`` 与路径参数。"""
    count = default
    paths: list[str] = []
    index = 0
    while index < len(args):
        argument = args[index]
        if argument in ("-n", "--lines") and index + 1 < len(args):
            try:
                count = max(0, int(args[index + 1]))
            except ValueError as exc:
                raise ToolError("head/tail 的行数必须是整数") from exc
            index += 2
            continue
        if argument.startswith("-") and argument[1:].isdigit():
            count = int(argument[1:])
        elif not argument.startswith("-"):
            paths.append(argument)
        index += 1
    return count, paths


def _parse_fields(value: str) -> list[int]:
    """解析 ``1,3`` / ``1-3`` 字段列表。"""
    fields: list[int] = []
    for part in str(value).split(","):
        text = part.strip()
        if not text:
            continue
        if "-" in text:
            start, end = text.split("-", 1)
            try:
                low, high = int(start), int(end)
            except ValueError as exc:
                raise ToolError(f"无效字段范围：{part}") from exc
            fields.extend(range(low, high + 1))
        else:
            try:
                fields.append(int(text))
            except ValueError as exc:
                raise ToolError(f"无效字段：{part}") from exc
    return fields
