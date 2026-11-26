"""LLM Judge 与 Win Rate：绝对质量 + 相对质量。

Judge 只负责打分，结果必须可解析；解析失败按 0 分处理并保留原因。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from agentflow.core.exceptions import EvaluationError

JUDGE_SYSTEM_PROMPT = """你是旅行行程评审专家。请按以下四个维度给候选行程打分（1–5 分）：
1. feasibility：交通、时间与体力是否可行；
2. constraint_compliance：是否满足天数、预算、同行人、偏好等约束；
3. information_quality：信息是否具体、可执行，是否标注依据；
4. format：结构是否清晰、可直接执行。

只输出一个 JSON 对象，不要输出其它内容：
{"feasibility": 4, "constraint_compliance": 5, "information_quality": 4, "format": 5, "comment": "一句话理由"}"""

WIN_RATE_SYSTEM_PROMPT = """你是旅行行程评审专家。请比较 A、B 两份行程，判断哪一份更好。
只输出一个 JSON 对象：
{"winner": "A" 或 "B" 或 "tie", "reason": "一句话理由"}"""

_SCORE_KEYS = (
    "feasibility",
    "constraint_compliance",
    "information_quality",
    "format",
)


@dataclass
class JudgeResult:
    """一次 LLM Judge 的结果。"""

    scores: dict[str, float] = field(default_factory=dict)
    average: float = 0.0
    passed: bool = False
    excellent: bool = False
    comment: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "scores": dict(self.scores),
            "average": self.average,
            "passed": self.passed,
            "excellent": self.excellent,
            "comment": self.comment,
            "error": self.error,
        }


def parse_judge_output(text: str) -> JudgeResult:
    """解析 Judge 输出；容忍代码围栏与少量前后缀。"""
    payload = _extract_json(text)
    scores: dict[str, float] = {}
    for key in _SCORE_KEYS:
        value = payload.get(key)
        if value is None and isinstance(payload.get("scores"), dict):
            value = payload["scores"].get(key)
        try:
            score = float(value)
        except (TypeError, ValueError):
            score = 0.0
        scores[key] = max(0.0, min(5.0, score))
    average = sum(scores.values()) / len(_SCORE_KEYS)
    return JudgeResult(
        scores=scores,
        average=round(average, 3),
        passed=average >= 3.5,
        excellent=average >= 4.5,
        comment=str(payload.get("comment", "") or ""),
    )


def _extract_json(text: str) -> dict[str, Any]:
    """从文本中提取第一个 JSON 对象；失败抛 ``EvaluationError``。"""
    source = str(text or "").strip()
    source = re.sub(r"^```(?:json)?\s*|\s*```$", "", source, flags=re.IGNORECASE)
    start = source.find("{")
    if start < 0:
        raise EvaluationError("Judge 输出中没有 JSON 对象")
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", source):
        try:
            payload, _ = decoder.raw_decode(source[match.start() :])
        except ValueError:
            continue
        if isinstance(payload, dict):
            return payload
    raise EvaluationError("Judge 输出中没有可解析的 JSON 对象")


class LLMJudge:
    """四维质量评委。"""

    def __init__(
        self,
        llm: Any,
        *,
        temperature: float = 0.0,
        system_prompt: str = JUDGE_SYSTEM_PROMPT,
    ) -> None:
        self.llm = llm
        self.temperature = temperature
        self.system_prompt = system_prompt

    def judge(
        self, task: str, answer: str, *, reference: str = ""
    ) -> JudgeResult:
        """调用 LLM 打分；任何失败都转为带 error 的 0 分结果。"""
        user = f"# 任务\n{task}\n\n# 参考\n{reference or '无'}\n\n# 候选行程\n{answer}"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user},
        ]
        try:
            raw = self._chat(messages)
        except Exception as exc:
            return JudgeResult(error=f"{type(exc).__name__}: {exc}")
        try:
            return parse_judge_output(raw)
        except EvaluationError as exc:
            return JudgeResult(error=str(exc))
        except Exception as exc:  # 防御性
            return JudgeResult(error=f"{type(exc).__name__}: {exc}")

    def _chat(self, messages: list[dict[str, str]]) -> str:
        """优先 JSON 模式；网关不支持时去掉后重试一次。"""
        try:
            return self.llm.chat(
                messages, temperature=self.temperature, json_mode=True
            )
        except Exception:
            return self.llm.chat(messages, temperature=self.temperature)


class WinRateJudge:
    """成对比较 A/B，交换顺序各跑一次以降低位置偏差。"""

    def __init__(self, llm: Any, *, temperature: float = 0.0) -> None:
        self.judge = LLMJudge(
            llm, temperature=temperature, system_prompt=WIN_RATE_SYSTEM_PROMPT
        )

    def compare(self, task: str, answer_a: str, answer_b: str) -> dict[str, Any]:
        forward = self._one_round(task, answer_a, answer_b)
        backward = self._one_round(task, answer_b, answer_a)
        # backward 中的 A 实际是 B，需要翻转。
        votes = [forward, _flip(backward)]
        wins_a = sum(1 for vote in votes if vote == "A")
        wins_b = sum(1 for vote in votes if vote == "B")
        ties = sum(1 for vote in votes if vote == "tie")
        total = len(votes)
        return {
            "wins_a": wins_a,
            "wins_b": wins_b,
            "ties": ties,
            "win_rate": (wins_a + 0.5 * ties) / total if total else 0.5,
            "rounds": total,
        }

    def _one_round(self, task: str, first: str, second: str) -> str:
        messages = [
            {"role": "system", "content": WIN_RATE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"# 任务\n{task}\n\n# 行程 A\n{first}\n\n# 行程 B\n{second}"
                ),
            },
        ]
        try:
            raw = self.judge._chat(messages)
            payload = _extract_json(raw)
        except Exception:
            return "tie"
        winner = str(payload.get("winner", "tie")).strip().upper()
        if winner in ("A", "1"):
            return "A"
        if winner in ("B", "2"):
            return "B"
        return "tie"


def _flip(winner: str) -> str:
    """交换 A/B 顺序后翻转胜者。"""
    if winner == "A":
        return "B"
    if winner == "B":
        return "A"
    return "tie"
