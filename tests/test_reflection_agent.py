"""ReflectionAgent（执行→反思→优化，双终止）测试。"""

from agentflow.agents import ReflectionAgent
from tests.fakes import ScriptedLLM


def test_reflection_converges_on_no_improvement():
    llm = ScriptedLLM(
        [
            "初稿",  # initial
            "发现效率问题，应优化",  # reflect（不含「无需改进」）
            "改进稿",  # refine
            "无需改进，已经最优",  # reflect → 触发终止
        ]
    )
    agent = ReflectionAgent("writer", llm, max_iterations=3)
    result = agent.run("写一篇短文")
    assert result == "改进稿"
    assert len(llm.calls) == 4
    assert "初稿" in agent.get_trajectory() and "无需改进" in agent.get_trajectory()


def test_reflection_stops_at_max_iterations():
    llm = ScriptedLLM(
        [
            "a",  # initial
            "继续改进",  # reflect
            "b",  # refine
            "继续改进",  # reflect
            "c",  # refine
        ]
    )
    agent = ReflectionAgent("w", llm, max_iterations=2)
    result = agent.run("task")
    assert result == "c"
    assert len(llm.calls) == 5


def test_reflection_custom_prompt_override():
    llm = ScriptedLLM(["result", "无需改进"])
    agent = ReflectionAgent(
        "w",
        llm,
        max_iterations=1,
        custom_prompts={"reflect": "评审任务: {task}\n结果: {result}\n给出反馈"},
    )
    agent.run("t")
    # 自定义 reflect 模板不含 reviewer_role 占位符，应当被采用
    assert "评审任务: t" in llm.calls[1]["messages"][0]["content"]
