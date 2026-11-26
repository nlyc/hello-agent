"""M0 冒烟测试：验证包可导入、版本号存在、分层结构齐全。"""

import agentflow


def test_version():
    assert agentflow.__version__ == "2.0.3"


def test_subpackages_importable():
    from agentflow import (
        agents,
        context,
        core,
        evaluation,
        llm,
        memory,
        notes,
        protocols,
        tools,
        trace,
        utils,
        workflow,
    )

    assert (
        agents
        and context
        and core
        and evaluation
        and llm
        and memory
        and notes
        and protocols
        and tools
        and trace
        and utils
        and workflow
    )
