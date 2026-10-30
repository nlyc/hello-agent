"""M0 冒烟测试：验证包可导入、版本号存在、分层结构齐全。"""

import agentflow


def test_version():
    assert agentflow.__version__ == "1.0.0"


def test_subpackages_importable():
    from agentflow import agents, core, llm, memory, tools, trace, utils, workflow

    assert agents and core and llm and memory and tools and trace and utils and workflow
