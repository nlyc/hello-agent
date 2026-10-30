"""Message 单元测试。"""

import pytest
from pydantic import ValidationError

from agentflow.core import Message


def test_to_dict_openai_format():
    assert Message(role="user", content="你好").to_dict() == {
        "role": "user",
        "content": "你好",
    }


def test_invalid_role_rejected():
    with pytest.raises(ValidationError):
        Message(role="invalid", content="x")


def test_timestamp_and_metadata_defaults():
    m = Message(role="assistant", content="hi")
    assert m.timestamp is not None
    assert m.metadata == {}


def test_name_included_in_dict_when_present():
    m = Message(role="tool", content="ok", name="get_weather")
    assert m.to_dict()["name"] == "get_weather"


def test_str():
    assert str(Message(role="system", content="x")) == "[system] x"
