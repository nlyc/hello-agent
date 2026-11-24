"""JSON-RPC 编解码测试。"""

import json

import pytest

from agentflow.protocols.mcp import (
    JsonRpcError,
    decode_line,
    encode_error,
    encode_request,
    encode_result,
)


def test_encode_request_and_decode():
    line = encode_request(7, "tools/list", {"x": 1})
    assert line.endswith(b"\n")
    payload = decode_line(line)
    assert payload == {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/list",
        "params": {"x": 1},
    }


def test_encode_result_and_error():
    result = json.loads(encode_result(1, {"tools": []}))
    assert result["result"] == {"tools": []}
    error = json.loads(encode_error(2, -32601, "未知方法", {"method": "x"}))
    assert error["error"]["code"] == -32601
    assert error["error"]["data"] == {"method": "x"}


def test_decode_invalid_json():
    with pytest.raises(JsonRpcError) as info:
        decode_line(b"{not json")
    assert info.value.code == -32700


def test_decode_non_object():
    with pytest.raises(JsonRpcError) as info:
        decode_line(b"[1, 2]")
    assert info.value.code == -32600


def test_jsonrpc_error_to_dict():
    error = JsonRpcError(-32602, "参数错误", data={"field": "name"})
    assert error.to_dict() == {
        "code": -32602,
        "message": "参数错误",
        "data": {"field": "name"},
    }
