"""agentflow.protocols.mcp —— 最小 MCP：JSON-RPC / Client / Server / MCPTool。"""

from agentflow.protocols.mcp.client import (
    InMemoryTransport,
    MCPClient,
    StdioTransport,
    Transport,
    stdio_command,
)
from agentflow.protocols.mcp.jsonrpc import (
    PROTOCOL_VERSION,
    JsonRpcError,
    decode_line,
    encode_error,
    encode_request,
    encode_result,
)
from agentflow.protocols.mcp.server import MCPServer
from agentflow.protocols.mcp.tool import (
    MCPRemoteTool,
    MCPTool,
    coerce_arguments,
    schema_to_parameters,
)

__all__ = [
    "PROTOCOL_VERSION",
    "InMemoryTransport",
    "JsonRpcError",
    "MCPClient",
    "MCPRemoteTool",
    "MCPServer",
    "MCPTool",
    "StdioTransport",
    "Transport",
    "coerce_arguments",
    "decode_line",
    "encode_error",
    "encode_request",
    "encode_result",
    "schema_to_parameters",
    "stdio_command",
]
