"""Agent 端的 MCP 客户端：通过 MCP 协议调用 mcp_server.py 里的工具。

LangGraph 的节点是同步函数，MCP SDK 是异步的，这里包一层同步调用。
每次调用都会启动一次 Server 子进程（约 1 秒）：check_official 每条简章最多调用一次，开销可以接受；
调用频繁时应改为常驻会话，复用同一个连接。
"""
import asyncio
import sys
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

SERVER = StdioServerParameters(command=sys.executable, args=[str(Path(__file__).with_name("mcp_server.py"))])


async def _call(tool: str, args: dict) -> dict:
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, args)
            if result.is_error:
                raise RuntimeError(" ".join(getattr(c, "text", "") for c in result.content))
            return result.structured_content


def lookup_jobs(apply_url: str) -> list[dict] | None:
    """与 ats.lookup 相同的返回值：[{name, cities}]；不支持的招聘系统返回 None。"""
    out = asyncio.run(_call("lookup_jobs", {"apply_url": apply_url}))
    return out["jobs"] if out["status"] == "ok" else None


def list_tools() -> list[str]:
    async def _list():
        async with stdio_client(SERVER) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return [t.name for t in (await session.list_tools()).tools]
    return asyncio.run(_list())
