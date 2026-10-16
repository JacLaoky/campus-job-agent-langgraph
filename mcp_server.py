"""MCP Server：把「招聘官网岗位查询」（ats.py）作为标准工具开放出去。

为什么做成 MCP 而不是直接 import：
- 工具和 Agent 解耦。以后接 Moka、飞书等其他招聘系统，只要再加一个 Server，Agent 代码不用改。
- 任何 MCP 客户端（本项目的 Agent、Claude、Cursor……）都能自动发现并调用它，不用各自写对接代码。

用法：由 MCP 客户端以子进程方式启动，走 stdio 传输（见 mcp_client.py）。
"""
from typing import Literal

from mcp.server.mcpserver import MCPServer
from mcp_types import ToolAnnotations
from pydantic import BaseModel, Field

import ats

server = MCPServer(
    name="campus-ats",
    instructions="查询校招岗位在招聘官网上的工作城市。目前支持北森（*.zhiye.com）；Moka 接口加密，不支持。",
)


class Job(BaseModel):
    name: str = Field(description="岗位名称，与官网一致")
    cities: list[str] = Field(description="该岗位的工作城市")


class LookupResult(BaseModel):
    status: Literal["ok", "unsupported"] = Field(
        description="ok：查到了；unsupported：不支持的招聘系统（如 Moka）或链接为空，需人工核实")
    jobs: list[Job] = Field(default_factory=list)


@server.tool(
    annotations=ToolAnnotations(
        read_only_hint=True,      # 只读：不会在招聘系统里做任何修改
        idempotent_hint=True,     # 重复调用结果相同（还有本地缓存）
        open_world_hint=True,     # 会访问外部网站
    ),
)
def lookup_jobs(apply_url: str) -> LookupResult:
    """根据校招投递链接，列出该公司招聘官网上的全部校招岗位及其工作城市。

    支持北森招聘系统（*.zhiye.com，包括跳转到北森的短链接）。用于判断「对口岗位是否在某个城市」：
    海报上的城市往往是公司整体的招聘城市，只有官网才有岗位级的城市。
    """
    jobs = ats.lookup(apply_url)
    if jobs is None:   # 空列表不算不支持：是北森，只是官网上没有校招岗位
        return LookupResult(status="unsupported")
    return LookupResult(status="ok", jobs=[Job(**j) for j in jobs])


if __name__ == "__main__":
    server.run()   # stdio
