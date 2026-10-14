"""对照组：不用 LangGraph，只看看板信息、一次模型调用就打分。

这就是 Job_Agent 最初的做法。它的问题是看不到简章原文——简章写的是「实习生招聘」、
或者岗位其实只在杭州，单次打分都发现不了。评测时用它和 agent 对比。
"""
from pydantic import BaseModel, Field

from agent import CANDIDATE, ask


class BaselineOut(BaseModel):
    score: int = Field(description="0-100 的匹配分")
    is_internship: bool = Field(description="是否为实习招聘")
    shenzhen: str = Field(description="深圳是否有对口岗位：yes / no / unknown")
    reason: str


def run(posting: dict) -> dict:
    out = ask(BaselineOut, 
        f"{CANDIDATE}\n\n只根据看板信息评估这份校招简章。\n"
        f"公司：{posting['company']}\n标题：{posting['title']}\n岗位：{posting['jobs'][:400]}\n"
        f"城市：{posting['cities']}\n届数要求：{posting['class_time']}")
    return out.model_dump()
