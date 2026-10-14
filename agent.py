"""校招简章精读 Agent（LangGraph）。

流程：triage 初评 → read_poster 读海报 → extract 结构化抽取 → reflect 校验
      （证据不足且还有原文没读完 → 回到 read_poster 多读一段；
      读完城市仍不确定 → check_official 查招聘官网的岗位城市）→ score 定分
      → 高分时 human_review 中断，等人确认。
"""
import json
import os
from pathlib import Path
from typing import Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

import ats

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

PAGE = 2500        # 每次读多少字海报原文
MAX_PAGES = 3      # 最多读几段（防止无限循环）
TRIAGE_PASS = 60   # 初评低于这个分数直接结束，不精读
REVIEW_AT = 85     # 定分达到这个分数时请人确认

PROFILE = json.loads((ROOT / os.getenv("PROFILE_PATH", "profile.example.json")).read_text(encoding="utf-8"))
CANDIDATE = (f"候选人：{PROFILE['graduation']}，{PROFILE['degree']}；目标城市：{'、'.join(PROFILE['target']['cities'])}；"
             f"方向：{PROFILE['target']['notes']}；亮点：{'；'.join(PROFILE['highlights'][:3])}")


def llm():
    return ChatOpenAI(model=os.getenv("MODEL", "deepseek-flash"), base_url="https://api.deepseek.com",
                      api_key=os.environ["DEEPSEEK_API_KEY"], temperature=0,
                      # DeepSeek V4 默认开推理模式，而推理模式不支持强制工具调用（结构化输出要用）；
                      # 抽取和打分也用不上深度推理，关掉更快更便宜
                      extra_body={"thinking": {"type": "disabled"}})


def ask(schema, prompt):
    """调用模型并拿到 Pydantic 结构化结果。
    DeepSeek 不支持 OpenAI 的 json_schema 响应格式，所以用 function calling：
    把 schema 声明成一个工具，让模型「调用」它，参数就是我们要的结构化结果。
    模型偶尔不调用工具、返回 None（第一轮评测 baseline 有 4 条这样失败），所以最多重试 3 次。"""
    chain = llm().with_structured_output(schema, method="function_calling")
    for _ in range(3):
        out = chain.invoke(prompt)
        if out is not None:
            return out
    raise RuntimeError(f"模型连续 3 次没有返回 {schema.__name__}")


# ---------- 结构化输出 ----------
class Triage(BaseModel):
    score: int = Field(description="0-100 的匹配分")
    reason: str = Field(description="一句话理由")


class Extraction(BaseModel):
    is_internship: bool = Field(description="这份简章是否为实习招聘（含转正实习、暑期实习），而不是正式校招")
    shenzhen: Literal["yes", "no", "unknown"] = Field(description="对口岗位的工作地点是否包含深圳；原文没写城市时填 unknown")
    # 第二轮新增：区分证据级别。第一轮错例里，海报写「工作地点：北京/上海/深圳…」是公司整体的招聘城市，
    # 但对口的 Agent 岗其实在别的城市（快手、联友、神州数码、海能达）。
    shenzhen_scope: Literal["job", "company", "none"] = Field(
        description="深圳出现在哪一级：job=对口岗位本身明确标注了深圳；company=深圳只出现在公司整体的工作地点/城市列表里，"
                    "没有和具体岗位对应；none=原文没有提到深圳")
    cities: list[str] = Field(description="原文明确写出的工作城市")
    grad_window: str = Field(description="原文写的毕业时间要求，没写则为空字符串")
    matched_jobs: list[str] = Field(description="原文中与候选人最匹配的 1-3 个岗位名")
    quotes: list[str] = Field(description="支撑以上判断的原文片段，必须逐字摘抄，每条不超过 40 字")


class Final(BaseModel):
    score: int
    reason: str


# ---------- 状态 ----------
# State 是图里所有节点共享的「黑板」。每个节点读需要的字段，只返回自己更新的字段，
# LangGraph 负责合并。好处：节点之间不直接调用，可以单独测试、随意调整顺序；
# 配合 checkpointer，每一步之后的 State 都能存下来，中断后可以从断点继续。
class State(TypedDict, total=False):
    posting: dict                     # 看板元数据 + poster_text
    triage: dict
    pages_read: int
    evidence: str
    extraction: dict
    issues: list[str]
    official: dict                    # 第三轮新增：官网岗位查询结果
    final: dict
    review: Optional[str]


# ---------- 节点 ----------
def triage(state: State):
    p = state["posting"]
    out = ask(Triage, 
        f"{CANDIDATE}\n\n只根据看板信息给这份校招简章打 0-100 匹配分。\n"
        f"公司：{p['company']}\n标题：{p['title']}\n岗位：{p['jobs'][:400]}\n城市：{p['cities']}\n届数要求：{p['class_time']}")
    return {"triage": out.model_dump(), "pages_read": 0}


def route_after_triage(state: State):
    # 条件边：初评就很低的（芯片、工程、医药…）直接结束，不浪费精读的模型调用。
    # 一天 100+ 条新简章里大部分在这里就被过滤掉，成本主要花在少数候选上。
    return "read_poster" if state["triage"]["score"] >= TRIAGE_PASS else "skip"


def read_poster(state: State):
    """工具节点：按段读取海报原文（正文 + OCR）。
    为什么分段读：大多数简章前 2500 字就有城市和岗位，没必要一次全塞给模型；
    只有 reflect 判断「证据不够」时才多读一段——这就是图里的循环。"""
    n = state.get("pages_read", 0) + 1
    return {"pages_read": n, "evidence": state["posting"]["poster_text"][: PAGE * n]}


def extract(state: State):
    out = ask(Extraction, 
        f"{CANDIDATE}\n\n下面是一份校招简章的原文（来自公众号正文和海报 OCR，可能有错字）。"
        f"请抽取信息，所有判断都要有原文依据，quotes 必须逐字摘抄。\n\n标题：{state['posting']['title']}\n\n{state['evidence']}")
    return {"extraction": out.model_dump()}


def reflect(state: State):
    """校验抽取结果：引用是否真的出自原文、关键字段是否有依据。
    关键设计：这一步是代码校验，不是再问一次模型。让模型「检查自己」容易自我认同，
    而「引用是否逐字出现在原文里」是可以确定判断的事——用代码查，结果可靠、零成本。
    发现问题就回到 read_poster 多读一段；读完了还不行，就带着问题去定分（score 会看到 issues）。"""
    ex = state["extraction"]
    ev = state["posting"]["title"] + "\n" + state["evidence"]
    issues = [f"引用不在原文中：{q}" for q in ex["quotes"] if q.strip() and q.strip() not in ev]
    if ex["shenzhen"] == "unknown":
        issues.append("城市未确定")
    if ex["shenzhen"] == "yes" and not any("深圳" in q for q in ex["quotes"]):
        issues.append("判定深圳有岗但没有引用依据")
    if ex["shenzhen"] == "yes" and ex["shenzhen_scope"] == "company":
        issues.append("深圳只出现在公司整体的城市列表里，没有对应到具体岗位")  # 原文没读完时会再读一段找岗位级证据
    return {"issues": issues}


def route_after_reflect(state: State):
    more_text = len(state["posting"]["poster_text"]) > PAGE * state["pages_read"]
    if state["issues"] and more_text and state["pages_read"] < MAX_PAGES:
        return "read_poster"
    # 海报读完了城市仍不确定（或只有公司级证据），且还没查过官网 → 去官网查岗位级城市
    ex = state["extraction"]
    city_unsure = ex["shenzhen"] == "unknown" or (ex["shenzhen"] == "yes" and ex["shenzhen_scope"] == "company")
    if city_unsure and "official" not in state and state["posting"].get("apply_url"):
        return "check_official"
    return "score"


class OfficialPick(BaseModel):
    matched: list[str] = Field(description="从岗位列表中挑出与候选人对口的岗位，必须逐字照抄列表里的岗位名，最多 8 个")


def check_official(state: State):
    """工具节点：查招聘官网，拿到「岗位 → 城市」的对应关系，解决海报只列公司城市的问题。
    和 reflect 同样的思路：模型只负责「挑对口岗位」这件需要理解的事；
    挑出的岗位名是否真在列表里、这些岗位的城市有没有深圳，都由代码判断。"""
    jobs = ats.lookup(state["posting"]["apply_url"])
    if not jobs:  # 不支持的招聘系统（如 Moka 加密接口）或查询失败：明确记为未查到，交给人工
        return {"official": {"status": "unsupported"}}
    listing = "\n".join(f"- {j['name']}" for j in jobs[:200])
    pick = ask(OfficialPick, f"{CANDIDATE}\n\n下面是 {state['posting']['company']} 招聘官网的全部校招岗位，"
                             f"挑出与候选人对口的岗位（不要挑实习岗、销售/职能岗）：\n{listing}")
    by_name = {j["name"]: j for j in jobs}
    matched = [by_name[n] for n in pick.matched if n in by_name]          # 代码校验：丢掉模型编造的岗位名
    sz = [j["name"] for j in matched if any("深圳" in c for c in j["cities"])]
    return {"official": {"status": "ok", "matched": [j["name"] for j in matched], "shenzhen_jobs": sz,
                         "shenzhen": ("yes" if sz else "no") if matched else "unknown"}}


def score(state: State):
    ex, issues = state["extraction"], state.get("issues", [])
    # 公司级证据不算「深圳有岗」：降级为 unknown，交给后续人工/官网确认
    shenzhen = "unknown" if ex["shenzhen"] == "yes" and ex["shenzhen_scope"] == "company" else ex["shenzhen"]
    official = state.get("official", {})
    if official.get("status") == "ok" and official["shenzhen"] != "unknown":
        shenzhen = official["shenzhen"]  # 官网的岗位级城市比海报更可靠，以官网为准
    out = ask(Final,
        f"{CANDIDATE}\n\n根据精读结果给最终匹配分（0-100）并写一句理由。\n"
        f"初评：{state['triage']}\n精读抽取：{json.dumps(ex, ensure_ascii=False)}\n"
        f"官网查询：{json.dumps(official, ensure_ascii=False) if official else '未查询'}\n"
        f"深圳有岗（按岗位级证据）：{shenzhen}\n校验发现的问题：{issues}")
    s = out.score
    # 硬规则：实习不算正式校招；明确不在深圳的不进第一梯队。
    # 为什么用规则而不是全交给模型：这两条是候选人的硬约束，模型偶尔会因为「岗位很对口」
    # 给高分，规则兜底保证结果一定满足约束，也让行为可解释。
    if ex["is_internship"]:
        s = min(s, 30)
    if shenzhen == "no":
        s = min(s, 60)
    return {"final": {"score": s, "reason": out.reason, "jobs": ex["matched_jobs"],
                      "shenzhen": shenzhen, "is_internship": ex["is_internship"]}}


def skip(state: State):
    t = state["triage"]
    return {"final": {"score": t["score"], "reason": "初评未通过：" + t["reason"], "jobs": [],
                      "shenzhen": "unknown", "is_internship": False}}


def human_review(state: State):
    """高分岗位暂停，等人确认（继续时用 Command(resume="approve"/"reject")）。
    interrupt() 会把当前 State 存进 checkpointer 并挂起；人确认后从这里继续，
    前面的模型调用不用重跑。对应投递环节「只填不交、最后由人决定」的原则。"""
    if state["final"]["score"] < REVIEW_AT:
        return {}
    decision = interrupt({"company": state["posting"]["company"], "final": state["final"]})
    return {"review": decision}


def build(checkpointer=None, with_review=True):
    g = StateGraph(State)
    for name, fn in [("triage", triage), ("read_poster", read_poster), ("extract", extract),
                     ("reflect", reflect), ("check_official", check_official), ("score", score),
                     ("skip", skip), ("human_review", human_review)]:
        g.add_node(name, fn)
    g.add_edge(START, "triage")
    g.add_conditional_edges("triage", route_after_triage, ["read_poster", "skip"])
    g.add_edge("read_poster", "extract")
    g.add_edge("extract", "reflect")
    g.add_conditional_edges("reflect", route_after_reflect, ["read_poster", "check_official", "score"])
    g.add_edge("check_official", "score")
    g.add_edge("score", "human_review" if with_review else END)
    g.add_edge("human_review", END)
    g.add_edge("skip", END)
    return g.compile(checkpointer=checkpointer)
