"""单条演示：逐节点打印执行过程，高分时在 human_review 处中断，等你在终端确认。

用法：.venv/bin/python run_one.py 44275      # 传简章 id（见 data/eval.jsonl）
"""
import json
import sqlite3
import sys
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from agent import build

ROOT = Path(__file__).parent


def show(update):
    """stream 每产出一次，就是一个节点刚执行完，update 是它返回的字段。"""
    for node, delta in update.items():
        if node == "__interrupt__":
            continue
        brief = {k: (v if k != "evidence" else f"<原文前 {len(v)} 字>") for k, v in (delta or {}).items()}
        print(f"── {node}: {json.dumps(brief, ensure_ascii=False)[:400]}")


def main():
    pid = int(sys.argv[1])
    posting = next(json.loads(l) for l in (ROOT / "data" / "eval.jsonl").open(encoding="utf-8")
                   if json.loads(l)["id"] == pid)
    print(f"简章：{posting['company']} · {posting['title']}\n人工标签：{posting['label']}\n")

    # checkpointer 把每一步的 State 存进 SQLite；thread_id 标识这一次运行，中断后靠它恢复
    saver = SqliteSaver(sqlite3.connect(ROOT / "checkpoints.sqlite", check_same_thread=False))
    graph = build(checkpointer=saver)
    config = {"configurable": {"thread_id": f"posting-{pid}"}}

    for update in graph.stream({"posting": posting}, config, stream_mode="updates"):
        show(update)

    # 如果停在 interrupt，State 里会有待处理的中断
    snap = graph.get_state(config)
    if snap.next:
        print(f"\n⏸  已暂停在 {snap.next}，最终分 {snap.values['final']['score']}，要标记为想投吗？[y/n] ", end="")
        decision = "approve" if input().strip().lower() == "y" else "reject"
        for update in graph.stream(Command(resume=decision), config, stream_mode="updates"):
            show(update)
        print(f"人工决定：{graph.get_state(config).values.get('review')}")

    print(f"\n最终结果：{json.dumps(graph.get_state(config).values['final'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
