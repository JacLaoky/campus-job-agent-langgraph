"""在 data/eval.jsonl 上对比 baseline（单次打分）和 agent（LangGraph 精读）。

用法：
  .venv/bin/python eval.py              # 两种方法都跑（已跑过的条目会跳过）
  .venv/bin/python eval.py --limit 5    # 先用 5 条试跑
  .venv/bin/python eval.py --fresh      # 清空缓存重跑（改了 prompt 或流程之后用）
结果：results/<方法>/<id>.json 逐条缓存，results/report.md 是汇总表和错例。
"""
import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import baseline
from agent import build

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
WORTH = 75  # 分数 ≥ 75 视为「值得投」


def load():
    return [json.loads(l) for l in (ROOT / "data" / "eval.jsonl").open(encoding="utf-8")]


def run_agent(posting):
    graph = build(with_review=False)  # 评测时不需要人工确认节点
    state = graph.invoke({"posting": posting})
    return {**state["final"], "pages_read": state.get("pages_read", 0), "triage": state["triage"]["score"]}


def run_all(method, fn, items, workers):
    out_dir = RESULTS / method
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [it for it in items if not (out_dir / f"{it['id']}.json").exists()]
    print(f"[{method}] 需要跑 {len(todo)} 条（已缓存 {len(items) - len(todo)} 条）")
    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(fn, it): it for it in todo}
        for i, f in enumerate(as_completed(futs), 1):
            it = futs[f]
            try:
                (out_dir / f"{it['id']}.json").write_text(json.dumps(f.result(), ensure_ascii=False), encoding="utf-8")
            except Exception as e:  # 单条失败不影响整体，下次重跑会补上
                print(f"  失败 {it['id']} {it['company']}: {type(e).__name__} {e}")
            if i % 10 == 0:
                print(f"  {i}/{len(todo)}")
    return {it["id"]: json.loads((out_dir / f"{it['id']}.json").read_text(encoding="utf-8"))
            for it in items if (out_dir / f"{it['id']}.json").exists()}


def prf(pairs):
    """pairs: [(预测, 真实)] 的布尔对 → 准确率/精确率/召回率/F1"""
    tp = sum(p and t for p, t in pairs); fp = sum(p and not t for p, t in pairs)
    fn = sum(t and not p for p, t in pairs); tn = sum(not p and not t for p, t in pairs)
    acc = (tp + tn) / len(pairs) if pairs else 0
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    return acc, prec, rec, f1


def metrics(items, preds):
    rows = [(it, preds[it["id"]]) for it in items if it["id"] in preds]
    worth = prf([(p["score"] >= WORTH, it["label"]["worth_applying"]) for it, p in rows])
    intern = prf([(bool(p["is_internship"]), it["label"]["is_internship"]) for it, p in rows])
    sz_rows = [(p["shenzhen"], it["label"]["shenzhen"]) for it, p in rows if it["label"]["shenzhen"] is not None]
    # 「unknown」是弃权，不等于判错：拆成覆盖率（敢下结论的比例）和结论准确率（下了结论的对多少）
    decided = [(p, t) for p, t in sz_rows if p in ("yes", "no")]
    sz_cov = len(decided) / len(sz_rows) if sz_rows else 0
    sz_acc = sum((p == "yes") == t for p, t in decided) / len(decided) if decided else 0
    return {"n": len(rows), "worth": worth, "intern": intern, "sz_acc": sz_acc, "sz_cov": sz_cov, "sz_n": len(sz_rows)}


def report(items, results):
    lines = ["# 评测结果\n", f"数据：{len(items)} 条人工精读标注的校招简章；「值得投」= 分数 ≥ {WORTH}\n",
             "| 方法 | 条数 | 值得投 精确率 | 值得投 召回率 | 值得投 F1 | 实习 召回率 | 实习 精确率 | 深圳 覆盖率 | 深圳 结论准确率 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, m in results.items():
        w, i = m["worth"], m["intern"]
        lines.append(f"| {name} | {m['n']} | {w[1]:.0%} | {w[2]:.0%} | {w[3]:.2f} | {i[2]:.0%} | {i[1]:.0%} | "
                     f"{m['sz_cov']:.0%} | {m['sz_acc']:.0%} |")
    lines.append(f"\n深圳指标只统计人工能确认城市的条目；覆盖率 = 给出 yes/no 结论的比例，unknown 视为弃权。")
    return "\n".join(lines)


def threshold_sweep(items, methods, thresholds=(60, 65, 70, 75, 80, 85)):
    """「值得投」的阈值不该拍脑袋定：列出不同阈值下的精确率/召回率/F1，看取舍。
    注意样本只有几十条，选阈值时别过度拟合这份数据。"""
    out = ["\n## 阈值扫描（值得投：预测分 ≥ 阈值）\n", "| 方法 | " + " | ".join(f"≥{t}" for t in thresholds) + " |",
           "|---|" + "---|" * len(thresholds)]
    for name, preds in methods.items():
        cells = []
        for t in thresholds:
            _, p, r, f1 = prf([(preds[it["id"]]["score"] >= t, it["label"]["worth_applying"])
                               for it in items if it["id"] in preds])
            cells.append(f"P{p:.0%} R{r:.0%} F1 {f1:.2f}")
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def errors(items, preds, limit=15):
    """列出「值得投」判断错误的条目，用来分析和迭代。"""
    out = ["\n## agent 判错的条目（值得投）\n", "| 公司 | 标签分 | 预测分 | 实习(标/预) | 深圳(标/预) | 理由 |", "|---|---|---|---|---|---|"]
    for it in items:
        p = preds.get(it["id"])
        if p and (p["score"] >= WORTH) != it["label"]["worth_applying"]:
            lb = it["label"]
            out.append(f"| {it['company']} | {lb['final_score']} | {p['score']} | {lb['is_internship']}/{p['is_internship']} | "
                       f"{lb['shenzhen']}/{p['shenzhen']} | {p['reason'][:50]} |")
    return "\n".join(out[: limit + 3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--tag", default="", help="结果目录后缀，用于对比实验，例如 MODEL=deepseek-v4-pro 时用 --tag _pro")
    args = ap.parse_args()
    if args.fresh and RESULTS.exists():
        shutil.rmtree(RESULTS)
    items = load()[: args.limit] if args.limit else load()

    base = run_all(f"baseline{args.tag}", baseline.run, items, args.workers)
    agent = run_all(f"agent{args.tag}", run_agent, items, args.workers)
    # 参考：Job_Agent 里最初人工+模型的初评分（只有分数，没有实习/城市字段）
    manual = {it["id"]: {"score": it["initial_score"], "is_internship": False, "shenzhen": "unknown"} for it in items}

    # 把 results/ 下所有跑过的版本都列出来：历史迭代（agent_v1…）和对比实验（*_pro…）
    runs = {d.name: {int(f.stem): json.loads(f.read_text(encoding="utf-8")) for f in d.glob("*.json")}
            for d in sorted(RESULTS.iterdir()) if d.is_dir() and d.name.startswith(("baseline", "agent"))}
    table = {"初评（Job_Agent 原流程）": metrics(items, manual)}
    table.update({name: metrics(items, preds) for name, preds in runs.items()})
    md = report(items, table)
    md += "\n" + threshold_sweep(items, runs)
    md += "\n" + errors(items, agent)
    avg_pages = sum(p.get("pages_read", 0) for p in agent.values()) / max(len(agent), 1)
    skipped = sum(1 for p in agent.values() if p.get("pages_read", 0) == 0)
    md += f"\n\n初评直接结束（未精读）：{skipped}/{len(agent)} 条；平均读取海报段数：{avg_pages:.2f}\n"
    (RESULTS / "report.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
