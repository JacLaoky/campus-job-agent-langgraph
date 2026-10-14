"""从 Job_Agent 的数据库导出评测集 data/eval.jsonl。

每条简章包含：看板元数据、海报原文（正文 + OCR）、人工精读得到的标签、精读前的初评分。
标签来源是人工精读（读海报 + 查官网），不是模型输出。
"""
import glob
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT.parent / "data"
OUT = ROOT / "data" / "eval.jsonl"


def shenzhen_label(text):
    """把精读记录里的「深圳有岗」描述归一成 True / False / None（无法确认）。"""
    t = text or ""
    if any(k in t for k in ("待确认", "未写", "简章层面", "无官网", "见 ", "无")) and not t.startswith(("是", "否")):
        return None
    if t.startswith("是"):
        return True
    if t.startswith(("否", "仅", "官网深圳")) or "没有" in t or "未见" in t:
        return False
    return None


# 「是否实习招聘」的人工复核名单。不用分数推断：有的实习简章精读时给了 40 分
#（如阿里云、ByteIntern），按分数阈值会被漏标。
INTERNSHIP_IDS = {40977, 42260, 42754, 44173, 44200, 44275, 44878, 45020, 45102, 45668, 46719, 46766, 47323}


def initial_scores():
    """精读前的第一次评分（data/scores/*.json，按文件名顺序取最早一次）。"""
    first = {}
    for f in sorted(glob.glob(str(SRC / "scores" / "[0-9]*.json"))):  # 跳过 _batch.json 等临时文件
        for it in json.load(open(f, encoding="utf-8")):
            first.setdefault(it["id"], it["score"])
    return first


def poster_text(pid, limit=8000):
    d = SRC / "posters" / str(pid)
    parts = [(d / name).read_text(encoding="utf-8").strip() for name in ("text.txt", "ocr.txt") if (d / name).exists()]
    text = "\n".join(p for p in parts if p and len(p) < 200_000)  # 图片消息类文章的 text.txt 是整页源码，跳过
    return text[:limit]


def main():
    conn = sqlite3.connect(SRC / "jobs.db")
    conn.row_factory = sqlite3.Row
    first = initial_scores()
    rows = conn.execute("""
        SELECT p.id, p.company, p.name, p.jobs, p.cities, p.class_time, p.expired_at, p.apply_url,
               s.score AS final_score, s.reason, d.shenzhen
        FROM deep_reads d JOIN postings p USING(id) JOIN scores s USING(id)""").fetchall()
    OUT.parent.mkdir(exist_ok=True)
    n = 0
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            text = poster_text(r["id"])
            if not text:
                continue
            item = {
                "id": r["id"], "company": r["company"], "title": r["name"], "jobs": r["jobs"],
                "cities": r["cities"], "class_time": r["class_time"], "apply_url": (r["apply_url"] or "").split("?")[0],  # 去掉看板带出的分享/追踪参数
                "poster_text": text,
                "initial_score": first.get(r["id"]),
                "label": {
                    "worth_applying": r["final_score"] >= 75,
                    "is_internship": r["id"] in INTERNSHIP_IDS,
                    "shenzhen": shenzhen_label(r["shenzhen"]),
                    "final_score": r["final_score"],
                },
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1
    print(f"写入 {n} 条 → {OUT}")


if __name__ == "__main__":
    main()
