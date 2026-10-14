"""官网岗位查询工具：根据投递链接，查招聘系统里每个岗位的工作城市。

目前支持北森（*.zhiye.com）：公开的 JSON 接口，无需登录。
不支持 Moka：它的岗位接口返回加密数据，属于对方的反爬保护，我们不去破解——返回 None，交给人工核实。
结果缓存在 data/official_cache/，同一家公司不会重复请求。
"""
import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

CACHE = Path(__file__).parent / "data" / "official_cache"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36"


def _resolve(url: str) -> str:
    """短链接（如 bsurl.cn）先跟随跳转，拿到真正的招聘系统地址。"""
    if "zhiye.com" in url:
        return url
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=15, allow_redirects=True)
        # 有的短链接（bsurl.cn）不是 HTTP 302，而是页面里用 JS 跳转，再跟一跳
        js = re.search(r'window\.location\.href\s*=\s*"([^"]+)"', r.text)
        if js:
            r = requests.get(js.group(1), headers={"User-Agent": UA}, timeout=15, allow_redirects=True)
        return r.url
    except requests.RequestException:
        return url


def _beisen(host: str) -> list[dict]:
    s = requests.Session()
    s.headers["User-Agent"] = UA
    html = s.get(f"https://{host}/campus/jobs", timeout=20).text
    m = re.search(r'"PortalId":"([0-9a-f-]{36})"', html)
    if not m:
        return []
    jobs, page = [], 0
    while True:  # 单页最多 100 条，需要翻页
        r = s.post(f"https://{host}/api/Jobad/GetJobAdPageList", timeout=20, json={
            "PageIndex": page, "PageSize": 100, "KeyWords": "", "SpecialType": 0,
            "PortalId": m.group(1), "Category": 2, "DisplayFields": ["LocId"]})
        data = r.json().get("Data") or []
        jobs += [{"name": x["JobAdName"], "cities": x.get("LocNames") or []} for x in data]
        if len(data) < 100:
            return jobs
        page += 1
        time.sleep(1)


def lookup(apply_url: str) -> list[dict] | None:
    """返回 [{name, cities}]；不支持的招聘系统返回 None。"""
    if not apply_url:
        return None
    url = _resolve(apply_url)
    host = urlparse(url).netloc
    if not host.endswith("zhiye.com"):
        return None
    cache = CACHE / f"{host}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    jobs = _beisen(host)
    CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(jobs, ensure_ascii=False), encoding="utf-8")
    return jobs
