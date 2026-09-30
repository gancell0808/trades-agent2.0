# -*- coding: utf-8 -*-
"""联网搜索：抓取公开搜索页的标题/链接/摘要（尽力而为，失败时给出明确原因）。

没有官方搜索接口可用时，解析 HTML 天然脆弱：页面改版、被识别为爬虫都会导致没有结果。
本模块不静默吞错，而是把「为什么没结果」返回给调用方，由界面告知用户。
"""
from __future__ import annotations

import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

TIMEOUT = 5  # 秒；两个引擎并发请求，总耗时约为一次超时

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

_BLOCK_HINTS = ("captcha", "unusual traffic", "verify you are a human", "验证码", "异常流量", "anomaly")


def _strip_tags(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s or "")).strip()


def parse_bing(html: str, limit: int) -> list[dict]:
    results: list[dict] = []
    for block in re.split(r'<li class="b_algo"', html, flags=re.I)[1:]:
        tm = re.search(r'<a class="tilk"[^>]*href="([^"]+)"', block, re.I)
        if tm:
            link = tm.group(1)
            label = re.search(r'aria-label="([^"]*)"', block, re.I)
            title = label.group(1) if label else ""
        else:
            hm = re.search(r'<h2>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.S | re.I)
            if not hm:
                continue
            link, title = hm.group(1), hm.group(2)
        if link.startswith("/"):
            link = "https://www.bing.com" + link
        if not link.startswith("http"):
            continue
        sm = re.search(r"<p[^>]*>(.*?)</p>", block, re.S | re.I) or re.search(
            r'class="b_caption"[^>]*>(.*?)</div>', block, re.S | re.I)
        snippet = _strip_tags(sm.group(1) if sm else "")[:220]
        title = _strip_tags(title)
        h2 = re.search(r"<h2[^>]*>\s*<a[^>]*>(.*?)</a>", block, re.S | re.I)
        if h2 and _strip_tags(h2.group(1)):
            title = _strip_tags(h2.group(1))
        if not title:
            title = urllib.parse.urlparse(link).netloc
        results.append({"title": title[:80], "url": link, "snippet": snippet})
        if len(results) >= limit:
            break
    return results


def parse_ddg(html: str, limit: int) -> list[dict]:
    results: list[dict] = []
    for m in re.finditer(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</',
        html, re.S | re.I,
    ):
        link, title, snippet = m.group(1), _strip_tags(m.group(2))[:80], _strip_tags(m.group(3))[:220]
        if "uddg=" in link:
            link = urllib.parse.unquote(link.split("uddg=")[-1].split("&")[0])
        if title and link:
            results.append({"title": title, "url": link, "snippet": snippet})
        if len(results) >= limit:
            break
    return results


def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers=HEADERS, method="GET")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _run_engine(name: str, url: str, parser: Callable[[str, int], list[dict]], limit: int) -> tuple[list[dict], str]:
    """返回 (结果, 失败原因)。结果为空时失败原因必不为空。"""
    try:
        html = _fetch(url)
    except Exception as e:
        return [], f"{name}：无法访问（{str(e)[:60]}）"
    hits = parser(html, limit)
    if hits:
        return hits, ""
    if any(h in html.lower() for h in _BLOCK_HINTS):
        return [], f"{name}：被搜索引擎拦截（需要验证）"
    return [], f"{name}：页面中没有解析到结果（可能是页面改版或关键词无结果）"


def web_search_ex(query: str, limit: int = 5) -> tuple[list[dict], list[str]]:
    """搜索并返回 (结果, 错误说明列表)。优先 Bing，其次 DuckDuckGo。"""
    q = urllib.parse.quote(query)
    engines = [
        ("Bing", f"https://www.bing.com/search?q={q}&setlang=zh-CN", parse_bing),
        ("DuckDuckGo", f"https://html.duckduckgo.com/html/?q={q}", parse_ddg),
    ]
    with ThreadPoolExecutor(max_workers=len(engines)) as pool:
        futures = [pool.submit(_run_engine, n, u, p, limit) for n, u, p in engines]
        outcomes = [f.result() for f in futures]
    errors: list[str] = []
    for hits, err in outcomes:  # 按引擎优先级取第一个有结果的
        if hits:
            return hits, []
        errors.append(err)
    return [], errors


def web_search(query: str, limit: int = 5) -> list[dict]:
    """兼容旧接口：只返回结果列表。"""
    return web_search_ex(query, limit)[0]
