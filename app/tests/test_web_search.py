# -*- coding: utf-8 -*-
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import web_search as ws  # noqa: E402

BING = """<ol><li class="b_algo"><h2><a href="https://example.com/a">示例 <b>标题</b></a></h2>
<div class="b_caption"><p>这是 <strong>摘要</strong> 文本</p></div></li>
<li class="b_algo"><h2><a href="/ck/x">相对链接</a></h2><p>第二条</p></li></ol>"""
DDG = """<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fp&rut=1">DDG <b>标题</b></a>
<a class="result__snippet" href="#">DDG 摘要</a>"""


class Parsers(unittest.TestCase):
    def test_bing(self):
        r = ws.parse_bing(BING, 5)
        self.assertEqual(len(r), 2)
        self.assertEqual(r[0]["url"], "https://example.com/a")
        self.assertEqual(r[0]["title"], "示例 标题")
        self.assertIn("摘要", r[0]["snippet"])
        self.assertTrue(r[1]["url"].startswith("https://www.bing.com/"))
        self.assertEqual(len(ws.parse_bing(BING, 1)), 1)

    def test_ddg(self):
        r = ws.parse_ddg(DDG, 5)
        self.assertEqual(r[0]["url"], "https://example.org/p")
        self.assertEqual(r[0]["title"], "DDG 标题")

    def test_unrecognized_page_yields_nothing(self):
        self.assertEqual(ws.parse_bing("<html>redesigned</html>", 5), [])
        self.assertEqual(ws.parse_ddg("<html>redesigned</html>", 5), [])


class Engines(unittest.TestCase):
    def test_failure_reasons_are_reported(self):
        with mock.patch.object(ws, "_fetch", side_effect=OSError("timed out")):
            hits, errs = ws.web_search_ex("x")
        self.assertEqual(hits, [])
        self.assertEqual(len(errs), 2)
        self.assertTrue(all("无法访问" in e for e in errs))

    def test_blocked_page_detected(self):
        with mock.patch.object(ws, "_fetch", return_value="<html>Please solve the CAPTCHA</html>"):
            hits, errs = ws.web_search_ex("x")
        self.assertEqual(hits, [])
        self.assertTrue(all("拦截" in e for e in errs))

    def test_layout_change_reported(self):
        with mock.patch.object(ws, "_fetch", return_value="<html>nothing here</html>"):
            _, errs = ws.web_search_ex("x")
        self.assertTrue(all("没有解析到结果" in e for e in errs))

    def test_fallback_to_second_engine(self):
        def fake(url):
            if "bing.com" in url:
                raise OSError("boom")
            return DDG
        with mock.patch.object(ws, "_fetch", side_effect=fake):
            hits, errs = ws.web_search_ex("x")
        self.assertEqual(hits[0]["url"], "https://example.org/p")
        self.assertEqual(errs, [])

    def test_legacy_wrapper(self):
        with mock.patch.object(ws, "_fetch", return_value=BING):
            self.assertEqual(len(ws.web_search("x")), 2)


class StreamNotice(unittest.TestCase):
    def test_user_is_told_when_search_fails(self):
        import http.client
        import json
        import threading
        from http.server import ThreadingHTTPServer

        import server

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            with mock.patch.object(server, "web_search_ex", return_value=([], ["Bing：无法访问（timed out）"])):
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                c.request("POST", "/api/qa/stream",
                          body=json.dumps({"question": "你好", "use_llm": False, "use_web_search": True}),
                          headers={"Host": f"127.0.0.1:{port}", "Content-Type": "application/json"})
                body = c.getresponse().read().decode("utf-8")
                c.close()
        finally:
            httpd.shutdown()
            httpd.server_close()
        done = [json.loads(l[5:]) for l in body.splitlines() if l.startswith("data:") and '"done"' in l][0]
        self.assertTrue(done["content"].startswith("⚠ 联网搜索未获得结果"))
        self.assertIn("timed out", done["content"])


if __name__ == "__main__":
    unittest.main()
