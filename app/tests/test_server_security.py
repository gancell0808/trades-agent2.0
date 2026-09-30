# -*- coding: utf-8 -*-
import json
import sys
import tempfile
import threading
import unittest
import http.client
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server  # noqa: E402


class SafeOutputDir(unittest.TestCase):
    def test_inside_home_ok(self):
        p = server._safe_output_dir(str(Path.home() / "out" / "x"))
        self.assertEqual(p, (Path.home() / "out" / "x").resolve())

    def test_relative_is_under_home(self):
        self.assertEqual(server._safe_output_dir("清洗"), (Path.home() / "清洗").resolve())

    def test_rejects_outside_home(self):
        for bad in ("/etc", "/etc/cron.d", "/", "/tmp/../etc", str(Path.home() / ".." / "other")):
            with self.assertRaises(ValueError, msg=bad):
                server._safe_output_dir(bad)

    def test_rejects_home_itself_and_traversal(self):
        with self.assertRaises(ValueError):
            server._safe_output_dir(str(Path.home()))
        with self.assertRaises(ValueError):
            server._safe_output_dir("../../etc")

    def test_filename_sanitized(self):
        self.assertNotIn("/", server._safe_filename("../../evil"))
        self.assertNotIn("\\", server._safe_filename("..\\evil"))
        self.assertEqual(server._safe_filename(""), "清洗结果")
        self.assertEqual(server._safe_filename("阿里巴巴"), "阿里巴巴")


class GuardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def req(self, method, path, headers=None, body=None, host=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        h = {"Host": host or f"127.0.0.1:{self.port}"}
        h.update(headers or {})
        c.request(method, path, body=body, headers=h)
        r = c.getresponse()
        data = r.read()
        c.close()
        return r.status, data

    def test_health_same_origin_ok(self):
        self.assertEqual(self.req("GET", "/api/health")[0], 200)
        self.assertEqual(self.req("GET", "/api/health", host=f"localhost:{self.port}")[0], 200)
        st, _ = self.req("GET", "/api/health", {"Origin": f"http://127.0.0.1:{self.port}"})
        self.assertEqual(st, 200)

    def test_dns_rebinding_host_rejected(self):
        self.assertEqual(self.req("GET", "/api/llm/config", host=f"evil.example.com:{self.port}")[0], 403)
        self.assertEqual(self.req("GET", "/api/health", host="127.0.0.1:1")[0], 403)

    def test_cross_origin_post_rejected(self):
        st, _ = self.req("POST", "/api/llm/config", {"Origin": "https://evil.example.com",
                         "Content-Type": "text/plain"}, body=json.dumps({"base_url": "http://x"}))
        self.assertEqual(st, 403)
        st, _ = self.req("POST", "/api/qa/sessions/clear", {"Origin": "null"})
        self.assertEqual(st, 403)

    def test_sec_fetch_site_cross_site_rejected(self):
        self.assertEqual(self.req("GET", "/api/health", {"Sec-Fetch-Site": "cross-site"})[0], 403)
        self.assertEqual(self.req("GET", "/api/health", {"Sec-Fetch-Site": "same-origin"})[0], 200)

    def test_export_outside_home_rejected(self):
        with mock.patch.dict(server._STATE, {"last_result": {"rows": None, "stats": {}}}):
            st, data = self.req("POST", "/api/export", {"Content-Type": "application/json"},
                                body=json.dumps({"output_dir": "/etc/cron.d"}))
        self.assertEqual(st, 400)
        self.assertIn("主目录", json.loads(data)["error"])


if __name__ == "__main__":
    unittest.main()
