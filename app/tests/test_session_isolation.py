# -*- coding: utf-8 -*-
import http.client
import json
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server  # noqa: E402


def csv_body(company, phone):
    return json.dumps({"filename": "a.csv", "content": f"公司名称,手机\n{company},{phone}\n"})


class Isolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        with server._SESSIONS_LOCK:
            server._SESSIONS.clear()

    def post(self, path, body, sid):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        h = {"Host": f"127.0.0.1:{self.port}", "Content-Type": "application/json"}
        if sid:
            h["X-Session-Id"] = sid
        c.request("POST", path, body=body, headers=h)
        r = c.getresponse()
        data = json.loads(r.read() or b"{}")
        c.close()
        return r.status, data

    def test_two_tabs_do_not_overwrite_each_other(self):
        self.post("/api/upload", csv_body("甲公司", "13800000001"), "tabA")
        self.post("/api/upload", csv_body("乙公司", "13900000002"), "tabB")
        _, a = self.post("/api/clean", "{}", "tabA")
        _, b = self.post("/api/clean", "{}", "tabB")
        self.assertEqual(a["preview"][0]["公司名称"], "甲公司")
        self.assertEqual(b["preview"][0]["公司名称"], "乙公司")

    def test_clean_without_upload_in_this_session(self):
        self.post("/api/upload", csv_body("甲公司", "13800000001"), "tabA")
        st, data = self.post("/api/clean", "{}", "tabC")
        self.assertEqual(st, 400)

    def test_invalid_sid_falls_back_to_default(self):
        self.assertIs(server.get_state("../../x"), server.get_state(None))
        self.assertIs(server.get_state("a" * 65), server.get_state("default"))

    def test_lru_eviction_caps_memory(self):
        for i in range(server.MAX_SESSIONS + 5):
            server.get_state(f"s{i}")
        self.assertEqual(len(server._SESSIONS), server.MAX_SESSIONS)
        self.assertNotIn("s0", server._SESSIONS)
        self.assertIn(f"s{server.MAX_SESSIONS + 4}", server._SESSIONS)

    def test_concurrent_clean_same_session_is_serialized(self):
        self.post("/api/upload", csv_body("甲公司", "13800000001"), "tabA")
        results = []

        def work():
            results.append(self.post("/api/clean", "{}", "tabA")[0])

        ts = [threading.Thread(target=work) for _ in range(6)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertEqual(results, [200] * 6)

    def test_removed_legacy_endpoints_are_gone(self):
        for path in ("/api/upload_b", "/api/compare"):
            st, _ = self.post(path, "{}", "tabA")
            self.assertEqual(st, 404, path)


if __name__ == "__main__":
    unittest.main()
