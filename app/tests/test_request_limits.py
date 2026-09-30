# -*- coding: utf-8 -*-
import http.client
import io
import json
import socket
import sys
import threading
import unittest
import zipfile
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import clean_engine  # noqa: E402
import server  # noqa: E402

B = "XXBOUNDARY"


def mp_body(files):
    body = b""
    for fname, data in files:
        body += (f'--{B}\r\nContent-Disposition: form-data; name="file"; filename="{fname}"\r\n'
                 "Content-Type: application/octet-stream\r\n\r\n").encode() + data + b"\r\n"
    return body + f"--{B}--\r\n".encode()


CSV = "公司名称,手机\nA,13800000001\nB,13800000002\nC,13800000003\n".encode()


class Limits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def request(self, path, body, ctype="application/json", extra=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        h = {"Host": f"127.0.0.1:{self.port}", "Content-Type": ctype, "X-Session-Id": "t-limits"}
        h.update(extra or {})
        c.request("POST", path, body=body, headers=h)
        r = c.getresponse()
        data = json.loads(r.read() or b"{}")
        c.close()
        return r.status, data

    def raw(self, head_lines, body=b"", half_close=False):
        """手写 HTTP 请求，用来构造 http.client 不会发的异常头。"""
        s = socket.create_connection(("127.0.0.1", self.port), timeout=10)
        req = "POST /api/upload HTTP/1.1\r\n" + f"Host: 127.0.0.1:{self.port}\r\n" + "".join(l + "\r\n" for l in head_lines) + "\r\n"
        s.sendall(req.encode() + body)
        if half_close:
            s.shutdown(socket.SHUT_WR)
        buf = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
        s.close()
        head, _, payload = buf.partition(b"\r\n\r\n")
        return int(head.split()[1]), json.loads(payload or b"{}")

    # ---- 大小上限 ----
    def test_oversized_upload_rejected_with_413(self):
        with mock.patch.object(server, "MAX_UPLOAD_BYTES", 1000):
            st, d = self.request("/api/upload", b"x" * 5000, "multipart/form-data; boundary=" + B)
        self.assertEqual(st, 413)
        self.assertIn("上限", d["error"])

    def test_oversized_json_endpoint_rejected(self):
        with mock.patch.object(server, "MAX_BODY_BYTES", 100):
            st, d = self.request("/api/qa", json.dumps({"question": "x" * 500}))
        self.assertEqual(st, 413)

    def test_upload_limit_is_larger_than_json_limit(self):
        # 同样大小的请求体：对 JSON 接口超限，对上传接口不超限
        body = mp_body([("a.csv", CSV)])
        with mock.patch.object(server, "MAX_BODY_BYTES", 50), mock.patch.object(server, "MAX_UPLOAD_BYTES", 10_000):
            st, d = self.request("/api/upload", body, "multipart/form-data; boundary=" + B)
        self.assertEqual(st, 200, d)

    def test_declared_size_checked_before_reading(self):
        # 声明 1GB 但只发一点：应当立刻 413，不会等着读 1GB
        st, d = self.raw(["Content-Type: application/json", "Content-Length: 1073741824"], b"{}")
        self.assertEqual(st, 413)

    # ---- 异常请求头 ----
    def test_invalid_content_length(self):
        for v in ("abc", "-5"):
            st, _ = self.raw(["Content-Type: application/json", f"Content-Length: {v}"])
            self.assertEqual(st, 400, v)

    def test_chunked_not_supported(self):
        st, _ = self.raw(["Content-Type: application/json", "Transfer-Encoding: chunked"], b"0\r\n\r\n")
        self.assertEqual(st, 411)

    def test_truncated_body_returns_400_not_hang(self):
        st, d = self.raw(["Content-Type: application/json", "Content-Length: 100"], b'{"a":1}', half_close=True)
        self.assertEqual(st, 400)
        self.assertIn("不完整", d["error"])

    def test_invalid_json_is_400(self):
        st, d = self.request("/api/qa", "{not json")
        self.assertEqual(st, 400)

    # ---- 上传内容限制 ----
    def test_too_many_files(self):
        with mock.patch.object(server, "MAX_UPLOAD_FILES", 2):
            st, d = self.request("/api/upload", mp_body([("a.csv", CSV)] * 3), "multipart/form-data; boundary=" + B)
        self.assertEqual(st, 400)
        self.assertIn("最多", d["error"])

    def test_too_many_rows(self):
        with mock.patch.object(server, "MAX_ROWS", 2):
            st, d = self.request("/api/upload", mp_body([("a.csv", CSV)]), "multipart/form-data; boundary=" + B)
        self.assertEqual(st, 413)
        self.assertIn("行", d["error"])

    def test_rows_at_limit_ok(self):
        with mock.patch.object(server, "MAX_ROWS", 3):
            st, d = self.request("/api/upload", mp_body([("a.csv", CSV)]), "multipart/form-data; boundary=" + B)
        self.assertEqual(st, 200)

    def test_unparseable_file_is_400_not_500(self):
        st, d = self.request("/api/upload", mp_body([("a.xlsx", b"PK\x03\x04garbage")]), "multipart/form-data; boundary=" + B)
        self.assertEqual(st, 400)
        self.assertIn("无法解析", d["error"])

    def test_health_exposes_limits(self):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request("GET", "/api/health", headers={"Host": f"127.0.0.1:{self.port}"})
        d = json.loads(c.getresponse().read())
        c.close()
        self.assertEqual(d["limits"]["max_upload_bytes"], server.MAX_UPLOAD_BYTES)
        self.assertEqual(d["limits"]["max_rows"], server.MAX_ROWS)


class ZipBomb(unittest.TestCase):
    def test_highly_compressible_xlsx_rejected(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("xl/worksheets/sheet1.xml", b"0" * (3 * 1024 * 1024))
        data = buf.getvalue()
        self.assertLess(len(data), 50_000)  # 压缩后很小
        with mock.patch.object(clean_engine, "MAX_UNCOMPRESSED_BYTES", 1024 * 1024):
            with self.assertRaises(ValueError) as cm:
                clean_engine.read_table_ex(data, "a.xlsx")
        self.assertIn("解压后", str(cm.exception))

    def test_normal_xlsx_unaffected(self):
        import pandas as pd

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            pd.DataFrame([["公司名称", "手机"], ["A", "1"]]).to_excel(w, index=False, header=False)
        self.assertEqual(len(clean_engine.read_table(buf.getvalue(), "a.xlsx")), 1)


if __name__ == "__main__":
    unittest.main()
