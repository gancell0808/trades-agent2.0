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

B = "XXBOUNDARY"


class Uploads(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def multipart(self, path, files):
        body = b""
        for fname, data in files:
            body += (f'--{B}\r\nContent-Disposition: form-data; name="file"; filename="{fname}"\r\n'
                     "Content-Type: application/octet-stream\r\n\r\n").encode() + data + b"\r\n"
        body += f"--{B}--\r\n".encode()
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request("POST", path, body=body, headers={
            "Host": f"127.0.0.1:{self.port}", "X-Session-Id": "t-upload",
            "Content-Type": f"multipart/form-data; boundary={B}"})
        r = c.getresponse()
        out = json.loads(r.read())
        c.close()
        return r.status, out

    def test_upload_reports_skipped_rows(self):
        st, d = self.multipart("/api/upload", [("a.csv", "公司名称,手机\nA,13800000001\nB, Inc,13800000002\n".encode())])
        self.assertEqual(st, 200)
        self.assertEqual(d["rows"], 1)
        self.assertEqual(len(d["warnings"]), 1)
        self.assertIn("未导入", d["warnings"][0])

    def test_upload_reports_ignored_empty_file(self):
        st, d = self.multipart("/api/upload", [("a.csv", "公司名称,手机\nA,13800000001\n".encode()), ("empty.csv", b"")])
        self.assertEqual(st, 200)
        self.assertEqual(d["filenames"], ["a.csv"])
        self.assertTrue(any("empty.csv" in w for w in d["warnings"]))

    def test_all_files_unreadable_returns_400_with_reason(self):
        st, d = self.multipart("/api/upload", [("empty.csv", b"")])
        self.assertEqual(st, 400)
        self.assertTrue(d["warnings"])

    def test_qa_attachment_text_is_extracted(self):
        # 回归：附件走 multipart 时曾因被当成表格解析而取不到文本
        st, d = self.multipart("/api/qa/upload", [("note.txt", "清洗规则说明".encode())])
        self.assertEqual(st, 200)
        self.assertEqual(d["files"][0]["chars"], 6)
        self.assertEqual(d["files"][0]["preview"], "清洗规则说明")


if __name__ == "__main__":
    unittest.main()
