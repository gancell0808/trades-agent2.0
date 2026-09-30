# -*- coding: utf-8 -*-
import json
import shutil
import ssl
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import llm_client  # noqa: E402


class _H(BaseHTTPRequestHandler):
    hits = 0

    def do_POST(self):
        type(self).hits += 1
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        body = json.dumps({"choices": [{"message": {"content": "OK"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@unittest.skipUnless(shutil.which("openssl"), "需要 openssl 生成自签名证书")
class TlsVerify(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        key, crt = f"{cls.tmp}/k.pem", f"{cls.tmp}/c.pem"
        subprocess.run(
            ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", key, "-out", crt,
             "-days", "1", "-subj", "/CN=localhost"],
            check=True, capture_output=True,
        )
        cls.httpd = HTTPServer(("127.0.0.1", 0), _H)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(crt, key)
        cls.httpd.socket = ctx.wrap_socket(cls.httpd.socket, server_side=True)
        cls.url = f"https://127.0.0.1:{cls.httpd.server_address[1]}/v1/chat/completions"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def call(self, verify):
        return llm_client._http_post_json(
            self.url, {"x": 1}, {"Content-Type": "application/json", "Authorization": "Bearer SECRET"},
            5, verify=verify)

    def test_untrusted_cert_rejected_by_default(self):
        _H.hits = 0
        with self.assertRaises(ValueError) as cm:
            self.call(True)
        self.assertIn("证书校验失败", str(cm.exception))
        self.assertEqual(_H.hits, 0, "校验失败时不得向服务器发送请求（含 API Key）")

    def test_explicit_opt_out_works(self):
        self.assertEqual(self.call(False)["choices"][0]["message"]["content"], "OK")

    def test_chat_defaults_to_verify(self):
        cfg = {"base_url": self.url, "api_key": "SECRET", "model": "m"}
        _H.hits = 0
        with self.assertRaises(ValueError):
            llm_client.chat([{"role": "user", "content": "hi"}], cfg)
        self.assertEqual(_H.hits, 0)
        self.assertEqual(llm_client.chat([{"role": "user", "content": "hi"}], {**cfg, "ssl_verify": False}), "OK")

    def test_stream_fallback_also_verifies(self):
        cfg = {"base_url": self.url, "api_key": "SECRET", "model": "m"}
        _H.hits = 0
        with self.assertRaises(ValueError):
            list(llm_client.chat_stream([{"role": "user", "content": "hi"}], cfg))
        self.assertEqual(_H.hits, 0)


class ProfileField(unittest.TestCase):
    def test_default_true_and_parsing(self):
        n = llm_client._normalize_profile
        self.assertIs(n({})["ssl_verify"], True)
        self.assertIs(n({"ssl_verify": False})["ssl_verify"], False)
        self.assertIs(n({"ssl_verify": "false"})["ssl_verify"], False)
        self.assertIs(n({"ssl_verify": "true"})["ssl_verify"], True)


if __name__ == "__main__":
    unittest.main()
