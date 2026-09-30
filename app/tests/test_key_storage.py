# -*- coding: utf-8 -*-
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import llm_client  # noqa: E402


class KeyStorage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "llm_config.json"
        self.patch = mock.patch.object(llm_client, "CONFIG_PATH", self.path)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    @unittest.skipIf(os.name == "nt", "POSIX 权限")
    def test_file_mode_600(self):
        llm_client.upsert_profile("deepseek", {"api_key": "sk-secret"})
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        self.assertFalse(list(self.path.parent.glob("*.tmp")))

    def test_key_roundtrip_and_masked_public(self):
        llm_client.upsert_profile("deepseek", {"api_key": "sk-1234567890abcd"})
        self.assertEqual(json.loads(self.path.read_text("utf-8"))["profiles"]["deepseek"]["api_key"], "sk-1234567890abcd")
        pub = json.dumps(llm_client.public_store())
        self.assertNotIn("sk-1234567890abcd", pub)

    def test_env_key_not_persisted(self):
        with mock.patch.dict(os.environ, {"TRADES_LLM_KEY": "ENV-SECRET", "TRADES_LLM_MODEL": "env-model"}):
            self.assertEqual(llm_client.load_config()["api_key"], "ENV-SECRET")  # 运行时可用
            llm_client.upsert_profile("deepseek", {"label": "x"})
            llm_client.set_active("deepseek")
            llm_client.save_config({"temperature": 0.5})
        text = self.path.read_text("utf-8")
        self.assertNotIn("ENV-SECRET", text)
        self.assertNotIn("env-model", text)


class RepoConfig(unittest.TestCase):
    def test_config_json_has_no_key_field(self):
        cfg = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text("utf-8"))
        self.assertNotIn("api_key", json.dumps(cfg))


if __name__ == "__main__":
    unittest.main()
