# -*- coding: utf-8 -*-
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clean_engine import DEFAULT_CLEANING, normalize_company_key as nk, run_maibang_clean  # noqa: E402

MAP = {"company_name": "公司", "mobile": "手机"}


class Normalize(unittest.TestCase):
    def test_equivalent_names(self):
        same = [
            ("ABC Co., Ltd.", "abc co ltd"),
            ("ABC  Trading", "ABC Trading"),
            ("深圳（华强）电子有限公司", "深圳(华强)电子有限公司"),
            ("深圳 华强 电子有限公司", "深圳华强电子有限公司"),
            ("ＡＢＣ　Ｌｔｄ", "abc ltd"),          # 全角字母与全角空格
            ("Ningbo Sea Star Import & Export", "ningbo sea star import export"),
        ]
        for a, b in same:
            self.assertEqual(nk(a), nk(b), (a, b))

    def test_different_entities_stay_different(self):
        # 不去后缀：不同法人主体不能被合并
        self.assertNotEqual(nk("深圳华强电子有限公司"), nk("深圳华强电子"))
        self.assertNotEqual(nk("ABC Ltd"), nk("ABC Inc"))
        self.assertNotEqual(nk("蓝海五金"), nk("蓝海五金二厂"))

    def test_empty(self):
        self.assertEqual(nk(None), "")
        self.assertEqual(nk(" ， "), "")


class DedupeFlow(unittest.TestCase):
    df = pd.DataFrame({
        "公司": ["ABC Co., Ltd.", "深圳（华强）电子", "abc co ltd", "深圳(华强)电子", "深圳华强电子有限公司"],
        "手机": ["13800000001", "13800000002", "13800000003", "13800000004", "13800000005"],
    })

    def test_dedupe_keeps_first_and_original_name(self):
        r = run_maibang_clean(self.df, MAP)
        self.assertEqual(list(r["rows"]["公司名称"]), ["ABC Co., Ltd.", "深圳（华强）电子", "深圳华强电子有限公司"])
        self.assertEqual(list(r["rows"]["常用电话"]), ["13800000001", "13800000002", "13800000005"])
        self.assertEqual(r["stats"]["duplicated_company"], 2)

    def test_can_disable_normalization(self):
        r = run_maibang_clean(self.df, MAP, cleaning={**DEFAULT_CLEANING, "dedupe_normalize": False})
        self.assertEqual(len(r["rows"]), 5)
        self.assertEqual(r["stats"]["duplicated_company"], 0)


if __name__ == "__main__":
    unittest.main()
