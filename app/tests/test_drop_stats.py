# -*- coding: utf-8 -*-
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clean_engine import classify_unusable, run_maibang_clean  # noqa: E402

MAP = {"company_name": "公司", "mobile": "手机"}
DROP_KEYS = ["dropped_no_mobile", "dropped_landline_only", "dropped_intl_only",
             "dropped_invalid", "dropped_no_company", "duplicated_company"]


class Classify(unittest.TestCase):
    def test_kinds(self):
        cases = {
            "": "empty", None: "empty", "nan": "empty", "abc": "empty",
            "021-12345678": "landline", "0571 88123456": "landline", "88123456": "landline",
            "400-186-1234": "landline",
            "+852 91234567": "intl", "+886 912345678": "intl", "0085291234567": "intl",
            "1381234567": "invalid",      # 10 位，不是座机
            "23812345678": "invalid",     # 11 位但不以 1 开头
            "138123456789": "invalid",    # 12 位
            "021-12345678 / +852 91234567": "landline",  # 混合：座机优先
            "+852 91234567 / 1234": "intl",
        }
        for raw, want in cases.items():
            self.assertEqual(classify_unusable(raw), want, repr(raw))


class Stats(unittest.TestCase):
    def test_each_row_counted_once_and_sums_up(self):
        df = pd.DataFrame({
            "公司": ["A", "B", "C", "D", "E", "", "", "A", "F", "G"],
            "手机": ["13812345678", "", "021-12345678", "+852 91234567", "1381234567",
                   "13912345678", "", "13712345678", "13612345678", "23812345678"],
        })
        r = run_maibang_clean(df, MAP)
        st = r["stats"]
        self.assertEqual(st["cleaned_count"], 2)  # A、F
        self.assertEqual(st["dropped_no_mobile"], 2)      # B、第7行(空公司空手机)
        self.assertEqual(st["dropped_landline_only"], 1)  # C
        self.assertEqual(st["dropped_intl_only"], 1)      # D
        self.assertEqual(st["dropped_invalid"], 2)        # E、G
        self.assertEqual(st["dropped_no_company"], 1)     # 第6行：有手机缺公司
        self.assertEqual(st["duplicated_company"], 1)     # 第二个 A
        self.assertEqual(st["cleaned_count"] + sum(st[k] for k in DROP_KEYS), st["raw_count"])

    def test_no_company_not_counted_as_no_mobile(self):
        df = pd.DataFrame({"公司": [""], "手机": ["13812345678"]})
        st = run_maibang_clean(df, MAP)["stats"]
        self.assertEqual(st["dropped_no_company"], 1)
        self.assertEqual(st["dropped_no_mobile"], 0)


if __name__ == "__main__":
    unittest.main()
