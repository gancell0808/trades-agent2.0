# -*- coding: utf-8 -*-
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clean_engine import extract_valid_mobiles as f  # noqa: E402


class ExtractMobiles(unittest.TestCase):
    def check(self, raw, expected):
        self.assertEqual(f(raw), expected, repr(raw))

    def test_plain(self):
        self.check("13812345678", ["13812345678"])
        self.check("18212345678", ["18212345678"])  # 含 82/81 子串不得被吞
        self.check("13381234567", ["13381234567"])

    def test_prefixes(self):
        self.check("+86 138-1234-5678", ["13812345678"])
        self.check("0086 13812345678", ["13812345678"])
        self.check("8613812345678", ["13812345678"])
        self.check("86-13812345678", ["13812345678"])
        self.check("+8613812345678", ["13812345678"])
        self.check("+86 (0)138 1234 5678", ["13812345678"])

    def test_multi(self):
        self.check("13812345678/13912345678", ["13812345678", "13912345678"])
        self.check("13812345678、13912345678；13712345678", ["13812345678", "13912345678", "13712345678"])
        self.check("13812345678 13912345678", ["13812345678", "13912345678"])
        self.check("13812345678\n13912345678,13712345678", ["13812345678", "13912345678", "13712345678"])
        self.check("13812345678/13812345678", ["13812345678"])  # 去重保序

    def test_no_concatenation(self):
        self.check("1381234567813912345678", [])  # 不得拼出假号
        self.check("138 1234 5678", ["13812345678"])  # 号码内部空格是同一个号
        self.check("1391234567890123", [])

    def test_landline_and_intl_dropped(self):
        self.check("021-12345678", [])
        self.check("400-186-1234", [])
        self.check("+852 91234567", [])
        self.check("+853 62812345", [])
        self.check("+886 912345678", [])
        self.check("+81 9012345678", [])
        self.check("+8613812345678、+85298765432", ["13812345678"])
        self.check("0571-88123456 18612345678", ["18612345678"])

    def test_empty_and_noise(self):
        for v in (None, "", "nan", "None", "-", "abc"):
            self.check(v, [])
        self.check(13812345678.0, ["13812345678"])
        self.check("13812345678.0", ["13812345678"])

    def test_cap(self):
        nums = [f"138{i:08d}" for i in range(7)]
        self.assertEqual(len(f("/".join(nums))), 5)


if __name__ == "__main__":
    unittest.main()
