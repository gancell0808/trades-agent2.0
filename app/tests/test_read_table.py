# -*- coding: utf-8 -*-
import io
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clean_engine import read_table, read_table_ex  # noqa: E402


def xlsx_bytes(sheets: dict) -> bytes:
    """sheets: {名称: 行列表（含表头）}"""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, rows in sheets.items():
            pd.DataFrame(rows).to_excel(w, sheet_name=name, index=False, header=False)
    return buf.getvalue()


class Csv(unittest.TestCase):
    def test_plain(self):
        df, warns = read_table_ex("公司名称,手机\nA,13800000001\nB,13800000002\n".encode("utf-8"), "a.csv")
        self.assertEqual(list(df.columns), ["公司名称", "手机"])
        self.assertEqual(len(df), 2)
        self.assertEqual(warns, [])

    def test_gbk_and_bom(self):
        gbk = "公司名称,手机\n深圳蓝海,13800000001\n".encode("gbk")
        self.assertEqual(read_table(gbk, "a.csv").iloc[0]["公司名称"], "深圳蓝海")
        bom = "﻿公司名称,手机\nA,1\n".encode("utf-8")
        self.assertEqual(list(read_table(bom, "a.csv").columns), ["公司名称", "手机"])

    def test_rows_with_too_many_fields_are_reported_not_silently_dropped(self):
        text = "公司名称,手机\nA,13800000001\nB, Inc,13800000002\nC,13800000003\n"
        df, warns = read_table_ex(text.encode("utf-8"), "a.csv")
        self.assertEqual(list(df["公司名称"]), ["A", "C"])
        self.assertEqual(len(warns), 1)
        self.assertIn("1 行", warns[0])
        self.assertIn("B", warns[0])

    def test_short_rows_are_kept_and_padded(self):
        df = read_table("公司名称,手机,姓名\nA,13800000001\nB,13800000002,乙\n".encode("utf-8"), "a.csv")
        self.assertEqual(len(df), 2)
        self.assertTrue(pd.isna(df.iloc[0]["姓名"]))
        self.assertEqual(df.iloc[1]["姓名"], "乙")

    def test_quoted_comma_and_newline(self):
        text = '公司名称,手机\n"A, Inc","13800000001"\n"B\nBranch",13800000002\n'
        df = read_table(text.encode("utf-8"), "a.csv")
        self.assertEqual(list(df["公司名称"]), ["A, Inc", "B\nBranch"])

    def test_semicolon_and_tab_delimiters(self):
        self.assertEqual(len(read_table("公司名称;手机\nA;1\nB;2\n".encode("utf-8"), "a.csv")), 2)
        self.assertEqual(len(read_table("公司名称\t手机\nA\t1\n".encode("utf-8"), "a.tsv")), 1)

    def test_title_rows_above_header_are_skipped_with_notice(self):
        text = "阿里巴巴导出 2026-09\n\n公司名称,手机,姓名\nA,13800000001,甲\n"
        df, warns = read_table_ex(text.encode("utf-8"), "a.csv")
        self.assertEqual(list(df.columns), ["公司名称", "手机", "姓名"])
        self.assertEqual(len(df), 1)
        self.assertTrue(any("被识别为表头" in w for w in warns))

    def test_unrecognizable_header_keeps_first_row(self):
        df, warns = read_table_ex("x,y\n1,2\n".encode("utf-8"), "a.csv")
        self.assertEqual(list(df.columns), ["x", "y"])
        self.assertEqual(warns, [])

    def test_duplicate_headers_kept_distinct(self):
        df = read_table("手机,手机,公司名称\n1,2,A\n".encode("utf-8"), "a.csv")
        self.assertEqual(list(df.columns), ["手机", "手机.1", "公司名称"])

    def test_blank_lines_and_empty_columns_dropped(self):
        df = read_table("公司名称,手机,\nA,1,\n,,\nB,2,\n".encode("utf-8"), "a.csv")
        self.assertEqual(list(df.columns), ["公司名称", "手机"])
        self.assertEqual(len(df), 2)

    def test_empty_file(self):
        self.assertTrue(read_table(b"", "a.csv").empty)


class Excel(unittest.TestCase):
    def test_all_sheets_are_read_and_merged(self):
        data = xlsx_bytes({
            "广交会": [["公司名称", "手机"], ["A", "13800000001"]],
            "中国制造": [["公司名称", "手机"], ["B", "13800000002"], ["C", "13800000003"]],
        })
        df, warns = read_table_ex(data, "线索.xlsx")
        self.assertEqual(sorted(df["公司名称"]), ["A", "B", "C"])
        self.assertTrue(any("2 个有数据的 sheet" in w for w in warns))

    def test_empty_sheet_ignored_with_notice(self):
        data = xlsx_bytes({"Sheet1": [["公司名称", "手机"], ["A", "1"]], "空": [[None]]})
        df, warns = read_table_ex(data, "a.xlsx")
        self.assertEqual(len(df), 1)
        self.assertTrue(any("空 sheet" in w for w in warns))

    def test_different_columns_across_sheets_warned(self):
        data = xlsx_bytes({
            "S1": [["公司名称", "手机"], ["A", "1"]],
            "S2": [["公司名称", "手机", "邮箱"], ["B", "2", "b@x.com"]],
        })
        df, warns = read_table_ex(data, "a.xlsx")
        self.assertEqual(len(df), 2)
        self.assertIn("邮箱", df.columns)
        self.assertTrue(any("不完全一致" in w for w in warns))

    def test_title_row_and_merged_header_area(self):
        data = xlsx_bytes({"Sheet1": [["2026-09 线索导出", None, None], [None, None, None],
                                      ["公司名称", "手机", "姓名"], ["A", "13800000001", "甲"]]})
        df, warns = read_table_ex(data, "a.xlsx")
        self.assertEqual(list(df.columns), ["公司名称", "手机", "姓名"])
        self.assertEqual(df.iloc[0]["手机"], "13800000001")
        self.assertTrue(any("被识别为表头" in w for w in warns))

    def test_excel_detected_by_content_when_extension_is_wrong(self):
        data = xlsx_bytes({"Sheet1": [["公司名称", "手机"], ["A", "1"]]})
        self.assertEqual(len(read_table(data, "upload.csv")), 1)

    def test_corrupt_excel_raises_clear_error(self):
        with self.assertRaises(ValueError):
            read_table(b"PK\x03\x04garbage", "a.xlsx")


class Pipeline(unittest.TestCase):
    def test_multi_sheet_rows_are_all_cleaned(self):
        from clean_engine import auto_map_columns, run_maibang_clean

        data = xlsx_bytes({
            "S1": [["公司名称", "手机"], ["A", "13800000001"]],
            "S2": [["公司名称", "手机"], ["B", "13800000002"]],
        })
        df = read_table(data, "a.xlsx")
        r = run_maibang_clean(df, auto_map_columns(list(df.columns)))
        self.assertEqual(r["stats"]["raw_count"], 2)
        self.assertEqual(r["stats"]["cleaned_count"], 2)


if __name__ == "__main__":
    unittest.main()
