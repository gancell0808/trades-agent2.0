# -*- coding: utf-8 -*-
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agent_core  # noqa: E402
from agent_core import parse_intent_rules, run_agent, verify_cleaned  # noqa: E402

DF = pd.DataFrame({
    "公司名称": ["A公司", "B公司", "A公司", "C公司"],
    "联系人": ["甲", "乙", "丙", "丁"],
    "手机": ["13812345678", "+86 13912345678/13712345678", "13612345678", "021-12345678"],
})


def steps(out, tool):
    return [x for x in out["steps"] if x["tool"] == tool]


class AgentRun(unittest.TestCase):
    def test_uses_cleaning_config_override(self):
        # 覆盖 template_columns 与手机规则，证明 Agent 模式确实读取了清洗配置
        out = run_agent(DF, "帮我清洗一下", use_llm=False,
                        cleaning={"mobile_slots": ["常用电话"]})
        rows = out["cleaned"]["rows"]
        self.assertEqual(list(rows["常用电话"]), ["13812345678", "13912345678"])
        self.assertEqual(out["cleaned"]["cleaning"]["mobile_slots"], ["常用电话"])
        self.assertEqual(rows["备用电话1"].tolist(), ["", ""])  # 槽位被限制后不再填备用

    def test_quality_check_reports_real_result(self):
        out = run_agent(DF, "清洗", use_llm=False)
        q = steps(out, "quality_check")[0]
        self.assertEqual(q["status"], "done")
        self.assertIn("可导出", out["result_summary"]["review"])

    def test_verify_detects_problems(self):
        bad = pd.DataFrame({"公司名称": ["X", "X", ""], "常用电话": ["13812345678", "12345", ""],
                            "备用电话1": ["", "", ""], "备用电话2": ["", "", ""]})
        v = verify_cleaned(bad, agent_core.DEFAULT_CLEANING, {"company_name": "c", "mobile": "m"})
        self.assertFalse(v["ok"])
        text = "；".join(v["problems"])
        for kw in ("不符合手机号规则", "常用电话为空", "公司名称为空", "重复"):
            self.assertIn(kw, text)

    def test_empty_result_not_exportable(self):
        df = pd.DataFrame({"公司名称": ["A"], "手机": ["021-12345678"]})
        out = run_agent(df, "清洗", use_llm=False)
        self.assertEqual(steps(out, "quality_check")[0]["status"], "warn")
        self.assertNotIn("可导出。", out["result_summary"]["review"])

    def test_missing_mapping_flagged(self):
        df = pd.DataFrame({"x": ["A"], "y": ["B"]})
        out = run_agent(df, "清洗", use_llm=False, mapping={})
        self.assertEqual(steps(out, "quality_check")[0]["status"], "warn")

    def test_unsupported_requests_are_reported(self):
        out = run_agent(DF, "把大表格拆分成多个小表格，每个表格约100条数据，并按中英文分表，只提取公司名称和手机",
                        use_llm=False)
        warn = [x for x in steps(out, "plan") if x["status"] == "warn"]
        self.assertTrue(warn)
        txt = warn[0]["detail"]
        self.assertIn("中/英文分表", txt)
        self.assertIn("100", txt)
        self.assertIn("只提取部分字段", txt)
        self.assertIn("暂不支持", out["result_summary"]["review"])

    def test_no_unsupported_warning_for_plain_clean(self):
        out = run_agent(DF, "帮我清洗一下", use_llm=False)
        self.assertFalse([x for x in steps(out, "plan") if x["status"] == "warn"])


class IntentRegex(unittest.TestCase):
    def test_number_not_mistaken_as_chunk(self):
        self.assertEqual(parse_intent_rules("删除第86条重复的")["chunk_size"], 0)
        self.assertEqual(parse_intent_rules("我有500条数据，帮我去重")["chunk_size"], 0)

    def test_chunk_detected(self):
        self.assertEqual(parse_intent_rules("每个表格约100条数据")["chunk_size"], 100)
        self.assertEqual(parse_intent_rules("拆成200条一份")["chunk_size"], 200)

    def test_ai_word_boundary(self):
        self.assertFalse(parse_intent_rules("MAIL 列不要动")["want_ai"])
        self.assertTrue(parse_intent_rules("用AI校验一下")["want_ai"])
        self.assertTrue(parse_intent_rules("用大模型复盘")["want_ai"])


if __name__ == "__main__":
    unittest.main()
