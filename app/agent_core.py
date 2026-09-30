# -*- coding: utf-8 -*-
"""TradeS Agent · 清洗智能体核心

目标：一句话指令 → 规划 → 调用清洗工具 → 自检 → 交付
- 有大模型：LLM 理解指令、生成计划、可选复盘
- 无大模型：规则意图解析（覆盖常用清洗指令），仍可自主跑完流水线
"""
from __future__ import annotations

import re
from typing import Any, Callable

import pandas as pd

from clean_engine import (
    DEFAULT_CLEANING,
    MOBILE_SLOTS,
    auto_map_columns,
    load_cleaning_config,
    run_maibang_clean,
)
import llm_client

# 兼容旧字段名
DEFAULT_OPTIONS = {
    "strip_86": True,
    "mobile_only": True,
    "dedupe": True,
    "keep_one_keyword": True,
    "drop_empty_mobile": True,
}

# ---------------------------------------------------------------------------
# 工具注册表（Agent 可调用的能力）
# ---------------------------------------------------------------------------

TOOL_DEFS = [
    {"name": "read_profile", "desc": "读取表格概况：行列数、列名、样例"},
    {"name": "map_fields", "desc": "识别并映射字段"},
    {"name": "clean_data", "desc": "执行清洗规则（电话/去重/关键词等）"},
    {"name": "quality_check", "desc": "生成数据质量简报"},
    {"name": "export_ready", "desc": "标记可导出多分表结果"},
    {"name": "self_review", "desc": "对清洗结果做自检汇总"},
]


class AgentRun:
    """一次 Agent 运行的轨迹。"""

    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []
        self.result: dict[str, Any] | None = None

    def step(self, name: str, title: str, detail: str = "", status: str = "done", payload: Any = None) -> None:
        self.steps.append({
            "tool": name,
            "title": title,
            "detail": detail,
            "status": status,
            "payload": payload,
        })

    def to_dict(self) -> dict[str, Any]:
        return {"steps": self.steps, "result_summary": self.result or {}}


# ---------------------------------------------------------------------------
# 意图解析：规则版（无 LLM 也能跑）
# ---------------------------------------------------------------------------

def parse_intent_rules(instruction: str) -> dict[str, Any]:
    """从中文指令提取清洗意图。"""
    text = (instruction or "").strip()
    low = text.lower()
    opt = dict(DEFAULT_OPTIONS)
    extract_core = False
    want_split_name = False
    want_split_phone = False
    want_quality = True
    want_ai = False
    chunk_size = 0

    # 字段提取
    if re.search(r"(提取|抽出|只保留?|只要).{0,8}(公司|联系人|电话|手机|链接)", text):
        extract_core = True
    if "提取" in text and re.search(r"公司名称|联系人|手机|链接", text):
        extract_core = True

    # 电话
    if re.search(r"\+86|(?<!\d)0086(?!\d)|(?<!\d)86(?=前缀|开头)|去掉.?前缀", text):
        opt["strip_86"] = True
    if re.search(r"只保留?.{0,6}手机|去掉座机|去除座机|只要手机", text):
        opt["mobile_only"] = True
        opt["extra_phones_to_landline"] = True
    if re.search(r"固定电话|座机列|多余.{0,6}电话", text):
        opt["extra_phones_to_landline"] = True
    if re.search(r"电话.{0,8}处理|处理.{0,6}电话", text):
        opt["strip_86"] = True
        opt["mobile_only"] = True
        opt["extra_phones_to_landline"] = True

    # 去重
    if re.search(r"去重|重复.{0,4}删|删除重复|dedup", text):
        opt["dedupe"] = True

    # 中英分表
    if re.search(r"中文|英文|中英", text) and re.search(r"分|拆|两个表|两个表格|表格", text):
        want_split_name = True
        opt["split_by_name"] = True
    if re.search(r"按.{0,6}公司名称.{0,6}分|分为三份|三份", text):
        want_split_name = True
        opt["split_by_name"] = True

    # 手机号数量分表
    if re.search(r"手机号数量|单个手机|多个手机|一个手机|多个号码", text):
        want_split_phone = True
        opt["split_by_phone_count"] = True

    # 按条数拆
    m = re.search(r"(?:每|各|约|拆成?|分成?)[^\d，,。]{0,8}?(\d+)\s*条", text)
    if m:
        chunk_size = int(m.group(1))
        opt["chunk_size"] = chunk_size

    # 关键词
    if re.search(r"关键词|关键字", text):
        opt["keep_one_keyword"] = True

    # 质量 / 验证
    if re.search(r"质量|分析一下|数据质量|怎么样", text):
        want_quality = True
    if re.search(r"大模型|(?<![A-Za-z])AI(?![A-Za-z])|智能校验|用模型", text, re.I):
        want_ai = True

    # 全流程组合指令
    if re.search(r"帮我处理|清洗一下|整理一下|按照.{0,8}要求", text):
        opt["dedupe"] = True
        opt["strip_86"] = True
        opt["keep_one_keyword"] = True
        want_quality = True

    opt["extract_core"] = extract_core
    opt["split_by_name"] = want_split_name or bool(opt.get("split_by_name"))
    opt["split_by_phone_count"] = want_split_phone or bool(opt.get("split_by_phone_count"))
    if chunk_size:
        opt["chunk_size"] = chunk_size

    return {
        "instruction": text,
        "options": opt,
        "extract_core": extract_core,
        "split_name": want_split_name,
        "split_phone": want_split_phone,
        "chunk_size": chunk_size,
        "want_quality": want_quality,
        "want_ai": want_ai,
        "notes": "规则意图解析",
    }


PLAN_SYSTEM = """你是数据清洗 Agent 的规划器。根据用户一句话指令，输出 JSON 执行计划。
只输出 JSON：
{
  "understand": "一句话说明你理解的任务",
  "plan": [
    {"tool": "read_profile", "reason": "…"},
    {"tool": "map_fields", "reason": "…"},
    {"tool": "clean_data", "options": {"strip_86": true, "mobile_only": true, "extra_phones_to_landline": true, "dedupe": true, "keep_one_keyword": true, "split_by_name": true, "split_by_phone_count": false, "chunk_size": 0, "extract_core": false}},
    {"tool": "quality_check"},
    {"tool": "export_ready"},
    {"tool": "self_review"}
  ],
  "options_override": {"extract_core": false}
}
可选 tool：read_profile, map_fields, clean_data, quality_check, export_ready, self_review。
options 字段可缺省。根据指令决定 extract_core / split_by_name / split_by_phone_count / chunk_size 等。"""


def parse_instruction(instruction: str, columns: list[str] | None = None, use_llm: bool = True) -> dict[str, Any]:
    """优先 LLM 规划，失败则规则解析。"""
    rule = parse_intent_rules(instruction)
    if not use_llm:
        rule["source"] = "rules"
        return rule

    try:
        cfg = llm_client.load_config()
        has_key = bool(cfg.get("api_key")) or "127.0.0.1" in str(cfg.get("base_url", "")) or "localhost" in str(cfg.get("base_url", ""))
        if not has_key:
            rule["source"] = "rules"
            return rule
        user = f"用户指令：{instruction}\n源表列名：{columns or []}"
        raw = llm_client.chat(
            [
                {"role": "system", "content": PLAN_SYSTEM},
                {"role": "user", "content": user},
            ],
            cfg,
        )
        parsed = llm_client.parse_json_reply(raw)
        plan = parsed.get("plan") or []
        options = dict(DEFAULT_OPTIONS)
        options.update(rule["options"])
        options.update(parsed.get("options_override") or {})
        # 从 plan 里合并 clean_data.options
        for item in plan:
            if isinstance(item, dict) and item.get("tool") == "clean_data":
                options.update(item.get("options") or {})
        extract_core = bool(options.get("extract_core") or rule["extract_core"])
        return {
            "instruction": instruction,
            "understand": parsed.get("understand") or rule.get("instruction"),
            "plan": plan or [
                {"tool": "read_profile"},
                {"tool": "map_fields"},
                {"tool": "clean_data"},
                {"tool": "quality_check"},
                {"tool": "export_ready"},
                {"tool": "self_review"},
            ],
            "options": options,
            "extract_core": extract_core,
            "split_name": bool(options.get("split_by_name")),
            "split_phone": bool(options.get("split_by_phone_count")),
            "chunk_size": int(options.get("chunk_size") or 0),
            "want_quality": any(i.get("tool") == "quality_check" for i in plan) if plan else True,
            "want_ai": rule["want_ai"] or "校验" in instruction or "验证" in instruction,
            "source": "llm",
            "notes": "大模型规划",
        }
    except Exception as e:
        rule["source"] = "rules"
        rule["notes"] = f"规则解析（LLM 不可用：{e}）"
        return rule


# ---------------------------------------------------------------------------
# 执行：按计划调用工具
# ---------------------------------------------------------------------------

def _slim_rows(df: pd.DataFrame, n: int = 5) -> list[dict]:
    show = [c for c in df.columns if not str(c).startswith("__")][:8]
    return df[show].head(n).astype(str).fillna("").to_dict(orient="records")


# ---------------------------------------------------------------------------
# 本地问答（无大模型时的兜底）
# ---------------------------------------------------------------------------

QA_FAQ = [
    (
        r"^(你好|您好|hi|hello|嗨|在吗|早上好|下午好|晚上好)[!！。~\s]*$",
        "您好！我是 TradeS Agent 问答助手。\n"
        "可以问我：清洗规则怎么执行、如何导出某公司模板、字段怎么映射，或直接说你的表格要怎么洗。",
    ),
    (
        r"你是谁|你是什么|介绍.{0,4}自己|who are you|你叫什么",
        "我是 TradeS Agent 的智能问答助手，专注外贸客户资料清洗：\n"
        "某公司模板导入、手机号处理、去重分表、质量分析。\n"
        "配置大模型后可开放闲聊与更复杂的业务问答。",
    ),
    (
        r"吃了吗|吃饭|天气|在干嘛|无聊",
        "我这边主要负责数据清洗问答～\n"
        "您可以问我手机号规则、去重分表、导出模板，或在「清洗」页给 Agent 下指令。",
    ),
    (
        r"怎么|如何|清洗流程|使用",
        "使用步骤：\n1) 双击 app\\启动.bat 打开本地网页\n2) 上传 Excel/CSV（或粘贴表格）\n"
        "3) 用「清洗 Agent」下一句话指令，或改「手动模式」勾选规则\n4) 看质检与轨迹后导出公司模板 Excel。",
    ),
    (
        r"86|手机|电话|座机|备用",
        "某公司电话规则：仅保留 1 开头 11 位；去掉号码前缀、座机、国际号；"
        "多号按常用电话 / 备用电话1 / 备用电话2 填写；无手机号记录删除。",
    ),
    (
        r"去重|重复",
        "按公司名称去重，保留第一条有效记录；重复公司会写入删除统计。",
    ),
    (
        r"中英|英文|中文|分表|拆分",
        "分表：可按中文/英文/其他公司名分三份，或按条数拆、按单/多手机号拆。\n"
        "指令示例：把大表格拆分成多个小表格，每个表格约100条数据。",
    ),
    (
        r"关键词",
        "多个关键词只保留第一个。指令示例：把关键词列筛选一下，多个关键词只保留一个。",
    ),
    (
        r"质量|分析|体检",
        "质量简报会看：缺公司名、缺联系人、无可识别电话、重复、多关键词、缺官网，并给质量分。\n"
        "指令示例：分析一下这个表格数据质量怎么样。",
    ),
    (
        r"大模型|模型|API|接入|配置",
        "右上角「设置」：选预设或填 Base URL + 模型名 + API Key，点测试连接后保存。"
        "不配模型也能完成规则清洗；配好后可用于指令理解、问答与 AI 校验。",
    ),
    (
        r"导出|模板|CRM",
        "导出为公司模板 Excel（多 sheet）。电话按文本保存，避免被 Excel 变成科学计数。可选「全部 / 仅可拨打 / 仅待修正」。",
    ),
]


def local_qa_answer(question: str) -> str:
    text = (question or "").strip()
    if not text:
        return "请输入你的问题。"
    for pattern, ans in QA_FAQ:
        if re.search(pattern, text, re.I):
            return ans
    return (
        "已收到：「" + text[:40] + ("…" if len(text) > 40 else "") + "」\n\n"
        "当前未走大模型时，我优先回答清洗相关问题：手机号规则、去重、分表、模板导出、操作步骤。\n"
        "若要开放问答，请在「设置」里配置大模型并测试连接；或换一个和清洗相关的问法。"
    )


def unsupported_requests(plan_info: dict[str, Any]) -> list[str]:
    """指令里提到、但当前清洗流程尚未实现的能力（如实告知，不悄悄忽略）。"""
    opt = plan_info.get("options") or {}
    out: list[str] = []
    if plan_info.get("extract_core") or opt.get("extract_core"):
        out.append("只提取部分字段（输出固定为某公司 18 列模板）")
    if plan_info.get("split_name") or opt.get("split_by_name"):
        out.append("按公司名称中/英文分表")
    if plan_info.get("split_phone") or opt.get("split_by_phone_count"):
        out.append("按手机号数量分表")
    chunk = int(plan_info.get("chunk_size") or opt.get("chunk_size") or 0)
    if chunk:
        out.append(f"按每表 {chunk} 条拆分（请在导出时设置「每表条数」）")
    return out


def verify_cleaned(rows: pd.DataFrame, cfg: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    """对清洗结果做真实检查，返回 {ok, problems, checks}。"""
    problems: list[str] = []
    checks: list[str] = []
    if not mapping.get("company_name"):
        problems.append("未识别到公司名称列")
    if not mapping.get("mobile"):
        problems.append("未识别到手机/电话列")
    n = len(rows)
    if n == 0:
        problems.append("清洗后没有任何有效记录")
        return {"ok": False, "problems": problems, "checks": checks}

    pat = re.compile(cfg.get("mobile_pattern") or r"^1\d{10}$")
    slots = [c for c in (cfg.get("mobile_slots") or MOBILE_SLOTS) if c in rows.columns]
    bad = 0
    for c in slots:
        bad += sum(1 for v in rows[c].astype(str).str.strip() if v and not pat.fullmatch(v))
    if bad:
        problems.append(f"{bad} 个电话不符合手机号规则")
    else:
        checks.append("电话均符合手机号规则")

    if "常用电话" in rows.columns and int((rows["常用电话"].astype(str).str.strip() == "").sum()):
        problems.append("存在常用电话为空的记录")
    names = rows["公司名称"].astype(str).str.strip()
    if int((names == "").sum()):
        problems.append("存在公司名称为空的记录")
    dup = int(names.duplicated().sum())
    if dup:
        problems.append(f"公司名称仍有 {dup} 条重复")
    else:
        checks.append("公司名称无重复")
    return {"ok": not problems, "problems": problems, "checks": checks}


def run_agent(
    df: pd.DataFrame,
    instruction: str,
    source: str = "其他",
    mapping: dict[str, str] | None = None,
    use_llm: bool = True,
    max_rows: int = 0,
    cleaning: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """执行清洗 Agent：规划 → 工具 → 自检。"""
    run = AgentRun()
    cols = list(df.columns)
    # 与手动模式一致：config.json 的 cleaning 段 + 请求里的覆盖项
    cfg = {**load_cleaning_config(), **(cleaning or {})}

    # 1) 理解与规划
    plan_info = parse_instruction(instruction, cols, use_llm=use_llm)
    run.step(
        "plan",
        "理解指令并生成计划",
        (plan_info.get("understand") or instruction) + f"（{plan_info.get('notes', '')}）",
        payload={"plan": plan_info.get("plan"), "source": plan_info.get("source")},
    )

    # 2) read_profile
    run.step(
        "read_profile",
        "读取表格概况",
        f"共 {len(df)} 行 × {len(cols)} 列；列：{', '.join(cols[:12])}",
        payload={"rows": int(len(df)), "columns": cols, "sample": _slim_rows(df, 3)},
    )

    # 3) map_fields
    mp = mapping or auto_map_columns(cols)
    mapped_desc = "；".join(f"{k}←{v}" for k, v in list(mp.items())[:8])
    run.step("map_fields", "自动识别字段映射", mapped_desc or "未识别到映射", payload={"mapping": mp})

    if not mp.get("company_name") and not mp.get("mobile"):
        run.step("map_fields", "字段映射不足", "缺少公司名称或手机映射，尝试启发式第一列/含电话列", status="warn")
        if "company_name" not in mp and cols:
            mp["company_name"] = cols[0]
        if "mobile" not in mp:
            for c in cols:
                if re.search(r"电话|手机|联系|mobile|phone", str(c), re.I):
                    mp["mobile"] = c
                    break

    # 4) clean_data（某公司规则）
    if max_rows and len(df) > max_rows:
        df = df.head(max_rows).copy()
        run.step("clean_data", "限制处理行数", f"仅处理前 {max_rows} 行", status="warn")

    try:
        cleaned = run_maibang_clean(df, mp, source=source, cleaning=cfg)
        cleaned["sheets"] = {"清洗结果": cleaned["rows"]}
        cleaned["report"] = {
            "score": cleaned["stats"].get("valid_rate"),
            "issues": [
                {"label": "删除·无手机", "count": cleaned["stats"].get("dropped_no_mobile", 0)},
                {"label": "删除·公司重复", "count": cleaned["stats"].get("duplicated_company", 0)},
                {"label": "删除·仅座机", "count": cleaned["stats"].get("dropped_landline_only", 0)},
                {"label": "删除·仅国际号", "count": cleaned["stats"].get("dropped_intl_only", 0)},
                {"label": "删除·无效号码", "count": cleaned["stats"].get("dropped_invalid", 0)},
                {"label": "删除·缺公司名", "count": cleaned["stats"].get("dropped_no_company", 0)},
            ],
        }
    except Exception as e:
        run.step("clean_data", "清洗执行失败", str(e), status="fail")
        raise

    stats = cleaned["stats"]
    run.step(
        "clean_data",
        "执行某公司清洗规则",
        f"原始 {stats.get('raw_count')} → 清洗后 {stats.get('cleaned_count')}（有效率 {stats.get('valid_rate')}%）；"
        f"无手机删除 {stats.get('dropped_no_mobile')}，公司重复 {stats.get('duplicated_company')}，"
        f"仅座机 {stats.get('dropped_landline_only')}，仅国际号 {stats.get('dropped_intl_only')}，"
        f"无效号码 {stats.get('dropped_invalid')}，缺公司名 {stats.get('dropped_no_company')}；有备用电话 {stats.get('has_backup_phone')} 家",
        payload={"stats": stats},
    )

    # 5) quality_check：基于真实结果检查，不再写死通过
    report = cleaned.get("report") or {}
    verdict = verify_cleaned(cleaned["rows"], cfg, mp)
    unsupported = unsupported_requests(plan_info)
    if unsupported:
        run.step("plan", "以下要求当前暂不支持，本次已忽略", "；".join(unsupported), status="warn",
                 payload={"unsupported": unsupported})
    run.step(
        "quality_check",
        "验收自检" if verdict["ok"] else "验收自检：发现问题",
        (f"有效率 {stats.get('valid_rate')}%；" + "；".join(verdict["checks"]))
        if verdict["ok"] else "；".join(verdict["problems"]),
        status="done" if verdict["ok"] else "warn",
        payload={"score": report.get("score"), "problems": verdict["problems"]},
    )

    # 6) export_ready
    sheet_names = ["清洗结果"]
    run.step("export_ready", "准备某公司模板导出", "列顺序按武汉某公司系统；大结果可按约 1500 条分表", payload={"sheets": sheet_names})

    main_df = cleaned.get("rows")
    preview_cols = list(getattr(main_df, "columns", []))
    preview = (
        main_df.head(20).astype(str).fillna("").to_dict(orient="records")
        if preview_cols and len(main_df)
        else []
    )

    # 7) self_review
    review = (
        f"自检完成：原始 {stats.get('raw_count')} 条 → 清洗后 {stats.get('cleaned_count')} 条；"
        f"有效率 {stats.get('valid_rate')}%；已去重 {stats.get('duplicated_company')} 条重复公司。"
        + ("可导出。" if verdict["ok"] else "存在问题，请先核对后再导出：" + "；".join(verdict["problems"]))
    )
    if unsupported:
        review += "（另有 " + str(len(unsupported)) + " 项要求暂不支持，已忽略）"
    run.step("self_review", "自检汇总", review)

    result = {
        "stats": stats,
        "report": report,
        "mapping": mp,
        "sheets": sheet_names,
        "preview": preview,
        "preview_columns": preview_cols,
    }
    run.result = {
        "raw": stats.get("raw_count"),
        "cleaned": stats.get("cleaned_count"),
        "rate": stats.get("valid_rate"),
        "review": review,
        "ok": verdict["ok"],
        "unsupported": unsupported,
    }

    # 可选 AI 复盘（不阻塞主流程）
    if plan_info.get("want_ai"):
        try:
            rows_records = cleaned["rows"].head(10).astype(str).fillna("").to_dict(orient="records")
            ai = llm_client.validate_rows(rows_records)
            run.step(
                "self_review",
                "大模型复盘",
                (ai.get("summary") or "完成")[:200],
                payload={"checked": ai.get("checked")},
            )
        except Exception as e:
            run.step("self_review", "大模型复盘跳过", str(e)[:160], status="warn")

    out = run.to_dict()
    out["intent"] = {
        "understand": plan_info.get("understand") or plan_info.get("instruction"),
        "source": plan_info.get("source"),
    }
    out["stats"] = stats
    out["report"] = report
    out["sheets"] = sheet_names
    out["preview"] = result["preview"]
    out["preview_columns"] = result["preview_columns"]
    out["mapping"] = mp
    out["cleaned"] = cleaned
    return out
