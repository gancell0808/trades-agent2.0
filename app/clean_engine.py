# -*- coding: utf-8 -*-
"""TradeS Agent · 清洗引擎（对齐武汉某公司导入模板 + 2026-09 工作记录规则）

硬性规则：
1. 手机号：1 开头 11 位
2. 去 +86 / 86 / 0086 前缀
3. 去掉座机
4. 去掉国际号（+853/+886/+852 等）
5. 多号码：常用电话 / 备用电话1 / 备用电话2
6. 按公司名称去重，保留第一条有效记录
7. 删除无手机号记录
"""
from __future__ import annotations

import io
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# 某公司模板列
# ---------------------------------------------------------------------------

MAIBANG_COLUMNS = [
    "公司名称", "部门", "公司网址", "网站网址", "姓名", "岗位",
    "常用电话", "备用电话1", "备用电话2", "性别", "生日",
    "微信", "钉钉", "邮箱", "QQ", "旺旺", "个人传真", "淘到客户时间",
]

MOBILE_SLOTS = ["常用电话", "备用电话1", "备用电话2"]

FIELD_ALIASES: dict[str, list[str]] = {
    "company_name": ["公司名称", "公司", "企业名称", "公司名", "店铺名", "客户名称", "单位名称", "企业", "company"],
    "department": ["部门", "department"],
    "website": ["公司网址", "网站网址", "网站", "网址", "官网", "链接", "website", "url", "site"],
    "contact_name": ["姓名", "联系人", "联系人姓名", "客户姓名", "name", "contact"],
    "title": ["岗位", "职位", "职务", "title"],
    "mobile": ["常用电话", "手机", "手机号", "手机号码", "移动电话", "联系电话", "电话", "联系方式", "mobile", "phone", "tel"],
    "mobile2": ["备用电话1", "备用电话", "备用", "电话2"],
    "mobile3": ["备用电话2", "电话3"],
    "email": ["邮箱", "电子邮件", "email"],
    "wechat": ["微信", "wechat"],
    "qq": ["qq"],
    "region": ["地区", "省", "市", "地址", "所属地区", "region", "city"],
    "industry": ["行业", "主营产品", "主营", "产品", "industry", "product"],
    "source_channel": ["来源", "渠道", "线索来源", "来源渠道", "source"],
}

DEFAULT_CLEANING: dict[str, Any] = {
    "mobile_pattern": r"^1\d{10}$",
    "strip_prefixes": ["+86", "86", "0086"],
    "drop_intl_prefixes": ["+853", "+886", "+852", "+81", "+82", "+852"],
    "drop_landline": True,
    "split_separators": ["、", "；", ";", "/", ",", " ", "　", "|", "｜", "\n", "\t"],
    "dedupe_key": "公司名称",
    "dedupe_normalize": True,  # 去重前规范化公司名（全/半角、大小写、空白、标点）
    "drop_empty_mobile": True,
    "mobile_slots": MOBILE_SLOTS,
    "template_columns": MAIBANG_COLUMNS,
    "split_rows": 1500,
}


def load_cleaning_config(path: Path | None = None) -> dict[str, Any]:
    """读取 config.json 中的 cleaning 段，缺省用 DEFAULT_CLEANING。"""
    cfg = dict(DEFAULT_CLEANING)
    p = path or Path(__file__).resolve().parent / "config.json"
    if p.is_file():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            cleaning = (data or {}).get("cleaning") or {}
            if isinstance(cleaning, dict):
                for k, v in cleaning.items():
                    cfg[k] = v
        except Exception:
            pass
    return cfg


# ---------------------------------------------------------------------------
# 读表 / 映射
# ---------------------------------------------------------------------------

_CSV_ENCODINGS = ("utf-8-sig", "gb18030")
HEADER_SCAN_ROWS = 10


def _decode_text(data: bytes) -> str:
    for enc in _CSV_ENCODINGS:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("无法识别文件编码（支持 UTF-8 / GBK）")


def _csv_rows(text: str, filename: str) -> list[list[str]]:
    import csv

    if filename.endswith((".tsv", ".tab")):
        delim = "\t"
    else:
        try:
            delim = csv.Sniffer().sniff(text[:4096], delimiters=",\t;|").delimiter
        except csv.Error:
            delim = ","
    return [list(r) for r in csv.reader(io.StringIO(text, newline=""), delimiter=delim)]


def _is_blank(row: list[str]) -> bool:
    return not any(str(c).strip() for c in row)


def _header_is_usable(row: list[str]) -> bool:
    """这一行像不像表头：能同时识别出公司名称和手机/电话列。"""
    m = auto_map_columns([str(c).strip() for c in row if str(c).strip()])
    return "company_name" in m and "mobile" in m


def _rows_to_df(rows: list[list[str]], label: str, warnings: list[str]) -> pd.DataFrame:
    """原始行 → DataFrame。自动找表头行；行长度与表头不一致时如实告警而不是静默丢弃。"""
    rows = [r for r in rows if not _is_blank(r)]
    if not rows:
        return pd.DataFrame()

    head_idx = 0
    if not _header_is_usable(rows[0]):
        for i in range(1, min(HEADER_SCAN_ROWS, len(rows))):
            if _header_is_usable(rows[i]):
                head_idx = i
                warnings.append(f"{label}：第 {i + 1} 行被识别为表头，其上方 {i} 行（标题/说明）已忽略")
                break

    header = [str(c).strip() for c in rows[head_idx]]
    width = len(header)
    while width and not header[width - 1]:
        width -= 1                      # 去掉表头右侧的空列
    header = header[:width]
    if not header:
        return pd.DataFrame()
    names, seen = [], {}
    for i, h in enumerate(header):
        h = h or f"Unnamed: {i}"
        n = seen.get(h, 0)
        seen[h] = n + 1
        names.append(h if n == 0 else f"{h}.{n}")

    body, too_long = [], []
    for r in rows[head_idx + 1:]:
        r = [str(c) if c is not None else "" for c in r]
        if len(r) > width and any(c.strip() for c in r[width:]):
            too_long.append(r)
            continue
        body.append((r + [""] * width)[:width])
    if too_long:
        sample = "；".join(",".join(x)[:40] for x in too_long[:3])
        warnings.append(
            f"{label}：有 {len(too_long)} 行的列数比表头多（常见原因：字段内含未加引号的逗号），"
            f"这些行未导入，请检查原文件。示例：{sample}"
        )

    df = pd.DataFrame(body, columns=names, dtype=str)
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    return df


def read_table_ex(data: bytes, filename: str) -> tuple[pd.DataFrame, list[str]]:
    """读取 CSV / Excel。返回 (表, 提示列表)；提示用于告诉用户读取时发生了什么。

    - Excel：读取全部 sheet 并按列名合并（不再只读第一个）
    - CSV：自动识别 UTF-8 / GBK 与分隔符，支持引号内换行
    - 自动跳过表头上方的标题/说明行；列数异常的行会被报告，而不是静默丢弃
    """
    name = (filename or "").lower()
    warnings: list[str] = []
    is_excel = (
        name.endswith((".xlsx", ".xls", ".xlsm"))
        or data[:4] == b"PK\x03\x04"
        or data[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
    )

    if is_excel:
        try:
            sheets = pd.read_excel(io.BytesIO(data), dtype=str, header=None, sheet_name=None)
        except Exception as e:
            raise ValueError(f"无法解析文件：{e}") from e
        frames, used, skipped = [], [], []
        first_cols = None
        for sheet_name, raw in sheets.items():
            rows = raw.fillna("").astype(str).values.tolist()
            df = _rows_to_df(rows, f"{filename} · {sheet_name}", warnings)
            if df.empty:
                skipped.append(str(sheet_name))
                continue
            if first_cols is None:
                first_cols = set(df.columns)
            elif set(df.columns) != first_cols:
                warnings.append(f"{filename} · {sheet_name}：列名与第一个有数据的 sheet 不完全一致，已按列名合并")
            frames.append(df)
            used.append(str(sheet_name))
        if len(used) > 1:
            warnings.append(f"{filename}：包含 {len(used)} 个有数据的 sheet（{'、'.join(used)}），已合并读取")
        if skipped and frames:
            warnings.append(f"{filename}：空 sheet 已忽略（{'、'.join(skipped)}）")
        df = merge_tables(frames) if frames else pd.DataFrame()
    else:
        df = _rows_to_df(_csv_rows(_decode_text(data), name), filename, warnings)

    return df, warnings


def read_table(data: bytes, filename: str) -> pd.DataFrame:
    """兼容旧接口：只返回表，不返回提示。"""
    return read_table_ex(data, filename)[0]


def merge_tables(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def suggest_source(filename: str, columns: list[str]) -> str:
    text = f"{filename} {' '.join(columns)}".lower()
    if "广交会" in text or "canton" in text:
        return "广交会"
    if any(k in text for k in ("中国制造", "made-in-china", "mic")):
        return "中国制造"
    if any(k in text for k in ("阿里", "alibaba", "国际站")):
        return "阿里巴巴"
    return "自定义"


def _norm_header(s: str) -> str:
    return re.sub(r"[\s　*（）()【】\[\]:：]+", "", str(s)).lower()


def auto_map_columns(columns: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    used: set[str] = set()
    norm_cols = {c: _norm_header(c) for c in columns}

    for field, aliases in FIELD_ALIASES.items():
        best, best_score = None, 0
        for col, ncol in norm_cols.items():
            if col in used or not ncol:
                continue
            score = 0
            for a in aliases:
                an = _norm_header(a)
                if ncol == an:
                    score = 100
                elif an and (an in ncol or ncol in an):
                    score = max(score, 70)
            if score > best_score:
                best, best_score = col, score
        if best and best_score >= 70:
            mapping[field] = best
            used.add(best)

    if "mobile" not in mapping:
        for col in columns:
            n = _norm_header(col)
            if any(k in n for k in ("手机", "电话", "联系方式", "mobile", "phone", "常用")):
                mapping["mobile"] = col
                break
    return mapping


# ---------------------------------------------------------------------------
# 电话清洗（工作记录硬规则）
# ---------------------------------------------------------------------------

def _clean_text(v: Any) -> str:
    s = str(v if v is not None else "").strip()
    if s.lower() in ("nan", "none", "null", "-", "—"):
        return ""
    return re.sub(r"\s+", " ", s)


def _cfg_bits(cfg: dict[str, Any]) -> tuple[re.Pattern, tuple[str, ...], re.Pattern]:
    """从清洗配置得到：手机号正则、可去除的国家码、强分隔符正则。"""
    pattern = re.compile(cfg.get("mobile_pattern") or r"^1\d{10}$")
    codes = {re.sub(r"\D", "", p) for p in (cfg.get("strip_prefixes") or ["+86", "86", "0086"])} - {""}
    codes = {c.lstrip("0") or c for c in codes} or {"86"}
    strip_cc = tuple(sorted(codes, key=len, reverse=True))
    # 强分隔符（不含空格：号码内部可能带空格，如 138 1234 5678）
    strong = [x for x in (cfg.get("split_separators") or DEFAULT_CLEANING["split_separators"])
              if x and not x.isspace()]
    strong += ["，", "\n", "\r", "\t"]
    splitter = re.compile("[" + "".join(re.escape(x) for x in dict.fromkeys(strong)) + "]+")
    return pattern, strip_cc, splitter


def _segment_digits(seg: str, strip_cc: tuple[str, ...]) -> tuple[str, bool]:
    """单个候选段 → (纯数字, 是否国际号)。前缀只在段首处理。"""
    seg = seg.strip()
    seg = re.sub(r"\.0+$", "", seg)              # Excel 数值遗留的 .0
    seg = re.sub(r"[（(]\s*0\s*[)）]", "", seg)   # +86 (0)138... 中的 (0)
    if not seg:
        return "", False
    m = re.match(r"^(\+|00)\s*(\d{1,3})", seg)
    if m:
        cc = m.group(2)
        # 国家码：86 保留后续；其它（852/853/886/81/82…）一律视为国际号
        hit = next((c for c in strip_cc if cc.startswith(c)), None)
        if hit is None:
            return re.sub(r"\D", "", seg), True
        seg = seg[m.end(1) + len(hit):]
    digits = re.sub(r"\D", "", seg)
    if len(digits) == 13 and digits.startswith("86"):
        digits = digits[2:]
    return digits, False


def _segment_kind(seg: str, pattern: re.Pattern, strip_cc: tuple[str, ...]) -> str:
    """'' 空 | mobile | intl | landline | invalid"""
    digits, intl = _segment_digits(seg, strip_cc)
    if not digits:
        return ""
    if intl:
        return "intl"
    if pattern.fullmatch(digits):
        return "mobile"
    if digits.startswith("0") or len(digits) in (7, 8) or digits.startswith(("400", "800")):
        return "landline"
    return "invalid"


def _segment_mobile(seg: str, pattern: re.Pattern, strip_cc: tuple[str, ...]) -> str | None:
    """单个候选段 → 合规手机号（1 开头 11 位）或 None。"""
    digits, intl = _segment_digits(seg, strip_cc)
    return digits if digits and not intl and pattern.fullmatch(digits) else None


def extract_valid_mobiles(raw: Any, cleaning: dict[str, Any] | None = None) -> list[str]:
    """从字段提取合规手机号（1 开头 11 位），保序去重，最多 5 个。

    先按分隔符切段，再逐段判断：仅在段首去 +86/0086/86，带其它国家码
    （+852/+853/+886…）的整段丢弃，座机（含区号）不匹配 11 位手机规则而丢弃。
    绝不把不同号码的数字拼接在一起。
    """
    cfg = cleaning or DEFAULT_CLEANING
    s = _clean_text(raw)
    if not s:
        return []
    pattern, strip_cc, splitter = _cfg_bits(cfg)

    found: list[str] = []
    for chunk in splitter.split(s):
        chunk = chunk.strip()
        if not chunk:
            continue
        hit = _segment_mobile(chunk, pattern, strip_cc)
        if hit:
            found.append(hit)
            continue
        # 整段不是一个号：可能是空格分隔的多个号码（如 "13812345678 13912345678"）
        for tok in re.split(r"\s+", chunk):
            hit = _segment_mobile(tok, pattern, strip_cc) if tok else None
            if hit:
                found.append(hit)

    return list(dict.fromkeys(found))[:5]


def classify_unusable(raw: Any, cleaning: dict[str, Any] | None = None) -> str:
    """一行没有提取到手机号时，判断原因：empty | landline | intl | invalid。

    多类混合时按 座机 > 国际号 > 无效号码 的优先级归类，每行只归一类。
    """
    cfg = cleaning or DEFAULT_CLEANING
    s = _clean_text(raw)
    if not re.sub(r"\D", "", s):
        return "empty"
    pattern, strip_cc, splitter = _cfg_bits(cfg)
    kinds: set[str] = set()
    for chunk in splitter.split(s):
        chunk = chunk.strip()
        if not chunk:
            continue
        k = _segment_kind(chunk, pattern, strip_cc)
        if k == "invalid" and re.search(r"\s", chunk):
            kinds |= {_segment_kind(t, pattern, strip_cc) for t in re.split(r"\s+", chunk) if t}
        else:
            kinds.add(k)
    kinds.discard("")
    for k in ("landline", "intl", "invalid"):
        if k in kinds:
            return k
    return "empty"


def normalize_company_key(name: str) -> str:
    """去重用的公司名规范化：NFKC（全角→半角）、忽略大小写、去掉空白和标点。

    刻意不去掉「有限公司 / Co., Ltd.」等后缀：后缀不同可能是不同的法人主体，宁可漏去重也不误删。
    """
    s = unicodedata.normalize("NFKC", str(name or "")).casefold()
    return "".join(ch for ch in s if unicodedata.category(ch)[0] not in ("Z", "P", "C"))


def _clean_website(v: Any) -> str:
    s = _clean_text(v)
    if not s:
        return ""
    s = re.split(r"[\s，,;；|｜]", s)[0]
    s = re.sub(r"^https?://", "", s, flags=re.I)
    s = s.split("?")[0].split("#")[0].strip("/")
    return s if s and "." in s else ""


# ---------------------------------------------------------------------------
# 主清洗流程
# ---------------------------------------------------------------------------

def run_maibang_clean(
    df: pd.DataFrame,
    mapping: dict[str, str],
    source: str = "自定义",
    cleaning: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """按某公司规则清洗。返回 rows / stats / dropped_samples。"""
    cfg = cleaning or DEFAULT_CLEANING
    slots = list(cfg.get("mobile_slots") or MOBILE_SLOTS)
    n = len(df)

    def col(field: str) -> pd.Series | None:
        c = mapping.get(field)
        if c and c in df.columns:
            return df[c]
        return None

    rows: list[dict[str, Any]] = []
    stats = {
        "raw_count": n,
        "cleaned_count": 0,
        "valid_rate": 0.0,
        "dropped_no_mobile": 0,       # 手机列为空 / 无任何数字
        "dropped_landline_only": 0,   # 只有座机
        "dropped_intl_only": 0,       # 只有国际号
        "dropped_invalid": 0,         # 有数字但不是有效手机号（位数不对等）
        "dropped_no_company": 0,      # 有手机号但缺公司名称
        "duplicated_company": 0,
        "has_backup_phone": 0,
        "multi_mobile": 0,
        "source": source,
    }
    dropped_samples: list[dict[str, str]] = []
    seen_company: set[str] = set()

    mobile_col = col("mobile")
    mobile2_col = col("mobile2")
    mobile3_col = col("mobile3")
    company_col = col("company_name")

    for i in range(n):
        company = _clean_text(company_col.iloc[i]) if company_col is not None else ""
        raw_cells = []
        for c in (mobile_col, mobile2_col, mobile3_col):
            if c is not None:
                raw_cells.append(str(c.iloc[i]))
        raw_join = " / ".join(raw_cells)
        mobiles = extract_valid_mobiles(raw_join, cfg)

        # 删除原因：每一行只归入一类（先看手机号，再看公司名）
        if not mobiles:
            kind = classify_unusable(raw_join, cfg)
            stat_key, reason = {
                "empty": ("dropped_no_mobile", "无手机号"),
                "landline": ("dropped_landline_only", "仅座机"),
                "intl": ("dropped_intl_only", "仅国际号"),
                "invalid": ("dropped_invalid", "无效号码"),
            }[kind]
            stats[stat_key] += 1
            if len(dropped_samples) < 12:
                dropped_samples.append({"公司名称": company, "原始电话": _clean_text(raw_join)[:40], "原因": reason})
            continue

        if not company:
            stats["dropped_no_company"] += 1
            if len(dropped_samples) < 12:
                dropped_samples.append({"公司名称": "(空)", "原始电话": mobiles[0], "原因": "缺公司名称"})
            continue

        # 公司名去重
        key = normalize_company_key(company) if cfg.get("dedupe_normalize", True) else company
        if key in seen_company:
            stats["duplicated_company"] += 1
            if len(dropped_samples) < 12:
                dropped_samples.append({"公司名称": company, "原始电话": mobiles[0], "原因": "公司重复"})
            continue
        seen_company.add(key)

        record = {c: "" for c in (cfg.get("template_columns") or MAIBANG_COLUMNS)}
        record["公司名称"] = company
        record["部门"] = _clean_text(col("department").iloc[i]) if col("department") is not None else ""
        web = _clean_website(col("website").iloc[i]) if col("website") is not None else ""
        record["公司网址"] = web
        record["网站网址"] = web
        record["姓名"] = _clean_text(col("contact_name").iloc[i]) if col("contact_name") is not None else ""
        record["岗位"] = _clean_text(col("title").iloc[i]) if col("title") is not None else ""
        record["邮箱"] = _clean_text(col("email").iloc[i]) if col("email") is not None else ""
        record["微信"] = _clean_text(col("wechat").iloc[i]) if col("wechat") is not None else ""
        record["QQ"] = _clean_text(col("qq").iloc[i]) if col("qq") is not None else ""

        for idx, slot in enumerate(slots[:3]):
            record[slot] = mobiles[idx] if idx < len(mobiles) else ""

        if len(mobiles) > 1:
            stats["has_backup_phone"] += 1
        if len(mobiles) >= 2:
            stats["multi_mobile"] += 1

        rows.append(record)

    cleaned = pd.DataFrame(rows, columns=(cfg.get("template_columns") or MAIBANG_COLUMNS))
    stats["cleaned_count"] = int(len(cleaned))
    stats["valid_rate"] = round(stats["cleaned_count"] / n * 100, 1) if n else 0.0
    stats["dropped_samples"] = dropped_samples

    return {"rows": cleaned, "stats": stats, "cleaning": cfg}


def export_maibang_bytes(df: pd.DataFrame) -> bytes:
    cols = [c for c in MAIBANG_COLUMNS if c in df.columns]
    data = df[cols].copy()
    for c in data.columns:
        data[c] = data[c].map(lambda x: "" if x is None else str(x))

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        data.to_excel(writer, index=False, sheet_name="清洗结果")
        ws = writer.book["清洗结果"]
        from openpyxl.utils import get_column_letter
        from openpyxl.styles import Font, PatternFill, Alignment

        widths = {
            "公司名称": 28, "部门": 12, "公司网址": 24, "网站网址": 24, "姓名": 12, "岗位": 12,
            "常用电话": 14, "备用电话1": 14, "备用电话2": 14, "性别": 8, "生日": 12,
            "微信": 14, "钉钉": 12, "邮箱": 20, "QQ": 12, "旺旺": 12, "个人传真": 12, "淘到客户时间": 16,
        }
        for i, col in enumerate(ws.iter_cols(min_row=1, max_row=1), start=1):
            name = col[0].value
            ws.column_dimensions[get_column_letter(i)].width = widths.get(name, 14)
            if name in MOBILE_SLOTS or name in ("QQ", "个人传真", "微信", "钉钉", "旺旺"):
                for r in range(2, ws.max_row + 1):
                    cell = ws.cell(row=r, column=i)
                    cell.number_format = "@"
                    if cell.value is not None:
                        cell.value = str(cell.value)
        fill = PatternFill("solid", fgColor="1F4E79")
        font = Font(color="FFFFFF", bold=True, name="微软雅黑", size=11)
        for cell in ws[1]:
            cell.fill = fill
            cell.font = font
            cell.alignment = Alignment(vertical="center")
        ws.auto_filter.ref = ws.dimensions
        ws.freeze_panes = "A2"
    return buf.getvalue()


def split_and_export(
    df: pd.DataFrame,
    out_dir: Path,
    base_name: str,
    split_rows: int = 1500,
) -> dict[str, Any]:
    """写出主结果 + 分表，返回文件路径列表。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    main_path = out_dir / f"{base_name}.xlsx"
    main_path.write_bytes(export_maibang_bytes(df))
    files.append(str(main_path))

    split_dir = out_dir / "分表"
    if split_rows and len(df) > split_rows:
        split_dir.mkdir(parents=True, exist_ok=True)
        for i in range(0, len(df), split_rows):
            part = df.iloc[i : i + split_rows]
            idx = i // split_rows + 1
            p = split_dir / f"{base_name}_分表_{idx:02d}.xlsx"
            p.write_bytes(export_maibang_bytes(part))
            files.append(str(p))

    return {"files": files, "main": str(main_path), "split_count": max(0, (len(df) - 1) // split_rows) if split_rows else 0}
