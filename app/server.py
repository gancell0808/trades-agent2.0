# -*- coding: utf-8 -*-
"""TradeS Agent · 本地数据清洗服务（武汉某公司模板 / 工作记录规则）"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import llm_client
from web_search import web_search_ex
from agent_core import local_qa_answer, run_agent
import qa_store
from clean_engine import (
    MAIBANG_COLUMNS,
    auto_map_columns,
    export_maibang_bytes,
    load_cleaning_config,
    merge_tables,
    read_table_ex,
    run_maibang_clean,
    split_and_export,
    suggest_source,
)

BASE = Path(__file__).resolve().parent
STATIC = BASE / "static"
CONFIG_PATH = BASE / "config.json"
OUT_DEFAULT = Path(os.path.expanduser("~")) / "Desktop" / "清洗"

QA_SYSTEM = """你是公司内部「数据清洗与线索整理」助手，服务外贸软件公司的业务同事。
你熟悉：广交会/中国制造/阿里巴巴线索清洗、手机号规则、武汉某公司导入模板。
清洗规则：手机号 1 开头 11 位；去掉号码前缀；去座机；去国际号；多号码按常用/备用1/备用2；
按公司名称去重；删除无手机号记录；导出列为某公司模板。
禁止编造清洗条数与路径；不确定写「待核实」；不泄露 Key。
回答中文、简洁、可执行。"""


def load_app_config() -> dict:
    cfg = {
        "server": {"host": "127.0.0.1", "port": 8787, "auto_open_browser": True},
        "cleaning": load_cleaning_config(CONFIG_PATH),
    }
    if CONFIG_PATH.is_file():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                if isinstance(data.get("server"), dict):
                    cfg["server"].update(data["server"])
                if isinstance(data.get("cleaning"), dict):
                    cfg["cleaning"].update(data["cleaning"])
        except Exception:
            pass
    return cfg


APP_CFG = load_app_config()
HOST = str(APP_CFG["server"].get("host") or "127.0.0.1")
PORT = int(os.environ.get("TRADES_PORT") or APP_CFG["server"].get("port") or 8787)

# 每个浏览器标签页一个独立会话（前端在 X-Session-Id 头里带 ID），互不覆盖。
MAX_SESSIONS = 8
_SID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SESSIONS: "OrderedDict[str, dict]" = OrderedDict()
_SESSIONS_LOCK = threading.Lock()


def _new_state() -> dict:
    return {
        "lock": threading.RLock(),   # 同一会话内的请求串行执行
        "dfs": [],
        "filenames": [],
        "source": "自定义",
        "mapping": {},
        "df_merged": None,
        "last_result": None,
        "last_export": None,  # {"files": [], "stats": {}}
    }


def get_state(sid: str | None) -> dict:
    """按会话 ID 取状态；ID 缺失/非法时用 default。超过上限淘汰最久未用的会话。"""
    sid = sid if sid and _SID_RE.match(sid) else "default"
    with _SESSIONS_LOCK:
        st = _SESSIONS.get(sid)
        if st is None:
            st = _SESSIONS[sid] = _new_state()
        _SESSIONS.move_to_end(sid)
        while len(_SESSIONS) > MAX_SESSIONS:
            _SESSIONS.popitem(last=False)
        return st


# 需要读写会话状态的接口（执行期间持有该会话的锁）
_STATEFUL_POST = {"/api/upload", "/api/clean", "/api/clean/run", "/api/export",
                  "/api/agent/run", "/api/llm/validate"}


def _safe_output_dir(raw: str) -> Path:
    """导出目录必须落在用户主目录内（含默认桌面\\清洗），防止任意路径写文件。"""
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        p = Path.home() / p
    p = p.resolve()
    home = Path.home().resolve()
    try:
        p.relative_to(home)
    except ValueError:
        raise ValueError(f"导出目录必须位于用户主目录内：{home}")
    if p == home:
        raise ValueError("导出目录不能直接是用户主目录，请指定子文件夹")
    return p


def _safe_filename(name: str, default: str = "清洗结果") -> str:
    """去掉路径分隔符与控制字符，避免文件名穿越。"""
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "_", str(name or "")).strip(" .")
    return name[:60] or default


class Handler(BaseHTTPRequestHandler):
    server_version = "TradeSClean/0.3"

    _st_cache: dict | None = None

    @property
    def st(self) -> dict:
        """本次请求所属会话的状态（每个请求只解析一次，避免中途被淘汰后重建）。"""
        if self._st_cache is None:
            self._st_cache = get_state(self.headers.get("X-Session-Id"))
        return self._st_cache

    def _guard(self) -> bool:
        """校验 Host / Origin / Sec-Fetch-Site，防 DNS 重绑定与跨站请求（CSRF）。"""
        self._st_cache = None
        port = self.server.server_address[1]
        allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        host = (self.headers.get("Host") or "").strip().lower()
        if host not in allowed_hosts:
            self._json({"error": "forbidden: bad host"}, 403)
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin.strip().lower() not in {f"http://{h}" for h in allowed_hosts}:
            self._json({"error": "forbidden: cross-origin request"}, 403)
            return False
        site = (self.headers.get("Sec-Fetch-Site") or "").lower()
        if site and site not in ("same-origin", "none"):
            self._json({"error": "forbidden: cross-site request"}, 403)
            return False
        return True

    def log_message(self, fmt, *args):
        sys.stderr.write("[TradeS] " + (fmt % args) + "\n")

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if extra:
            for k, v in extra.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _read_body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def do_GET(self):
        if not self._guard():
            return
        path = unquote(urlparse(self.path).path)
        if path == "/api/health":
            return self._json({"ok": True, "port": PORT, "host": HOST})
        if path == "/api/llm/config":
            return self._json({"store": llm_client.public_store()})
        if path == "/api/qa/sessions":
            return self._json({"sessions": qa_store.list_sessions()})
        if path.startswith("/api/qa/session/"):
            sid = path.split("/api/qa/session/", 1)[1]
            item = qa_store.get_session(sid)
            if not item:
                return self._json({"error": "对话不存在"}, 404)
            return self._json({"session": item})
        if path.startswith("/api/clean/download/"):
            with self.st["lock"]:
                return self._download(path.split("/api/clean/download/", 1)[1])
        if path in ("/", "/index.html"):
            return self._file(STATIC / "index.html", "text/html; charset=utf-8")
        if path.startswith("/static/"):
            rel = path[len("/static/"):]
            fp = (STATIC / rel).resolve()
            if not str(fp).startswith(str(STATIC.resolve())) or not fp.is_file():
                return self._json({"error": "not found"}, 404)
            ctype = "application/octet-stream"
            suffix = fp.suffix.lower()
            mimes = {
                ".css": "text/css; charset=utf-8",
                ".js": "application/javascript; charset=utf-8",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".webp": "image/webp",
                ".svg": "image/svg+xml",
                ".woff2": "font/woff2",
            }
            ctype = mimes.get(suffix, ctype)
            return self._file(fp, ctype)
        return self._json({"error": "not found"}, 404)

    def _file(self, fp: Path, ctype: str):
        if not fp.is_file():
            return self._json({"error": "not found"}, 404)
        self._send(200, fp.read_bytes(), ctype)

    def do_POST(self):
        if not self._guard():
            return
        path = unquote(urlparse(self.path).path)
        try:
            routes = {
                "/api/upload": self._upload,
                "/api/clean": self._clean,
                "/api/clean/run": self._clean,
                "/api/export": self._export,
                "/api/agent/run": self._agent_run,
                "/api/qa": self._qa,
                "/api/qa/stream": self._qa_stream,
                "/api/qa/upload": self._qa_upload,
                "/api/qa/search": self._qa_search,
                "/api/qa/session/save": self._qa_save,
                "/api/qa/session/delete": self._qa_delete,
                "/api/qa/sessions/clear": self._qa_clear,
                "/api/llm/config": self._llm_save_config,
                "/api/llm/profile": self._llm_save_profile,
                "/api/llm/profile/activate": self._llm_activate,
                "/api/llm/profile/delete": self._llm_delete_profile,
                "/api/llm/preset": self._llm_preset,
                "/api/llm/test": self._llm_test,
                "/api/llm/validate": self._llm_validate,
            }
            if path in routes:
                if path in _STATEFUL_POST:
                    with self.st["lock"]:
                        return routes[path]()
                return routes[path]()
            return self._json({"error": "not found"}, 404)
        except Exception as e:
            self._json({"error": str(e)}, 500)

    # -------- upload / clean / export --------

    def _parse_upload(self):
        body = self._read_body()
        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" in ctype:
            return _parse_multipart_multi(body, ctype)
        payload = json.loads(body.decode("utf-8") or "{}")
        raw = payload.get("content", "")
        filename = payload.get("filename") or "pasted.csv"
        df, warns = read_table_ex(raw.encode("utf-8"), filename)
        return [(filename, df, warns)]

    def _upload(self):
        items = self._parse_upload()
        if not items:
            return self._json({"error": "未读到文件"}, 400)
        frames, names, warnings = [], [], []
        for filename, df, warns in items:
            warnings.extend(warns)
            if df is None or df.empty:
                warnings.append(f"{filename}：没有读到任何数据行，已忽略")
                continue
            frames.append(df)
            names.append(filename)
        if not frames:
            return self._json({"error": "文件为空或无法解析", "warnings": warnings}, 400)

        merged = merge_tables(frames) if len(frames) > 1 else frames[0]
        columns = list(merged.columns)
        mapping = auto_map_columns(columns)
        source = suggest_source(" ".join(names), columns)

        st = self.st
        st["dfs"] = frames
        st["filenames"] = names
        st["mapping"] = mapping
        st["source"] = source
        st["last_result"] = None
        st["last_export"] = None
        st["df_merged"] = merged

        sample = merged.head(8).astype(str).fillna("").to_dict(orient="records")
        return self._json({
            "filename": " + ".join(names),
            "filenames": names,
            "source": source,
            "rows": int(len(merged)),
            "columns": columns,
            "mapping": mapping,
            "sample": sample,
            "warnings": warnings,
            "template_columns": MAIBANG_COLUMNS,
        })

    def _clean(self):
        merged = self.st.get("df_merged")
        if merged is None or merged.empty:
            return self._json({"error": "请先上传表格"}, 400)
        payload = json.loads(self._read_body().decode("utf-8") or "{}") or {}
        mapping = payload.get("mapping") or self.st.get("mapping") or {}
        source = payload.get("source") or self.st.get("source") or "自定义"
        cleaning = {**load_cleaning_config(CONFIG_PATH), **(payload.get("cleaning") or {})}

        result = run_maibang_clean(merged, mapping, source=source, cleaning=cleaning)
        self.st["last_result"] = result

        rows = result["rows"]
        preview = rows.head(20).astype(str).fillna("").to_dict(orient="records")
        cols = list(rows.columns)
        return self._json({
            "stats": result["stats"],
            "columns": cols,
            "preview": preview,
            "source": source,
            "filename": " + ".join(self.st.get("filenames") or []),
            "sheets": ["清洗结果"],
        })

    def _export(self):
        result = self.st.get("last_result")
        if not result:
            return self._json({"error": "请先执行清洗"}, 400)
        payload = json.loads(self._read_body().decode("utf-8") or "{}") or {}
        source = _safe_filename(payload.get("source") or self.st.get("source") or "自定义", "自定义")
        split_rows = int(payload.get("split_rows") or load_cleaning_config(CONFIG_PATH).get("split_rows") or 1500)
        out_dir = payload.get("output_dir") or load_cleaning_config(CONFIG_PATH).get("output_dir") or str(OUT_DEFAULT)
        try:
            out_dir_p = _safe_output_dir(out_dir)
        except ValueError as e:
            return self._json({"error": str(e)}, 400)

        base_name = f"{source}_清洗结果"
        written = split_and_export(result["rows"], out_dir_p, base_name, split_rows=split_rows)
        self.st["last_export"] = written
        self.st["last_export"]["stats"] = result["stats"]

        # 同时提供浏览器下载主文件
        main_path = Path(written["main"])
        raw = main_path.read_bytes()
        self._send(
            200,
            raw,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            {
                "Content-Disposition": f"attachment; filename*=UTF-8''{_url_quote(main_path.name)}",
                "X-Row-Count": str(result["stats"].get("cleaned_count", 0)),
                "X-Output-Dir": _url_quote(str(out_dir_p)),
                "X-Files": _url_quote(" | ".join(written["files"])),
            },
        )

    def _download(self, token: str):
        # 简单 token：文件名编码
        name = unquote(token)
        last = self.st.get("last_export") or {}
        for fp in last.get("files") or []:
            p = Path(fp)
            if p.name == name:
                if p.is_file():
                    return self._send(
                        200,
                        p.read_bytes(),
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        {"Content-Disposition": f"attachment; filename*=UTF-8''{_url_quote(p.name)}"},
                    )
        return self._json({"error": "文件不存在"}, 404)

    # -------- agent / qa --------

    def _agent_run(self):
        merged = self.st.get("df_merged")
        if merged is None or merged.empty:
            return self._json({"error": "请先上传表格"}, 400)
        payload = json.loads(self._read_body().decode("utf-8") or "{}") or {}
        instruction = (payload.get("instruction") or "").strip()
        if not instruction:
            return self._json({"error": "请填写清洗指令"}, 400)
        out = run_agent(
            merged,
            instruction,
            source=self.st.get("source") or "自定义",
            mapping=payload.get("mapping") or self.st.get("mapping"),
            use_llm=bool(payload.get("use_llm", True)),
            cleaning=payload.get("cleaning") if isinstance(payload.get("cleaning"), dict) else None,
        )
        cleaned = out.get("cleaned") or {}
        self.st["last_result"] = cleaned
        return self._json({
            "steps": out.get("steps"),
            "intent": out.get("intent"),
            "stats": out.get("stats"),
            "columns": out.get("preview_columns") or [],
            "preview": out.get("preview") or [],
            "sheets": out.get("sheets") or ["清洗结果"],
            "review": (out.get("result_summary") or {}).get("review"),
        })

    def _qa(self):
        payload = json.loads(self._read_body().decode("utf-8") or "{}") or {}
        question = (payload.get("question") or "").strip()
        history = payload.get("history") or []
        role = payload.get("role") or "clean"
        if not question:
            return self._json({"error": "请输入问题"}, 400)
        role_hint = {
            "clean": "重点回答清洗规则：手机号、去重、删除、导出。",
            "map": "重点回答字段映射与某公司模板列。",
            "quality": "重点回答有效率、删除原因、质量口径。",
            "ops": "重点回答工作台操作步骤。",
        }.get(role, "")
        try:
            cfg = llm_client.load_config()
            has_llm = bool(cfg.get("api_key")) or "127.0.0.1" in str(cfg.get("base_url", "")) or "localhost" in str(cfg.get("base_url", ""))
            if has_llm and payload.get("use_llm", True):
                msgs = [{"role": "system", "content": QA_SYSTEM + ("\n" + role_hint if role_hint else "")}]
                for h in history[-6:]:
                    r = "assistant" if h.get("role") == "bot" else "user"
                    msgs.append({"role": r, "content": str(h.get("text") or "")[:500]})
                msgs.append({"role": "user", "content": question})
                reply = llm_client.chat(msgs, cfg)
                return self._json({"reply": reply, "source": "llm"})
        except Exception as e:
            return self._json({"reply": local_qa_answer(question), "source": "local", "note": f"模型不可用：{e}"})
        return self._json({"reply": local_qa_answer(question), "source": "local"})

    def _qa_search(self):
        """简易联网搜索：抓取公开搜索结果摘要，供问答引用。"""
        payload = json.loads(self._read_body().decode("utf-8") or "{}") or {}
        q = (payload.get("query") or "").strip()
        if not q:
            return self._json({"error": "请输入搜索词"}, 400)
        results, errors = web_search_ex(q, limit=int(payload.get("limit") or 5))
        out = {"ok": bool(results), "query": q, "results": results, "count": len(results)}
        if not results:
            out["error"] = "联网搜索未获得结果：" + "；".join(errors)
        return self._json(out)

    def _qa_upload(self):
        """附件上传：提取文本内容，供问答引用。"""
        body = self._read_body()
        ctype = self.headers.get("Content-Type", "")
        try:
            if "multipart/form-data" in ctype:
                items = _multipart_files(body, ctype)
            else:
                payload = json.loads(body.decode("utf-8") or "{}") or {}
                filename = payload.get("filename") or "file.txt"
                content = payload.get("content") or ""
                items = [(filename, content)]
        except Exception as e:
            return self._json({"error": f"上传失败：{e}"}, 400)

        files = []
        for filename, raw in items:
            text = ""
            if isinstance(raw, str):
                text = raw[:20000]
            else:
                try:
                    text = raw.decode("utf-8", errors="replace")[:20000]
                except Exception:
                    text = ""
            # 只保留可读文本片段
            text = "".join(ch if ch == "\n" or ch == "\t" or (32 <= ord(ch) < 127) or ord(ch) > 127 else " " for ch in text)
            files.append({
                "filename": filename,
                "chars": len(text),
                "preview": text[:2000],
                "content": text[:12000],
            })
        return self._json({"ok": True, "files": [{"filename": f["filename"], "chars": f["chars"], "preview": f["preview"]} for f in files], "contents": files})

    def _build_qa_messages(self, payload: dict) -> tuple[list, str]:
        """返回 (消息列表, 联网搜索提示)。提示非空表示已开启联网但没有拿到结果。"""
        question = (payload.get("question") or "").strip()
        history = payload.get("history") or []
        role = payload.get("role") or "clean"
        attachments = payload.get("attachments") or []
        use_web = bool(payload.get("use_web_search"))
        role_hint = {
            "clean": "重点回答清洗规则：手机号、去重、删除、导出。",
            "map": "重点回答字段映射与某公司模板列。",
            "quality": "重点回答有效率、删除原因、质量口径。",
            "ops": "重点回答工作台操作步骤。",
        }.get(role, "")
        system = QA_SYSTEM + ("\n" + role_hint if role_hint else "")
        if use_web:
            system += "\n已启用联网搜索，回答时优先依据下方「联网检索结果」，并标注来源；无结果时说明未能联网获取。"
        msgs = [{"role": "system", "content": system}]
        for h in history[-6:]:
            r = "assistant" if h.get("role") == "bot" else "user"
            msgs.append({"role": r, "content": str(h.get("text") or "")[:500]})
        user_content = question
        web_notice = ""
        if use_web and question:
            hits, errors = web_search_ex(question, limit=5)
            if not hits:
                web_notice = "⚠ 联网搜索未获得结果（" + "；".join(errors) + "），以下回答未使用联网信息。\n\n"
                msgs[0]["content"] += "\n本次联网搜索失败，没有检索结果：请在回答开头说明这一点，不要编造来源或链接。"
            if hits:
                lines = ["【联网检索结果】"]
                for i, h in enumerate(hits, 1):
                    lines.append(f"{i}. {h.get('title')} | {h.get('url')}\n   {h.get('snippet')}")
                user_content = "\n".join(lines) + "\n\n请结合上述结果回答用户问题。\n" + question
        if attachments:
            parts = []
            for a in attachments[:3]:
                parts.append(f"【附件 {a.get('filename', 'file')}】\n{str(a.get('content') or a.get('preview') or '')[:4000]}")
            user_content = ("\n\n".join(parts) + "\n\n用户问题：" + user_content) if user_content else "\n\n".join(parts)
        msgs.append({"role": "user", "content": user_content[:8000]})
        return msgs, web_notice

    def _qa_stream(self):
        """SSE 流式问答（HTTP chunked）。"""
        payload = json.loads(self._read_body().decode("utf-8") or "{}") or {}
        question = (payload.get("question") or "").strip()
        if not question and not payload.get("attachments"):
            return self._json({"error": "请输入问题"}, 400)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def emit(obj: dict):
            data = f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")
            try:
                self.wfile.write(f"{len(data):x}\r\n".encode("ascii") + data + b"\r\n")
                self.wfile.flush()
            except Exception:
                pass

        def emit_done():
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except Exception:
                pass

        emit({"type": "meta", "ok": True})
        msgs, web_notice = self._build_qa_messages(payload)
        full: list[str] = []
        if web_notice:
            full.append(web_notice)
            emit({"type": "content", "content": web_notice})
        try:
            cfg = llm_client.load_config()
            has_llm = bool(cfg.get("api_key")) or "127.0.0.1" in str(cfg.get("base_url", "")) or "localhost" in str(cfg.get("base_url", ""))
            if has_llm and payload.get("use_llm", True):
                for piece in llm_client.chat_stream(msgs, cfg):
                    full.append(piece)
                    emit({"type": "content", "content": piece})
            else:
                text = local_qa_answer(question or "请分析附件")
                for i in range(0, len(text), 12):
                    piece = text[i : i + 12]
                    full.append(piece)
                    emit({"type": "content", "content": piece})
        except Exception as e:
            text = local_qa_answer(question or "请分析附件")
            if not full:
                emit({"type": "content", "content": text})
                full.append(text)
            emit({"type": "note", "content": f"（模型不可用：{str(e)[:120]}）"})
        emit({"type": "done", "content": "".join(full)})
        emit_done()

    def _qa_save(self):
        payload = json.loads(self._read_body().decode("utf-8") or "{}") or {}
        sid = payload.get("id") or ""
        messages = payload.get("messages") or []
        if not isinstance(messages, list):
            return self._json({"error": "messages 须为数组"}, 400)
        item = qa_store.save_messages(sid, messages, title=payload.get("title"))
        return self._json({"ok": True, "session": {
            "id": item["id"],
            "title": item.get("title"),
            "count": len(item.get("messages") or []),
            "updated_at": item.get("updated_at"),
        }})

    def _qa_delete(self):
        payload = json.loads(self._read_body().decode("utf-8") or "{}") or {}
        sid = payload.get("id") or ""
        if not sid:
            return self._json({"error": "缺少 id"}, 400)
        ok = qa_store.delete_session(str(sid))
        if not ok:
            return self._json({"error": "对话不存在"}, 404)
        return self._json({"ok": True, "sessions": qa_store.list_sessions()})

    def _qa_clear(self):
        n = qa_store.clear_all()
        return self._json({"ok": True, "removed": n, "sessions": []})

    # -------- llm --------

    def _llm_save_config(self):
        payload = json.loads(self._read_body().decode("utf-8") or "{}") or {}
        llm_client.save_config(payload)
        return self._json({"ok": True, "store": llm_client.public_store()})

    def _llm_save_profile(self):
        payload = json.loads(self._read_body().decode("utf-8") or "{}") or {}
        pid = payload.get("id") or payload.get("profile_id") or "custom"
        data = payload.get("profile") or payload
        llm_client.upsert_profile(pid, data, activate=bool(payload.get("activate", True)))
        return self._json({"ok": True, "store": llm_client.public_store()})

    def _llm_activate(self):
        payload = json.loads(self._read_body().decode("utf-8") or "{}") or {}
        pid = payload.get("id") or payload.get("profile_id")
        if not pid:
            return self._json({"error": "缺少 id"}, 400)
        try:
            llm_client.set_active(str(pid))
        except ValueError as e:
            return self._json({"error": str(e)}, 400)
        return self._json({"ok": True, "store": llm_client.public_store()})

    def _llm_delete_profile(self):
        payload = json.loads(self._read_body().decode("utf-8") or "{}") or {}
        pid = payload.get("id") or payload.get("profile_id")
        if not pid:
            return self._json({"error": "缺少 id"}, 400)
        llm_client.delete_profile(str(pid))
        return self._json({"ok": True, "store": llm_client.public_store()})

    def _llm_preset(self):
        payload = json.loads(self._read_body().decode("utf-8") or "{}") or {}
        return self._json({"ok": True, "profile": llm_client.clone_preset(payload.get("preset_id") or "custom")})

    def _llm_test(self):
        payload = json.loads(self._read_body().decode("utf-8") or "{}") or {}
        # 允许「未保存就测试」：表单配置优先，id 仅用于取本地已存 Key
        store = llm_client.load_store()
        pid = str(payload.get("id") or payload.get("profile_id") or "")
        if pid and pid in store["profiles"]:
            cfg = dict(store["profiles"][pid])
        else:
            cfg = llm_client.load_config()
        for k in ("base_url", "model", "chat_path", "api_style", "temperature", "max_tokens", "extra_headers", "ssl_verify"):
            if payload.get(k) is not None:
                cfg[k] = payload[k]
        if payload.get("api_key"):
            cfg["api_key"] = payload["api_key"]
        if payload.get("timeout"):
            cfg["timeout"] = int(payload["timeout"])
        try:
            return self._json(llm_client.test_connection(cfg))
        except Exception as e:
            return self._json({"error": str(e)}, 502)

    def _llm_validate(self):
        result = self.st.get("last_result")
        if not result:
            return self._json({"error": "请先执行清洗"}, 400)
        rows = result["rows"]
        records = rows.head(15).astype(str).fillna("").to_dict(orient="records")
        try:
            ai = llm_client.validate_rows(records)
        except Exception as e:
            return self._json({"error": f"AI 校验失败：{e}"}, 502)
        return self._json({
            "summary": ai.get("summary", ""),
            "checked": ai.get("checked"),
            "fail_count": 0,
            "preview": records[:10],
        })


def _url_quote(s: str) -> str:
    from urllib.parse import quote
    return quote(s)


def _multipart_files(body: bytes, ctype: str) -> list[tuple[str, bytes]]:
    """解析 multipart，返回 [(文件名, 原始字节)]。"""
    m = re.search(r'boundary="?([^";]+)"?', ctype)
    if not m:
        raise ValueError("multipart boundary 缺失")
    boundary = b"--" + m.group(1).encode()
    items: list[tuple[str, bytes]] = []
    for part in body.split(boundary):
        if b"Content-Disposition" not in part:
            continue
        header, _, data = part.partition(b"\r\n\r\n")
        if b'name="file"' not in header and b'name="files"' not in header:
            continue
        fn = re.search(rb'filename="([^"]+)"', header)
        filename = fn.group(1).decode("utf-8", errors="replace") if fn else "upload.xlsx"
        items.append((filename, data.rstrip(b"\r\n")))
    return items


def _parse_multipart_multi(body: bytes, ctype: str):
    """multipart → [(文件名, 表, 读取提示)]；空文件返回 (文件名, None, [])。"""
    items = []
    for filename, file_bytes in _multipart_files(body, ctype):
        if not file_bytes:
            items.append((filename, None, []))
            continue
        df, warns = read_table_ex(file_bytes, filename)
        items.append((filename, df, warns))
    return items


def _find_free_port(host: str, start: int) -> int:
    import socket

    port = start
    for _ in range(20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                port += 1
    return start


def main():
    os.chdir(BASE)
    global PORT
    PORT = _find_free_port(HOST, PORT)
    url = f"http://{HOST}:{PORT}"
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print("=" * 48)
    print("  [TradeS Agent] 服务已启动")
    print(f"  访问地址: {url}")
    print("  （浏览器应自动打开；若未打开请手动复制上方地址）")
    print("  按 Ctrl+C 可停止服务。")
    print("=" * 48)
    sys.stdout.flush()

    def _open():
        try:
            if APP_CFG["server"].get("auto_open_browser", True):
                webbrowser.open(url)
        except Exception:
            pass

    threading.Timer(0.8, _open).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[TradeS] 已停止")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
