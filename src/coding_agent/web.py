from __future__ import annotations

"""
轻量 Web UI 服务：
- 提供静态前端页面
- 暴露基础 JSON API（health/state/prompt）
"""

import argparse
import asyncio
import json
import mimetypes
import os
import re
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Sequence
from urllib.parse import parse_qs, urlparse

from ai.overflow import estimate_context_tokens
from ai.types import AssistantMessage, TextContent, ThinkingContent

from .factory import create_agent_session
from .types import CreateAgentSessionOptions


@dataclass
class WebServerOptions:
    host: str
    port: int
    workspace: Path
    provider: str | None
    model_id: str | None
    session_id: str | None
    system_prompt: str
    thinking_level: str
    tool_execution: str
    read_only: bool
    allow_dangerous_bash: bool
    disable_workspace_resources: bool


class WebServerState:
    def __init__(self, session, options: WebServerOptions) -> None:
        self.session = session
        self.options = options
        self.lock = threading.Lock()

    def replace_session(self, session) -> None:
        old = self.session
        self.session = session
        old.close()

    def create_fresh_session(self, *, session_id: str | None = None):
        return create_agent_session(
            CreateAgentSessionOptions(
                workspace_dir=self.options.workspace,
                provider=self.options.provider,
                model_id=self.options.model_id,
                system_prompt=self.options.system_prompt,
                session_id=session_id,
                thinking_level=self.options.thinking_level,
                tool_execution=self.options.tool_execution,
                read_only_mode=self.options.read_only,
                block_dangerous_bash=not self.options.allow_dangerous_bash,
                load_workspace_resources=not self.options.disable_workspace_resources,
            )
        )


def _strip_wrapped_quotes(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1]
    return text


def _load_env_file(env_path: Path) -> int:
    if not env_path.exists() or not env_path.is_file():
        return 0
    loaded = 0
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = _strip_wrapped_quotes(value)
        loaded += 1
    return loaded


def _load_env_ps1_file(env_path: Path) -> int:
    if not env_path.exists() or not env_path.is_file():
        return 0
    loaded = 0
    pattern = re.compile(r"^\s*\$env:([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$")
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        matched = pattern.match(line)
        if not matched:
            continue
        key = matched.group(1)
        value = _strip_wrapped_quotes(matched.group(2))
        if key in os.environ:
            continue
        os.environ[key] = value
        loaded += 1
    return loaded


def _load_workspace_env(workspace: Path) -> int:
    loaded = 0
    loaded += _load_env_file(workspace / ".env")
    loaded += _load_env_ps1_file(workspace / ".env.ps1")
    return loaded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LiaoClaw coding-agent Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host")
    parser.add_argument("--port", type=int, default=8787, help="HTTP bind port")
    parser.add_argument("--workspace", default=".", help="Workspace directory")
    parser.add_argument("--session-id", default=None, help="Existing session id to resume")
    parser.add_argument("--provider", default=None, help="Model provider, e.g. anthropic/openai-standard")
    parser.add_argument("--model-id", default=None, help="Model id")
    parser.add_argument("--system-prompt", default="", help="System prompt")
    parser.add_argument("--thinking-level", default="off", help="Thinking level: off/minimal/low/medium/high/xhigh")
    parser.add_argument("--tool-execution", choices=["parallel", "sequential"], default="parallel")
    parser.add_argument("--read-only", action="store_true", help="Enable read-only mode (disable write/edit/bash)")
    parser.add_argument("--allow-dangerous-bash", action="store_true", help="Disable dangerous bash blocking")
    parser.add_argument(
        "--disable-workspace-resources",
        action="store_true",
        help="Disable reading .liaoclaw/{settings,prompt,tools}",
    )
    return parser


def _extract_assistant_text(message: AssistantMessage) -> str:
    text = "".join(block.text for block in message.content if isinstance(block, TextContent)).strip()
    if text:
        return text
    thinking = "".join(block.thinking for block in message.content if isinstance(block, ThinkingContent)).strip()
    return thinking


def _build_reply_text(final_assistant: AssistantMessage | None) -> str:
    if final_assistant is None:
        return "未收到助手回复，请重试。"

    answer = _extract_assistant_text(final_assistant)
    if answer:
        return answer

    if final_assistant.error_message:
        return f"执行失败：{final_assistant.error_message}"
    if final_assistant.stop_reason == "toolUse":
        return "模型触发了工具调用，但未返回可展示文本。请稍后重试。"
    return "助手本轮没有返回文本内容。"


def _json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _safe_read_json(handler: BaseHTTPRequestHandler) -> dict:
    length_raw = handler.headers.get("Content-Length", "0")
    try:
        length = int(length_raw)
    except ValueError:
        length = 0
    raw = handler.rfile.read(max(length, 0)) if length > 0 else b"{}"
    if not raw:
        return {}
    decoded = raw.decode("utf-8")
    data = json.loads(decoded)
    if not isinstance(data, dict):
        raise ValueError("request body must be a JSON object")
    return data


def _compute_token_state(session) -> dict:
    usage = session.cumulative_usage
    actual_total = int(usage.get("total_tokens", 0) or 0)
    system_prompt = ""
    agent = getattr(session, "agent", None)
    agent_state = getattr(agent, "state", None)
    if agent_state is not None:
        system_prompt = str(getattr(agent_state, "system_prompt", "") or "")
    estimated_total = estimate_context_tokens(session.messages, system_prompt)
    display_total = actual_total if actual_total > 0 else estimated_total
    return {
        "actual_total_tokens": actual_total,
        "estimated_total_tokens": estimated_total,
        "display_total_tokens": display_total,
        "token_source": "usage" if actual_total > 0 else "estimated",
    }


def _create_handler(state: WebServerState):
    static_dir = Path(__file__).resolve().parent / "web_static"

    class Handler(BaseHTTPRequestHandler):
        server_version = "LiaoClawWeb/1.0"

        def _parse(self):
            parsed = urlparse(self.path)
            return parsed.path, parse_qs(parsed.query)

        def do_GET(self) -> None:  # noqa: N802
            route, query = self._parse()
            if route in {"/", "/index.html"}:
                self._serve_static("index.html")
                return
            if route == "/assets/style.css":
                self._serve_static("style.css")
                return
            if route == "/assets/app.js":
                self._serve_static("app.js")
                return
            if route == "/api/health":
                self._send_json(HTTPStatus.OK, {"status": "ok", "session_id": state.session.session_id})
                return
            if route == "/api/state":
                token_state = _compute_token_state(state.session)
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "session_id": state.session.session_id,
                        "leaf_id": state.session.get_leaf_id(),
                        "message_count": len(state.session.messages),
                        "cumulative_usage": state.session.cumulative_usage,
                        "tokens": token_state,
                    },
                )
                return
            if route == "/api/session/entries":
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "session_id": state.session.session_id,
                        "leaf_id": state.session.get_leaf_id(),
                        "entries": state.session.list_entries(),
                    },
                )
                return
            if route == "/api/session/tree":
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "session_id": state.session.session_id,
                        "leaf_id": state.session.get_leaf_id(),
                        "tree": state.session.get_session_tree(),
                    },
                )
                return
            if route == "/api/messages":
                raw_limit = (query.get("limit") or ["80"])[0]
                try:
                    limit = max(1, min(300, int(raw_limit)))
                except ValueError:
                    limit = 80
                messages = state.session.messages[-limit:]
                data: list[dict] = []
                for msg in messages:
                    role = getattr(msg, "role", "unknown")
                    text = ""
                    if isinstance(msg, AssistantMessage):
                        text = _extract_assistant_text(msg)
                    elif hasattr(msg, "content"):
                        content = getattr(msg, "content", "")
                        if isinstance(content, str):
                            text = content
                        elif isinstance(content, list):
                            text = "".join(block.text for block in content if isinstance(block, TextContent))
                    data.append({"role": role, "text": (text or "").strip()})
                self._send_json(HTTPStatus.OK, {"status": "ok", "messages": data})
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"status": "error", "message": "Not found"})

        def do_POST(self) -> None:  # noqa: N802
            route, _ = self._parse()
            if route == "/api/prompt":
                self._handle_prompt()
                return
            if route == "/api/continue":
                self._handle_continue()
                return
            if route == "/api/session/switch":
                self._handle_switch_entry()
                return
            if route == "/api/session/fork":
                self._handle_fork_session()
                return
            if route == "/api/session/new":
                self._handle_new_session()
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"status": "error", "message": "Not found"})

        def _build_prompt_response(self, final_assistant: AssistantMessage | None) -> dict:
            return {
                "status": "ok",
                "session_id": state.session.session_id,
                "leaf_id": state.session.get_leaf_id(),
                "reply": _build_reply_text(final_assistant),
                "stop_reason": getattr(final_assistant, "stop_reason", None),
                "error_message": getattr(final_assistant, "error_message", None),
                "last_usage": state.session.last_usage,
                "tokens": _compute_token_state(state.session),
            }

        def _handle_prompt(self) -> None:
            try:
                body = _safe_read_json(self)
                text = str(body.get("text", "")).strip()
                if not text:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"status": "error", "message": "text is required"})
                    return

                with state.lock:
                    asyncio.run(state.session.prompt(text))

                final_assistant = next(
                    (m for m in reversed(state.session.messages) if isinstance(m, AssistantMessage)),
                    None,
                )
                self._send_json(HTTPStatus.OK, self._build_prompt_response(final_assistant))
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"status": "error", "message": str(exc)})
            except Exception as exc:  # pragma: no cover - defensive fallback
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"status": "error", "message": f"server error: {exc}"})

        def _handle_continue(self) -> None:
            try:
                with state.lock:
                    asyncio.run(state.session.continue_run())
                final_assistant = next(
                    (m for m in reversed(state.session.messages) if isinstance(m, AssistantMessage)),
                    None,
                )
                self._send_json(HTTPStatus.OK, self._build_prompt_response(final_assistant))
            except Exception as exc:  # pragma: no cover - defensive fallback
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"status": "error", "message": f"server error: {exc}"})

        def _handle_switch_entry(self) -> None:
            try:
                body = _safe_read_json(self)
                entry_id = str(body.get("entry_id", "")).strip()
                if not entry_id:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"status": "error", "message": "entry_id is required"})
                    return
                with state.lock:
                    state.session.switch_to_entry(entry_id)
                self._send_json(HTTPStatus.OK, {"status": "ok", "session_id": state.session.session_id, "leaf_id": state.session.get_leaf_id()})
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"status": "error", "message": str(exc)})

        def _handle_fork_session(self) -> None:
            try:
                body = _safe_read_json(self)
                entry_id_raw = body.get("entry_id")
                entry_id = str(entry_id_raw).strip() if entry_id_raw is not None else ""
                from_entry = entry_id or (state.session.get_leaf_id() or None)
                with state.lock:
                    forked = state.session.fork_session(from_entry_id=from_entry)
                    state.replace_session(forked)
                self._send_json(HTTPStatus.OK, {"status": "ok", "session_id": state.session.session_id, "leaf_id": state.session.get_leaf_id()})
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"status": "error", "message": str(exc)})

        def _handle_new_session(self) -> None:
            with state.lock:
                fresh = state.create_fresh_session(session_id=None)
                state.replace_session(fresh)
            self._send_json(HTTPStatus.OK, {"status": "ok", "session_id": state.session.session_id, "leaf_id": state.session.get_leaf_id()})

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

        def _serve_static(self, filename: str) -> None:
            target = static_dir / filename
            if not target.exists() or not target.is_file():
                self._send_json(HTTPStatus.NOT_FOUND, {"status": "error", "message": "asset not found"})
                return
            data = target.read_bytes()
            content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, status: HTTPStatus, payload: dict) -> None:
            body = _json_bytes(payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def create_server(options: WebServerOptions):
    session = create_agent_session(
        CreateAgentSessionOptions(
            workspace_dir=options.workspace,
            provider=options.provider,
            model_id=options.model_id,
            system_prompt=options.system_prompt,
            session_id=options.session_id,
            thinking_level=options.thinking_level,
            tool_execution=options.tool_execution,
            read_only_mode=options.read_only,
            block_dangerous_bash=not options.allow_dangerous_bash,
            load_workspace_resources=not options.disable_workspace_resources,
        )
    )
    state = WebServerState(session=session, options=options)
    server = ThreadingHTTPServer((options.host, options.port), _create_handler(state))
    return server, session


def serve(options: WebServerOptions) -> int:
    server, session = create_server(options)
    print(f"LiaoClaw Web UI running at http://{options.host}:{options.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        session.close()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    opts = WebServerOptions(
        host=args.host,
        port=args.port,
        workspace=Path(args.workspace),
        provider=args.provider,
        model_id=args.model_id,
        session_id=args.session_id,
        system_prompt=args.system_prompt,
        thinking_level=args.thinking_level,
        tool_execution=args.tool_execution,
        read_only=bool(args.read_only),
        allow_dangerous_bash=bool(args.allow_dangerous_bash),
        disable_workspace_resources=bool(args.disable_workspace_resources),
    )

    loaded = _load_workspace_env(opts.workspace)
    if loaded > 0:
        print(f"Loaded {loaded} env vars from workspace env files")

    try:
        return serve(opts)
    except ValueError as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
