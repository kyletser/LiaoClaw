from __future__ import annotations

import json
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib import request

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai.types import AssistantMessage, TextContent, UserMessage  # noqa: E402
from coding_agent.web import WebServerOptions, create_server  # noqa: E402


class _FakeSession:
    def __init__(self) -> None:
        self.session_id = "session-web-test"
        self.messages = []
        self._leaf_id = "leaf-1"

    @property
    def cumulative_usage(self) -> dict:
        return {
            "input_tokens": 3,
            "output_tokens": 5,
            "total_tokens": 8,
            "total_cost": 0.0,
        }

    @property
    def last_usage(self) -> dict:
        return {
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
            "cache_read": 0,
            "cache_write": 0,
            "cost": {"input": 0.0, "output": 0.0, "total": 0.0},
        }

    def get_leaf_id(self) -> str:
        return self._leaf_id

    def list_entries(self) -> list[dict]:
        return [
            {
                "id": "leaf-1",
                "parent_id": None,
                "timestamp": "2026-01-01T00:00:00Z",
                "role": "user",
                "preview": "hello",
                "depth": 0,
                "is_leaf": self._leaf_id == "leaf-1",
            }
        ]

    def get_session_tree(self) -> list[dict]:
        return [{"id": "leaf-1", "parent_id": None, "children": []}]

    def switch_to_entry(self, entry_id: str) -> None:
        self._leaf_id = entry_id

    def fork_session(self, from_entry_id: str | None = None):
        _ = from_entry_id
        cloned = _FakeSession()
        cloned.session_id = "session-web-forked"
        return cloned

    async def continue_run(self):
        self.messages.append(AssistantMessage(content=[TextContent(text="continued")]))
        return self.messages

    async def prompt(self, text: str):
        self.messages.append(UserMessage(content=text))
        self.messages.append(AssistantMessage(content=[TextContent(text=f"echo: {text}")]))
        return self.messages

    def close(self) -> None:
        return


class _EmptyReplySession(_FakeSession):
    async def prompt(self, text: str):
        self.messages.append(UserMessage(content=text))
        self.messages.append(AssistantMessage(content=[]))
        return self.messages


class _ZeroUsageSession(_FakeSession):
    @property
    def cumulative_usage(self) -> dict:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
        }


class CodingAgentWebTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake_session = _FakeSession()
        options = WebServerOptions(
            host="127.0.0.1",
            port=0,
            workspace=ROOT,
            provider="anthropic",
            model_id="glm-4.7",
            session_id=None,
            system_prompt="",
            thinking_level="off",
            tool_execution="parallel",
            read_only=False,
            allow_dangerous_bash=False,
            disable_workspace_resources=False,
        )

        with patch("coding_agent.web.create_agent_session", return_value=self.fake_session):
            self.server, self.session = create_server(options)

        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.session.close()

    def test_health_endpoint(self) -> None:
        with request.urlopen(f"{self.base_url}/api/health", timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            payload = json.loads(resp.read().decode("utf-8"))
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["session_id"], "session-web-test")

    def test_index_page_served(self) -> None:
        with request.urlopen(f"{self.base_url}/", timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            html = resp.read().decode("utf-8")
        self.assertIn("LiaoClaw 网页控制台", html)

    def test_prompt_endpoint_returns_reply(self) -> None:
        req = request.Request(
            f"{self.base_url}/api/prompt",
            data=json.dumps({"text": "hello web"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            payload = json.loads(resp.read().decode("utf-8"))

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["reply"], "echo: hello web")

    def test_prompt_endpoint_returns_non_empty_fallback_when_model_has_no_text(self) -> None:
        options = WebServerOptions(
            host="127.0.0.1",
            port=0,
            workspace=ROOT,
            provider="anthropic",
            model_id="glm-4.5-air",
            session_id=None,
            system_prompt="",
            thinking_level="off",
            tool_execution="parallel",
            read_only=False,
            allow_dangerous_bash=False,
            disable_workspace_resources=False,
        )

        with patch("coding_agent.web.create_agent_session", return_value=_EmptyReplySession()):
            server, session = create_server(options)

        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address
        base_url = f"http://{host}:{port}"
        try:
            req = request.Request(
                f"{base_url}/api/prompt",
                data=json.dumps({"text": "no text"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                payload = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["reply"], "助手本轮没有返回文本内容。")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)
            session.close()

    def test_state_contains_token_display_fields(self) -> None:
        with request.urlopen(f"{self.base_url}/api/state", timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            payload = json.loads(resp.read().decode("utf-8"))
        self.assertEqual(payload["status"], "ok")
        self.assertIn("tokens", payload)
        self.assertIn("display_total_tokens", payload["tokens"])

    def test_entries_endpoint(self) -> None:
        with request.urlopen(f"{self.base_url}/api/session/entries", timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            payload = json.loads(resp.read().decode("utf-8"))
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["entries"])

    def test_switch_entry_endpoint(self) -> None:
        req = request.Request(
            f"{self.base_url}/api/session/switch",
            data=json.dumps({"entry_id": "leaf-x"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            payload = json.loads(resp.read().decode("utf-8"))
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["leaf_id"], "leaf-x")

    def test_state_token_fallback_uses_estimation_when_usage_is_zero(self) -> None:
        options = WebServerOptions(
            host="127.0.0.1",
            port=0,
            workspace=ROOT,
            provider="anthropic",
            model_id="glm-4.5-air",
            session_id=None,
            system_prompt="",
            thinking_level="off",
            tool_execution="parallel",
            read_only=False,
            allow_dangerous_bash=False,
            disable_workspace_resources=False,
        )

        with patch("coding_agent.web.create_agent_session", return_value=_ZeroUsageSession()):
            server, session = create_server(options)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address
        base_url = f"http://{host}:{port}"
        try:
            req = request.Request(
                f"{base_url}/api/prompt",
                data=json.dumps({"text": "token fallback"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                payload = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["tokens"]["token_source"], "estimated")
            self.assertGreater(payload["tokens"]["display_total_tokens"], 0)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)
            session.close()


if __name__ == "__main__":
    unittest.main()
