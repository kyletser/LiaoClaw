"""
MCP Client implementation - supports stdio-connected MCP servers
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from coding_agent.mcp import MCPClient


@dataclass
class MCPServerConfig:
    """MCP server configuration"""
    name: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    args: list[str] = field(default_factory=list)


class StdioMCPClient:
    """Client that connects to MCP servers via stdio"""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._proc: asyncio.subprocess.Process | None = None
        self._request_id = 0
        self._lock = asyncio.Lock()
        self._initialized = False

    async def _initialize(self) -> None:
        """Initialize MCP protocol with server"""
        if self._initialized:
            return
            
        proc = await self._ensure_running()
        
        async with self._lock:
            self._request_id += 1
            request = {
                "jsonrpc": "2.0",
                "id": self._request_id,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "liaoclaw",
                        "version": "0.2.0"
                    }
                },
            }
            
            request_str = json.dumps(request) + "\n"
            proc.stdin.write(request_str.encode())
            await proc.stdin.drain()
            
            # Read response
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=30)
            if not line:
                stderr = await self._read_stderr_snippet(proc)
                detail = f": {stderr}" if stderr else ""
                raise RuntimeError(f"MCP server died during initialization{detail}")
            
            response = json.loads(line.decode())
            if "error" in response:
                raise RuntimeError(f"MCP init error: {response['error']}")
            
            # Send initialized notification
            notify = {"jsonrpc": "2.0", "method": "initialized", "params": {}}
            proc.stdin.write((json.dumps(notify) + "\n").encode())
            await proc.stdin.drain()
            
            self._initialized = True

    async def _ensure_running(self) -> asyncio.subprocess.Process:
        """Ensure MCP server process is running"""
        if self._proc is None or self._proc.returncode is not None:
            # Merge command and args
            full_command = self._resolve_command(self.config.command + self.config.args)
            
            # Merge environment variables
            env = {**os.environ.copy(), **self.config.env}
            
            self._proc = await asyncio.create_subprocess_exec(
                *full_command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            
            # Initialize the connection
            await self._initialize()
            
        return self._proc

    @staticmethod
    def _resolve_command(command: list[str]) -> list[str]:
        if not command:
            return command
        first = Path(command[0]).name.lower()
        if first in {"python", "python.exe", "python3", "py", "py.exe"}:
            return [sys.executable, *command[1:]]
        return command

    @staticmethod
    async def _read_stderr_snippet(proc: asyncio.subprocess.Process, max_bytes: int = 4096) -> str:
        if not proc.stderr:
            return ""
        try:
            raw = await asyncio.wait_for(proc.stderr.read(max_bytes), timeout=0.3)
        except Exception:
            return ""
        text = raw.decode(errors="ignore").strip()
        if len(text) > 500:
            return text[:500] + "...<truncated>"
        return text

    async def call_tool(self, server: str, tool: str, arguments: dict[str, Any]) -> Any:
        """Call MCP server tool"""
        proc = await self._ensure_running()
        
        async with self._lock:
            self._request_id += 1
            request_id = self._request_id

            request = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {
                    "name": tool,
                    "arguments": arguments,
                },
            }

            # Send request
            request_str = json.dumps(request) + "\n"
            proc.stdin.write(request_str.encode())
            await proc.stdin.drain()

            # Read response line - use async timeout
            try:
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=30)
            except asyncio.TimeoutError:
                raise RuntimeError(f"MCP server timeout for tool {tool}")

            if not line:
                stderr = await self._read_stderr_snippet(proc)
                raise RuntimeError(f"MCP server died: {stderr or 'unknown'}")

            response = json.loads(line.decode())
            
            if "error" in response:
                error = response["error"]
                raise RuntimeError(f"MCP error: {error.get('message', error)}")

            return response.get("result", {})

    async def list_tools(self) -> list[dict[str, Any]]:
        """List tools provided by MCP server"""
        proc = await self._ensure_running()
        
        async with self._lock:
            self._request_id += 1
            request_id = self._request_id

            request = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/list",
                "params": {},
            }

            request_str = json.dumps(request) + "\n"
            proc.stdin.write(request_str.encode())
            await proc.stdin.drain()

            line = await asyncio.wait_for(proc.stdout.readline(), timeout=30)
            if not line:
                stderr = await self._read_stderr_snippet(proc)
                detail = f": {stderr}" if stderr else ""
                raise RuntimeError(f"MCP server died{detail}")

            response = json.loads(line.decode())
            return response.get("result", {}).get("tools", [])

    def close_sync(self) -> None:
        """Synchronous close - for use during process shutdown"""
        if self._proc:
            try:
                if self._proc.stdin:
                    try:
                        self._proc.stdin.close()
                    except Exception:
                        pass
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=2)
                except Exception:
                    try:
                        self._proc.kill()
                    except Exception:
                        pass
            except Exception:
                pass
            self._proc = None

    async def close(self) -> None:
        """Close MCP server connection"""
        self.close_sync()


class MultiMCPClient:
    """管理多个 MCP 服务器的客户端"""

    def __init__(self):
        self._clients: dict[str, StdioMCPClient] = {}

    def add_server(self, config: MCPServerConfig) -> None:
        """添加 MCP 服务器"""
        self._clients[config.name] = StdioMCPClient(config)

    async def call_tool(self, server: str, tool: str, arguments: dict[str, Any]) -> Any:
        """调用指定服务器的-tool"""
        if server not in self._clients:
            raise ValueError(f"Unknown MCP server: {server}")
        return await self._clients[server].call_tool(server, tool, arguments)

    async def list_all_tools(self) -> dict[str, list[dict[str, Any]]]:
        """列出所有服务器的工具"""
        result = {}
        for name, client in self._clients.items():
            try:
                result[name] = await client.list_tools()
            except Exception as e:
                result[name] = [{"error": str(e)}]
        return result

    async def close(self) -> None:
        """关闭所有 MCP 服务器连接"""
        for client in self._clients.values():
            await client.close()
        self._clients.clear()


# 便捷函数：从配置创建客户端
async def create_mcp_client_from_config(mcp_servers: list[dict[str, Any]]) -> MultiMCPClient:
    """从配置字典创建 MCP 客户端"""
    client = MultiMCPClient()
    
    for server_config in mcp_servers:
        name = server_config.get("name", "")
        command = server_config.get("command", [])
        env = server_config.get("env", {})
        args = server_config.get("args", [])
        
        if not name or not command:
            continue
            
        config = MCPServerConfig(
            name=name,
            command=command,
            env=env,
            args=args,
        )
        client.add_server(config)
    
    return client


if __name__ == "__main__":
    import sys
    
    async def test_client():
        # 测试：列出所有可用工具
        config = MCPServerConfig(
            name="weather",
            command=["npx", "-y", "@modelcontextprotocol/server-weather"],
        )
        client = StdioMCPClient(config)
        
        try:
            tools = await client.list_tools()
            print("Available tools:")
            for tool in tools:
                print(f"  - {tool.get('name')}: {tool.get('description', '')}")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            await client.close()
    
    asyncio.run(test_client())
