"""
Debug test - list all available MCP tools
"""
import asyncio
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai.mcp_client import MCPServerConfig, create_mcp_client_from_config


async def main():
    config = json.loads(Path(".Liaoclaw/settings.json").read_text(encoding="utf-8"))
    mcp_servers = config.get("mcp_servers", [])
    
    client = await create_mcp_client_from_config(mcp_servers)
    
    # List all tools from all servers
    all_tools = await client.list_all_tools()
    
    print("=" * 60)
    print("All available MCP tools:")
    print("=" * 60)
    
    for server_name, tools in all_tools.items():
        print(f"\n[{server_name}]")
        if isinstance(tools, list):
            for t in tools:
                name = t.get("name", "unknown")
                desc = t.get("description", "")[:50]
                print(f"  - {name}: {desc}...")
        else:
            print(f"  Error: {tools}")
    
    print("\n" + "=" * 60)
    print("Testing get_file_contents:")
    result = await client.call_tool(
        server="github",
        tool="get_file_contents",
        arguments={"owner": "kyletser", "repo": "LiaoClaw", "path": "README.md"}
    )
    print(f"Success! File size: {len(str(result))} bytes")
    
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())