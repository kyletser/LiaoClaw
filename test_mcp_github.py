"""
Debug test for GitHub MCP
"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai.mcp_client import MCPServerConfig, StdioMCPClient


async def test_github():
    # GitHub MCP config - use full path
    config = MCPServerConfig(
        name="github",
        command=["node", "C:\\Users\\廖晓平\\AppData\\Roaming\\npm\\node_modules\\@modelcontextprotocol\\server-github\\dist\\index.js"],
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": "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"},
    )
    
    client = StdioMCPClient(config)
    
    try:
        print("[1] Listing tools...")
        tools = await client.list_tools()
        print(f"Tools: {len(tools)}")
        for t in tools[:5]:
            print(f"  - {t.get('name')}")
        
        print("\n[2] Trying get_file_contents...")
        result = await client.call_tool(
            server="github",
            tool="get_file_contents",
            arguments={"owner": "kyletser", "repo": "LiaoClaw", "path": "README.md"}
        )
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.close()
        print("\n[Done]")


if __name__ == "__main__":
    asyncio.run(test_github())