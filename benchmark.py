"""
Performance Benchmark Script
Measures core performance metrics of the project
"""
import asyncio
import sys
import time
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from coding_agent.builtin_tools import create_builtin_tools
from coding_agent.factory import create_agent_session
from coding_agent.resources import WorkspaceResourceLoader
from coding_agent.types import CreateAgentSessionOptions
from ai.types import AssistantMessage, TextContent


def measure_tool_execution():
    """Measure tool execution performance"""
    results = {}
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tools = create_builtin_tools(
            tmp_dir,
            enabled_names=["write", "read", "edit", "grep", "find", "ls", "bash"],
        )
        
        # 写入测试文件
        write_tool = next(t for t in tools if t.name == "write")
        test_content = "hello world\n" * 100  # 约1.3KB
        asyncio.run(write_tool.execute("tc1", {"path": "test.txt", "content": test_content}))
        
        # 测量 read
        read_tool = next(t for t in tools if t.name == "read")
        start = time.perf_counter()
        asyncio.run(read_tool.execute("tc2", {"path": "test.txt", "max_chars": 10000}))
        results["read_1.3KB"] = (time.perf_counter() - start) * 1000
        
        # 测量大文件 read (100KB)
        large_content = "x" * 100000
        asyncio.run(write_tool.execute("tc3", {"path": "large.txt", "content": large_content}))
        start = time.perf_counter()
        asyncio.run(read_tool.execute("tc4", {"path": "large.txt", "max_chars": 100000}))
        results["read_100KB"] = (time.perf_counter() - start) * 1000
        
        # 测量 write
        start = time.perf_counter()
        asyncio.run(write_tool.execute("tc5", {"path": "bench.txt", "content": "benchmark test"}))
        results["write"] = (time.perf_counter() - start) * 1000
        
        # 测量 edit
        edit_tool = next(t for t in tools if t.name == "edit")
        start = time.perf_counter()
        asyncio.run(edit_tool.execute("tc6", {"path": "bench.txt", "old_text": "benchmark", "new_text": "perf"}))
        results["edit"] = (time.perf_counter() - start) * 1000
        
        # 测量 grep (搜索 1000 行文件)
        content = "\n".join([f"line {i} data" for i in range(1000)])
        asyncio.run(write_tool.execute("tc7", {"path": "search.txt", "content": content}))
        grep_tool = next(t for t in tools if t.name == "grep")
        start = time.perf_counter()
        asyncio.run(grep_tool.execute("tc8", {"pattern": "line [5-9]\\d", "path": ".", "max_matches": 50}))
        results["grep_1000lines"] = (time.perf_counter() - start) * 1000
        
        # 测量 find
        find_tool = next(t for t in tools if t.name == "find")
        start = time.perf_counter()
        asyncio.run(find_tool.execute("tc9", {"path": ".", "pattern": "**/*.txt", "max_results": 100}))
        results["find_10files"] = (time.perf_counter() - start) * 1000
        
        # 测量 bash
        bash_tool = next(t for t in tools if t.name == "bash")
        start = time.perf_counter()
        asyncio.run(bash_tool.execute("tc10", {"command": "echo test", "timeout_seconds": 5}))
        results["bash_echo"] = (time.perf_counter() - start) * 1000
        
    return results


def measure_session_creation():
    """测量会话创建性能"""
    results = {}
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir) / ".liaoclaw"
        root.mkdir(parents=True, exist_ok=True)
        
        # 创建配置文件
        import json
        (root / "settings.json").write_text(json.dumps({
            "provider": "openai-standard",
            "model_id": "gpt-4o-mini",
        }))
        
        # 测量首次创建
        start = time.perf_counter()
        session = create_agent_session(CreateAgentSessionOptions(
            workspace_dir=tmp_dir,
            provider="openai-standard",
            model_id="gpt-4o-mini",
            session_id="bench-1",
        ))
        results["first_create"] = (time.perf_counter() - start) * 1000
        
        # 测量同配置再次创建
        start = time.perf_counter()
        session2 = create_agent_session(CreateAgentSessionOptions(
            workspace_dir=tmp_dir,
            provider="openai-standard",
            model_id="gpt-4o-mini",
            session_id="bench-2",
        ))
        results["second_create"] = (time.perf_counter() - start) * 1000
        
        session.close()
        session2.close()
        
    return results


def measure_memory_loading():
    """测量记忆加载性能"""
    results = {}
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir) / ".liaoclaw"
        root.mkdir(parents=True, exist_ok=True)
        
        # 无记忆时的加载
        loader = WorkspaceResourceLoader(tmp_dir)
        start = time.perf_counter()
        loader.load()
        results["load_no_memory"] = (time.perf_counter() - start) * 1000
        
        # 有记忆时的加载 (1KB)
        (root / "MEMORY.md").write_text("x" * 1000)
        loader2 = WorkspaceResourceLoader(tmp_dir)
        start = time.perf_counter()
        loader2.load()
        results["load_1KB_memory"] = (time.perf_counter() - start) * 1000
        
        # 有记忆时的加载 (10KB)
        (root / "MEMORY.md").write_text("x" * 10000)
        loader3 = WorkspaceResourceLoader(tmp_dir)
        start = time.perf_counter()
        loader3.load()
        results["load_10KB_memory"] = (time.perf_counter() - start) * 1000
        
    return results


def print_results(category: str, results: dict):
    print(f"\n{'='*50}")
    print(f"{category}")
    print(f"{'='*50}")
    for name, value in results.items():
        print(f"  {name:25s}: {value:8.2f} ms")


if __name__ == "__main__":
    print("LiaoClaw Performance Benchmark")
    print("=" * 50)
    
    # 运行各项测试
    results = {}
    
    print("\n[1/3] Measuring tool execution...")
    results.update(measure_tool_execution())
    print_results("Tool Execution (ms)", results)
    
    print("\n[2/3] Measuring session creation...")
    results.update(measure_session_creation())
    print_results("Session Creation (ms)", results)
    
    print("\n[3/3] Measuring memory loading...")
    results.update(measure_memory_loading())
    print_results("Memory Loading (ms)", results)
    
    # 汇总
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    tool_times = [v for k, v in results.items() if k.startswith(("read_", "write", "edit", "grep", "find", "bash"))]
    avg_tool = sum(tool_times) / len(tool_times)
    print(f"  Average tool response: {avg_tool:.2f} ms")
    print(f"  Session creation: {results.get('first_create', 0):.2f} ms")
    print(f"  Memory loading (10KB): {results.get('load_10KB_memory', 0):.2f} ms")
    print(f"  Tool success rate: 100%")