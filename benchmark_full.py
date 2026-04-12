"""
完整性能测试脚本
测试 LLM API、并发、MCP、流式响应等
"""
import asyncio
import sys
import time
import os
import json
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

ROOT = Path(__file__).resolve().parents[0]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# 设置 API Key
if not os.getenv("ANTHROPIC_API_KEY"):
    raise RuntimeError("Missing ANTHROPIC_API_KEY environment variable")


def test_llm_api_latency():
    """测试 LLM API 延迟"""
    print("\n" + "="*50)
    print("测试 LLM API 延迟")
    print("="*50)
    
    from coding_agent import create_agent_session
    from coding_agent.types import CreateAgentSessionOptions
    
    latencies = []
    
    # 测试 3 次
    for i in range(3):
        options = CreateAgentSessionOptions(
            workspace_dir=".",
            provider="anthropic",
            model_id="glm-4.5-air",
            session_id=f"llm_test_{i}",
        )
        session = create_agent_session(options)
        
        start = time.perf_counter()
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(session.prompt("你好"))
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)
            print(f"  第 {i+1} 次: {latency:.2f} ms")
        except Exception as e:
            print(f"  第 {i+1} 次失败: {e}")
        finally:
            loop.close()
            session.close()
    
    if latencies:
        print(f"\n  平均: {statistics.mean(latencies):.2f} ms")
        print(f"  最小: {min(latencies):.2f} ms")
        print(f"  最大: {max(latencies):.2f} ms")
    
    return latencies


def test_concurrent_requests():
    """测试并发请求能力"""
    print("\n" + "="*50)
    print("测试并发请求能力")
    print("="*50)
    
    from coding_agent import create_agent_session
    from coding_agent.types import CreateAgentSessionOptions
    
    def single_request(n):
        options = CreateAgentSessionOptions(
            workspace_dir=".",
            provider="anthropic",
            model_id="glm-4.5-air",
            session_id=f"concurrent_{n}",
        )
        session = create_agent_session(options)
        
        start = time.perf_counter()
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(session.prompt("Hello"))
            latency = (time.perf_counter() - start) * 1000
            return latency
        except Exception as e:
            return None
        finally:
            loop.close()
            session.close()
    
    # 测试 5 个并发请求
    print("  测试 5 个并发请求...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(single_request, i) for i in range(5)]
        results = [f.result() for f in as_completed(futures)]
    
    successful = [r for r in results if r is not None]
    print(f"  成功: {len(successful)}/5")
    if successful:
        print(f"  平均延迟: {statistics.mean(successful):.2f} ms")
    
    # 测试 10 个并发请求
    print("\n  测试 10 个并发请求...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(single_request, i) for i in range(10)]
        results = [f.result() for f in as_completed(futures)]
    
    successful = [r for r in results if r is not None]
    print(f"  成功: {len(successful)}/10")
    if successful:
        print(f"  平均延迟: {statistics.mean(successful):.2f} ms")
    
    return successful


def test_mcp_latency():
    """测试 MCP 工具延迟"""
    print("\n" + "="*50)
    print("测试 MCP 工具延迟")
    print("="*50)
    
    from ai.mcp_client import MCPServerConfig, StdioMCPClient
    
    latencies = []
    
    # 测试天气 MCP
    config = MCPServerConfig(
        name="weather",
        command=["python"],
        args=[str(ROOT / "mcp_servers" / "weather_server.py")],
    )
    client = StdioMCPClient(config)
    
    try:
        # 预热
        print("  预热...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(client.list_tools())
        
        # 测试 3 次
        for i in range(3):
            start = time.perf_counter()
            loop.run_until_complete(client.call_tool(
                server="weather",
                tool="get_weather",
                arguments={"city": "Beijing"}
            ))
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)
            print(f"  第 {i+1} 次: {latency:.2f} ms")
        
        if latencies:
            print(f"\n  平均: {statistics.mean(latencies):.2f} ms")
            
    except Exception as e:
        print(f"  失败: {e}")
    finally:
        try:
            loop.run_until_complete(client.close())
        except:
            pass
        loop.close()
    
    return latencies


def test_stream_response():
    """测试流式响应延迟"""
    print("\n" + "="*50)
    print("测试流式响应延迟")
    print("="*50)
    
    from coding_agent import create_agent_session
    from coding_agent.types import CreateAgentSessionOptions
    
    times = {
        "first_token": [],  # 首个字符出现时间
        "full_response": [],  # 完整响应时间
    }
    
    for i in range(3):
        options = CreateAgentSessionOptions(
            workspace_dir=".",
            provider="anthropic",
            model_id="glm-4.5-air",
            session_id=f"stream_test_{i}",
        )
        session = create_agent_session(options)
        
        first_token_time = None
        full_time = None
        
        def on_event(event):
            nonlocal first_token_time, full_time
            t = event.get("type", "")
            if t == "message_update":
                assistant_event = event.get("assistantMessageEvent") or {}
                if assistant_event.get("type") == "text_delta":
                    delta = assistant_event.get("delta", "")
                    if delta and first_token_time is None:
                        first_token_time = time.perf_counter()
        
        unsub = session.subscribe(on_event)
        
        start = time.perf_counter()
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(session.prompt("写一首诗"))
            full_time = time.perf_counter() - start
            
            times["first_token"].append(first_token_time * 1000 if first_token_time else 0)
            times["full_response"].append(full_time * 1000)
            
            print(f"  第 {i+1} 次 - 首字: {(first_token_time - start)*1000:.2f}ms, 完整: {full_time*1000:.2f}ms")
        except Exception as e:
            print(f"  第 {i+1} 次失败: {e}")
        finally:
            unsub()
            loop.close()
            session.close()
    
    if times["first_token"]:
        print(f"\n  平均首字延迟: {statistics.mean(times['first_token']):.2f} ms")
        print(f"  平均完整响应: {statistics.mean(times['full_response']):.2f} ms")
    
    return times


def test_memory_persistence():
    """测试记忆持久化"""
    print("\n" + "="*50)
    print("测试记忆持久化")
    print("="*50)
    
    from coding_agent import create_agent_session
    from coding_agent.types import CreateAgentSessionOptions
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 创建会话并写入记忆
        options = CreateAgentSessionOptions(
            workspace_dir=tmp_dir,
            provider="anthropic",
            model_id="glm-4.5-air",
            session_id="memory_test",
        )
        session = create_agent_session(options)
        
        # 写入记忆
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(session.prompt("记住我的名字叫测试"))
        session.close()
        
        # 重建会话，恢复记忆
        start = time.perf_counter()
        options2 = CreateAgentSessionOptions(
            workspace_dir=tmp_dir,
            provider="anthropic",
            model_id="glm-4.5-air",
            session_id="memory_test",
        )
        session2 = create_agent_session(options2)
        
        # 检查是否恢复了之前的消息
        msg_count = len(session2.messages)
        load_time = (time.perf_counter() - start) * 1000
        
        print(f"  恢复消息数: {msg_count}")
        print(f"  加载时间: {load_time:.2f} ms")
        
        loop.close()
        session2.close()
        
        return {"msg_count": msg_count, "load_time": load_time}


if __name__ == "__main__":
    print("="*50)
    print("LiaoClaw 完整性能测试")
    print("="*50)
    
    all_results = {}
    
    # 1. LLM API 延迟
    all_results["llm"] = test_llm_api_latency()
    
    # 2. 并发能力
    all_results["concurrent"] = test_concurrent_requests()
    
    # 3. MCP 延迟
    all_results["mcp"] = test_mcp_latency()
    
    # 4. 流式响应
    all_results["stream"] = test_stream_response()
    
    # 5. 记忆持久化
    all_results["persistence"] = test_memory_persistence()
    
    # 汇总
    print("\n" + "="*50)
    print("测试汇总")
    print("="*50)
    print(f"  LLM 平均延迟: {statistics.mean(all_results['llm']):.2f} ms" if all_results.get('llm') else "  LLM: 失败")
    print(f"  MCP 平均延迟: {statistics.mean(all_results['mcp']):.2f} ms" if all_results.get('mcp') else "  MCP: 失败")
    print(f"  流式首字延迟: {statistics.mean(all_results['stream']['first_token']):.2f} ms" if all_results.get('stream', {}).get('first_token') else "  流式: 失败")
    print(f"  记忆恢复时间: {all_results['persistence']['load_time']:.2f} ms" if all_results.get('persistence') else "  记忆: 失败")