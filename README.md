# LiaoClaw

> 一个可运行的编程 Agent 项目：统一 LLM 接口 + 工具调用编排 + Web/CLI/IM 多入口。

## 项目运行（先看这个）

### Windows（推荐）

在项目根目录执行：

```powershell
py -3.10 -m venv .venv
if ($LASTEXITCODE -ne 0) { python -m venv .venv }

.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

if (!(Test-Path .env.ps1)) {
  Copy-Item .env.ps1.example .env.ps1
}
```

在 `.env.ps1` 至少配置一个可用 Key（推荐先用 Anthropic 通道）：

```powershell
$env:ANTHROPIC_API_KEY = "你的key"
$env:LIAOCLAW_PROVIDER = "anthropic"
$env:LIAOCLAW_MODEL_ID = "glm-4.5-air"
```

启动方式：

1. Web（建议显式带模型参数）  
`liaoclaw-web --provider anthropic --model-id glm-4.5-air --host 127.0.0.1 --port 8787`
2. CLI（交互式）  
`.\dev.ps1 -Mode cli`
3. IM 长连接（飞书）  
`.\dev.ps1 -Mode im -Transport longconn`

### macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

启动：

1. Web  
`liaoclaw-web --provider anthropic --model-id glm-4.5-air --host 127.0.0.1 --port 8787`
2. CLI  
`chmod +x ./dev.sh && ./dev.sh --mode cli`
3. IM 长连接  
`chmod +x ./dev.sh && ./dev.sh --mode im --transport longconn`

## 核心功能（重点）

1. 统一模型调用层  
同一套消息/工具协议，兼容 Anthropic Messages 与 OpenAI-Compatible，不用为每家 API 重写业务逻辑。

2. 可执行的 Agent 循环  
模型不仅“回答”，还能在一次任务中多轮执行工具（读文件、写文件、命令行），再把结果继续喂回模型推理。

3. Web/CLI/IM 三种入口  
同一个 Agent 内核，支持浏览器、终端、飞书三种交互方式，适合从本地调试到团队协作。

4. 会话与分支能力  
支持会话持久化、分叉、切换叶子节点，适合做“多方案并行尝试”。

5. Skills 与 MCP 扩展  
可通过 Skills 和 MCP 工具代理扩展能力，不需要改内核即可接入新工具。

## 这个项目适合什么场景

1. 想快速搭一套“可运行”的编程助手雏形。  
2. 想学习 Agent 项目如何把“模型调用、工具执行、会话持久化”串起来。  
3. 想把一个 Agent 同时接入 CLI、Web、IM，复用同一套后端能力。

## 常用命令

```powershell
# 运行测试
pytest tests/ -v

# 仅启动 Web
liaoclaw-web --provider anthropic --model-id glm-4.5-air --host 127.0.0.1 --port 8787

# 仅启动 CLI
python -m coding_agent --mode interactive --provider anthropic --model-id glm-4.5-air
```

## 项目结构

```text
src/
  ai/             # 统一 LLM 接口与 provider 实现
  agent_core/     # Agent 循环与工具协议
  coding_agent/   # CLI/Web 编程 Agent 应用层
  im/             # 飞书 IM 桥接
examples/         # 各层示例
tests/            # 单测
```

## 参考文档

1. 项目运行手册：`项目运行.md`
2. 飞书接入参考：https://www.runoob.com/ai-agent/openclaw-feishu.html
3. OpenClaw 教程参考：https://www.runoob.com/ai-agent/openclaw-clawdbot-tutorial.html
