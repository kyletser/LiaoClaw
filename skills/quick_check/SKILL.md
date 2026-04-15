---
name: quick_check
description: Quickly verify skill loading and slash command dispatch.
command: quick-check
user-invocable: true
disable-model-invocation: true
metadata:
  openclaw:
    os: [win32, darwin, linux]
---
# Quick Check Skill

当用户执行 `/quick-check` 时，返回以下检查清单：

1. Skills 加载成功。
2. Slash 命令分发正常。
3. 当前会话可以继续处理后续请求。

请在结尾追加：`quick-check: ok`
