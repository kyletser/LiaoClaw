const els = {
  sessionId: document.getElementById("sessionId"),
  leafId: document.getElementById("leafId"),
  messageCount: document.getElementById("messageCount"),
  tokenCount: document.getElementById("tokenCount"),
  tokenMeta: document.getElementById("tokenMeta"),
  chatLog: document.getElementById("chatLog"),
  chatForm: document.getElementById("chatForm"),
  promptInput: document.getElementById("promptInput"),
  sendBtn: document.getElementById("sendBtn"),
  refreshBtn: document.getElementById("refreshBtn"),
  continueBtn: document.getElementById("continueBtn"),
  newSessionBtn: document.getElementById("newSessionBtn"),
  forkBtn: document.getElementById("forkBtn"),
  entrySelect: document.getElementById("entrySelect"),
  switchEntryBtn: document.getElementById("switchEntryBtn"),
  reloadEntriesBtn: document.getElementById("reloadEntriesBtn"),
  msgTpl: document.getElementById("msgTpl"),
};

const MAX_RENDER_MESSAGES = 200;

function smoothScrollToBottom() {
  requestAnimationFrame(() => {
    els.chatLog.scrollTop = els.chatLog.scrollHeight;
  });
}

function addMessage(role, text) {
  const node = els.msgTpl.content.firstElementChild.cloneNode(true);
  node.classList.add(role === "user" ? "user" : "agent");
  node.querySelector(".msg-role").textContent = role === "user" ? "用户" : "助手";
  node.querySelector(".msg-body").textContent = text || "（空内容）";
  els.chatLog.appendChild(node);
  while (els.chatLog.children.length > MAX_RENDER_MESSAGES) {
    els.chatLog.removeChild(els.chatLog.firstElementChild);
  }
  smoothScrollToBottom();
}

function renderEntries(entries, leafId) {
  els.entrySelect.innerHTML = "";
  if (!entries || entries.length === 0) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "暂无分支";
    els.entrySelect.appendChild(option);
    return;
  }

  for (const item of entries) {
    const option = document.createElement("option");
    option.value = item.id;
    const indent = "  ".repeat(Math.max(0, item.depth || 0));
    const mark = item.is_leaf ? " *" : "";
    option.textContent = `${indent}${item.id}${mark}`;
    if (item.id === leafId) {
      option.selected = true;
    }
    els.entrySelect.appendChild(option);
  }
}

function setBusy(busy) {
  els.sendBtn.disabled = busy;
  els.promptInput.disabled = busy;
  els.sendBtn.textContent = busy ? "发送中..." : "发送";
}

async function refreshState() {
  const res = await fetch("/api/state");
  const data = await res.json();
  if (data.status !== "ok") {
    throw new Error(data.message || "读取状态失败");
  }
  els.sessionId.textContent = data.session_id || "-";
  els.leafId.textContent = `分支: ${data.leaf_id || "-"}`;
  els.messageCount.textContent = String(data.message_count || 0);
  const displayTokens = data.tokens?.display_total_tokens || 0;
  els.tokenCount.textContent = String(displayTokens);
  const source = data.tokens?.token_source === "usage" ? "模型统计" : "估算";
  const actual = data.tokens?.actual_total_tokens || 0;
  const estimated = data.tokens?.estimated_total_tokens || 0;
  els.tokenMeta.textContent = `当前会话累计消耗（来源: ${source}，usage=${actual}，estimate=${estimated}）`;
}

async function loadEntries() {
  const res = await fetch("/api/session/entries");
  const data = await res.json();
  if (!res.ok || data.status !== "ok") {
    throw new Error(data.message || "读取分支失败");
  }
  renderEntries(data.entries || [], data.leaf_id || "");
}

async function postJson(url, payload = {}) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok || data.status !== "ok") {
    throw new Error(data.message || "请求失败");
  }
  return data;
}

async function sendPrompt(text) {
  return postJson("/api/prompt", { text });
}

els.chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = els.promptInput.value.trim();
  if (!text) {
    return;
  }

  addMessage("user", text);
  els.promptInput.value = "";
  setBusy(true);
  try {
    const result = await sendPrompt(text);
    addMessage("assistant", result.reply || "助手没有返回可展示内容");
    await refreshState();
    await loadEntries();
  } catch (error) {
    addMessage("assistant", `错误：${error.message}`);
  } finally {
    setBusy(false);
    els.promptInput.focus();
  }
});

els.refreshBtn.addEventListener("click", async () => {
  try {
    await refreshState();
    await loadEntries();
  } catch (error) {
    addMessage("assistant", `错误：${error.message}`);
  }
});

els.continueBtn.addEventListener("click", async () => {
  try {
    setBusy(true);
    const result = await postJson("/api/continue");
    addMessage("assistant", result.reply || "助手没有返回可展示内容");
    await refreshState();
    await loadEntries();
  } catch (error) {
    addMessage("assistant", `错误：${error.message}`);
  } finally {
    setBusy(false);
  }
});

els.newSessionBtn.addEventListener("click", async () => {
  try {
    await postJson("/api/session/new");
    els.chatLog.innerHTML = "";
    addMessage("assistant", "已创建新会话，历史上下文已清空。");
    await refreshState();
    await loadEntries();
  } catch (error) {
    addMessage("assistant", `错误：${error.message}`);
  }
});

els.forkBtn.addEventListener("click", async () => {
  try {
    const entryId = els.entrySelect.value || undefined;
    await postJson("/api/session/fork", entryId ? { entry_id: entryId } : {});
    addMessage("assistant", "已基于当前分支创建新会话。" );
    await refreshState();
    await loadEntries();
  } catch (error) {
    addMessage("assistant", `错误：${error.message}`);
  }
});

els.switchEntryBtn.addEventListener("click", async () => {
  const entryId = els.entrySelect.value;
  if (!entryId) {
    addMessage("assistant", "请先选择一个分支。" );
    return;
  }
  try {
    await postJson("/api/session/switch", { entry_id: entryId });
    const messagesResp = await fetch("/api/messages?limit=120");
    const messagesData = await messagesResp.json();
    if (!messagesResp.ok || messagesData.status !== "ok") {
      throw new Error(messagesData.message || "读取消息失败");
    }
    els.chatLog.innerHTML = "";
    for (const item of messagesData.messages || []) {
      if (item.role === "user" || item.role === "assistant") {
        addMessage(item.role, item.text);
      }
    }
    await refreshState();
    await loadEntries();
  } catch (error) {
    addMessage("assistant", `错误：${error.message}`);
  }
});

els.reloadEntriesBtn.addEventListener("click", async () => {
  try {
    await loadEntries();
  } catch (error) {
    addMessage("assistant", `错误：${error.message}`);
  }
});

(async function init() {
  addMessage("assistant", "网页控制台已就绪，可以开始提问。\n支持继续生成、切换分支、分叉会话和新建会话。" );
  try {
    await refreshState();
    await loadEntries();
  } catch (error) {
    addMessage("assistant", `错误：${error.message}`);
  }
})();
