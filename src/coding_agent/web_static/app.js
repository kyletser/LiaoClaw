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
  historySessionSelect: document.getElementById("historySessionSelect"),
  openHistorySessionBtn: document.getElementById("openHistorySessionBtn"),
  reloadSessionsBtn: document.getElementById("reloadSessionsBtn"),
  msgTpl: document.getElementById("msgTpl"),
};

const MAX_RENDER_MESSAGES = 200;
let eventStream = null;

function formatTimestamp(value) {
  let d = null;
  if (typeof value === "number") {
    d = new Date(value);
  } else if (typeof value === "string" && value.trim()) {
    d = new Date(value);
  }
  if (!d || Number.isNaN(d.getTime())) {
    d = new Date();
  }
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(
    d.getMinutes()
  )}:${pad(d.getSeconds())}`;
}

function formatMessageWithTimestamp(text, timestamp) {
  const body = String(text || "").trim();
  return `${body}\n\n[${formatTimestamp(timestamp)}]`;
}

function hasRenderableText(text) {
  return String(text || "").trim().length > 0;
}

function smoothScrollToBottom() {
  requestAnimationFrame(() => {
    els.chatLog.scrollTop = els.chatLog.scrollHeight;
  });
}

function addMessage(role, text, options = {}) {
  if (!hasRenderableText(text)) {
    return;
  }
  const node = els.msgTpl.content.firstElementChild.cloneNode(true);
  const normalizedRole =
    role === "user" || role === "assistant" || role === "system" || role === "tool" ? role : "assistant";
  const className =
    normalizedRole === "user"
      ? "user"
      : normalizedRole === "system"
      ? "system"
      : normalizedRole === "tool"
      ? "tool"
      : "agent";
  node.classList.add(className);
  const roleLabel =
    normalizedRole === "user"
      ? "User"
      : normalizedRole === "system"
      ? "System"
      : normalizedRole === "tool"
      ? "Tool"
      : "Assistant";
  node.querySelector(".msg-role").textContent = roleLabel;
  node.querySelector(".msg-body").textContent = formatMessageWithTimestamp(text, options.timestamp);

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
    option.textContent = "No branches";
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

function renderSessions(sessions, currentSessionId) {
  els.historySessionSelect.innerHTML = "";
  if (!sessions || sessions.length === 0) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No sessions";
    els.historySessionSelect.appendChild(option);
    return;
  }

  for (const item of sessions) {
    const option = document.createElement("option");
    option.value = item.session_id;
    const mark = item.session_id === currentSessionId ? " *" : "";
    const updatedAt = item.updated_at ? formatTimestamp(item.updated_at) : "unknown";
    const count = Number(item.message_count || 0);
    option.textContent = `${item.session_id}${mark} | ${updatedAt} | ${count} msgs`;
    if (item.session_id === currentSessionId) {
      option.selected = true;
    }
    els.historySessionSelect.appendChild(option);
  }
}

function setBusy(busy) {
  els.sendBtn.disabled = busy;
  els.promptInput.disabled = busy;
  els.sendBtn.textContent = busy ? "Sending..." : "Send";
}

function normalizeHistoryRole(item) {
  const role = String(item?.role || "");
  if (role === "user" || role === "assistant") {
    return role;
  }
  if (role === "toolResult") {
    return "tool";
  }
  const text = String(item?.text || "").trim();
  if (
    text.startsWith("Tool started:") ||
    text.startsWith("Tool finished:") ||
    text.startsWith("Auto retry (") ||
    text === "Context compacted to avoid overflow."
  ) {
    return "tool";
  }
  return "system";
}

function normalizeHistoryText(item) {
  const role = String(item?.role || "");
  const text = String(item?.text || "");
  if (role === "toolResult") {
    return `[toolResult] ${text}`;
  }
  return text;
}

function renderMessages(messages) {
  els.chatLog.innerHTML = "";
  for (const item of messages || []) {
    const role = normalizeHistoryRole(item);
    const text = normalizeHistoryText(item);
    if (!hasRenderableText(text)) {
      continue;
    }
    addMessage(role, text, { timestamp: item.timestamp });
  }
}

function handleStreamEvent(payload) {
  if (!payload || typeof payload !== "object") {
    return;
  }
  const ts = payload.timestamp;
  const type = payload.type;
  if (type === "tool_execution_start") {
    const toolName = payload.toolName || "unknown";
    addMessage("tool", `Tool started: ${toolName}`, { timestamp: ts });
    return;
  }
  if (type === "tool_execution_end") {
    const toolName = payload.toolName || "unknown";
    addMessage("tool", `Tool finished: ${toolName}`, { timestamp: ts });
    return;
  }
  if (type === "auto_retry_start") {
    const attempt = payload.attempt || "?";
    const maxAttempts = payload.max_attempts || "?";
    const delayMs = payload.delay_ms || 0;
    addMessage("tool", `Auto retry (${attempt}/${maxAttempts}), waiting ${delayMs}ms`, { timestamp: ts });
    return;
  }
  if (type === "context_compacted") {
    addMessage("tool", "Context compacted to avoid overflow.", { timestamp: ts });
  }
}

function connectEventStream() {
  if (eventStream) {
    eventStream.close();
  }
  eventStream = new EventSource("/api/events/stream");
  eventStream.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);
      handleStreamEvent(payload);
    } catch (_) {
      // ignore malformed stream event
    }
  };
  eventStream.onerror = () => {
    // EventSource auto-reconnects by default.
  };
}

async function refreshState() {
  const res = await fetch("/api/state");
  const data = await res.json();
  if (data.status !== "ok") {
    throw new Error(data.message || "Failed to load state");
  }
  els.sessionId.textContent = data.session_id || "-";
  els.leafId.textContent = `Leaf: ${data.leaf_id || "-"}`;
  els.messageCount.textContent = String(data.message_count || 0);
  const displayTokens = data.tokens?.display_total_tokens || 0;
  els.tokenCount.textContent = String(displayTokens);
  const source = data.tokens?.token_source === "usage" ? "usage" : "estimate";
  const actual = data.tokens?.actual_total_tokens || 0;
  const estimated = data.tokens?.estimated_total_tokens || 0;
  els.tokenMeta.textContent = `source=${source}, usage=${actual}, estimate=${estimated}`;
  return data;
}

async function loadEntries() {
  const res = await fetch("/api/session/entries");
  const data = await res.json();
  if (!res.ok || data.status !== "ok") {
    throw new Error(data.message || "Failed to load branches");
  }
  renderEntries(data.entries || [], data.leaf_id || "");
}

async function loadSessions() {
  const res = await fetch("/api/sessions");
  const data = await res.json();
  if (!res.ok || data.status !== "ok") {
    throw new Error(data.message || "Failed to load sessions");
  }
  renderSessions(data.sessions || [], data.current_session_id || "");
}

async function fetchMessages(limit = 120) {
  const messagesResp = await fetch(`/api/messages?limit=${limit}`);
  const messagesData = await messagesResp.json();
  if (!messagesResp.ok || messagesData.status !== "ok") {
    throw new Error(messagesData.message || "Failed to load messages");
  }
  return messagesData.messages || [];
}

async function postJson(url, payload = {}) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok || data.status !== "ok") {
    throw new Error(data.message || "Request failed");
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

  addMessage("user", text, { timestamp: Date.now() });
  els.promptInput.value = "";
  setBusy(true);
  try {
    const result = await sendPrompt(text);
    addMessage("assistant", result.reply || "Assistant returned no text", { timestamp: result.reply_timestamp });
    await refreshState();
    await loadEntries();
    await loadSessions();
  } catch (error) {
    addMessage("assistant", `Error: ${error.message}`, { timestamp: Date.now() });
  } finally {
    setBusy(false);
    els.promptInput.focus();
  }
});

els.refreshBtn.addEventListener("click", async () => {
  try {
    await refreshState();
    await loadEntries();
    await loadSessions();
  } catch (error) {
    addMessage("assistant", `Error: ${error.message}`, { timestamp: Date.now() });
  }
});

els.continueBtn.addEventListener("click", async () => {
  try {
    setBusy(true);
    const result = await postJson("/api/continue");
    addMessage("assistant", result.reply || "Assistant returned no text", { timestamp: result.reply_timestamp });
    await refreshState();
    await loadEntries();
    await loadSessions();
  } catch (error) {
    addMessage("assistant", `Error: ${error.message}`, { timestamp: Date.now() });
  } finally {
    setBusy(false);
  }
});

els.newSessionBtn.addEventListener("click", async () => {
  try {
    await postJson("/api/session/new");
    els.chatLog.innerHTML = "";
    addMessage("assistant", "New session created. Context cleared.", { timestamp: Date.now() });
    await refreshState();
    await loadEntries();
    await loadSessions();
  } catch (error) {
    addMessage("assistant", `Error: ${error.message}`, { timestamp: Date.now() });
  }
});

els.forkBtn.addEventListener("click", async () => {
  try {
    const entryId = els.entrySelect.value || undefined;
    await postJson("/api/session/fork", entryId ? { entry_id: entryId } : {});
    addMessage("assistant", "Session forked from current branch.", { timestamp: Date.now() });
    await refreshState();
    await loadEntries();
    await loadSessions();
  } catch (error) {
    addMessage("assistant", `Error: ${error.message}`, { timestamp: Date.now() });
  }
});

els.switchEntryBtn.addEventListener("click", async () => {
  const entryId = els.entrySelect.value;
  if (!entryId) {
    addMessage("assistant", "Please select a branch first.", { timestamp: Date.now() });
    return;
  }
  try {
    await postJson("/api/session/switch", { entry_id: entryId });
    const messages = await fetchMessages(200);
    renderMessages(messages);
    await refreshState();
    await loadEntries();
    await loadSessions();
  } catch (error) {
    addMessage("assistant", `Error: ${error.message}`, { timestamp: Date.now() });
  }
});

els.reloadEntriesBtn.addEventListener("click", async () => {
  try {
    await loadEntries();
  } catch (error) {
    addMessage("assistant", `Error: ${error.message}`, { timestamp: Date.now() });
  }
});

els.openHistorySessionBtn.addEventListener("click", async () => {
  const sessionId = els.historySessionSelect.value;
  if (!sessionId) {
    addMessage("assistant", "Please select a session first.", { timestamp: Date.now() });
    return;
  }
  try {
    await postJson("/api/session/open", { session_id: sessionId });
    const messages = await fetchMessages(200);
    renderMessages(messages);
    await refreshState();
    await loadEntries();
    await loadSessions();
  } catch (error) {
    addMessage("assistant", `Error: ${error.message}`, { timestamp: Date.now() });
  }
});

els.reloadSessionsBtn.addEventListener("click", async () => {
  try {
    await loadSessions();
  } catch (error) {
    addMessage("assistant", `Error: ${error.message}`, { timestamp: Date.now() });
  }
});

window.addEventListener("beforeunload", () => {
  if (eventStream) {
    eventStream.close();
  }
});

(async function init() {
  addMessage("assistant", "Web console is ready. You can chat now.", { timestamp: Date.now() });
  try {
    connectEventStream();
    await refreshState();
    await loadEntries();
    await loadSessions();
  } catch (error) {
    addMessage("assistant", `Error: ${error.message}`, { timestamp: Date.now() });
  }
})();
