const state = {
  sessionId: createSessionId(),
  file: null,
  fileUrl: "",
  isSending: false,
  sessions: [],
  messages: [],
  historyQuery: "",
};

const STORAGE_KEY = "medical-agent-chat-sessions";
const API_STORAGE_KEY = "medical-agent-api-base";

const els = {
  apiBase: document.querySelector("#apiBase"),
  appShell: document.querySelector("#appShell"),
  attachBtn: document.querySelector("#attachBtn"),
  clearHistoryBtn: document.querySelector("#clearHistoryBtn"),
  composer: document.querySelector("#composer"),
  connectionDot: document.querySelector("#connectionDot"),
  connectionState: document.querySelector("#connectionState"),
  conversation: document.querySelector("#conversation"),
  currentSessionTitle: document.querySelector("#currentSessionTitle"),
  fileInput: document.querySelector("#fileInput"),
  historyList: document.querySelector("#historyList"),
  historySearch: document.querySelector("#historySearch"),
  newSessionBtn: document.querySelector("#newSessionBtn"),
  pendingAttachment: document.querySelector("#pendingAttachment"),
  queryInput: document.querySelector("#queryInput"),
  sendBtn: document.querySelector("#sendBtn"),
  serverSettings: document.querySelector("#serverSettings"),
  sessionLabel: document.querySelector("#sessionLabel"),
  sidebarBackdrop: document.querySelector("#sidebarBackdrop"),
  sidebarCloseBtn: document.querySelector("#sidebarCloseBtn"),
  sidebarToggleBtn: document.querySelector("#sidebarToggleBtn"),
};

async function init() {
  loadApiBase();
  await Clinic.login();
  await loadSessions();
  renderSession();
  bindEvents();
  resizeTextarea();
  renderIcons();
  checkBackendHealth();
  document.querySelector("#tasksBtn").onclick = () => Clinic.tasks().catch(e => alert(e.message));
  document.querySelector("#logoutBtn").onclick = async () => { await Clinic.api("/api/auth/logout", {method: "POST"}); location.reload(); };
  document.querySelector("#demoBtn").onclick = async () => {
    if (state.isSending) return;
    try {
      const task = await Clinic.api("/api/demo", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({session_id: state.sessionId})});
      await window.openServerTask(task.task_id, state.sessionId);
    } catch (e) { alert(e.message); }
  };
  const active = (await Clinic.api("/api/tasks")).find(t => ["queued", "running"].includes(t.status));
  if (active) void window.openServerTask(active.id, active.session_id);
}

function bindEvents() {
  els.attachBtn.addEventListener("click", () => els.fileInput.click());

  els.fileInput.addEventListener("change", (event) => {
    const [file] = event.target.files;
    setPendingFile(file);
  });

  els.newSessionBtn.addEventListener("click", createNewSession);
  els.clearHistoryBtn.addEventListener("click", clearAllSessions);

  els.historySearch.addEventListener("input", (event) => {
    state.historyQuery = event.target.value.trim().toLowerCase();
    renderHistoryList();
  });

  els.apiBase.addEventListener("change", () => {
    els.apiBase.value = els.apiBase.value.trim().replace(/\/$/, "");
    localStorage.setItem(API_STORAGE_KEY, els.apiBase.value);
    checkBackendHealth();
  });

  els.sidebarToggleBtn.addEventListener("click", openSidebar);
  els.sidebarCloseBtn.addEventListener("click", closeSidebar);
  els.sidebarBackdrop.addEventListener("click", closeSidebar);

  els.queryInput.addEventListener("input", resizeTextarea);
  els.queryInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      els.composer.requestSubmit();
    }
  });

  els.composer.addEventListener("submit", async (event) => {
    event.preventDefault();
    await sendMessage();
  });

  ["dragenter", "dragover"].forEach((name) => {
    window.addEventListener(name, (event) => {
      event.preventDefault();
      els.composer.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((name) => {
    window.addEventListener(name, (event) => {
      event.preventDefault();
      els.composer.classList.remove("dragover");
    });
  });

  window.addEventListener("drop", (event) => {
    const [file] = event.dataTransfer.files;
    setPendingFile(file);
  });
}

function renderIcons() {
  if (window.lucide) window.lucide.createIcons();
}

function openSidebar() {
  if (window.matchMedia("(max-width: 800px)").matches) {
    els.appShell.classList.add("sidebar-open");
  } else {
    els.appShell.classList.remove("sidebar-collapsed");
  }
}

function closeSidebar() {
  if (window.matchMedia("(max-width: 800px)").matches) {
    els.appShell.classList.remove("sidebar-open");
  } else {
    els.appShell.classList.add("sidebar-collapsed");
  }
}

function createSessionId() {
  return `sess-${Math.random().toString(16).slice(2, 10)}`;
}

function makeSession() {
  const now = Date.now();
  return {
    id: createSessionId(),
    title: "新对话",
    createdAt: now,
    updatedAt: now,
    messages: [],
  };
}

async function loadSessions() {
  state.sessions = await Clinic.api("/api/sessions");
  if (!state.sessions.length) state.sessions = [makeSession()];
  state.sessionId = state.sessions[0].id;
  state.messages = state.sessions[0].messages;
}

function getCurrentSession() {
  let session = state.sessions.find((item) => item.id === state.sessionId);
  if (!session) {
    session = makeSession();
    state.sessions.unshift(session);
    state.sessionId = session.id;
    state.messages = session.messages;
  }
  return session;
}

function createNewSession() {
  if (state.isSending) return;
  clearPendingFile();

  const current = getCurrentSession();
  if (current.messages.length === 0) {
    renderSession();
    els.queryInput.focus();
    closeSidebarOnMobile();
    return;
  }

  const session = makeSession();
  state.sessions.unshift(session);
  state.sessionId = session.id;
  state.messages = session.messages;
  persistSessions();
  renderSession();
  closeSidebarOnMobile();
  els.queryInput.focus();
}

function switchSession(sessionId) {
  if (state.isSending || sessionId === state.sessionId) {
    closeSidebarOnMobile();
    return;
  }

  clearPendingFile();
  const session = state.sessions.find((item) => item.id === sessionId);
  if (!session) return;
  state.sessionId = session.id;
  state.messages = session.messages;
  renderSession();
  closeSidebarOnMobile();
}

function deleteSession(sessionId) {
  if (state.isSending && sessionId === state.sessionId) return;

  state.sessions = state.sessions.filter((session) => session.id !== sessionId);
  void deleteBackendSession(sessionId);

  if (state.sessions.length === 0) state.sessions = [makeSession()];
  if (state.sessionId === sessionId) {
    state.sessionId = state.sessions[0].id;
    state.messages = state.sessions[0].messages;
    clearPendingFile();
  }

  persistSessions();
  renderSession();
}

function clearAllSessions() {
  if (state.isSending) return;
  if (!window.confirm("清空全部历史对话？此操作无法撤销。")) return;

  const oldSessionIds = state.sessions.map((session) => session.id);
  state.sessions = [makeSession()];
  state.sessionId = state.sessions[0].id;
  state.messages = state.sessions[0].messages;
  state.historyQuery = "";
  els.historySearch.value = "";
  clearPendingFile();
  persistSessions();
  renderSession();
  oldSessionIds.forEach((sessionId) => void deleteBackendSession(sessionId));
}

async function deleteBackendSession(sessionId) {
  try {
    const endpoint = `${getApiBase()}/api/session/${encodeURIComponent(sessionId)}`;
    await fetch(endpoint, { method: "DELETE" });
  } catch {
    // Local history remains authoritative even if the backend is unavailable.
  }
}

function saveSessions() {
  const session = getCurrentSession();
  session.messages = state.messages.map(serializeMessage);
  session.updatedAt = Date.now();

  if (!session.title || session.title === "新会话" || session.title === "新对话") {
    const firstUser = session.messages.find((message) => message.role === "user");
    if (firstUser?.text) session.title = compact(stripMarkdown(firstUser.text), 28);
  }

  state.sessions.sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
  persistSessions();
  renderHistoryList();
  updateCurrentSessionHeader();
}

function persistSessions() {
  // Server task records persist accepted messages and results; no patient text in localStorage.
}

function serializeMessage(message) {
  const copy = JSON.parse(JSON.stringify(message));
  if (copy.attachment?.previewUrl?.startsWith("blob:")) delete copy.attachment.previewUrl;
  if (copy.result?.raw) delete copy.result.raw;
  return copy;
}

function renderHistoryList() {
  const sessions = state.sessions.filter(sessionMatchesSearch);
  els.historyList.innerHTML = "";

  if (sessions.length === 0) {
    els.historyList.innerHTML = '<div class="history-empty">没有找到相关对话</div>';
    return;
  }

  const groups = new Map();
  sessions.forEach((session) => {
    const groupName = getHistoryGroup(session.updatedAt);
    if (!groups.has(groupName)) groups.set(groupName, []);
    groups.get(groupName).push(session);
  });

  groups.forEach((groupSessions, groupName) => {
    const group = document.createElement("section");
    group.className = "history-group";
    group.innerHTML = `<div class="history-group-label">${escapeHtml(groupName)}</div>`;

    groupSessions.forEach((session) => {
      const item = document.createElement("div");
      item.className = `history-item ${session.id === state.sessionId ? "active" : ""}`;
      item.innerHTML = `
        <button class="history-select" type="button" aria-label="打开 ${escapeAttribute(session.title || "新对话")}">
          <span class="history-title">${escapeHtml(session.title || "新对话")}</span>
          <span class="history-preview">${escapeHtml(getSessionPreview(session))}</span>
        </button>
        <button class="history-delete" type="button" title="删除对话" aria-label="删除对话">
          <span data-lucide="trash-2"></span>
        </button>
      `;
      item.querySelector(".history-select").addEventListener("click", () => switchSession(session.id));
      item.querySelector(".history-delete").addEventListener("click", () => deleteSession(session.id));
      group.appendChild(item);
    });

    els.historyList.appendChild(group);
  });

  renderIcons();
}

function sessionMatchesSearch(session) {
  if (!state.historyQuery) return true;
  const messageText = session.messages.map((message) => {
    if (message.role === "user") return message.text || "";
    return message.result?.answer || "";
  }).join(" ");
  return `${session.title || ""} ${messageText}`.toLowerCase().includes(state.historyQuery);
}

function getSessionPreview(session) {
  const lastMessage = [...session.messages].reverse().find((message) => {
    if (message.role === "user") return Boolean(message.text);
    return Boolean(cleanInternalProtocol(message.result?.answer));
  });

  if (!lastMessage) return `${formatTime(session.updatedAt)} · 暂无消息`;
  const content = lastMessage.role === "user"
    ? lastMessage.text
    : cleanInternalProtocol(lastMessage.result?.answer);
  return `${formatTime(session.updatedAt)} · ${compact(stripMarkdown(content), 34)}`;
}

function getHistoryGroup(timestamp) {
  const date = new Date(timestamp || Date.now());
  const today = new Date();
  const dayStart = new Date(today.getFullYear(), today.getMonth(), today.getDate());
  const targetStart = new Date(date.getFullYear(), date.getMonth(), date.getDate());
  const days = Math.round((dayStart - targetStart) / 86400000);
  if (days <= 0) return "今天";
  if (days === 1) return "昨天";
  if (days < 7) return "最近 7 天";
  return "更早";
}

function renderSession() {
  clearConversation();
  updateCurrentSessionHeader();

  if (state.messages.length === 0) {
    renderEmptyState();
  } else {
    state.messages.forEach(renderStoredMessage);
  }

  renderHistoryList();
  scrollToBottom();
  renderIcons();
}

function updateCurrentSessionHeader() {
  const session = getCurrentSession();
  els.currentSessionTitle.textContent = session.title || "新对话";
  els.sessionLabel.textContent = `session ${state.sessionId}`;
}

function clearConversation() {
  els.conversation.innerHTML = "";
}

function renderEmptyState() {
  els.conversation.innerHTML = `
    <div class="empty-state">
      <div class="empty-mark">M</div>
      <h1>今天想分析什么？</h1>
      <p>Medical Agent · Supervisor</p>
    </div>
  `;
}

function renderStoredMessage(message) {
  if (message.role === "user") {
    addUserMessage(message.text, message.attachment, false);
  } else if (message.role === "assistant") {
    addAssistantMessage(message.result, false);
  }
}

function closeSidebarOnMobile() {
  if (window.matchMedia("(max-width: 800px)").matches) closeSidebar();
}

function resizeTextarea() {
  els.queryInput.style.height = "auto";
  els.queryInput.style.height = `${Math.min(els.queryInput.scrollHeight, 180)}px`;
}

function setPendingFile(file) {
  if (!file) return;
  clearPendingFile();
  state.file = file;
  state.fileUrl = URL.createObjectURL(file);
  renderPendingAttachment();
}

function clearPendingFile() {
  if (state.fileUrl) URL.revokeObjectURL(state.fileUrl);
  state.file = null;
  state.fileUrl = "";
  els.fileInput.value = "";
  els.pendingAttachment.hidden = true;
  els.pendingAttachment.innerHTML = "";
}

function renderPendingAttachment() {
  if (!state.file) return;
  els.pendingAttachment.hidden = false;
  els.pendingAttachment.innerHTML = `
    <div class="attachment-chip">
      ${renderAttachmentPreview(state.file, state.fileUrl)}
      <div class="attachment-text">
        <strong>${escapeHtml(state.file.name)}</strong>
        <span>${formatBytes(state.file.size)}</span>
      </div>
    </div>
    <button class="remove-attachment" type="button" title="移除附件" aria-label="移除附件">
      <span data-lucide="x">×</span>
    </button>
  `;
  els.pendingAttachment.querySelector(".remove-attachment").addEventListener("click", clearPendingFile);
  renderIcons();
}

async function sendMessage() {
  const query = els.queryInput.value.trim();
  if (state.isSending || (!query && !state.file)) return;

  const fileForRequest = state.file;
  const fileUrlForMessage = state.fileUrl;
  const userText = query || "请分析这个文件。";

  state.isSending = true;
  els.sendBtn.disabled = true;
  setConnectionState("正在分析", "pending");
  els.queryInput.value = "";
  resizeTextarea();

  const attachment = fileForRequest ? {
    name: fileForRequest.name,
    size: fileForRequest.size,
    previewUrl: fileUrlForMessage,
    isImage: isImageFile(fileForRequest.name),
  } : null;

  addUserMessage(userText, attachment, true);
  const assistantEl = addAssistantLoadingMessage();
  const stopProgress = () => {};
  const request = buildRequest(userText, fileForRequest);
  clearPendingFile();

  try {
    const response = await postToBackend(request, fileForRequest, assistantEl);
    if (attachment && response.input_file?.url) {
      attachment.previewUrl = absolutizeUrl(response.input_file.url);
    }
    const normalized = normalizeBackendResponse(response);
    stopProgress();
    updateAssistantMessage(assistantEl, normalized, true);
    setConnectionState("已连接", "connected");
  } catch (error) {
    stopProgress();
    const fallback = buildErrorResponse(error, request);
    updateAssistantMessage(assistantEl, fallback, true);
    setConnectionState("连接失败", "error");
    els.serverSettings.open = true;
  } finally {
    state.isSending = false;
    els.sendBtn.disabled = false;
    renderIcons();
  }
}

function buildRequest(query, file) {
  const body = {
    session_id: state.sessionId,
    graph: "supervisor",
    agent: "auto",
    query,
    task: inferTask(query, file),
    max_iterations: 12,
    show_tools: false,
  };

  if (file) {
    body.file_name = file.name;
    body.file_type = file.type || "application/octet-stream";
    body.file_size = file.size;
  }

  return {
    endpoint: `${getApiBase()}/api/tasks`,
    body,
  };
}

function inferTask(query, file) {
  const text = query.toLowerCase();
  if (text.includes("分割") || text.includes("segment")) return "segment";
  if (text.includes("姿态") || text.includes("pose")) return "pose";
  if (text.includes("分类") || text.includes("classify")) return "classify";
  if (file && !isImageFile(file.name)) return "detect";
  return "detect";
}

async function postToBackend({ endpoint, body }, file, article) {
  const formData = new FormData();
  formData.append("payload", JSON.stringify(body));
  if (file) formData.append("file", file);
  const task = await Clinic.api(endpoint, {method: "POST", body: formData});
  return Clinic.wait(task.task_id, article);
}

window.openServerTask = async function(taskId, sessionId) {
  if (state.isSending) { alert("请先等待或取消当前任务"); return; }
  await loadSessions();
  if (sessionId && state.sessions.some(s => s.id === sessionId)) {
    state.sessionId = sessionId;
    state.messages = state.sessions.find(s => s.id === sessionId).messages;
  }
  renderSession();
  const task = await Clinic.api(`/api/tasks/${taskId}`);
  if (task.status === "completed") return;
  state.isSending = true; els.sendBtn.disabled = true;
  const article = addAssistantLoadingMessage();
  try { updateAssistantMessage(article, normalizeBackendResponse(await Clinic.wait(taskId, article)), true); }
  catch (error) { updateAssistantMessage(article, normalizeBackendResponse({answer: error.message}), false); }
  finally { state.isSending = false; els.sendBtn.disabled = false; }
};

function addUserMessage(text, attachment, shouldPersist = true) {
  const article = document.createElement("article");
  article.className = "message user-message";
  article.innerHTML = `
    <div class="message-body">
      <div class="bubble markdown-body">
        ${renderMarkdown(text)}
        ${attachment ? `<div class="attachment-row">${renderAttachmentChip(attachment)}</div>` : ""}
      </div>
    </div>
  `;
  els.conversation.appendChild(article);

  if (shouldPersist) {
    state.messages.push({ role: "user", text, attachment });
    saveSessions();
  }

  scrollToBottom();
  renderIcons();
}

function addAssistantLoadingMessage() {
  const article = document.createElement("article");
  article.className = "message assistant-message";
  article.innerHTML = `
    <div class="avatar">AI</div>
    <div class="message-body">
      <div class="bubble">
        <div class="typing-dots" aria-label="正在分析"><i></i><i></i><i></i></div>
      </div>
      <div class="timeline-line"><span>正在提交任务</span></div>
    </div>
  `;
  els.conversation.appendChild(article);
  scrollToBottom();
  return article;
}

function addAssistantMessage(result, shouldPersist = true) {
  const article = addAssistantLoadingMessage();
  updateAssistantMessage(article, normalizeBackendResponse(result), shouldPersist);
}

function updateAssistantMessage(article, result, shouldPersist = false) {
  const body = article.querySelector(".message-body");
  body.innerHTML = `
    <div class="bubble markdown-body">
      ${renderAnswer(result.answer)}
      ${renderResultImages(result)}
      ${renderFileLinks(result.files)}
      ${renderToolDetails(result.tools)}
    </div>
    ${renderTimeline(result.timeline)}
    <div class="message-actions">
      <button class="message-action copy-message" type="button" title="复制回答" aria-label="复制回答">
        <span data-lucide="copy"></span>
      </button>
    </div>
  `;

  Clinic.bindResult(article, result);
  const copyButton = body.querySelector(".copy-message");
  copyButton.addEventListener("click", () => copyAnswer(copyButton, result.answer));

  if (shouldPersist) {
    state.messages.push({ role: "assistant", result });
    saveSessions();
  }

  scrollToBottom();
  renderIcons();
}

async function copyAnswer(button, answer) {
  const text = cleanInternalProtocol(answer);
  try {
    await navigator.clipboard.writeText(text);
    button.innerHTML = '<span data-lucide="check"></span>';
    button.title = "已复制";
    renderIcons();
    window.setTimeout(() => {
      button.innerHTML = '<span data-lucide="copy"></span>';
      button.title = "复制回答";
      renderIcons();
    }, 1400);
  } catch {
    console.error("Unable to copy message");
  }
}


function normalizeBackendResponse(response = {}) {
  return {
    answer: response.answer || response.final_answer || response.message || "",
    timeline: Array.isArray(response.timeline) ? response.timeline : [],
    tools: Array.isArray(response.tools || response.tool_calls)
      ? response.tools || response.tool_calls
      : [],
    files: normalizeFiles(response.files || response.outputs || []),
    resultImageUrl: response.resultImageUrl || response.result_image_url || response.result_image || "",
    viewer: response.viewer,
    task_id: response.task_id,
    report_data: response.report_data,
    raw: response,
  };
}

function buildErrorResponse(error, request) {
  return {
    answer: `请求后端失败：${error.message}`,
    timeline: [
      { node: "supervisor", text: "请求已发出" },
      { node: "error", text: "后端未返回可用结果" },
    ],
    tools: [],
    files: [],
    raw: {
      endpoint: request.endpoint,
      payload: request.body,
      error: error.message,
    },
  };
}

function normalizeFiles(files) {
  return files.map((file) => {
    const path = file.url || file.path || "";
    return {
      name: file.name || getNameFromPath(path) || "输出文件",
      path: file.path || path,
      url: path,
    };
  });
}

function renderAnswer(answer) {
  const text = cleanInternalProtocol(answer) || "工具执行完成，但模型没有生成最终文字说明。";
  return renderMarkdown(text);
}

function cleanInternalProtocol(answer) {
  const text = String(answer || "");
  if (/DSML/i.test(text) && /tool_calls|invoke/i.test(text)) return "";
  return text;
}

function renderMarkdown(text) {
  const raw = String(text || "");
  const markdownParser = window.marked;
  if (typeof markdownParser?.parse === "function") {
    markdownParser.setOptions({ breaks: true, gfm: true });
    return sanitizeHtml(markdownParser.parse(raw));
  }

  console.error("Markdown parser failed to load; rendering plain text.");
  return raw
    .split(/\n{2,}/)
    .map((part) => `<p>${escapeHtml(part).replaceAll("\n", "<br>")}</p>`)
    .join("");
}

function sanitizeHtml(html) {
  const template = document.createElement("template");
  template.innerHTML = html;
  const allowed = new Set(["P","BR","STRONG","EM","DEL","CODE","PRE","BLOCKQUOTE","UL","OL","LI","H1","H2","H3","H4","TABLE","THEAD","TBODY","TR","TH","TD","HR","A"]);
  template.content.querySelectorAll("*").forEach(node => {
    if (!allowed.has(node.tagName)) { node.remove(); return; }
    const href = node.tagName === "A" ? node.getAttribute("href") : null;
    [...node.attributes].forEach(attr => node.removeAttribute(attr.name));
    if (href) {
      try {
        const url = new URL(href, location.origin);
        if (["http:", "https:"].includes(url.protocol)) {
          node.href = url.href; node.target = "_blank"; node.rel = "noopener noreferrer";
        }
      } catch { /* invalid link */ }
    }
  });
  return template.innerHTML;
}

function renderResultImages(result) {
  const urls = new Set();
  if (result.resultImageUrl) urls.add(absolutizeUrl(result.resultImageUrl));
  result.files.forEach((file) => {
    const source = file.url || file.path || "";
    if (isImageFile(file.name || source)) {
      urls.add(absolutizeUrl(source));
    }
  });

  return [...urls]
    .filter(Boolean)
    .map((url) => `<img class="result-image" src="${escapeAttribute(url)}" alt="模型输出结果" />`)
    .join("");
}

function renderFileLinks(files) {
  if (!files.length) return "";
  return `
    <div class="file-list">
      ${files.map((file) => `
        <a class="file-link" href="${escapeAttribute(absolutizeUrl(file.url || file.path))}" target="_blank" rel="noreferrer">
          <span data-lucide="paperclip"></span>
          <span>${escapeHtml(file.name)} · ${escapeHtml(file.path || file.url)}</span>
        </a>
      `).join("")}
    </div>
  `;
}

function renderToolDetails(tools) {
  if (!tools.length) return "";
  return `
    <details class="tool-details">
      <summary>工具调用 ${tools.length} 个</summary>
      <pre>${escapeHtml(JSON.stringify(tools, null, 2))}</pre>
    </details>
  `;
}

function renderTimeline(timeline) {
  const items = timeline.length ? timeline : [{ node: "supervisor", text: "完成" }];
  return `
    <div class="timeline-line">
      ${items.map((item) => {
        const node = item.node || item.name || "step";
        const text = item.text || item.description || item.status || "";
        return `<span>${escapeHtml(node)}${text ? `: ${escapeHtml(compact(text))}` : ""}</span>`;
      }).join("")}
    </div>
  `;
}

function renderAttachmentChip(attachment) {
  return `
    <div class="attachment-chip">
      ${renderAttachmentPreview(attachment)}
      <div class="attachment-text">
        <strong>${escapeHtml(attachment.name)}</strong>
        <span>${formatBytes(attachment.size)}</span>
      </div>
    </div>
  `;
}

function renderAttachmentPreview(attachment, fileUrl) {
  const previewUrl = fileUrl || attachment.previewUrl;
  if (previewUrl && isImageFile(attachment.name)) {
    return `<div class="attachment-thumb"><img src="${escapeAttribute(absolutizeUrl(previewUrl))}" alt="上传图片预览" /></div>`;
  }
  return '<div class="attachment-icon"><span data-lucide="file-image"></span></div>';
}

function loadApiBase() {
  els.apiBase.value = window.location.origin;
  els.apiBase.readOnly = true;
}

function getApiBase() {
  return els.apiBase.value.trim().replace(/\/$/, "");
}

async function checkBackendHealth() {
  setConnectionState("正在连接", "pending");
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 4000);

  try {
    const response = await fetch(`${getApiBase()}/api/health`, { signal: controller.signal });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    setConnectionState("已连接", "connected");
  } catch {
    setConnectionState("连接失败", "error");
  } finally {
    window.clearTimeout(timeout);
  }
}

function setConnectionState(text, status) {
  els.connectionState.textContent = text;
  els.connectionDot.classList.remove("connected", "error");
  if (status === "connected") els.connectionDot.classList.add("connected");
  if (status === "error") els.connectionDot.classList.add("error");
}

function isImageFile(name = "") {
  return /\.(jpg|jpeg|png|jfif|webp|bmp)$/i.test(name);
}

function absolutizeUrl(path) {
  if (!path) return "";
  const value = String(path);
  if (value.startsWith("blob:")) return value;

  if (/^https?:\/\//i.test(value)) {
    try {
      const url = new URL(value);
      if (["127.0.0.1", "localhost"].includes(url.hostname) && /^\/(outputs|uploads)\//.test(url.pathname)) {
        return `${getApiBase()}${url.pathname}`;
      }
    } catch {
      return value;
    }
    return value;
  }

  if (value.startsWith("/")) return `${getApiBase()}${value}`;
  return `${getApiBase()}/${value.replace(/^\.?\//, "")}`;
}

function stripMarkdown(value) {
  return String(value || "")
    .replace(/```[\s\S]*?```/g, " 代码 ")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, " 图片 ")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/[#>*_`~|-]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function getNameFromPath(path) {
  return String(path).split("/").filter(Boolean).pop() || "";
}

function compact(text, limit = 72) {
  const normalized = String(text || "").replace(/\s+/g, " ").trim();
  return normalized.length > limit ? `${normalized.slice(0, limit)}...` : normalized;
}

function scrollToBottom() {
  els.conversation.scrollTop = els.conversation.scrollHeight;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "";
  const units = ["B", "KB", "MB", "GB"];
  let size = bytes;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  return `${size.toFixed(size >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function formatTime(timestamp) {
  if (!timestamp) return "";
  return new Date(timestamp).toLocaleTimeString("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttribute(value) {
  return escapeHtml(value).replaceAll("`", "&#096;");
}

init().catch(error => { setConnectionState(error.message, "error"); });
