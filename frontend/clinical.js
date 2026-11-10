/* Same-origin authentication, durable tasks and CT review controls. */
window.Clinic = {
  currentTask: null,
  async api(path, options = {}) {
    const response = await fetch(path, { ...options, credentials: "same-origin" });
    if (!response.ok) {
      let message = `HTTP ${response.status}`;
      try { message = (await response.json()).detail || message; } catch { /* plain response */ }
      throw new Error(typeof message === "string" ? message : JSON.stringify(message));
    }
    return response.json();
  },
  async login() {
    try { return await this.api("/api/auth/me"); } catch { /* show sign in */ }
    const dialog = document.querySelector("#loginDialog");
    dialog.showModal();
    dialog.addEventListener("cancel", event => event.preventDefault());
    return new Promise(resolve => {
      document.querySelector("#loginForm").addEventListener("submit", async event => {
        event.preventDefault();
        const submit = event.submitter;
        submit.disabled = true;
        try {
          const account = await this.api(`/api/auth/${submit.value}`, {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ username: document.querySelector("#username").value,
              password: document.querySelector("#password").value }),
          });
          document.querySelector("#password").value = "";
          dialog.close(); resolve(account);
        } catch (error) { document.querySelector("#loginError").textContent = error.message; }
        finally { submit.disabled = false; }
      });
    });
  },
  async wait(taskId, article) {
    this.currentTask = taskId;
    const cancel = document.querySelector("#cancelTaskBtn");
    cancel.hidden = false;
    cancel.onclick = () => this.api(`/api/tasks/${taskId}/cancel`, { method: "POST" }).catch(error => alert(error.message));
    const line = article.querySelector(".timeline-line");
    const source = new EventSource(`/api/tasks/${taskId}/events`);
    source.addEventListener("progress", event => {
      const data = JSON.parse(event.data);
      line.textContent = data.message;
    });
    source.onerror = () => { line.textContent = "进度连接暂时中断，正在重新连接；任务仍在服务器执行"; };
    return new Promise((resolve, reject) => {
      let finished = false;
      const finish = (task) => {
        if (finished || !["completed", "failed", "cancelled", "interrupted"].includes(task.status)) return;
        finished = true; source.close(); clearInterval(timer);
        cancel.hidden = true; this.currentTask = null;
        if (task.status === "completed") resolve(task.result);
        else reject(new Error(task.error || "任务已取消或中断，可在任务记录中重试"));
      };
      const poll = () => this.api(`/api/tasks/${taskId}`).then(finish).catch(() => {
        line.textContent = "服务器暂时无法连接，恢复后会继续显示任务结果";
      });
      const timer = setInterval(poll, 3000);
      source.addEventListener("done", poll);
      poll();
    });
  },
  async tasks() {
    const dialog = document.querySelector("#tasksDialog");
    const list = document.querySelector("#taskList");
    list.replaceChildren();
    for (const task of await this.api("/api/tasks")) {
      const item = document.createElement("div");
      item.className = "task-record";
      const text = document.createElement("span");
      text.textContent = `${new Date(task.created * 1000).toLocaleString()} · ${task.status}${task.error ? ' · ' + task.error : ''}`;
      item.append(text);
      const button = document.createElement("button");
      button.textContent = ["failed", "cancelled", "interrupted"].includes(task.status) ? "重试" : "查看";
      button.onclick = async () => {
        try {
          let id = task.id;
          if (button.textContent === "重试") id = (await this.api(`/api/tasks/${id}/retry`, { method: "POST" })).task_id;
          dialog.close();
          await window.openServerTask(id, task.session_id);
        } catch (e) { alert(e.message); }
      };
      item.append(button); list.append(item);
    }
    if (!list.children.length) list.textContent = "还没有任务";
    dialog.showModal();
  },
  bindResult(article, result) {
    const original = result.raw || result;
    const viewer = original.viewer;
    if (!viewer) return;
    const box = document.createElement("section");
    box.className = "ct-review";
    const title = document.createElement("strong"); title.textContent = "CT 切片与分割叠加（RAS 轴向显示）";
    const img = document.createElement("img"); img.className = "ct-slice"; img.alt = "CT 与分割 mask 叠加";
    const slider = document.createElement("input"); slider.type = "range"; slider.min = 0; slider.max = viewer.slices - 1; slider.value = Math.floor(viewer.slices / 2);
    const label = document.createElement("span");
    const overlay = document.createElement("input"); overlay.type = "checkbox"; overlay.checked = true;
    const overlayLabel = document.createElement("label"); overlayLabel.append(overlay, " 显示分割 mask");
    const windowSelect = document.createElement("select");
    for (const [text,value] of [["软组织窗", "400,50"], ["肺窗", "1500,-600"], ["骨窗", "1800,400"]]) {
      const option = document.createElement("option"); option.textContent = text; option.value = value; windowSelect.append(option);
    }
    const show = () => {
      const [windowWidth, level] = windowSelect.value.split(",");
      img.src = `/api/files/${viewer.image_id}/slice?index=${slider.value}&window=${windowWidth}&level=${level}${overlay.checked ? '&mask_id=' + viewer.mask_id : ''}`;
      label.textContent = `切片 ${Number(slider.value) + 1} / ${viewer.slices}`;
    };
    slider.onchange = show; overlay.onchange = show; windowSelect.onchange = show;
    show(); box.append(title, img, label, slider, overlayLabel, windowSelect);
    const note = document.createElement("textarea"); note.placeholder = "复核意见（不会覆盖原始测量值）";
    const reviewStatus = document.createElement("p");
    const reload = async () => {
      const rows = await this.api(`/api/tasks/${original.task_id}/reviews`);
      reviewStatus.textContent = rows.length ? rows.map(row => `${row.reviewer} · ${row.status === 'reviewed' ? '已复核' : '待修改'}：${row.note}`).join("\n") : "待人工复核";
    };
    for (const [text, status] of [["确认已复核", "reviewed"], ["标记需修改", "needs_changes"]]) {
      const button = document.createElement("button"); button.textContent = text;
      button.onclick = async () => {
        try {
          await this.api(`/api/tasks/${original.task_id}/review`, { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({ status, note: note.value }) });
          await reload();
        } catch (e) { alert(e.message); }
      };
      box.append(button);
    }
    const upload = document.createElement("input"); upload.type = "file"; upload.accept = ".nii,.nii.gz";
    const uploadLabel = document.createElement("label"); uploadLabel.textContent = "上传修订 mask 并重新计算（保留旧版报告）："; uploadLabel.append(upload);
    upload.onchange = async () => {
      if (!upload.files[0]) return;
      const body = new FormData(); body.append("file", upload.files[0]);
      try {
        const task = await this.api(`/api/tasks/${original.task_id}/mask`, { method: "POST", body });
        await window.openServerTask(task.task_id);
      } catch (e) { alert(e.message); }
    };
    box.append(note, reviewStatus, uploadLabel); article.querySelector(".message-body").append(box);
    reload().catch(() => { reviewStatus.textContent = "复核记录暂时无法加载"; });
  },
};
