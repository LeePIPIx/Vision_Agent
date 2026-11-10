import asyncio
import hashlib
import json
import os
import re
import shutil
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from medical_app.imaging import load_volume, metadata, slice_png
from medical_app.security import new_login, password_hash, token_hash, verify_password
from medical_app.storage import ROOT, Store
from medical_app.tasks import MODEL, MODEL_DIR, TERMINAL, TaskRunner

MAX_UPLOAD = 256 * 1024 * 1024


class ChatPayload(BaseModel):
    session_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    query: str = Field(default="", max_length=8000)
    workflow: str = Field(default="auto", pattern="^(auto|chat|ct_spleen)$")
    file_id: str | None = None
    mask_id: str | None = None
    max_iterations: int = Field(default=12, ge=1, le=30)


class Credentials(BaseModel):
    username: str = Field(pattern=r"^[a-zA-Z0-9_-]{3,40}$")
    password: str = Field(min_length=10, max_length=128)


class Review(BaseModel):
    status: str = Field(pattern="^(reviewed|needs_changes)$")
    note: str = Field(min_length=1, max_length=4000)


def create_app(store=None, start_worker=True):
    store = store or Store()
    runner = TaskRunner(store)

    @asynccontextmanager
    async def lifespan(app):
        if start_worker:
            runner.start()
        yield
        if start_worker:
            runner.close()

    app = FastAPI(title="Medical Agent", lifespan=lifespan)
    app.state.store, app.state.runner = store, runner
    attempts = defaultdict(deque)

    @app.middleware("http")
    async def request_guard(request, call_next):
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
                return Response("跨站请求被拒绝", status_code=403)
        length = request.headers.get("content-length")
        if length and (not length.isdigit() or int(length) > MAX_UPLOAD + 1024 * 1024):
            return Response("请求过大", status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'"
        return response

    def user(request: Request):
        token = request.cookies.get("medical_session", "")
        row = store.query("SELECT users.id,users.username FROM logins JOIN users ON users.id=logins.user_id WHERE token=? AND expires>?",
                          (token_hash(token), time.time()), one=True)
        if not row:
            raise HTTPException(401, "请先登录")
        return row

    def own_task(task_id, account):
        task = store.task(task_id, account["id"])
        if not task:
            raise HTTPException(404, "任务不存在")
        return task

    def own_file(file_id, account):
        file = store.query("SELECT * FROM files WHERE id=? AND owner=?", (file_id, account["id"]), one=True)
        if not file:
            raise HTTPException(404, "文件不存在")
        if not Path(file["path"]).resolve().is_relative_to(store.root.resolve()):
            raise HTTPException(403, "无效文件位置")
        return file

    def auth_rate(request):
        key = request.client.host
        now = time.time()
        queue = attempts[key]
        while queue and queue[0] < now - 300:
            queue.popleft()
        if len(queue) >= 20:
            raise HTTPException(429, "登录请求过于频繁，请五分钟后重试")
        queue.append(now)

    def cookie(response, account):
        response.set_cookie("medical_session", new_login(store, account["id"]), httponly=True,
                            samesite="strict", secure=os.getenv("MEDICAL_COOKIE_SECURE") == "1", max_age=86400)
        return {"id": account["id"], "username": account["username"]}

    @app.post("/api/auth/register")
    def register(data: Credentials, request: Request, response: Response):
        import sqlite3
        auth_rate(request)
        if os.getenv("MEDICAL_ALLOW_REGISTRATION", "1") != "1":
            raise HTTPException(403, "已关闭注册")
        account = {"id": uuid4().hex, "username": data.username}
        try:
            store.execute("INSERT INTO users VALUES(?,?,?)", (account["id"], data.username, password_hash(data.password)))
        except sqlite3.IntegrityError:
            raise HTTPException(409, "用户名已存在")
        return cookie(response, account)

    @app.post("/api/auth/login")
    def login(data: Credentials, request: Request, response: Response):
        auth_rate(request)
        account = store.query("SELECT * FROM users WHERE username=?", (data.username,), one=True)
        if not account or not verify_password(data.password, account["password"]):
            raise HTTPException(401, "用户名或密码错误")
        return cookie(response, account)

    @app.post("/api/auth/logout")
    def logout(request: Request, response: Response):
        store.execute("DELETE FROM logins WHERE token=?", (token_hash(request.cookies.get("medical_session", "")),))
        response.delete_cookie("medical_session")
        return {"ok": True}

    @app.get("/api/auth/me")
    def me(account=Depends(user)):
        return account

    @app.get("/api/health")
    def health():
        return {"status": "ok", "ct_model_ready": (MODEL_DIR / "fold_0/model_final_checkpoint.model").is_file(),
                "demo_ready": (ROOT / "data/demo/ct.nii.gz").is_file(), "model": MODEL,
                "registration": os.getenv("MEDICAL_ALLOW_REGISTRATION", "1") == "1"}

    @app.get("/api/sessions")
    def sessions(account=Depends(user)):
        rows = store.query("SELECT id,title,messages,case_id,updated FROM sessions WHERE owner=? ORDER BY updated DESC", (account["id"],))
        for row in rows:
            row["messages"] = json.loads(row["messages"])
            row["updatedAt"] = row["updated"] * 1000
            row["createdAt"] = row["updatedAt"]
        return rows

    @app.delete("/api/session/{session_id}")
    def delete_session(session_id, account=Depends(user)):
        if store.query("SELECT id FROM tasks WHERE owner=? AND session_id=? AND status IN ('queued','running')", (account["id"], session_id)):
            raise HTTPException(409, "请先取消正在执行的任务")
        with store.connect() as db:
            db.execute("DELETE FROM sessions WHERE owner=? AND id=?", (account["id"], session_id))
        # Case files and review history are retained separately from the conversation.
        return {"ok": True}

    @app.get("/api/cases")
    def cases(account=Depends(user)):
        return store.query("SELECT * FROM cases WHERE owner=? ORDER BY created DESC", (account["id"],))

    def ensure_session(data, account):
        with store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM sessions WHERE id=? AND owner=?", (data.session_id, account["id"])).fetchone()
            if row:
                return row["case_id"]
            case_id = uuid4().hex
            db.execute("INSERT INTO cases VALUES(?,?,?,?)", (case_id, account["id"], data.query[:50] or "CT Demo", time.time()))
            db.execute("INSERT INTO sessions(id,owner,title,case_id,updated) VALUES(?,?,?,?,?)",
                       (data.session_id, account["id"], data.query[:40] or "CT Demo", case_id, time.time()))
            return case_id

    def upload(file, account, case_id, kind="input"):
        name = Path(file.filename or "").name
        if not re.search(r"\.(nii(\.gz)?|png|jpg|jpeg|jfif|webp|bmp)$", name, re.I):
            raise HTTPException(400, "支持 NIfTI 和常见图片格式")
        directory = store.root / "cases" / case_id / "uploads" / uuid4().hex
        directory.mkdir(parents=True)
        path = directory / name
        total = 0
        try:
            with path.open("wb") as stream:
                while block := file.file.read(1024 * 1024):
                    total += len(block)
                    if total > MAX_UPLOAD:
                        raise ValueError("文件超过 256 MB")
                    stream.write(block)
            if name.lower().endswith((".nii", ".nii.gz")):
                load_volume(path)
            else:
                from PIL import Image
                with Image.open(path) as image:
                    image.verify()
            from medical_app.imaging import digest
            return store.register_file(account["id"], case_id, path, kind, digest(path))
        except Exception as exc:
            shutil.rmtree(directory)
            raise HTTPException(400, f"文件校验失败：{type(exc).__name__}") from exc

    def enqueue(data, account, case_id):
        req = data.model_dump()
        if data.file_id:
            f = own_file(data.file_id, account)
            if f["case_id"] != case_id:
                raise HTTPException(400, "文件不属于当前病例")
        else:
            f = store.query("SELECT * FROM files WHERE case_id=? AND owner=? AND kind='input' ORDER BY created DESC LIMIT 1", (case_id, account["id"]), one=True)
            if f:
                req["file_id"] = f["id"]
        if data.mask_id:
            mask = own_file(data.mask_id, account)
            if mask["case_id"] != case_id:
                raise HTTPException(400, "mask 不属于当前病例")
        if req["workflow"] == "auto":
            is_ct = f and f["name"].lower().endswith((".nii", ".nii.gz"))
            req["workflow"] = "ct_spleen" if is_ct and (data.file_id or re.search("分割|重新分析|重新计算", data.query)) else "chat"
        if req["workflow"] == "ct_spleen" and (not f or not f["name"].lower().endswith((".nii", ".nii.gz"))):
            raise HTTPException(400, "CT 分割需要上传 .nii 或 .nii.gz 文件")
        task_id = uuid4().hex
        with store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            active = db.execute("SELECT id FROM tasks WHERE owner=? AND session_id=? AND status IN ('queued','running')", (account["id"], data.session_id)).fetchone()
            if active:
                raise HTTPException(409, "当前会话已有任务，请等待完成或取消")
            if db.execute("SELECT count(*) FROM tasks WHERE status IN ('queued','running')").fetchone()[0] >= 32:
                raise HTTPException(429, "任务队列已满，请稍后再试")
            now = time.time()
            db.execute("INSERT INTO tasks(id,owner,session_id,case_id,request,status,created,updated) VALUES(?,?,?,?,?,'queued',?,?)",
                       (task_id, account["id"], data.session_id, case_id, json.dumps(req, ensure_ascii=False), now, now))
            row = db.execute("SELECT messages FROM sessions WHERE owner=? AND id=?", (account["id"], data.session_id)).fetchone()
            messages = json.loads(row[0]); messages.append({"role": "user", "text": data.query or "CT 脾脏分割"})
            db.execute("UPDATE sessions SET messages=?,updated=? WHERE owner=? AND id=?", (json.dumps(messages, ensure_ascii=False), now, account["id"], data.session_id))
        store.event(task_id, "queued", "任务已接收，等待处理")
        return {"task_id": task_id, "case_id": case_id, "status": "queued"}

    @app.post("/api/tasks", status_code=202)
    def submit(payload: str = Form(...), file: UploadFile | None = File(None), account=Depends(user)):
        try:
            data = ChatPayload.model_validate_json(payload)
        except ValueError:
            raise HTTPException(400, "请求参数无效")
        case_id = ensure_session(data, account)
        if file:
            data.file_id = upload(file, account, case_id)["id"]
        return enqueue(data, account, case_id)

    @app.post("/api/demo", status_code=202)
    def demo(data: ChatPayload, account=Depends(user)):
        source = ROOT / "data/demo/ct.nii.gz"
        if not source.exists():
            raise HTTPException(503, "公开 CT 样例尚未下载")
        data.workflow, data.query = "ct_spleen", "运行公开腹部 CT 脾脏分割 Demo"
        case_id = ensure_session(data, account)
        target = store.root / "cases" / case_id / "uploads" / uuid4().hex / "demo_ct.nii.gz"
        target.parent.mkdir(parents=True)
        shutil.copyfile(source, target)
        data.file_id = store.register_file(account["id"], case_id, target, "input")["id"]
        return enqueue(data, account, case_id)

    @app.get("/api/tasks")
    def task_list(account=Depends(user)):
        return store.query("SELECT id,session_id,case_id,status,error,created FROM tasks WHERE owner=? ORDER BY created DESC LIMIT 100", (account["id"],))

    @app.get("/api/tasks/{task_id}")
    def get_task(task_id, account=Depends(user)):
        task = own_task(task_id, account)
        return {k: task[k] for k in ("id", "status", "result", "error", "session_id", "case_id")}

    @app.get("/api/tasks/{task_id}/events")
    async def events(task_id: str, request: Request, after: int = 0, account=Depends(user)):
        own_task(task_id, account)
        try:
            cursor = max(after, int(request.headers.get("last-event-id", "0")))
        except ValueError:
            cursor = 0
        async def stream():
            nonlocal cursor
            while not await request.is_disconnected():
                rows = store.query("SELECT id,data FROM events WHERE task_id=? AND id>? ORDER BY id", (task_id, cursor))
                for row in rows:
                    cursor = row["id"]
                    yield f"id: {cursor}\nevent: progress\ndata: {row['data']}\n\n"
                task = store.task(task_id)
                if task["status"] in TERMINAL:
                    yield "event: done\ndata: " + json.dumps({"status": task["status"]}) + "\n\n"
                    break
                yield ": heartbeat\n\n"
                await asyncio.sleep(1)
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})

    @app.post("/api/tasks/{task_id}/cancel")
    def cancel(task_id, account=Depends(user)):
        own_task(task_id, account)
        with store.connect() as db:
            db.execute("UPDATE tasks SET cancel=1, status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END WHERE id=? AND status IN ('queued','running')", (task_id,))
        store.event(task_id, "cancel_requested", "已请求取消；正在终止当前步骤")
        return {"ok": True}

    @app.post("/api/tasks/{task_id}/retry", status_code=202)
    def retry(task_id, account=Depends(user)):
        task = own_task(task_id, account)
        if task["status"] not in {"failed", "interrupted", "cancelled"}:
            raise HTTPException(409, "只能重试失败、中断或取消的任务")
        data = ChatPayload(**task["request"])
        ensure_session(data, account)
        return enqueue(data, account, task["case_id"])

    @app.post("/api/tasks/{task_id}/review")
    def review(task_id, data: Review, account=Depends(user)):
        task = own_task(task_id, account)
        if task["status"] != "completed" or not task["result"].get("report_data"):
            raise HTTPException(409, "只有已完成的定量报告可以复核")
        store.execute("INSERT INTO reviews VALUES(?,?,?,?,?,?)", (uuid4().hex, task_id, account["id"], data.status, data.note, time.time()))
        return {"status": data.status, "reviewer": account["username"]}

    @app.get("/api/tasks/{task_id}/reviews")
    def reviews(task_id, account=Depends(user)):
        own_task(task_id, account)
        return store.query("SELECT status,note,created,users.username AS reviewer FROM reviews JOIN users ON users.id=reviews.owner WHERE task_id=? ORDER BY created", (task_id,))

    @app.post("/api/tasks/{task_id}/mask", status_code=202)
    def replace_mask(task_id: str, file: UploadFile = File(...), account=Depends(user)):
        task = own_task(task_id, account)
        if task["status"] != "completed" or not task["result"].get("report_data"):
            raise HTTPException(409, "请先完成 CT 分析")
        mask = upload(file, account, task["case_id"], "mask")
        data = ChatPayload(**task["request"])
        data.mask_id, data.workflow, data.query = mask["id"], "ct_spleen", "根据修订 mask 重新计算并生成新版报告"
        return enqueue(data, account, task["case_id"])

    @app.get("/api/files/{file_id}")
    def download(file_id, account=Depends(user)):
        f = own_file(file_id, account)
        return FileResponse(f["path"], filename=f["name"], content_disposition_type="inline" if f["name"].endswith(".png") else "attachment")

    @app.get("/api/files/{file_id}/slice")
    def view_slice(file_id: str, index: int = 0, mask_id: str | None = None,
                   window: int = 400, level: int = 50, account=Depends(user)):
        f = own_file(file_id, account)
        mask = own_file(mask_id, account) if mask_id else None
        if mask and mask["case_id"] != f["case_id"]:
            raise HTTPException(400, "mask 不属于当前病例")
        try:
            data, count = slice_png(f["path"], mask["path"] if mask else None, index, max(1, min(window, 4000)), max(-1500, min(level, 2000)))
        except Exception as exc:
            raise HTTPException(400, str(exc))
        return Response(data, media_type="image/png", headers={"X-Slice-Count": str(count)})

    app.mount("/", StaticFiles(directory=str(ROOT / "frontend"), html=True), name="frontend")
    return app
