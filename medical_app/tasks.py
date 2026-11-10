import json
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

from medical_app.context import TaskCancelled, current_task
from medical_app.imaging import digest, label_statistics, metadata, report_markdown, slice_png
from medical_app.storage import ROOT

MODEL_DIR = ROOT / "models" / "nnunet" / "3d_fullres" / "Task009_Spleen" / "nnUNetTrainerV2__nnUNetPlansv2.1"
MODEL = {"name": "nnU-Net v1 Task009_Spleen", "configuration": "3d_fullres / fold_0 / no TTA",
         "source": "https://zenodo.org/records/4003545", "labels": {"1": "脾脏"}}
TERMINAL = {"completed", "failed", "cancelled", "interrupted"}


class TaskContext:
    def __init__(self, store, task):
        self.store, self.task = store, task
        self.deadline = time.monotonic() + int(os.getenv("MEDICAL_TASK_TIMEOUT", "900"))
        self.output = store.root / "cases" / task["case_id"] / task["id"]
        self.output.mkdir(parents=True, exist_ok=True)
        self.allowed_files = {r["path"] for r in store.query(
            "SELECT path FROM files WHERE owner=? AND case_id=?", (task["owner"], task["case_id"]))}

    def check(self):
        row = self.store.query("SELECT cancel FROM tasks WHERE id=?", (self.task["id"],), one=True)
        if row["cancel"]:
            raise TaskCancelled("用户已取消任务")
        if time.monotonic() > self.deadline:
            raise TimeoutError("任务超过执行时限，请缩小影像范围或稍后重试")

    def event(self, stage, message, **extra):
        self.check()
        self.store.event(self.task["id"], stage, message, **extra)

    def file(self, path, kind):
        self.allowed_files.add(str(Path(path).resolve()))
        return self.store.register_file(self.task["owner"], self.task["case_id"], path, kind, digest(path))


class TaskRunner:
    def __init__(self, store):
        self.store = store
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        import fcntl
        self.lock = open(self.store.root / "worker.lock", "w")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for row in self.store.query("SELECT id FROM tasks WHERE status='running'"):
            self.store.execute("UPDATE tasks SET status='interrupted',error=?,updated=? WHERE id=?",
                               ("服务重启中断，可重试；已完成的文件仍保留", time.time(), row["id"]))
            self.store.event(row["id"], "interrupted", "服务重启中断，可重试")
        self.thread = threading.Thread(target=self.loop, daemon=True, name="medical-worker")
        self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)
        if hasattr(self, "lock"):
            self.lock.close()

    def loop(self):
        while not self.stop_event.is_set():
            with self.store.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute("SELECT id FROM tasks WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
                if row:
                    db.execute("UPDATE tasks SET status='running',updated=? WHERE id=?", (time.time(), row[0]))
            if row:
                self.run(row[0])
            else:
                self.stop_event.wait(0.5)

    def run(self, task_id):
        task = self.store.task(task_id)
        ctx = TaskContext(self.store, task)
        token = current_task.set(ctx)
        try:
            ctx.event("running", "任务开始处理")
            result = self.ct(ctx) if task["request"]["workflow"] == "ct_spleen" else self.chat(ctx)
            ctx.check()
            result.update(task_id=task_id, case_id=task["case_id"])
            # Commit the final answer and task status together so refresh cannot duplicate it.
            with self.store.connect() as db:
                row = db.execute("SELECT messages FROM sessions WHERE owner=? AND id=?",
                                 (task["owner"], task["session_id"])).fetchone()
                messages = json.loads(row[0])
                messages.append({"role": "assistant", "result": result})
                db.execute("UPDATE sessions SET messages=?,updated=? WHERE owner=? AND id=?",
                           (json.dumps(messages, ensure_ascii=False), time.time(), task["owner"], task["session_id"]))
                db.execute("UPDATE tasks SET status='completed',result=?,updated=? WHERE id=?",
                           (json.dumps(result, ensure_ascii=False), time.time(), task_id))
            self.store.event(task_id, "completed", "处理完成，结果已保存")
        except Exception as exc:
            status = "cancelled" if isinstance(exc, TaskCancelled) else "failed"
            # No query, patient identifiers or raw model output in application logs.
            message = str(exc)[:500] if isinstance(exc, (ValueError, TimeoutError, TaskCancelled)) else f"{type(exc).__name__}：处理失败，请检查服务配置或重试"
            self.store.execute("UPDATE tasks SET status=?,error=?,updated=? WHERE id=?", (status, message, time.time(), task_id))
            self.store.event(task_id, status, message)
        finally:
            current_task.reset(token)

    def ct(self, ctx):
        req = ctx.task["request"]
        image = self.store.query("SELECT * FROM files WHERE id=? AND owner=? AND case_id=?",
                                (req["file_id"], ctx.task["owner"], ctx.task["case_id"]), one=True)
        if not image:
            raise ValueError("请先上传腹部 CT（.nii 或 .nii.gz）")
        ctx.event("reading", "读取 CT 与空间信息")
        info = metadata(image["path"])
        mask_path = ctx.output / "mask.nii.gz"
        model = dict(MODEL)
        if req.get("mask_id"):
            source = self.store.query("SELECT * FROM files WHERE id=? AND owner=? AND case_id=?",
                                     (req["mask_id"], ctx.task["owner"], ctx.task["case_id"]), one=True)
            if not source:
                raise ValueError("找不到当前病例的修订 mask")
            shutil.copyfile(source["path"], mask_path)
            model = {"name": "人工上传/修订 mask", "configuration": "label-count-v1", "source": "病例附件", "labels": {}}
        else:
            self.predict(ctx, image["path"], mask_path)
            model = dict(MODEL)
        ctx.event("measuring", "计算分割体积、连通区域与空间位置")
        stats = label_statistics(image["path"], mask_path, model["labels"])
        mask_file = ctx.file(mask_path, "mask")
        ctx.event("reporting", "生成可追溯报告与预览")
        report = report_markdown(ctx.task["case_id"], stats, model)
        payload = {"case_id": ctx.task["case_id"], "image_id": image["id"], "mask_id": mask_file["id"],
                   "model": model, "statistics": stats, "image": info, "review_status": "pending"}
        json_path, md_path = ctx.output / "report.json", ctx.output / "report.md"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        md_path.write_text(report)
        preview, count = slice_png(image["path"], mask_path)
        preview_path = ctx.output / "overlay.png"
        preview_path.write_bytes(preview)
        return {"answer": report, "route": "ct_spleen", "files": [mask_file, ctx.file(json_path, "report"), ctx.file(md_path, "report"), ctx.file(preview_path, "preview")],
                "report_data": payload, "viewer": {"image_id": image["id"], "mask_id": mask_file["id"], "slices": count},
                "timeline": [{"node": "CT", "text": "分割 → 定量计算 → 待人工复核"}]}

    def predict(self, ctx, image, output):
        checkpoint = MODEL_DIR / "fold_0" / "model_final_checkpoint.model"
        if not checkpoint.is_file():
            raise ValueError("官方 nnU-Net 权重尚未安装，请先运行 scripts/setup_demo.py")
        ctx.event("segmenting", "nnU-Net 正在分割脾脏；耗时取决于影像大小")
        inp, out = ctx.output / "input", ctx.output / "prediction"
        inp.mkdir(exist_ok=True); out.mkdir(exist_ok=True)
        import nibabel as nib
        nib.save(nib.load(image), inp / "case_0000.nii.gz")
        cmd = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/nnunet_predict.py"),
               "--model", str(MODEL_DIR), "--input", str(inp), "--output", str(out)]
        env = dict(os.environ, nnUNet_n_proc_DA="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        with open(ctx.output / "inference.log", "w") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=log, env=env, start_new_session=True)
            try:
                while proc.poll() is None:
                    if self.stop_event.wait(0.5):
                        raise TaskCancelled("服务正在停止")
                    ctx.check()
                if proc.returncode:
                    raise ValueError("nnU-Net 推理失败；请检查服务器任务目录中的 inference.log")
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL); proc.wait()
        shutil.move(out / "case.nii.gz", output)
        shutil.rmtree(inp)
        model_hash = digest(checkpoint)
        MODEL["checkpoint_sha256"] = model_hash

    def chat(self, ctx):
        from graphs.session import SessionManager
        from langgraph.checkpoint.sqlite import SqliteSaver
        import sqlite3
        req, task = ctx.task["request"], ctx.task
        conn = sqlite3.connect(self.store.root / "checkpoints.sqlite3", check_same_thread=False, timeout=30)
        try:
            session = SessionManager(checkpointer=SqliteSaver(conn))
            session.thread_id = f"{task['owner']}:{task['session_id']}"
            query = req["query"]
            if req.get("file_id"):
                file = self.store.query("SELECT * FROM files WHERE id=? AND owner=?", (req["file_id"], task["owner"]), one=True)
                query = f"当前授权文件路径：{file['path']}\n{query}"
            recent = self.store.query("SELECT result FROM tasks WHERE case_id=? AND owner=? AND status='completed' ORDER BY updated DESC LIMIT 1", (task["case_id"], task["owner"]), one=True)
            context = json.loads(recent["result"]).get("report_data", {}) if recent else {}
            ctx.event("routing", "判断请求意图并检查病例上下文")
            answer = session.chat(query, max_iterations=req.get("max_iterations", 12), case_context=context)
            files = []
            for path in ctx.output.rglob("*"):
                if path.is_file():
                    files.append(ctx.file(path, "output"))
            return {"answer": answer, "route": session.last_route, "files": files,
                    "timeline": [{"node": "Agent", "text": "回答已保存"}]}
        finally:
            conn.close()
