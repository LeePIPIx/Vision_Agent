"""Task-local cancellation, progress and allowed files, also visible in graph nodes."""
from contextvars import ContextVar

current_task = ContextVar("medical_task", default=None)


class TaskCancelled(Exception):
    pass


def checkpoint():
    ctx = current_task.get()
    if ctx:
        ctx.check()


def allowed_path(path):
    from pathlib import Path
    ctx = current_task.get()
    path = Path(path).resolve()
    if ctx and str(path) not in ctx.allowed_files:
        raise PermissionError("只能访问当前病例中已授权的文件")
    return path
