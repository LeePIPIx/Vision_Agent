"""SQLite records are the source of truth for users, cases, tasks and events."""
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.getenv("MEDICAL_DATA_DIR", ROOT / "runtime")).resolve()


class Store:
    def __init__(self, root=DATA):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.root.chmod(0o700)
        self.path = self.root / "app.sqlite3"
        with self.connect() as db:
            db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY, username TEXT UNIQUE, password TEXT);
            CREATE TABLE IF NOT EXISTS logins(token TEXT PRIMARY KEY, user_id TEXT, expires REAL);
            CREATE TABLE IF NOT EXISTS cases(id TEXT PRIMARY KEY, owner TEXT, title TEXT, created REAL);
            CREATE TABLE IF NOT EXISTS files(id TEXT PRIMARY KEY, owner TEXT, case_id TEXT, name TEXT,
              path TEXT UNIQUE, kind TEXT, sha256 TEXT, created REAL);
            CREATE TABLE IF NOT EXISTS sessions(id TEXT, owner TEXT, title TEXT, messages TEXT DEFAULT '[]',
              state TEXT DEFAULT '{}', case_id TEXT, updated REAL, PRIMARY KEY(id,owner));
            CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, owner TEXT, session_id TEXT, case_id TEXT,
              request TEXT, status TEXT, result TEXT, error TEXT, cancel INTEGER DEFAULT 0, created REAL, updated REAL);
            CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT, data TEXT);
            CREATE INDEX IF NOT EXISTS events_task ON events(task_id,id);
            CREATE TABLE IF NOT EXISTS reviews(id TEXT PRIMARY KEY, task_id TEXT, owner TEXT,
              status TEXT, note TEXT, created REAL);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def query(self, sql, args=(), one=False):
        with self.connect() as db:
            rows = db.execute(sql, args).fetchall()
        values = [dict(r) for r in rows]
        return (values[0] if values else None) if one else values

    def execute(self, sql, args=()):
        with self.connect() as db:
            db.execute(sql, args)

    def event(self, task_id, stage, message, **extra):
        data = dict(stage=stage, message=message, time=time.time(), **extra)
        self.execute("INSERT INTO events(task_id,data) VALUES(?,?)", (task_id, json.dumps(data, ensure_ascii=False)))

    def task(self, task_id, owner=None):
        row = self.query("SELECT * FROM tasks WHERE id=?" + (" AND owner=?" if owner else ""),
                         (task_id, owner) if owner else (task_id,), one=True)
        if row:
            for key in ("request", "result"):
                row[key] = json.loads(row[key]) if row[key] else None
        return row

    def messages(self, owner, session_id, message):
        with self.connect() as db:
            row = db.execute("SELECT messages FROM sessions WHERE owner=? AND id=?", (owner, session_id)).fetchone()
            messages = json.loads(row[0])
            messages.append(message)
            db.execute("UPDATE sessions SET messages=?,updated=? WHERE owner=? AND id=?",
                       (json.dumps(messages, ensure_ascii=False), time.time(), owner, session_id))

    def register_file(self, owner, case_id, path, kind, digest=""):
        path = Path(path).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("文件必须位于应用数据目录")
        fid = uuid4().hex
        self.execute("INSERT INTO files VALUES(?,?,?,?,?,?,?,?)",
                     (fid, owner, case_id, path.name, str(path), kind, digest, time.time()))
        return {"id": fid, "name": path.name, "url": f"/api/files/{fid}", "kind": kind, "sha256": digest}
