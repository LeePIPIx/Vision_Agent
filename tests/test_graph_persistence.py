import sqlite3
from types import SimpleNamespace


def test_graph_recovers_history_from_sqlite(tmp_path, monkeypatch):
    from langgraph.checkpoint.sqlite import SqliteSaver
    from graphs.session import SessionManager
    import graphs.node as nodes
    calls = []
    def create(**kwargs):
        calls.append(kwargs["messages"])
        message = SimpleNamespace(content="已记录", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])
    monkeypatch.setattr(nodes, "_get_client", lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    db = tmp_path / "checkpoints.sqlite3"
    for text in ["记住病例代号 alpha", "刚才的代号是什么"]:
        with sqlite3.connect(db, check_same_thread=False) as conn:
            sess = SessionManager(graph="react", checkpointer=SqliteSaver(conn))
            sess.thread_id = "owner:session"
            assert sess.chat(text) == "已记录"
    assert any(message.get("content") == "记住病例代号 alpha" for message in calls[-1])
    assert any(message.get("content") == "刚才的代号是什么" for message in calls[-1])
