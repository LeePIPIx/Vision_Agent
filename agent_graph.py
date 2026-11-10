"""
Agent Graph 入口：演示 Supervisor 路由图 + 会话记忆。

支持多轮对话：每次 chat() 自动携带之前所有轮次的历史。
"""

from graphs.session import SessionManager


def demo_memory():
    """演示多轮对话记忆：后续问题能引用前文的上下文"""
    print("=" * 60)
    print("【会话记忆演示】Supervisor 路由图 + MemorySaver")
    print("=" * 60)

    sess = SessionManager(graph="supervisor")
    print(f"会话 ID: {sess.thread_id}\n")

    # ---- 第 1 轮 ----
    print(">>> 第 1 轮")
    query1 = "现在几点了？"
    print(f"用户: {query1}")
    answer1 = sess.chat(query1)
    print(f"Agent: {answer1[:300]}\n")

    # ---- 第 2 轮（引用前文，无需重复说"当前时间"）----
    print(">>> 第 2 轮（自动携带上文）")
    query2 = "那时明天是周几？"
    print(f"用户: {query2}")
    answer2 = sess.chat(query2)
    print(f"Agent: {answer2[:300]}\n")

    # ---- 查看会话状态 ----
    print(f"会话信息: {sess.info}")


def demo_new_session():
    """演示 new_session()：开启新会话后，历史被清空"""
    print("=" * 60)
    print("【new_session 演示】清空历史，重新开始")
    print("=" * 60)

    sess = SessionManager(graph="react")

    # 第一轮
    print(">>> 会话 A - 第 1 轮")
    answer = sess.chat("什么是肺结节？")
    print(f"Agent: {answer[:200]}\n")

    # 开启新会话
    sess.new_session()
    print(f">>> 切换到新会话: {sess.thread_id}\n")

    # 新会话第一轮（历史为空）
    print(">>> 会话 B - 第 1 轮（没有前文）")
    answer = sess.chat("它严重吗？")  # "它" 没有指代对象，因为历史已清空
    print(f"Agent: {answer[:300]}\n")


def demo_plan_llm_memory():
    """演示 plan_llm 图的多轮记忆"""
    print("=" * 60)
    print("【plan_llm 图 + 记忆】多步骤任务后追问")
    print("=" * 60)

    sess = SessionManager(graph="plan_llm")

    # 第一轮：图片检测
    query1 = "帮我检测这张图片，路径为:data/pexels-athletes-1867185_1920.jpg，并解释检测结果"
    print(f">>> 第 1 轮\n用户: {query1}")
    answer1 = sess.chat(query1)
    print(f"Agent: {answer1[:300]}\n")

    # 第二轮：追问（自动携带上文）
    query2 = "这些检测到的物体在医学上有什么意义？"
    print(f">>> 第 2 轮（追问）\n用户: {query2}")
    answer2 = sess.chat(query2)
    print(f"Agent: {answer2[:300]}\n")


if __name__ == "__main__":
    import sys
    
    while  True:
        # 交互模式
        sess = SessionManager()
        print(f"会话已创建: {sess.thread_id}")
        print("输入 'quit' 退出，'new' 开启新会话\n")
        while True:
            query = input("你: ").strip()
            if query.lower() == "quit":
                break
            if query.lower() == "new":
                sid = sess.new_session()
                print(f"[新会话: {sid}]")
                continue
            if not query:
                continue
            answer = sess.chat(query)
            print(f"Agent: {answer}\n")
