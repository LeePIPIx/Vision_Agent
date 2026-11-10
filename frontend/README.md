# Medical Agent 前端

前端随 `backend_api.py` 在同一端口提供。先运行根目录 README 中的安装和启动命令，再在 VS Code“端口”面板点击服务器 8000 对应的转发地址。页面会自动使用同一地址访问 API。

登录后可直接点击“运行公开 CT Demo”，也可上传三维腹部 CT 后直接发送分析请求，系统会自动进入脾脏分割流程。任务进度由 `/api/tasks/{id}/events` 推送，结果、复核意见和历史会话保存在服务端。浏览器刷新后会重新读取服务器数据。点击“任务记录”可查看失败原因和重试。文件通过需登录的 `/api/files/{id}` 下载。

新接口：`/api/auth/register`、`/api/auth/login`、`/api/sessions`、`/api/tasks`、`/api/tasks/{id}/events`、`/api/tasks/{id}/review`。任务提交采用 multipart 表单，`payload` 为 JSON，`file` 为可选附件。
