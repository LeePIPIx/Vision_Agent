import logging
import os
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

# ANSI 颜色码
RESET = "\033[0m"
COLORS = {
    logging.DEBUG:    "\033[36m",   # 青色
    logging.INFO:     "\033[32m",   # 绿色
    logging.WARNING:  "\033[33m",   # 黄色
    logging.ERROR:    "\033[31m",   # 红色
    logging.CRITICAL: "\033[1;31m", # 加粗红色
}


class ColoredFormatter(logging.Formatter):
    """给不同日志级别上色，仅用于控制台输出。"""

    def format(self, record):
        color = COLORS.get(record.levelno, RESET)
        # 给整行日志包上颜色
        return color + super().format(record) + RESET


_configured = False


def get_logger(name: str) -> logging.Logger:
    """
    获取一个配置好的 logger。

    首次调用时自动初始化日志系统：
    - 同时输出到控制台和日志文件
    - 日志文件按天轮转，保留最近 7 天
    - 日志级别通过环境变量 LOG_LEVEL 控制，默认 INFO

    用法:
        from utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("something happened")
    """
    global _configured
    if not _configured:
        _setup()
        _configured = True
    return logging.getLogger(name)


def _setup():
    level = _parse_level()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台输出（带颜色）
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(ColoredFormatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    # 文件输出：按天轮转，保留 7 天
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = TimedRotatingFileHandler(
        filename=str(log_dir / "app.log"),
        when="midnight",
        backupCount=7,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)

    # 配置根 logger 同时输出到控制台和文件
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(console)
    root.addHandler(file_handler)

    # 抑制第三方库的日志噪音
    for lib in ["httpx", "httpcore", "openai", "urllib3", "PIL"]:
        logging.getLogger(lib).setLevel(max(level, logging.WARNING))


def _parse_level() -> int:
    """从环境变量 LOG_LEVEL 解析日志级别，默认 INFO。"""
    raw = os.getenv("LOG_LEVEL", "INFO").upper()
    # 尝试将字符串转换为 logging 模块的级别常量
    level = getattr(logging, raw, None)
    if isinstance(level, int):
        return level
    # 无效值时回退到 INFO 并提示
    fallback = logging.INFO
    # 这里不能用 logger（尚未初始化），直接用 print
    print(f"[logger] 无效的 LOG_LEVEL='{raw}'，回退为 INFO")
    return fallback
