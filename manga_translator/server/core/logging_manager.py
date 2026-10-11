"""
日志管理模块

负责日志队列管理、任务日志隔离和日志导出功能。
"""

import contextvars
import logging
import threading
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Optional

# Log queues by task ID (each task has a queue of its own)
task_logs = defaultdict(lambda: deque(maxlen=1000))  # At most 1000 log entries are kept per task
task_logs_lock = threading.Lock()

# Global log queue (for the administrator to view all logs)
# Limited to 1000 entries, so too many logs do not cost memory or cause stutter
global_log_queue = deque(maxlen=1000)

# Thread-local storage for the current task ID
current_task_id = contextvars.ContextVar('current_task_id', default=None)

# Context variable for the current session ID (for filtering logs by session)
current_session_id = contextvars.ContextVar('current_session_id', default=None)


def generate_task_id() -> str:
    """生成唯一的任务ID"""
    return str(uuid.uuid4())


def set_task_id(task_id: str):
    """设置当前任务ID"""
    current_task_id.set(task_id)


def get_task_id() -> Optional[str]:
    """获取当前任务ID"""
    return current_task_id.get()


def set_session_id(session_id: str):
    """设置当前会话ID"""
    current_session_id.set(session_id)


def get_session_id() -> Optional[str]:
    """获取当前会话ID"""
    return current_session_id.get()


def add_log(message: str, level: str = "INFO", task_id: Optional[str] = None, session_id: Optional[str] = None, skip_print: bool = False):
    """
    添加日志到队列（支持任务隔离和会话隔离）
    
    Args:
        message: 日志消息
        level: 日志级别
        task_id: 任务ID（可选，如果不提供则从上下文获取）
        session_id: 会话ID（可选，如果不提供则从上下文获取）
        skip_print: 是否跳过控制台输出（避免与 logging handler 重复输出）
    """
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "message": message
    }
    
    # When no task_id is given, try to get it from the context
    if task_id is None:
        task_id = get_task_id()
    
    # When no session_id is given, try to get it from the context
    if session_id is None:
        session_id = get_session_id()
    
    # Add the session ID to the log entry
    if session_id:
        log_entry['session_id'] = session_id
    
    with task_logs_lock:
        # Add to the global log queue
        global_log_queue.append(log_entry)
        
        # With a task_id, add to the queue of that task as well
        if task_id:
            log_entry_with_id = log_entry.copy()
            log_entry_with_id['task_id'] = task_id
            task_logs[task_id].append(log_entry_with_id)
    
    # Print to the console too (unless skip_print=True, to avoid duplicating the logging handler)
    if not skip_print:
        task_prefix = task_id[:8] if task_id else 'GLOBAL'
        session_prefix = f" S:{session_id[:8]}" if session_id else ""
        print(f"[{level}] [{task_prefix}{session_prefix}] {message}")


def get_logs(level: Optional[str] = None, limit: int = 100, task_id: Optional[str] = None, session_id: Optional[str] = None) -> list:
    """
    获取日志
    
    Args:
        level: 日志级别过滤（INFO, WARNING, ERROR等）
        limit: 返回的日志数量限制
        task_id: 任务ID（如果指定，只返回该任务的日志）
        session_id: 会话ID（如果指定，只返回该会话的日志）
    
    Returns:
        日志列表
    """
    with task_logs_lock:
        if task_id:
            # Return the logs of the given task
            logs = list(task_logs.get(task_id, []))
        else:
            # Return the global logs
            logs = list(global_log_queue)
    
    # Filter by session ID
    if session_id:
        logs = [log for log in logs if log.get('session_id') == session_id]
    
    # Filter by level
    if level and level.lower() != 'all':
        logs = [log for log in logs if log['level'].lower() == level.lower()]
    
    # Limit the number (the newest are returned)
    if len(logs) > limit:
        logs = logs[-limit:]
    
    return logs


def get_task_logs(task_id: str, limit: int = 50) -> list:
    """
    获取指定任务的日志（简化接口）
    
    Args:
        task_id: 任务ID
        limit: 返回的日志数量限制
    
    Returns:
        日志列表
    """
    return get_logs(task_id=task_id, limit=limit)


def export_logs(task_id: Optional[str] = None) -> tuple[str, str]:
    """
    导出日志为文本文件
    
    Args:
        task_id: 任务ID（可选）
    
    Returns:
        (filename, log_text) 元组
    """
    with task_logs_lock:
        if task_id:
            logs = list(task_logs.get(task_id, []))
            filename = f"logs_{task_id[:8]}.txt"
        else:
            logs = list(global_log_queue)
            from datetime import timezone
            filename = f"logs_all_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.txt"
    
    # Build the log text
    log_text = "\n".join([
        f"[{log['timestamp']}] [{log['level']}] {log['message']}"
        for log in logs
    ])
    
    return filename, log_text


class WebLogHandler(logging.Handler):
    """自定义日志处理器，捕获manga_translator的日志"""
    
    def __init__(self):
        super().__init__()
        # Filter out unwanted log sources
        self.ignored_loggers = {'uvicorn.access', 'uvicorn.error', 'httpcore', 'httpx'}
    
    def emit(self, record):
        try:
            # Skip noise such as the uvicorn access log
            if record.name in self.ignored_loggers:
                return
            
            msg = self.format(record)
            # Extract the log level and the message
            level = record.levelname
            # Get task_id and session_id from the context
            task_id = get_task_id()
            session_id = get_session_id()
            # skip_print=True avoids duplicating the output of logging's root handler
            add_log(msg, level, task_id, session_id, skip_print=True)
        except Exception:
            self.handleError(record)


# Whether the log handlers have been set up already
_log_handler_initialized = False


def setup_log_handler():
    """设置日志处理器"""
    global _log_handler_initialized
    
    # Guard against initialising twice
    if _log_handler_initialized:
        return
    
    # Initialise the translator's logging (to make sure the manga-translator logger is set up correctly)
    try:
        from manga_translator.utils.log import init_logging
        init_logging()
    except ImportError:
        pass
    
    web_log_handler = WebLogHandler()
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    web_log_handler.setFormatter(formatter)
    web_log_handler.setLevel(logging.INFO)  # Capture records at INFO level and above
    
    # Add to the manga_translator logger (the namespace with an underscore, for the server modules)
    mt_logger = logging.getLogger('manga_translator')
    mt_logger.addHandler(web_log_handler)
    mt_logger.propagate = False
    if mt_logger.level == logging.NOTSET or mt_logger.level > logging.INFO:
        mt_logger.setLevel(logging.INFO)
    
    # Add to the manga-translator logger (the namespace with a hyphen, for core modules such as the translators)
    # This is the namespace the translator, OCR, detection and similar modules use
    mt_hyphen_logger = logging.getLogger('manga-translator')
    mt_hyphen_logger.addHandler(web_log_handler)
    # propagate is not set to False, so records of child loggers can propagate here
    if mt_hyphen_logger.level == logging.NOTSET or mt_hyphen_logger.level > logging.INFO:
        mt_hyphen_logger.setLevel(logging.INFO)
    
    # Make sure the records of submodules are captured as well
    submodules = ['translators', 'detection', 'ocr', 'inpainting', 'rendering', 'upscaling', 'colorization']
    
    # List of translator class names (these are the logger names actually used)
    translator_names = [
        'OpenAITranslator', 'OpenAIHighQualityTranslator', 
        'GeminiTranslator', 'GeminiHighQualityTranslator',
        'VertexHighQualityTranslator',
        'SakuraTranslator', 'Qwen2Translator',
        'DeepLTranslator', 'GoogleTranslator', 'BaiduTranslator',
        'PapagoTranslator', 'YandexTranslator', 'ChatGPTTranslator'
    ]
    
    for submodule in submodules:
        # Namespace with an underscore
        sub_logger = logging.getLogger(f'manga_translator.{submodule}')
        if sub_logger.level == logging.NOTSET or sub_logger.level > logging.INFO:
            sub_logger.setLevel(logging.INFO)
        # Namespace with a hyphen
        sub_logger_hyphen = logging.getLogger(f'manga-translator.{submodule}')
        if sub_logger_hyphen.level == logging.NOTSET or sub_logger_hyphen.level > logging.INFO:
            sub_logger_hyphen.setLevel(logging.INFO)
    
    # Set the logger level for each translator class name
    for name in translator_names:
        translator_logger = logging.getLogger(f'manga-translator.{name}')
        if translator_logger.level == logging.NOTSET or translator_logger.level > logging.INFO:
            translator_logger.setLevel(logging.INFO)
    
    _log_handler_initialized = True
    
    # Add a test log entry to confirm the system works
    add_log("Logging system initialized", "INFO")
