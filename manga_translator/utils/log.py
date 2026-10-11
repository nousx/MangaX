import logging
import sys

import colorama

from .generic import replace_prefix

ROOT_TAG = 'manga-translator'

class Formatter(logging.Formatter):
    def formatMessage(self, record: logging.LogRecord) -> str:
        if record.levelno >= logging.ERROR:
            self._style._fmt = f'{colorama.Fore.RED}%(levelname)s:{colorama.Fore.RESET} [%(name)s] %(message)s'
        elif record.levelno >= logging.WARN:
            self._style._fmt = f'{colorama.Fore.YELLOW}%(levelname)s:{colorama.Fore.RESET} [%(name)s] %(message)s'
        elif record.levelno == logging.DEBUG:
            self._style._fmt = '[%(name)s] %(message)s'
        else:
            self._style._fmt = '[%(name)s] %(message)s'
        result = super().formatMessage(record)
        # ✅ Flush the output after every formatted record
        sys.stdout.flush()
        return result

class Filter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Try to filter out logs from imported modules
        if not (record.name.startswith(ROOT_TAG) or record.name.startswith('desktop-ui')):
            return False
        # Shorten the name
        record.name = replace_prefix(record.name, ROOT_TAG + '.', '')
        record.name = replace_prefix(record.name, 'desktop-ui.', '')
        return super().filter(record)

root = logging.getLogger(ROOT_TAG)
_initialized = False

def init_logging():
    global _initialized
    # Before anything is logged: no credential may reach a log file. Idempotent.
    from .log_redaction import install_log_redaction
    install_log_redaction()
    if _initialized:
        return
    _initialized = True
    
    # ✅ Force standard output to flush (fixes a log that appears stuck)
    import os
    import sys
    # Set the environment variable that turns off Python output buffering
    os.environ['PYTHONUNBUFFERED'] = '1'
    # When stdout has a reconfigure method, make it unbuffered
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass
    
    # Add the handler directly (without relying on basicConfig)
    if not logging.root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        logging.root.addHandler(handler)
    
    # Only the StreamHandler (console) is changed, not the FileHandler (log file),
    # so the original format of the file log handler configured by main.py is kept
    for h in logging.root.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            h.setFormatter(Formatter())
            h.addFilter(Filter())
            h.setLevel(logging.INFO)
            # ✅ Flush the output after every log record
            h.flush()
    
    # Explicitly set the root logger level
    root.setLevel(logging.INFO)
    logging.getLogger().setLevel(logging.INFO)

def set_log_level(level):
    root.setLevel(level)
    # Also set the root logger level to ensure DEBUG messages pass through
    logging.getLogger().setLevel(level)
    # Set the level of every handler as well
    for handler in logging.root.handlers:
        handler.setLevel(level)

def get_logger(name: str):
    return root.getChild(name)

file_handlers = {}

def add_file_logger(path: str):
    if path in file_handlers:
        return
    file_handlers[path] = logging.FileHandler(path, encoding='utf8')
    logging.root.addHandler(file_handlers[path])

def remove_file_logger(path: str):
    if path in file_handlers:
        logging.root.removeHandler(file_handlers[path])
        file_handlers[path].close()
        del file_handlers[path]
