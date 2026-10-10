"""Errors that the pipeline ignores on purpose must still leave a trace."""
import _bootstrap  # noqa: F401, I001

import ast
import logging
from pathlib import Path

import pytest

from manga_translator.utils.swallowed import note_ignored_error

PACKAGE = Path(__file__).resolve().parents[1] / "manga_translator"
# The translation path. Vendored model code and the logging modules themselves are left alone.
CHECKED = ["manga_translator.py", "translators", "rendering", "ocr", "detection", "inpainting", "mode", "utils"]
SKIPPED_PARTS = {"ldm", "panel", "default_utils"}
SKIPPED_FILES = {"log.py", "log_redaction.py", "swallowed.py"}
REPORTING_CALLS = {"debug", "info", "warning", "warn", "error", "exception", "critical", "print", "log", "note_ignored_error"}


def _checked_files():
    for entry in CHECKED:
        target = PACKAGE / entry
        for path in [target] if target.is_file() else sorted(target.rglob("*.py")):
            if path.name not in SKIPPED_FILES and not SKIPPED_PARTS & set(path.parts):
                yield path


def _is_broad(handler):
    if handler.type is None:
        return True
    names = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(isinstance(name, ast.Name) and name.id in ("Exception", "BaseException") for name in names)


def _leaves_a_trace(handler):
    for node in ast.walk(ast.Module(body=handler.body, type_ignores=[])):
        if isinstance(node, ast.Raise):
            return True
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if name in REPORTING_CALLS:
                return True
        if handler.name and isinstance(node, ast.Name) and node.id == handler.name:
            return True
    return False


def test_should_record_the_place_and_the_error_at_debug_level(caplog):
    with caplog.at_level(logging.DEBUG, logger="manga-translator"):
        note_ignored_error(ValueError("bad size"), "module.py:function")

    assert caplog.records[0].levelno == logging.DEBUG
    assert caplog.records[0].getMessage() == "module.py:function ignored ValueError: bad size"


def test_should_never_raise_into_the_caller():
    class Unprintable(Exception):
        def __str__(self):
            raise RuntimeError("cannot be shown")

    note_ignored_error(Unprintable(), "module.py:function")


@pytest.mark.parametrize("path", list(_checked_files()), ids=lambda path: path.relative_to(PACKAGE).as_posix())
def test_translation_path_should_have_no_silent_broad_handler(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))

    silent = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler) and _is_broad(node) and not _leaves_a_trace(node)
    ]

    assert silent == [], f"handlers that drop the error without a trace at lines {silent}"
