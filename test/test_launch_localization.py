"""The maintenance tool must not show Chinese-only text or invented graphics cards."""
import _bootstrap  # noqa: F401

import ast
import importlib.util
import re

ROOT = _bootstrap.ROOT
LAUNCH = ROOT / "packaging" / "launch.py"
CJK = re.compile(r"[一-鿿]")

# Literals that are not shown on their own: a marker matched against detector
# output, and labels whose English text is passed alongside them.
ALLOWED_CHINESE_ONLY = {"安装损坏", "安装", "更新"}

REGISTRY_OUTPUT = """
HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}\\0000
    DriverDesc    REG_SZ    NVIDIA GeForce RTX 4090

HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}\\0001
    DriverDesc    REG_SZ    AMD Radeon(TM) Graphics

End of search: 4 match(es) found.
"""


def load_launch(name):
    spec = importlib.util.spec_from_file_location(name, LAUNCH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def chinese_only_literals():
    tree = ast.parse(LAUNCH.read_text(encoding="utf-8"))
    translated = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "L":
            translated.update(id(child) for child in ast.walk(node))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                translated.add(id(body[0].value))
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and CJK.search(node.value)
        and id(node) not in translated
    }


def test_should_give_every_chinese_message_an_english_text():
    assert chinese_only_literals() <= ALLOWED_CHINESE_ONLY


def test_should_use_english_text_when_language_is_not_chinese():
    launch = load_launch("launch_language_test")

    launch.LANG = "en"
    assert launch.L("已是最新", "up to date") == "up to date"
    launch.LANG = "zh"
    assert launch.L("已是最新", "up to date") == "已是最新"


def test_should_not_list_registry_summary_line_as_a_graphics_card(monkeypatch, capsys):
    launch = load_launch("launch_registry_gpu_test")

    def fake_check_output(command, *args, **kwargs):
        if "reg query" in command:
            return REGISTRY_OUTPUT
        raise OSError("tool unavailable")

    monkeypatch.setattr(launch.sys, "platform", "win32")
    monkeypatch.setattr(launch.subprocess, "check_output", fake_check_output)
    # Choosing entry 3 would pick the invented card if it were still listed.
    monkeypatch.setenv("MANGAT_SELECTED_GPU", "3")

    gpu_type, gpu_name = launch.detect_gpu(interactive=False)[:2]

    assert (gpu_type, gpu_name) == ("NVIDIA", "NVIDIA GeForce RTX 4090")
    assert "End of search" not in capsys.readouterr().out
