"""Common test set-up: sys.path, offscreen, and the loading order of torch and PyQt6.

Any test script that uses Qt or the code of this repository writes this as its **first** import:

    import _bootstrap  # noqa: F401

It gathers three things every test would otherwise have to remember by itself:

1. ``sys.path`` - the repository root + ``desktop_qt_ui`` (otherwise ``No module named 'editor'``);
2. ``QT_QPA_PLATFORM=offscreen`` - must come before any PyQt6 import;
3. **torch must be loaded before PyQt6**. This is a hard constraint on Windows: the Qt DLL
   search path of PyQt6 displaces the dependency resolution of ``c10.dll``, and importing the other way round gives
   ``OSError: [WinError 1114]`` (a dynamic link library (DLL) initialisation routine failed).
   The real entry point of the desktop app, ``desktop_qt_ui/main.py``, does exactly this (see the issue quoted there,
   https://github.com/pytorch/pytorch/issues/166628), and the tests follow the same approach.

An environment without torch runs as usual - when the preload fails it is skipped, and pure Qt tests are not affected.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "desktop_qt_ui"):
    entry = str(path)
    if entry not in sys.path:
        sys.path.insert(0, entry)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Must come before any PyQt6 import; the module docstring explains why.
try:
    import torch  # noqa: F401
except ImportError:
    pass

__all__ = ["ROOT"]
