"""Model and reasoning-effort choices for the Codex and Claude command-line translators.

The values end up as command-line arguments, so everything that comes from a
settings file or from the CLI's own cache is checked here before it is used.
"""

import json
import os
import re
from pathlib import Path

DEFAULT_EFFORT = "low"
# Higher effort means slower and more expensive requests; translation rarely needs it.
CODEX_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
CLAUDE_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
# Aliases the Claude Code CLI resolves to the newest model of each family.
CLAUDE_MODEL_ALIASES = ("haiku", "sonnet", "opus")

_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:\[\]-]{0,99}")
_MODELS_CACHE_NAME = "models_cache.json"
_MODELS_CACHE_MAX_BYTES = 2 * 1024 * 1024
_MAX_LISTED_MODELS = 50


def safe_model_name(value) -> str:
    """Return the model name when it is one a CLI could accept, otherwise an empty string."""
    text = value.strip() if isinstance(value, str) else ""
    return text if _MODEL_NAME.fullmatch(text) else ""


def safe_effort(value, levels=CLAUDE_EFFORT_LEVELS) -> str:
    """Return the effort level when it is one of the allowed levels, otherwise the default."""
    text = value.strip().lower() if isinstance(value, str) else ""
    return text if text in levels else DEFAULT_EFFORT


def codex_home(env=None) -> Path:
    """The folder where the Codex CLI keeps its account files."""
    env = os.environ if env is None else env
    configured = env.get("CODEX_HOME")
    if configured:
        return Path(configured)
    return Path(env.get("USERPROFILE") or env.get("HOME") or Path.home()) / ".codex"


def list_codex_models(home: Path | None = None) -> list[tuple[str, str]]:
    """Models the signed-in Codex CLI offers, as (name, display name), from the CLI's own cache.

    Returns an empty list when the cache is missing or cannot be read: the
    model setting then only offers the CLI default.
    """
    cache = (home or codex_home()) / _MODELS_CACHE_NAME
    try:
        if not cache.is_file() or cache.stat().st_size > _MODELS_CACHE_MAX_BYTES:
            return []
        data = json.loads(cache.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    entries = data.get("models") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        return []
    models = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("visibility") != "list":
            continue
        name = safe_model_name(entry.get("slug") or entry.get("id"))
        if not name or any(name == listed for listed, _ in models):
            continue
        label = entry.get("display_name")
        label = " ".join(label.split())[:60] if isinstance(label, str) and label.strip() else name
        models.append((name, label))
        if len(models) >= _MAX_LISTED_MODELS:
            break
    return models
