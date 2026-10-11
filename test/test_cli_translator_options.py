"""Choosing the model and the reasoning effort of the Codex and Claude command-line translators."""
import _bootstrap  # noqa: F401, I001

import json

import pytest

from manga_translator.cli_translator_options import (
    CLAUDE_EFFORT_LEVELS,
    CODEX_EFFORT_LEVELS,
    list_codex_models,
    safe_effort,
    safe_model_name,
)
from manga_translator.config import Config, TranslatorConfig
from manga_translator.translators.claude_cli import ClaudeCLITranslator
from manga_translator.translators.codex_cli import CodexCLITranslator


def configured(translator, **settings):
    translator.parse_args(Config(translator=TranslatorConfig(**settings)))
    return translator


@pytest.mark.parametrize("name", ["haiku", "sonnet", "gpt-6-astra", "claude-opus-5-5", "sonnet[1m]", "vendor:model.v2"])
def test_model_name_should_be_kept_when_a_cli_could_accept_it(name):
    assert safe_model_name(f"  {name} ") == name


@pytest.mark.parametrize("name", ["", "   ", "--dangerously-skip-permissions", "a b", 'x" -c y="z', "x" * 101, None, 7])
def test_model_name_should_be_dropped_when_it_is_not_a_plain_name(name):
    assert safe_model_name(name) == ""


@pytest.mark.parametrize("level", CLAUDE_EFFORT_LEVELS)
def test_effort_should_be_kept_when_it_is_an_allowed_level(level):
    assert safe_effort(level.upper(), CLAUDE_EFFORT_LEVELS) == level


@pytest.mark.parametrize("level", ["", "extreme", 'low" -c x="y', None, 3])
def test_effort_should_fall_back_to_low_when_it_is_not_an_allowed_level(level):
    assert safe_effort(level, CODEX_EFFORT_LEVELS) == "low"


def test_settings_should_default_to_low_effort():
    settings = TranslatorConfig()

    assert (settings.codex_effort, settings.claude_effort) == ("low", "low")


def test_claude_command_should_use_the_chosen_model_and_effort():
    translator = configured(ClaudeCLITranslator(), claude_model="haiku", claude_effort="high")

    command = translator._build_command("claude-bin", "English", "Thai")

    assert command[command.index("--model") + 1] == "haiku"
    assert command[command.index("--effort") + 1] == "high"


def test_claude_command_should_leave_the_model_to_the_cli_when_none_is_chosen():
    translator = configured(ClaudeCLITranslator(), claude_model="")

    command = translator._build_command("claude-bin", "English", "Thai")

    assert "--model" not in command
    assert command[command.index("--effort") + 1] == "low"


def test_claude_command_should_not_pass_on_a_model_that_looks_like_an_option():
    translator = configured(ClaudeCLITranslator(), claude_model="--dangerously-skip-permissions",
                            claude_effort="whatever")

    command = translator._build_command("claude-bin", "English", "Thai")

    assert "--dangerously-skip-permissions" not in command
    assert "--model" not in command
    assert command[command.index("--effort") + 1] == "low"


def test_codex_should_take_the_chosen_model_and_effort_from_the_settings():
    translator = configured(CodexCLITranslator(), codex_model="gpt-6-astra", codex_effort="medium")

    assert (translator.model, translator.effort) == ("gpt-6-astra", "medium")


def test_codex_should_ignore_an_effort_that_would_break_out_of_its_config_value():
    translator = configured(CodexCLITranslator(), codex_model='x" -c y="z', codex_effort='low" -c x="y')

    assert (translator.model, translator.effort) == ("", "low")


def test_codex_models_should_come_from_the_cache_of_the_cli(tmp_path):
    (tmp_path / "models_cache.json").write_text(json.dumps({"models": [
        {"slug": "gpt-6-astra", "display_name": "GPT-6-Astra", "visibility": "list"},
        {"slug": "gpt-reserve", "display_name": "GPT-Reserve", "visibility": "hide"},
        {"slug": "bad name", "display_name": "Bad", "visibility": "list"},
        {"slug": "gpt-6-astra", "display_name": "Duplicate", "visibility": "list"},
        {"slug": "gpt-6-luna", "visibility": "list"},
        "not an entry",
    ]}), encoding="utf-8")

    assert list_codex_models(tmp_path) == [("gpt-6-astra", "GPT-6-Astra"), ("gpt-6-luna", "gpt-6-luna")]


@pytest.mark.parametrize("content", [None, "not json", "[]", '{"models": "nope"}'])
def test_codex_models_should_be_empty_when_the_cache_is_missing_or_unreadable(tmp_path, content):
    if content is not None:
        (tmp_path / "models_cache.json").write_text(content, encoding="utf-8")

    assert list_codex_models(tmp_path) == []
