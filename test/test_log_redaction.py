"""Credentials must never reach a log file or an error message."""
import inspect
import logging
import re

import pytest

from manga_translator.utils.log_redaction import REDACTED, redact_secrets, safe_url_for_log
from manga_translator.utils.retry import (
    is_retryable_api_error,
    summarize_exception_message,
    summarize_response_text,
    summarize_text,
)

# Made-up values in the shapes real credentials have. None of them is a real key.
FAKE_OPENAI_KEY = "sk-" + "testonly" * 4
FAKE_GOOGLE_KEY = "AIza" + "TestOnlyValue" * 3
FAKE_GATEWAY_KEY = "gateway-key-0123456789"
FAKE_ACCOUNT_ID = "0123456789abcdef0123456789abcdef"
FAKE_JWT = "eyJ" + "a" * 20 + "." + "b" * 20 + "." + "c" * 20
ALL_FAKES = (FAKE_OPENAI_KEY, FAKE_GOOGLE_KEY, FAKE_GATEWAY_KEY, FAKE_ACCOUNT_ID, FAKE_JWT)


def assert_clean(text):
    for secret in ALL_FAKES:
        assert secret not in text, f"leaked {secret!r} in {text!r}"


class TestUrls:
    @pytest.mark.parametrize("url", [
        "https://api.example.com/v1/chat/completions",
        "https://api.example.com:8443/v1/models",
        "http://127.0.0.1:11434/v1/models",
        "https://[::1]:8080/v1/models",
        "https://generativelanguage.example.com/v1beta/models/gemini-2.5-pro:generateContent",
        "https://api.example.com/v1/models/gpt-4o-mini",
    ])
    def test_should_keep_plain_endpoint_urls_readable(self, url):
        assert safe_url_for_log(url) == url

    def test_should_drop_a_password_embedded_in_the_url(self):
        logged = safe_url_for_log(f"https://user:{FAKE_GATEWAY_KEY}@proxy.example.com/v1/models")

        assert logged == "https://proxy.example.com/v1/models"

    def test_should_drop_the_query_string(self):
        logged = safe_url_for_log(f"https://host.example.com/v1beta/models?key={FAKE_GOOGLE_KEY}&alt=sse")

        assert logged == f"https://host.example.com/v1beta/models?{REDACTED}"

    def test_should_drop_the_fragment(self):
        assert_clean(safe_url_for_log(f"https://host.example.com/v1/models#{FAKE_GATEWAY_KEY}"))

    @pytest.mark.parametrize("url", [
        f"https://gateway.example.com/v1/{FAKE_ACCOUNT_ID}/my-gateway/openai/chat/completions",
        f"https://proxy.example.com/{FAKE_GATEWAY_KEY}/v1/chat/completions",
        f"https://proxy.example.com/key/{FAKE_OPENAI_KEY}/v1/models",
        f"https://proxy.example.com/v1/models/{FAKE_OPENAI_KEY}",
        f"https://proxy.example.com/v1/models/{FAKE_GOOGLE_KEY}:generateContent",
        f"https://proxy.example.com/v1/models/gemini:{FAKE_GATEWAY_KEY}",
    ])
    def test_should_replace_path_segments_that_are_not_endpoint_words(self, url):
        logged = safe_url_for_log(url)

        assert_clean(logged)
        assert REDACTED in logged
        assert logged.startswith("https://")

    def test_should_keep_the_endpoint_after_a_redacted_base_path(self):
        logged = safe_url_for_log(f"https://proxy.example.com/{FAKE_GATEWAY_KEY}/v1/chat/completions")

        assert logged == f"https://proxy.example.com/{REDACTED}/v1/chat/completions"

    @pytest.mark.parametrize("value", ["not a url", "//missing-scheme/path", f"key={FAKE_GATEWAY_KEY}", "https://"])
    def test_should_not_echo_text_that_is_not_a_plain_url(self, value):
        assert safe_url_for_log(value) == REDACTED

    def test_should_not_echo_a_url_with_an_invalid_port(self):
        assert safe_url_for_log(f"https://host.example.com:notaport/{FAKE_GATEWAY_KEY}") == REDACTED

    @pytest.mark.parametrize("value", ["", None])
    def test_should_return_empty_text_for_a_missing_url(self, value):
        assert safe_url_for_log(value) == ""


class TestText:
    def test_should_remove_the_configured_key_wherever_it_appears(self):
        body = f'{{"error": "Incorrect API key provided: {FAKE_GATEWAY_KEY}."}}'

        assert_clean(redact_secrets(body, (FAKE_GATEWAY_KEY,)))

    @pytest.mark.parametrize("body", [
        f"Incorrect API key provided: {FAKE_OPENAI_KEY}",
        f"API key not valid: {FAKE_GOOGLE_KEY}",
        f"Authorization: Bearer {FAKE_GATEWAY_KEY}",
        f'{{"api_key": "{FAKE_GATEWAY_KEY}"}}',
        f"x-goog-api-key: {FAKE_GATEWAY_KEY}",
        f"access_token={FAKE_GATEWAY_KEY}&x=1",
        f"password: {FAKE_GATEWAY_KEY}",
        f"session token {FAKE_JWT} expired",
    ])
    def test_should_remove_credential_shaped_values_without_knowing_them(self, body):
        redacted = redact_secrets(body)

        assert_clean(redacted)
        assert REDACTED in redacted

    def test_should_make_urls_inside_a_message_safe(self):
        message = (
            f"Failed to connect to https://user:{FAKE_GATEWAY_KEY}@proxy.example.com/"
            f"{FAKE_ACCOUNT_ID}/v1/chat/completions?key={FAKE_GOOGLE_KEY} after 3 tries"
        )

        redacted = redact_secrets(message)

        assert_clean(redacted)
        assert "proxy.example.com" in redacted
        assert redacted.endswith("after 3 tries")

    def test_should_keep_the_rest_of_the_message_readable(self):
        redacted = redact_secrets(f"Rate limit exceeded for model x. Authorization: Bearer {FAKE_GATEWAY_KEY}")

        assert redacted.startswith("Rate limit exceeded for model x.")

    @pytest.mark.parametrize("text", [
        "Text translator returned empty queries",
        "Output file already exists: 003.webp",
        "HTTP 429 Too Many Requests",
        "status 401 unauthorized",
    ])
    def test_should_leave_ordinary_messages_untouched(self, text):
        assert redact_secrets(text) == text

    @pytest.mark.parametrize("secret", ["", "abc", None, 12345])
    def test_should_ignore_known_secrets_too_short_or_not_text(self, secret):
        assert redact_secrets("status abc 12345 ok", (secret,)) == "status abc 12345 ok"

    @pytest.mark.parametrize("text", ["", None])
    def test_should_handle_missing_text(self, text):
        assert redact_secrets(text) == ""


class TestSharedSummarizer:
    """Every error path builds its message with these helpers, so they must redact."""

    BODY = f'{{"error": {{"message": "Incorrect API key provided: {FAKE_OPENAI_KEY}"}}}}'

    @pytest.mark.parametrize("summarize", [summarize_text, summarize_response_text])
    def test_should_redact_response_bodies(self, summarize):
        assert_clean(summarize(self.BODY))

    def test_should_redact_exception_messages(self):
        error = RuntimeError(f"POST https://proxy.example.com/{FAKE_GATEWAY_KEY}/v1/models?key={FAKE_GOOGLE_KEY} failed")

        assert_clean(summarize_exception_message(error))

    def test_should_redact_before_truncating(self):
        """A key cut in half by the length limit would no longer be recognized."""
        text = "x" * 30 + f" Authorization: Bearer {FAKE_GATEWAY_KEY}"

        summary = summarize_text(text, limit=60)

        assert FAKE_GATEWAY_KEY[:8] not in summary

    def test_should_still_classify_errors_after_redaction(self):
        assert is_retryable_api_error(RuntimeError(f"HTTP 429 for key {FAKE_OPENAI_KEY}")) is True
        assert is_retryable_api_error(RuntimeError(f"HTTP 401 invalid api key {FAKE_OPENAI_KEY}")) is False

    def test_should_keep_the_placeholder_for_empty_text(self):
        assert summarize_response_text("", empty_placeholder="(empty response)") == "(empty response)"


class TestCallSites:
    def test_api_client_error_logs_should_not_contain_credentials(self, caplog):
        """The log lines the API clients write on a failed request, built the same way."""
        from manga_translator.translators import common

        url = (f"https://user:{FAKE_GATEWAY_KEY}@proxy.example.com/{FAKE_ACCOUNT_ID}"
               f"/v1/chat/completions?key={FAKE_GOOGLE_KEY}")
        body = f'{{"error": {{"message": "Incorrect API key provided: {FAKE_OPENAI_KEY}"}}}}'

        with caplog.at_level(logging.ERROR):
            common._http_logger.error(f"[AsyncOpenAICurlCffi] Error - URL: {safe_url_for_log(url)}")
            common._http_logger.error(
                f"[AsyncOpenAICurlCffi] Error - Response: "
                f"{redact_secrets(summarize_response_text(body), (FAKE_OPENAI_KEY,))}"
            )

        logged = "\n".join(record.getMessage() for record in caplog.records)
        assert_clean(logged)
        assert "proxy.example.com" in logged

    def test_api_clients_should_not_log_a_raw_url_or_response_body(self):
        """Every error log line of the API clients must go through the redaction helpers."""
        from manga_translator.translators import common

        source = inspect.getsource(common)
        url_lines = re.findall(r"_http_logger\.\w+\(.*Error - URL: .*", source)
        body_lines = re.findall(r"_http_logger\.\w+\(.*Error - Response: .*", source)

        assert len(url_lines) == 4 and all("safe_url_for_log(url)" in line for line in url_lines)
        assert len(body_lines) == 4 and all("redact_secrets(" in line for line in body_lines)

    def test_no_translator_code_should_put_a_raw_response_body_in_a_message(self):
        """A response body may only reach a message through the redacting summarizer."""
        from manga_translator.translators import common
        from manga_translator.utils import openai_image_interface

        for module in (common, openai_image_interface):
            source = inspect.getsource(module)
            raw_uses = [
                line.strip() for line in source.splitlines()
                if re.search(r"\{(?:response|resp)\.text\}", line)
            ]
            assert raw_uses == [], f"{module.__name__} formats a raw response body: {raw_uses}"

    def test_codex_failure_text_should_not_carry_credentials(self):
        from manga_translator.translators.codex_cli import CodexCLITranslator

        stderr = (
            f"user: some manga line\n"
            f"ERROR: request to https://proxy.example.com/{FAKE_GATEWAY_KEY}/v1/responses failed\n"
            f"error: unauthorized, token {FAKE_JWT} rejected\n"
            f"Authorization: Bearer {FAKE_OPENAI_KEY}\n"
        ).encode("utf-8")

        detail = CodexCLITranslator.failure_lines(stderr)

        assert_clean(detail)
        assert "unauthorized" in detail

    def test_claude_failure_text_should_not_carry_credentials(self):
        from manga_translator.translators.claude_cli import ClaudeCLITranslator

        detail = ClaudeCLITranslator._failure_detail(
            "", f"API error for https://api.example.com/v1/messages?key={FAKE_GOOGLE_KEY}: "
                f"invalid x-api-key: {FAKE_OPENAI_KEY}")

        assert_clean(detail)
        assert "API error" in detail
