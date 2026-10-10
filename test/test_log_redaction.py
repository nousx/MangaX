"""Credentials must never reach a log file."""
import logging

import pytest

from manga_translator.utils.log_redaction import REDACTED, redact_secrets, safe_url_for_log

# Made-up values in the shapes real credentials have. None of them is a real key.
FAKE_OPENAI_KEY = "sk-" + "testonly" * 4
FAKE_GOOGLE_KEY = "AIza" + "TestOnlyValue" * 3
FAKE_GATEWAY_KEY = "gateway-key-0123456789"


@pytest.mark.parametrize("url,expected", [
    ("https://api.example.com/v1/chat/completions", "https://api.example.com/v1/chat/completions"),
    ("https://api.example.com:8443/v1/models", "https://api.example.com:8443/v1/models"),
    ("http://127.0.0.1:11434/v1/models", "http://127.0.0.1:11434/v1/models"),
    ("https://[::1]:8080/v1/models", "https://[::1]:8080/v1/models"),
])
def test_should_keep_urls_that_carry_no_credentials(url, expected):
    assert safe_url_for_log(url) == expected


def test_should_drop_a_password_embedded_in_the_url():
    logged = safe_url_for_log(f"https://user:{FAKE_GATEWAY_KEY}@proxy.example.com/v1/models")

    assert logged == "https://proxy.example.com/v1/models"


def test_should_drop_the_query_string():
    logged = safe_url_for_log(f"https://host.example.com/v1beta/models?key={FAKE_GOOGLE_KEY}&alt=sse")

    assert FAKE_GOOGLE_KEY not in logged
    assert logged == f"https://host.example.com/v1beta/models?{REDACTED}"


@pytest.mark.parametrize("value", ["not a url", "//missing-scheme/path", f"key={FAKE_GATEWAY_KEY}"])
def test_should_not_echo_text_that_is_not_a_plain_url(value):
    assert safe_url_for_log(value) == REDACTED


@pytest.mark.parametrize("value", ["", None])
def test_should_return_empty_text_for_a_missing_url(value):
    assert safe_url_for_log(value) == ""


def test_should_remove_the_configured_key_wherever_it_appears():
    body = f'{{"error": "Incorrect API key provided: {FAKE_GATEWAY_KEY}."}}'

    assert FAKE_GATEWAY_KEY not in redact_secrets(body, (FAKE_GATEWAY_KEY,))


@pytest.mark.parametrize("body", [
    f"Incorrect API key provided: {FAKE_OPENAI_KEY}",
    f"API key not valid: {FAKE_GOOGLE_KEY}",
    f"Authorization: Bearer {FAKE_GATEWAY_KEY}",
    f'{{"api_key": "{FAKE_GATEWAY_KEY}"}}',
    f"access_token={FAKE_GATEWAY_KEY}&x=1",
    f"password: {FAKE_GATEWAY_KEY}",
])
def test_should_remove_credential_shaped_values_without_knowing_them(body):
    redacted = redact_secrets(body)

    assert FAKE_OPENAI_KEY not in redacted
    assert FAKE_GOOGLE_KEY not in redacted
    assert FAKE_GATEWAY_KEY not in redacted
    assert REDACTED in redacted


def test_should_keep_the_rest_of_the_message_readable():
    redacted = redact_secrets(f"Rate limit exceeded for model x. Authorization: Bearer {FAKE_GATEWAY_KEY}")

    assert redacted.startswith("Rate limit exceeded for model x.")


@pytest.mark.parametrize("secret", ["", "abc", None, 12345])
def test_should_ignore_known_secrets_too_short_or_not_text(secret):
    assert redact_secrets("status abc 12345 ok", (secret,)) == "status abc 12345 ok"


@pytest.mark.parametrize("text", ["", None])
def test_should_handle_missing_text(text):
    assert redact_secrets(text) == ""


def test_api_client_error_logs_should_not_contain_the_key_or_url_credentials(caplog):
    """The log lines the API clients write on a failed request, built the same way."""
    from manga_translator.translators import common
    from manga_translator.utils.retry import summarize_response_text

    url = f"https://user:{FAKE_GATEWAY_KEY}@proxy.example.com/v1/chat/completions?key={FAKE_GOOGLE_KEY}"
    body = f'{{"error": {{"message": "Incorrect API key provided: {FAKE_OPENAI_KEY}"}}}}'

    with caplog.at_level(logging.ERROR):
        common._http_logger.error(f"[AsyncOpenAICurlCffi] Error - URL: {safe_url_for_log(url)}")
        common._http_logger.error(
            f"[AsyncOpenAICurlCffi] Error - Response: "
            f"{redact_secrets(summarize_response_text(body), (FAKE_OPENAI_KEY,))}"
        )

    logged = "\n".join(record.getMessage() for record in caplog.records)
    for secret in (FAKE_GATEWAY_KEY, FAKE_GOOGLE_KEY, FAKE_OPENAI_KEY):
        assert secret not in logged
    assert "proxy.example.com/v1/chat/completions" in logged


def test_api_clients_should_not_log_a_raw_url_or_response_body():
    """Guard the call sites: every error log line must go through the redaction helpers."""
    import inspect
    import re

    from manga_translator.translators import common

    source = inspect.getsource(common)
    url_lines = re.findall(r"_http_logger\.\w+\(.*Error - URL: .*", source)
    body_lines = re.findall(r"_http_logger\.\w+\(.*Error - Response: .*", source)

    assert len(url_lines) == 4 and all("safe_url_for_log(url)" in line for line in url_lines)
    assert len(body_lines) == 4 and all("redact_secrets(" in line for line in body_lines)
