"""Credentials must never reach a log file or an error message.

Most cases are generated from the registries in log_redaction, so a name or a
prefix added there is tested without anyone remembering to add a case.
"""
import inspect
import logging
import re

import pytest

from manga_translator.utils import log_redaction
from manga_translator.utils.log_redaction import (
    MODEL_METHODS,
    REDACTED,
    SAFE_PATH_SEGMENTS,
    SECRET_NAME_WORDS,
    SECRET_PREFIX_PATTERNS,
    install_log_redaction,
    redact_secrets,
    safe_url_for_log,
)
from manga_translator.utils.retry import (
    is_retryable_api_error,
    summarize_exception_message,
    summarize_response_text,
    summarize_text,
)

# Made-up values. None of them is a real credential.
OPAQUE = "Zx9Kq7Lm2Np4Rt6Vw8Yb0Cd1"            # no name, no known prefix
SHORT = "hunter2pass"                           # too short and plain for the shape rule
HEX_ID = "0123456789abcdef0123456789abcdef"
PREFIXED = {
    "sk-": "sk-" + "testonly" * 4,
    "AIza": "AIza" + "TestOnlyValue" * 3,
    "ghp_": "ghp_" + "TestOnly1" * 4,
    "gho_": "gho_" + "TestOnly2" * 4,
    "xoxb-": "xoxb-" + "testonly-123" * 2,
    "eyJ": "eyJ" + "a" * 20 + "." + "b" * 20 + "." + "c" * 20,
}
ALL_FAKES = (OPAQUE, SHORT, HEX_ID, *PREFIXED.values())
# Names as they appear in real headers and payloads, beyond the bare registry words.
COMPOUND_NAMES = (
    "x-goog-api-key", "x-api-key", "api_key", "api-key", "apikey", "access_token", "refresh_token",
    "id_token", "client_secret", "private_key", "secret_key", "proxy-authorization", "set-cookie",
    "sessionid", "session_id", "auth_token", "db_password", "X-Auth-Token", "OPENAI_API_KEY",
)


def assert_clean(text):
    for secret in ALL_FAKES:
        assert secret not in text, f"leaked {secret!r} in {text!r}"
        assert secret.lower() not in text.lower(), f"leaked {secret!r} (case changed) in {text!r}"


class TestRegistryCoverage:
    """One case per registry entry, so an entry that does nothing cannot hide."""

    @pytest.mark.parametrize("prefix", sorted(SECRET_PREFIX_PATTERNS))
    def test_every_prefix_has_a_sample(self, prefix):
        assert prefix in PREFIXED, f"add a made-up sample for the new prefix {prefix!r}"

    @pytest.mark.parametrize("prefix", sorted(SECRET_PREFIX_PATTERNS))
    def test_every_prefix_is_redacted_in_prose(self, prefix):
        assert_clean(redact_secrets(f"the value {PREFIXED[prefix]} was rejected"))

    @pytest.mark.parametrize("name", [*SECRET_NAME_WORDS, *COMPOUND_NAMES])
    @pytest.mark.parametrize("form", [
        "{name}: {value}", "{name}:{value}", "{name}={value}", '"{name}": "{value}"',
        "'{name}': '{value}'", "{name} = {value}", "{upper}: {value}", "{name} {value}",
    ])
    def test_every_name_is_redacted_in_every_form(self, name, form):
        text = "prefix " + form.format(name=name, upper=name.upper(), value=OPAQUE) + " suffix"

        redacted = redact_secrets(text)

        assert_clean(redacted)
        assert redacted.startswith("prefix ")

    @pytest.mark.parametrize("name", [*SECRET_NAME_WORDS, *COMPOUND_NAMES])
    def test_every_name_redacts_a_short_value_after_a_separator(self, name):
        assert_clean(redact_secrets(f"{name}={SHORT}"))

    @pytest.mark.parametrize("segment", sorted(SAFE_PATH_SEGMENTS))
    def test_every_allowed_path_word_is_kept(self, segment):
        assert safe_url_for_log(f"https://api.example.com/{segment}") == f"https://api.example.com/{segment}"

    @pytest.mark.parametrize("method", sorted(MODEL_METHODS))
    def test_every_model_method_is_kept_without_the_model_name(self, method):
        logged = safe_url_for_log(f"https://api.example.com/v1beta/models/{OPAQUE}:{method}")

        assert logged == f"https://api.example.com/v1beta/models/{REDACTED}:{method}"


class TestUrls:
    @pytest.mark.parametrize("url", [
        "https://api.example.com/v1/chat/completions",
        "https://api.example.com:8443/v1/models",
        "http://127.0.0.1:11434/v1/models",
        "https://[::1]:8080/v1/models",
    ])
    def test_should_keep_plain_endpoint_urls_readable(self, url):
        assert safe_url_for_log(url) == url

    def test_should_drop_a_password_embedded_in_the_url(self):
        assert safe_url_for_log(f"https://user:{OPAQUE}@proxy.example.com/v1/models") == "https://proxy.example.com/v1/models"

    def test_should_drop_the_query_string_and_fragment(self):
        logged = safe_url_for_log(f"https://host.example.com/v1beta/models?key={OPAQUE}&alt=sse#{SHORT}")

        assert logged == f"https://host.example.com/v1beta/models?{REDACTED}"

    @pytest.mark.parametrize("secret", ALL_FAKES)
    @pytest.mark.parametrize("template", [
        "https://gateway.example.com/v1/{s}/my-gateway/openai/chat/completions",
        "https://proxy.example.com/{s}/v1/chat/completions",
        "https://proxy.example.com/key/{s}/v1/models",
        "https://proxy.example.com/v1/models/{s}",
        "https://proxy.example.com/v1/models/{s}:generateContent",
        "https://proxy.example.com/v1/models/gemini:{s}",
        "https://proxy.example.com/v1/models/{s}/versions",
    ])
    def test_should_replace_every_path_segment_that_is_not_an_endpoint_word(self, secret, template):
        logged = safe_url_for_log(template.format(s=secret))

        assert_clean(logged)
        assert REDACTED in logged

    def test_should_not_keep_a_model_name_because_it_cannot_be_told_from_a_token(self):
        logged = safe_url_for_log("https://api.example.com/v1/models/some-model-name")

        assert logged == f"https://api.example.com/v1/models/{REDACTED}"

    def test_should_replace_a_host_label_that_looks_like_a_token(self):
        logged = safe_url_for_log(f"https://{HEX_ID}.gateway.example.com/v1/models")

        assert logged == f"https://{REDACTED}.gateway.example.com/v1/models"

    @pytest.mark.parametrize("value", ["not a url", "//missing-scheme/path", f"key={OPAQUE}", "https://"])
    def test_should_not_echo_text_that_is_not_a_plain_url(self, value):
        assert safe_url_for_log(value) == REDACTED

    def test_should_not_echo_a_url_with_an_invalid_port(self):
        assert safe_url_for_log(f"https://host.example.com:notaport/{OPAQUE}") == REDACTED

    @pytest.mark.parametrize("value", ["", None])
    def test_should_return_empty_text_for_a_missing_url(self, value):
        assert safe_url_for_log(value) == ""


class TestText:
    def test_should_remove_the_configured_key_wherever_it_appears(self):
        assert_clean(redact_secrets(f'{{"error": "Incorrect API key provided: {SHORT}."}}', (SHORT,)))

    def test_should_remove_the_configured_key_in_its_url_encoded_form(self):
        secret = "p@ss/word 1+x"

        redacted = redact_secrets("got p%40ss%2Fword%201%2Bx and p%40ss%2Fword+1%2Bx back", (secret,))

        assert "p%40ss" not in redacted

    def test_should_drop_every_pair_of_a_cookie_header(self):
        redacted = redact_secrets(f"Cookie: a=first{SHORT}; b=second{SHORT}; c={OPAQUE}")

        assert_clean(redacted)
        assert redacted == f"Cookie: {REDACTED}"

    @pytest.mark.parametrize("scheme", ["Bearer", "Basic", "Digest", "bearer"])
    def test_should_drop_the_value_of_an_authorization_scheme(self, scheme):
        assert_clean(redact_secrets(f"Authorization: {scheme} {SHORT}value"))

    def test_should_redact_an_unnamed_random_looking_value(self):
        assert_clean(redact_secrets(f'{{"error":"bad credential {OPAQUE}"}}'))
        assert_clean(redact_secrets(f"connect to proxy.example/{OPAQUE}/v1/chat failed"))

    def test_should_make_urls_inside_a_message_safe(self):
        message = (f"Failed to connect to https://user:{SHORT}@proxy.example.com/{HEX_ID}"
                   f"/v1/chat/completions?key={OPAQUE} after 3 tries")

        redacted = redact_secrets(message)

        assert_clean(redacted)
        assert "proxy.example.com" in redacted
        assert redacted.endswith("after 3 tries")

    @pytest.mark.parametrize("text", [
        "Text translator returned empty queries",
        "Output file already exists: 003.webp",
        "HTTP 429 Too Many Requests",
        "status 401 unauthorized",
        "invalid api key",
        "token expired, please sign in again",
        "The session was closed by the server",
        "Processing rolling batch 88/504 (images 436-440)",
        "sha256 f0e1d2c3b4a5968778695a4b3c2d1e0ff0e1d2c3b4a5968778695a4b3c2d1e0f",
        "Saved successfully: 003.webp",
        "Detection resolution: 760x16088",
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

    @pytest.mark.parametrize("summarize", [summarize_text, summarize_response_text])
    def test_should_redact_response_bodies(self, summarize):
        assert_clean(summarize(f'{{"error": {{"message": "Incorrect API key provided: {PREFIXED["sk-"]}"}}}}'))

    def test_should_redact_exception_messages(self):
        error = RuntimeError(f"POST https://proxy.example.com/{OPAQUE}/v1/models?key={SHORT} failed")

        assert_clean(summarize_exception_message(error))

    def test_should_redact_before_truncating(self):
        """A key cut in half by the length limit would no longer be recognized."""
        summary = summarize_text("x" * 30 + f" Authorization: Bearer {OPAQUE}", limit=60)

        assert OPAQUE[:8] not in summary

    def test_should_still_classify_errors_after_redaction(self):
        assert is_retryable_api_error(RuntimeError(f"HTTP 429 for key {PREFIXED['sk-']}")) is True
        assert is_retryable_api_error(RuntimeError(f"HTTP 401 invalid api key {PREFIXED['sk-']}")) is False

    def test_should_keep_the_placeholder_for_empty_text(self):
        assert summarize_response_text("", empty_placeholder="(empty response)") == "(empty response)"


class TestProcessWideSafetyNet:
    """A call site that forgot to redact must still not reach the log."""

    @pytest.fixture
    def installed(self):
        previous = logging.getLogRecordFactory()
        install_log_redaction()
        yield
        logging.setLogRecordFactory(previous)

    def test_should_redact_a_message_logged_without_any_helper(self, installed, caplog):
        logger = logging.getLogger("test-unredacted-call-site")
        with caplog.at_level(logging.DEBUG):
            logger.error(f"request to https://user:{SHORT}@h.example/{OPAQUE}/v1/models?key={HEX_ID} failed")
            logger.warning("retrying with %s=%s", "api_key", OPAQUE)
            logger.info(f"upstream said: Authorization: Bearer {PREFIXED['sk-']}")

        logged = "\n".join(record.getMessage() for record in caplog.records)
        assert len(caplog.records) == 3
        assert_clean(logged)

    def test_should_redact_an_exception_logged_as_an_argument(self, installed, caplog):
        logger = logging.getLogger("test-unredacted-call-site")
        with caplog.at_level(logging.ERROR):
            logger.error("Error during batch translation stage: %s", RuntimeError(f"token {OPAQUE} rejected"))

        assert_clean(caplog.records[0].getMessage())

    def test_should_leave_ordinary_records_and_their_arguments_alone(self, installed, caplog):
        logger = logging.getLogger("test-unredacted-call-site")
        with caplog.at_level(logging.INFO):
            logger.info("Processing rolling batch %d/%d", 88, 504)

        record = caplog.records[0]
        assert record.getMessage() == "Processing rolling batch 88/504"
        assert record.args == (88, 504)

    def test_should_not_raise_when_a_message_cannot_be_formatted(self, installed):
        """A malformed log call is the handler's problem to report, not a reason to crash here."""
        factory = logging.getLogRecordFactory()

        record = factory("name", logging.INFO, __file__, 1, "value %d", ("not a number",), None)

        assert record.msg == "value %d"
        assert record.args == ("not a number",)

    def test_should_install_only_once(self, installed):
        first = logging.getLogRecordFactory()
        install_log_redaction()

        assert logging.getLogRecordFactory() is first

    def test_backend_and_desktop_logging_should_install_it(self):
        from manga_translator.utils import log as backend_log

        assert "install_log_redaction()" in inspect.getsource(backend_log.init_logging)
        desktop_source = (
            __import__("pathlib").Path(__file__).resolve().parent.parent
            / "desktop_qt_ui" / "services" / "log_service.py"
        ).read_text(encoding="utf-8")
        assert "install_log_redaction()" in desktop_source


class TestCallSites:
    def test_api_clients_should_not_log_a_raw_url_or_response_body(self):
        """Every error log line of the API clients must go through the redaction helpers."""
        from manga_translator.translators import common

        source = inspect.getsource(common)
        url_lines = re.findall(r"_http_logger\.\w+\(.*Error - URL: .*", source)
        body_lines = re.findall(r"_http_logger\.\w+\(.*Error - Response: .*", source)

        assert len(url_lines) == 4 and all("safe_url_for_log(url)" in line for line in url_lines)
        assert len(body_lines) == 4 and all("redact_secrets(" in line for line in body_lines)

    def test_no_translator_code_should_format_a_raw_response_body(self):
        """A response body may only reach a message through the redacting summarizer."""
        from manga_translator.translators import common
        from manga_translator.utils import openai_image_interface

        for module in (common, openai_image_interface):
            raw_uses = [
                line.strip() for line in inspect.getsource(module).splitlines()
                if re.search(r"\{(?:response|resp)\.text\}", line)
            ]
            assert raw_uses == [], f"{module.__name__} formats a raw response body: {raw_uses}"

    def test_codex_failure_text_should_not_carry_credentials(self):
        from manga_translator.translators.codex_cli import CodexCLITranslator

        stderr = (
            f"user: some manga line\n"
            f"ERROR: request to https://proxy.example.com/{OPAQUE}/v1/responses failed\n"
            f"error: unauthorized, token {PREFIXED['eyJ']} rejected\n"
            f"Authorization: Bearer {PREFIXED['sk-']}\n"
        ).encode("utf-8")

        detail = CodexCLITranslator.failure_lines(stderr)

        assert_clean(detail)
        assert "unauthorized" in detail

    def test_claude_failure_text_should_not_carry_credentials(self):
        from manga_translator.translators.claude_cli import ClaudeCLITranslator

        detail = ClaudeCLITranslator._failure_detail(
            "", f"API error for https://api.example.com/v1/messages?key={OPAQUE}: invalid x-api-key: {SHORT}")

        assert_clean(detail)
        assert "API error" in detail


class TestTracebacksAndStacks:
    """The handler formats tracebacks after the record is created; they must be redacted too."""

    @pytest.fixture
    def installed(self):
        previous = logging.getLogRecordFactory()
        install_log_redaction()
        yield
        logging.setLogRecordFactory(previous)

    @staticmethod
    def _formatted(record):
        return logging.Formatter("%(levelname)s %(message)s").format(record)

    def test_should_redact_the_traceback_of_a_logged_exception(self, installed, caplog):
        logger = logging.getLogger("test-traceback")
        with caplog.at_level(logging.ERROR):
            try:
                raise RuntimeError(f"GET https://user:{SHORT}@h.example/{OPAQUE}/v1/models?key={HEX_ID} failed")
            except RuntimeError:
                logger.exception("Error pre-processing image")

        output = self._formatted(caplog.records[0])
        assert "Traceback (most recent call last)" in output
        assert "RuntimeError" in output
        assert_clean(output)

    def test_should_redact_a_chained_exception(self, installed, caplog):
        logger = logging.getLogger("test-traceback")
        with caplog.at_level(logging.ERROR):
            try:
                try:
                    raise ValueError(f"api_key={OPAQUE}")
                except ValueError as inner:
                    raise RuntimeError("request failed") from inner
            except RuntimeError:
                logger.error("Error during batch translation stage", exc_info=True)

        output = self._formatted(caplog.records[0])
        assert "request failed" in output and "ValueError" in output
        assert_clean(output)

    def test_should_redact_stack_information(self, installed, caplog):
        logger = logging.getLogger("test-traceback")
        with caplog.at_level(logging.INFO):
            logger.info("checkpoint", stack_info=True)
        record = caplog.records[0]
        record.stack_info = f"Stack: token={OPAQUE}"
        log_redaction._redact_record(record)

        assert_clean(record.stack_info)

    def test_should_withhold_a_record_that_cannot_be_checked(self, installed, monkeypatch):
        def broken(text, known_secrets=()):
            raise RuntimeError("redaction is broken")

        monkeypatch.setattr(log_redaction, "redact_secrets", broken)
        factory = logging.getLogRecordFactory()

        record = factory("name", logging.ERROR, __file__, 1, f"api_key={OPAQUE}", None, None)

        assert record.getMessage() == log_redaction.WITHHELD
        assert record.exc_info is None and record.exc_text is None


class TestBoundedCost:
    """Response bodies are controlled by a remote server and can be very large."""

    # Built inside the test: a 400,000-character parameter would become the test's name.
    ADVERSARIAL = {
        "one long run": lambda: "A" * 400_000,
        "dashes": lambda: "a-" * 200_000,
        "repeated secret word": lambda: "key" * 130_000,
        "secret word then space": lambda: "token " * 60_000,
        "compound name": lambda: "session_" * 50_000 + "x",
        "name then spaces": lambda: ("api_key" + " " * 7) * 30_000,
        "token prefix": lambda: "eyJ" * 130_000,
        "url scheme": lambda: "https://" * 50_000,
        "many random-looking runs": lambda: "x" * 23 + " " + ("Ab1" * 9 + " ") * 14_000,
        "quotes and separators": lambda: '"' * 200_000 + ":" * 200_000,
    }

    @pytest.mark.parametrize("case", sorted(ADVERSARIAL))
    def test_should_finish_quickly_on_adversarial_input(self, case):
        import time

        text = self.ADVERSARIAL[case]()
        started = time.perf_counter()
        redact_secrets(text)
        summarize_text(text, limit=200)

        assert time.perf_counter() - started < 2.0

    def test_should_not_process_or_keep_text_beyond_the_cap(self):
        text = "ordinary words " * 3000 + f" api_key={OPAQUE}"

        redacted = redact_secrets(text)

        assert len(text) > log_redaction.MAX_REDACTED_CHARS
        assert len(redacted) < log_redaction.MAX_REDACTED_CHARS + 100
        assert "more characters not logged" in redacted
        assert_clean(redacted)

    def test_should_not_leak_a_credential_split_by_the_cap(self):
        head = "x " * ((log_redaction.MAX_REDACTED_CHARS - 20) // 2)
        text = head + f"api_key={OPAQUE}{OPAQUE}"

        redacted = redact_secrets(text)

        assert OPAQUE[:6] not in redacted

    def test_should_refuse_an_oversized_url(self):
        assert safe_url_for_log("https://h.example/" + "a" * (log_redaction.MAX_REDACTED_CHARS + 1)) == REDACTED


class TestShortNameWords:
    """Short words are common inside ordinary words and must not trigger on them."""

    @pytest.mark.parametrize("text", [
        "monkey: banana-split", "author: someone", "design=flat-layout", "keyboard: layout-us",
        "signal: interrupt", "hotkey=ctrl+s", "Translator: claude", "font_family: Sarabun",
    ])
    def test_should_not_redact_values_after_ordinary_words(self, text):
        assert redact_secrets(text) == text

    @pytest.mark.parametrize("name", ["key", "api_key", "x-api-key", "auth", "x-auth", "pwd", "db.pwd", "sig"])
    def test_should_redact_when_the_short_word_is_a_whole_part_of_the_name(self, name):
        assert_clean(redact_secrets(f"{name}={SHORT}"))


class TestCoverageKeptByTheLinearRewrite:
    """Cases the first linear version let through, which the version before it had caught."""

    @pytest.mark.parametrize("name", [
        "accessKey", "privateKey", "apiKey", "secretAccessKey", "authKey", "accesskey", "privatekey",
        "authkey", "appkey", "passkey", "keyid", "key_id", "sshkeys", "dbpwd", "authz", "oauth_token",
        "X-Amz-Signature", "sig",
    ])
    @pytest.mark.parametrize("form", ["{name}={value}", "{name}: {value}", '"{name}": "{value}"'])
    def test_should_redact_joined_and_camel_case_names(self, name, form):
        assert_clean(redact_secrets(form.format(name=name, value=OPAQUE)))
        assert_clean(redact_secrets(form.format(name=name, value=SHORT)))

    @pytest.mark.parametrize("gap", [0, 1, 8, 9, 20, 60])
    def test_should_redact_whatever_the_spacing_around_the_separator(self, gap):
        spaces = " " * gap

        assert_clean(redact_secrets(f"api_key{spaces}:{spaces}{SHORT}"))
        assert_clean(redact_secrets(f"api_key{spaces}={spaces}{SHORT}"))
        assert_clean(redact_secrets(f"Authorization: Bearer{' ' * max(gap, 1)}{SHORT}"))
        assert_clean(redact_secrets(f"token{' ' * max(gap, 1)}{OPAQUE}"))

    @pytest.mark.parametrize("text,leaked", [
        ('"password": "my pass phrase 12"', "pass phrase"),
        ("password: 'two words here'", "words"),
        ('{"client_secret": "a b"}', "a b"),
        ('secret="x y z"', "x y z"),
    ])
    def test_should_redact_a_quoted_value_that_contains_spaces(self, text, leaked):
        redacted = redact_secrets(text)

        assert leaked not in redacted
        assert REDACTED in redacted

    def test_should_keep_text_after_a_quoted_value(self):
        assert redact_secrets('{"api_key": "abc def", "model": "x"}') == f'{{"api_key": "{REDACTED}", "model": "x"}}'

    def test_should_drop_a_cookie_value_that_contains_quotes(self):
        assert_clean(redact_secrets(f'Cookie: a="{SHORT}"; b={OPAQUE}'))


class TestOneSetOfRules:
    """No second, cheaper judge may decide what the redaction gets to see."""

    @pytest.fixture
    def installed(self):
        previous = logging.getLogRecordFactory()
        install_log_redaction()
        yield
        logging.setLogRecordFactory(previous)

    def test_should_have_no_pre_check(self):
        assert not hasattr(log_redaction, "_TRIGGER_RE")

    @pytest.mark.parametrize("message", [
        f"Basic\t{SHORT}value",
        f"BEARER\n{SHORT}value",
        f"pwd\t=\t{SHORT}",
        f"dbPwd {OPAQUE}",
        f"x {OPAQUE[:23]}",
    ])
    def test_a_logged_record_should_match_direct_redaction(self, installed, caplog, message):
        logger = logging.getLogger("test-one-set-of-rules")
        with caplog.at_level(logging.INFO):
            logger.info(message)

        assert caplog.records[0].getMessage() == redact_secrets(message)

    @pytest.mark.parametrize("password", ["pa)ss" + SHORT, 'pa"ss' + SHORT, "pa'ss" + SHORT, "pa]ss}" + SHORT, "pa<ss>" + SHORT])
    def test_should_redact_a_url_whose_password_contains_punctuation(self, password):
        redacted = redact_secrets(f"GET https://user:{password}@proxy.example.com/v1/models failed")

        assert SHORT not in redacted
        assert "ss" + SHORT[:3] not in redacted
        assert redacted.endswith(" failed")

    @pytest.mark.parametrize("wrapped", ["({url})", "[{url}]", "<{url}>", '"{url}"', "see {url}.", "{url}, then"])
    def test_should_keep_punctuation_around_a_url(self, wrapped):
        text = wrapped.format(url="https://api.example.com/v1/models")

        assert redact_secrets(text) == text

    @pytest.mark.parametrize("value", ["1", "ab", "abc"])
    def test_should_redact_a_short_bare_value_like_a_quoted_one(self, value):
        assert redact_secrets(f"password={value}") == f"password={REDACTED}"
        assert redact_secrets(f'password="{value}"') == f'password="{REDACTED}"'


class TestNoTrustInOneParser:
    """A URL or a name/value pair must be safe however a client or a decoder would read it."""

    @pytest.mark.parametrize("url", [
        "https://host.example.com\\" + OPAQUE + "/v1/models",
        "https://host.example.com\\@" + SHORT + "/v1/models",
        "https:\\\\host.example.com\\" + SHORT,
        "https://host.example.com/v1\\..\\" + SHORT,
        "https://host.example.com\t" + SHORT + "/v1/models",
        "https://host_" + SHORT + ".example.com/v1/models",
        "https://host.example.com%2F" + SHORT + "/v1/models",
        "https://" + SHORT + "%40host.example.com/v1/models",
    ])
    def test_should_refuse_a_url_that_clients_read_differently(self, url):
        assert_clean(safe_url_for_log(url))

    @pytest.mark.parametrize("scheme", ["javascript", "data", "file", SHORT, "x" + SHORT[:10]])
    def test_should_refuse_an_unknown_scheme(self, scheme):
        assert safe_url_for_log(f"{scheme}://host.example.com/v1/models") == REDACTED

    @pytest.mark.parametrize("scheme", sorted(log_redaction.SAFE_URL_SCHEMES))
    def test_should_keep_every_known_scheme(self, scheme):
        assert safe_url_for_log(f"{scheme}://proxy.example.com:8080/v1") == f"{scheme}://proxy.example.com:8080/v1"

    @pytest.mark.parametrize("text", [
        "proxy user:" + SHORT + "@proxy.example.com:8080 refused",
        "http_proxy=user:" + SHORT + "@10.0.0.1:3128",
        "//user:" + SHORT + "@host.example.com/path",
        "connect user:" + OPAQUE + "@[::1]:8080",
    ])
    def test_should_redact_credentials_written_without_a_scheme(self, text):
        redacted = redact_secrets(text)

        assert_clean(redacted)
        assert REDACTED + "@" in redacted

    @pytest.mark.parametrize("text", ["at 10:30 see user@example.com", "ratio 16:9 on screen", "a: b"])
    def test_should_leave_times_and_addresses_alone(self, text):
        assert redact_secrets(text) == text

    @pytest.mark.parametrize("text", [
        '{"body": "{\\"api_key\\": \\"' + SHORT + '\\"}"}',
        '{"body": "{\\"password\\":\\"' + SHORT + '\\"}"}',
        "api_key\\\\\\\": \\\\\\\"" + SHORT,
        "api_key%3D" + SHORT + "%26next%3D1",
        "api_key%22%3A%22" + SHORT + "%22",
        "password%3A%20" + SHORT,
        "token%3d" + OPAQUE,
    ])
    def test_should_redact_values_in_escaped_or_encoded_text(self, text):
        redacted = redact_secrets(text)

        assert_clean(redacted)
        assert REDACTED in redacted
