"""Username allowlist, first-admin setup restriction and CORS defaults of the web server."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import _bootstrap  # noqa: F401

from fastapi import FastAPI
from fastapi.testclient import TestClient

from manga_translator.args import create_parser
from manga_translator.server.core import setup_guard
from manga_translator.server.core.account_service import AccountService
from manga_translator.server.core.audit_service import AuditService
from manga_translator.server.core.request_rate_limiter import SlidingWindowRateLimiter
from manga_translator.server.core.session_service import SessionService
from manga_translator.server.core.username_policy import (
    USERNAME_MAX_LENGTH,
    InvalidUsernameError,
    is_valid_username,
    validate_username,
)
from manga_translator.server.routes import auth as auth_routes

XSS_USERNAME = "');alert(1);//"
XSS_IMG_USERNAME = '"><img src=x onerror=alert(1)>'
SETUP_TOKEN = "unit-test-setup-token-0123456789"


class UsernamePolicyTest(unittest.TestCase):
    def test_should_accept_ascii_and_unicode_names(self):
        for name in ("admin", "user_01", "a.b-c", "Ünïcode", "สมชาย", "น้ำใส", "田中太郎", "ab"):
            with self.subTest(name=name):
                self.assertEqual(validate_username(name), name)

    def test_should_reject_markup_and_script_characters(self):
        for name in (XSS_USERNAME, XSS_IMG_USERNAME, "a'b", 'a"b', "a<b", "a>b", "a&b", "a`b", "a b", "a/b", "a\\b", "a\nb", "a\x00b"):
            with self.subTest(name=name):
                with self.assertRaises(InvalidUsernameError):
                    validate_username(name)

    def test_should_reject_out_of_range_lengths(self):
        self.assertFalse(is_valid_username("a"))
        self.assertFalse(is_valid_username(""))
        self.assertFalse(is_valid_username("a" * (USERNAME_MAX_LENGTH + 1)))
        self.assertTrue(is_valid_username("a" * USERNAME_MAX_LENGTH))

    def test_should_reject_leading_or_trailing_punctuation(self):
        for name in (".hidden", "-flag", "_x", "name.", "..", "ิab"):
            with self.subTest(name=name):
                self.assertFalse(is_valid_username(name))

    def test_should_reject_non_string_and_non_nfc_input(self):
        self.assertFalse(is_valid_username(None))
        self.assertFalse(is_valid_username(123))
        self.assertFalse(is_valid_username("éric"))  # decomposed form of "éric"
        self.assertTrue(is_valid_username("éric"))


class AccountServiceUsernameTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.accounts_file = os.path.join(self._tmp.name, "accounts.json")

    def test_should_reject_invalid_username_on_create(self):
        service = AccountService(accounts_file=self.accounts_file)

        with self.assertRaises(ValueError):
            service.create_user(XSS_USERNAME, "password1", "user")

        self.assertEqual(service.list_users(), [])

    def test_should_keep_legacy_accounts_loadable_and_able_to_log_in(self):
        service = AccountService(accounts_file=self.accounts_file)
        service.create_user("legacy", "password1", "user")
        # Simulate an account created before the allowlist existed.
        account = service.accounts.pop("legacy")
        account.username = XSS_USERNAME
        service.accounts[XSS_USERNAME] = account
        service._save_accounts()

        reloaded = AccountService(accounts_file=self.accounts_file)

        self.assertIsNotNone(reloaded.get_user(XSS_USERNAME))
        self.assertTrue(reloaded.verify_password(XSS_USERNAME, "password1"))
        self.assertTrue(reloaded.delete_user(XSS_USERNAME))


def _fake_request(client_host="127.0.0.1", headers=None):
    base = {"host": "127.0.0.1:8000"}
    base.update(headers or {})
    return SimpleNamespace(
        client=SimpleNamespace(host=client_host) if client_host else None,
        headers=base,
    )


class SetupGuardTest(unittest.TestCase):
    def test_should_recognise_loopback_hosts(self):
        for host in ("127.0.0.1", "127.8.9.1", "::1", "[::1]", "localhost", "::ffff:127.0.0.1"):
            with self.subTest(host=host):
                self.assertTrue(setup_guard.is_loopback_host(host))
        for host in ("0.0.0.0", "192.168.1.5", "10.0.0.1", "example.com", "::", "", None, "localhost.evil.com"):
            with self.subTest(host=host):
                self.assertFalse(setup_guard.is_loopback_host(host))

    def test_should_treat_direct_local_request_as_loopback(self):
        self.assertTrue(setup_guard.is_direct_loopback_request(_fake_request()))
        self.assertTrue(setup_guard.is_direct_loopback_request(
            _fake_request(headers={"host": "localhost:8000", "origin": "http://localhost:8000"})
        ))

    def test_should_not_treat_remote_or_unknown_client_as_loopback(self):
        self.assertFalse(setup_guard.is_direct_loopback_request(_fake_request(client_host="192.168.1.20")))
        self.assertFalse(setup_guard.is_direct_loopback_request(_fake_request(client_host=None)))

    def test_should_not_trust_forwarding_headers(self):
        # A proxy on the same host connects from 127.0.0.1; a spoofed
        # X-Forwarded-For claiming loopback must not grant access either.
        for header in ("x-forwarded-for", "forwarded", "x-real-ip", "via"):
            with self.subTest(header=header):
                request = _fake_request(headers={header: "127.0.0.1"})
                self.assertFalse(setup_guard.is_direct_loopback_request(request))
        spoofed = _fake_request(client_host="192.168.1.20", headers={"x-forwarded-for": "127.0.0.1"})
        self.assertFalse(setup_guard.is_direct_loopback_request(spoofed))

    def test_should_reject_non_loopback_host_and_origin_headers(self):
        self.assertFalse(setup_guard.is_direct_loopback_request(
            _fake_request(headers={"host": "rebind.attacker.example:8000"})
        ))
        self.assertFalse(setup_guard.is_direct_loopback_request(
            _fake_request(headers={"origin": "https://attacker.example"})
        ))

    def test_should_ignore_short_setup_token(self):
        with mock.patch.dict(os.environ, {setup_guard.SETUP_TOKEN_ENV: "short"}):
            self.assertIsNone(setup_guard.get_setup_token())
            self.assertFalse(setup_guard.setup_token_matches("short"))
            self.assertTrue(setup_guard.setup_token_is_too_short())

    def test_should_evaluate_setup_access(self):
        remote = _fake_request(client_host="192.168.1.20")
        with mock.patch.dict(os.environ, {setup_guard.SETUP_TOKEN_ENV: ""}):
            self.assertEqual(setup_guard.evaluate_setup_access(_fake_request(), None), (True, "loopback"))
            self.assertEqual(setup_guard.evaluate_setup_access(remote, None), (False, "token_not_configured"))
            self.assertEqual(setup_guard.evaluate_setup_access(remote, "anything"), (False, "token_not_configured"))
        with mock.patch.dict(os.environ, {setup_guard.SETUP_TOKEN_ENV: SETUP_TOKEN}):
            self.assertEqual(setup_guard.evaluate_setup_access(remote, None), (False, "token_required"))
            self.assertEqual(setup_guard.evaluate_setup_access(remote, "wrong-token"), (False, "token_invalid"))
            self.assertEqual(setup_guard.evaluate_setup_access(remote, SETUP_TOKEN), (True, "token"))

    def test_should_describe_setup_steps_for_non_loopback_bind(self):
        with mock.patch.dict(os.environ, {setup_guard.SETUP_TOKEN_ENV: ""}):
            local_hint = "\n".join(setup_guard.initial_setup_hint("127.0.0.1", 8000))
            lan_hint = "\n".join(setup_guard.initial_setup_hint("0.0.0.0", 8000))
        self.assertNotIn(setup_guard.SETUP_TOKEN_ENV, local_hint)
        self.assertIn(setup_guard.SETUP_TOKEN_ENV, lan_hint)
        self.assertIn("http://127.0.0.1:8000/", lan_hint)


class CorsOptionsTest(unittest.TestCase):
    def test_should_default_to_loopback_origins_only(self):
        import re

        options = setup_guard.build_cors_options(None)

        self.assertEqual(options["allow_origins"], [])
        pattern = re.compile(options["allow_origin_regex"])
        for origin in ("http://localhost:8000", "http://127.0.0.1:5173", "https://localhost", "http://[::1]:8000"):
            with self.subTest(origin=origin):
                self.assertIsNotNone(pattern.fullmatch(origin))
        for origin in ("https://evil.example", "http://localhost.evil.example", "http://127.0.0.1.evil.example:8000", "null"):
            with self.subTest(origin=origin):
                self.assertIsNone(pattern.fullmatch(origin))

    def test_should_honour_explicit_origins_and_wildcard(self):
        self.assertIsNone(setup_guard.parse_cors_origins(None))
        self.assertIsNone(setup_guard.parse_cors_origins("  "))
        origins = setup_guard.parse_cors_origins("https://a.example/, https://b.example")
        self.assertEqual(origins, ["https://a.example", "https://b.example"])

        explicit = setup_guard.build_cors_options(origins)
        self.assertEqual(explicit["allow_origins"], origins)
        self.assertNotIn("allow_origin_regex", explicit)

        wildcard = setup_guard.build_cors_options(["*"])
        self.assertEqual(wildcard["allow_origins"], ["*"])
        self.assertFalse(wildcard["allow_credentials"])


class WebArgsDefaultsTest(unittest.TestCase):
    def test_should_bind_to_loopback_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MT_WEB_HOST", None)
            os.environ.pop("MT_WEB_CORS_ORIGINS", None)
            args = create_parser().parse_args(["web"])

        self.assertEqual(args.host, "127.0.0.1")
        self.assertIsNone(args.cors_origins)

    def test_should_allow_opting_in_to_wider_binding(self):
        args = create_parser().parse_args(["web", "--host", "0.0.0.0", "--cors-origins", "https://a.example"])

        self.assertEqual(args.host, "0.0.0.0")
        self.assertEqual(args.cors_origins, "https://a.example")

        with mock.patch.dict(os.environ, {"MT_WEB_HOST": "0.0.0.0"}):
            env_args = create_parser().parse_args(["web"])
        self.assertEqual(env_args.host, "0.0.0.0")


class AuthRoutesTest(unittest.TestCase):
    """Request-level tests of /auth/* against temporary account storage."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp = self._tmp.name

        self.account_service = AccountService(accounts_file=os.path.join(tmp, "accounts.json"))
        session_service = SessionService(
            sessions_file=os.path.join(tmp, "sessions.json"),
            session_timeout_minutes=60,
            enable_persistence=False,
        )
        audit_service = AuditService(audit_log_file=os.path.join(tmp, "audit.log"))

        previous = (
            auth_routes._account_service,
            auth_routes._session_service,
            auth_routes._audit_service,
            auth_routes._auth_rate_limiter,
        )
        self.addCleanup(self._restore_auth_module, previous)
        auth_routes.init_auth_services(self.account_service, session_service, audit_service)
        auth_routes._auth_rate_limiter = SlidingWindowRateLimiter()

        env = mock.patch.dict(os.environ, {setup_guard.SETUP_TOKEN_ENV: ""})
        env.start()
        self.addCleanup(env.stop)

        self.app = FastAPI()
        self.app.include_router(auth_routes.router)

    @staticmethod
    def _restore_auth_module(previous):
        (
            auth_routes._account_service,
            auth_routes._session_service,
            auth_routes._audit_service,
            auth_routes._auth_rate_limiter,
        ) = previous

    def _client(self, host="127.0.0.1"):
        return TestClient(self.app, base_url="http://127.0.0.1:8000", client=(host, 50000))

    def test_should_create_first_admin_from_loopback(self):
        client = self._client()

        status = client.get("/auth/status").json()
        response = client.post("/auth/setup", json={"username": "admin", "password": "password1"})

        self.assertTrue(status["need_setup"])
        self.assertFalse(status["setup_token_required"])
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["user"]["role"], "admin")
        self.assertEqual(self.account_service.get_user("admin").role, "admin")

    def test_should_refuse_setup_from_remote_client_without_token(self):
        client = self._client(host="192.168.1.20")

        status = client.get("/auth/status").json()
        response = client.post("/auth/setup", json={"username": "admin", "password": "password1"})

        self.assertTrue(status["setup_token_required"])
        self.assertEqual(response.status_code, 403)
        self.assertIn(setup_guard.SETUP_TOKEN_ENV, response.json()["detail"])
        self.assertEqual(self.account_service.list_users(), [])

    def test_should_refuse_setup_when_forwarding_header_claims_loopback(self):
        client = self._client()

        response = client.post(
            "/auth/setup",
            json={"username": "admin", "password": "password1"},
            headers={"X-Forwarded-For": "127.0.0.1"},
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.account_service.list_users(), [])

    def test_should_refuse_setup_with_wrong_token_and_accept_correct_token(self):
        client = self._client(host="192.168.1.20")
        payload = {"username": "admin", "password": "password1"}

        with mock.patch.dict(os.environ, {setup_guard.SETUP_TOKEN_ENV: SETUP_TOKEN}):
            wrong = client.post("/auth/setup", json=payload, headers={"X-Setup-Token": "wrong-token-wrong-token"})
            missing = client.post("/auth/setup", json=payload)
            self.assertEqual(self.account_service.list_users(), [])
            correct = client.post("/auth/setup", json=payload, headers={"X-Setup-Token": SETUP_TOKEN})

        self.assertEqual(wrong.status_code, 403)
        self.assertEqual(missing.status_code, 403)
        self.assertEqual(correct.status_code, 200, correct.text)
        self.assertIsNotNone(self.account_service.get_user("admin"))

    def test_should_accept_setup_token_in_request_body(self):
        client = self._client(host="192.168.1.20")

        with mock.patch.dict(os.environ, {setup_guard.SETUP_TOKEN_ENV: SETUP_TOKEN}):
            response = client.post(
                "/auth/setup",
                json={"username": "admin", "password": "password1", "setup_token": SETUP_TOKEN},
            )

        self.assertEqual(response.status_code, 200, response.text)

    def test_should_rate_limit_repeated_denied_setup_attempts(self):
        client = self._client(host="192.168.1.20")
        payload = {"username": "admin", "password": "password1"}

        statuses = [
            client.post("/auth/setup", json=payload).status_code
            for _ in range(auth_routes.SETUP_IP_MAX_DENIED_ATTEMPTS + 1)
        ]

        self.assertEqual(statuses[0], 403)
        self.assertEqual(statuses[-1], 429)

    def test_should_return_400_for_xss_style_username_on_setup(self):
        client = self._client()

        for username in (XSS_USERNAME, XSS_IMG_USERNAME):
            with self.subTest(username=username):
                response = client.post("/auth/setup", json={"username": username, "password": "password1"})
                self.assertEqual(response.status_code, 400)

        self.assertEqual(self.account_service.list_users(), [])

    def test_should_refuse_second_setup_once_an_account_exists(self):
        client = self._client()
        client.post("/auth/setup", json={"username": "admin", "password": "password1"})

        response = client.post("/auth/setup", json={"username": "other", "password": "password1"})

        self.assertEqual(response.status_code, 400)
        self.assertIsNone(self.account_service.get_user("other"))

    def test_should_return_400_for_xss_style_username_on_register(self):
        client = self._client(host="192.168.1.20")
        registration = {"registration": {"enabled": True, "default_group": "default"}}

        with mock.patch.dict(auth_routes.admin_settings, registration):
            rejected = client.post("/auth/register", json={"username": XSS_USERNAME, "password": "password1"})
            accepted = client.post("/auth/register", json={"username": "สมชาย_01", "password": "password1"})

        self.assertEqual(rejected.status_code, 400)
        self.assertIn("Username", rejected.json()["detail"])
        self.assertEqual(accepted.status_code, 200, accepted.text)
        self.assertEqual([u.username for u in self.account_service.list_users()], ["สมชาย_01"])

    def test_should_let_legacy_account_with_unusual_name_log_in(self):
        self.account_service.create_user("legacy", "password1", "user")
        account = self.account_service.accounts.pop("legacy")
        account.username = XSS_USERNAME
        self.account_service.accounts[XSS_USERNAME] = account
        client = self._client(host="192.168.1.20")

        response = client.post("/auth/login", json={"username": XSS_USERNAME, "password": "password1"})

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])


if __name__ == "__main__":
    unittest.main()
