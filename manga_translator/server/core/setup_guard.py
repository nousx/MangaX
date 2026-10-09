"""
Guards for the first-run admin setup and for cross-origin access.

The first account created through ``/auth/setup`` becomes the administrator,
so that endpoint must not be reachable by "whoever gets there first" on a
network.  Initial setup is allowed only:

* from a direct loopback connection (the operator sitting at the machine), or
* from anywhere, when the caller presents the setup token configured through
  the ``MANGA_TRANSLATOR_SETUP_TOKEN`` environment variable.
"""

import hmac
import ipaddress
import os
import re
from typing import Iterable, List, Optional, Tuple

SETUP_TOKEN_ENV = "MANGA_TRANSLATOR_SETUP_TOKEN"
SETUP_TOKEN_HEADER = "x-setup-token"
SETUP_TOKEN_MIN_LENGTH = 16

CORS_ORIGINS_ENV = "MT_WEB_CORS_ORIGINS"
# Default: only pages served from this machine may call the API cross-origin.
# The bundled web UI is same-origin and does not depend on CORS at all.
LOOPBACK_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d{1,5})?$"

# Headers added by reverse proxies.  A request carrying any of them did not
# originate on this machine even if the TCP peer is 127.0.0.1 (a proxy running
# on the same host), so it is never treated as a loopback request.  The header
# VALUES are deliberately not trusted or parsed.
_PROXY_HEADERS = (
    "forwarded",
    "x-forwarded-for",
    "x-forwarded-host",
    "x-forwarded-proto",
    "x-forwarded-server",
    "x-real-ip",
    "x-client-ip",
    "true-client-ip",
    "cf-connecting-ip",
    "via",
)

_LOOPBACK_NAMES = {"localhost", "localhost."}


def is_loopback_host(host: Optional[str]) -> bool:
    """Return True if `host` (an IP literal or host name) denotes loopback."""
    if not host:
        return False
    candidate = host.strip().lower()
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]
    if candidate in _LOOPBACK_NAMES:
        return True
    try:
        address = ipaddress.ip_address(candidate)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_loopback


def _split_host_header(value: str) -> str:
    """Return the host part of a Host header / URL authority (port removed)."""
    value = value.strip()
    if value.startswith("["):
        end = value.find("]")
        return value[: end + 1] if end != -1 else value
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value


def _origin_host(origin: str) -> Optional[str]:
    match = re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://([^/]*)", origin.strip())
    if not match:
        return None
    return _split_host_header(match.group(1))


def is_direct_loopback_request(request) -> bool:
    """
    True only for a request made directly from this machine.

    Checks, all of which must hold:
    * the TCP peer address is a loopback address;
    * no reverse-proxy headers are present (see ``_PROXY_HEADERS``);
    * the Host header names a loopback host (defeats DNS rebinding);
    * the Origin header, if any, is a loopback origin (defeats cross-site
      requests issued by a web page open in the operator's browser).
    """
    client = getattr(request, "client", None)
    if client is None or not is_loopback_host(client.host):
        return False

    headers = request.headers
    if any(name in headers for name in _PROXY_HEADERS):
        return False

    host_header = headers.get("host")
    if not host_header or not is_loopback_host(_split_host_header(host_header)):
        return False

    origin = headers.get("origin")
    if origin is not None:
        origin_host = _origin_host(origin)
        if origin_host is None or not is_loopback_host(origin_host):
            return False

    return True


def get_setup_token() -> Optional[str]:
    """Return the configured setup token, or None if unset or too short."""
    token = os.environ.get(SETUP_TOKEN_ENV, "").strip()
    if len(token) < SETUP_TOKEN_MIN_LENGTH:
        return None
    return token


def setup_token_is_too_short() -> bool:
    token = os.environ.get(SETUP_TOKEN_ENV, "").strip()
    return 0 < len(token) < SETUP_TOKEN_MIN_LENGTH


def setup_token_matches(supplied: Optional[str]) -> bool:
    expected = get_setup_token()
    if expected is None or not supplied:
        return False
    return hmac.compare_digest(supplied.strip().encode("utf-8"), expected.encode("utf-8"))


def evaluate_setup_access(request, supplied_token: Optional[str]) -> Tuple[bool, str]:
    """
    Decide whether `request` may perform the initial admin setup.

    Returns:
        (allowed, reason) where reason is one of
        "loopback", "token", "token_required", "token_invalid", "token_not_configured".
    """
    if supplied_token:
        if setup_token_matches(supplied_token):
            return True, "token"
        if get_setup_token() is not None:
            return False, "token_invalid"
    if is_direct_loopback_request(request):
        return True, "loopback"
    if get_setup_token() is None:
        return False, "token_not_configured"
    return False, "token_required"


def initial_setup_hint(host: str, port) -> List[str]:
    """Log lines explaining how to finish first-run setup for a given bind address."""
    lines = [
        "No user accounts exist yet: the first account created becomes the administrator.",
    ]
    if is_loopback_host(host):
        lines.append(f"Open http://{host}:{port}/ on this machine to create it.")
        return lines
    lines.append(
        f"The server is listening on a non-loopback address ({host}); initial setup is "
        "NOT available to remote clients by default."
    )
    lines.append(f"  - Either open http://127.0.0.1:{port}/ on this machine to create the admin account,")
    if get_setup_token() is not None:
        lines.append(
            f"  - or enter the setup token from the {SETUP_TOKEN_ENV} environment variable "
            "on the setup page (header: X-Setup-Token)."
        )
    else:
        lines.append(
            f"  - or restart with the {SETUP_TOKEN_ENV} environment variable set to a random "
            f"secret of at least {SETUP_TOKEN_MIN_LENGTH} characters and enter it on the setup page."
        )
    if setup_token_is_too_short():
        lines.append(
            f"WARNING: {SETUP_TOKEN_ENV} is set but shorter than {SETUP_TOKEN_MIN_LENGTH} "
            "characters, so it is ignored."
        )
    lines.append("Requests that pass through a reverse proxy always need the setup token.")
    return lines


def parse_cors_origins(raw: Optional[str]) -> Optional[List[str]]:
    """Parse a comma/space separated origin list. Returns None when nothing is configured."""
    if raw is None:
        return None
    origins = [item.strip().rstrip("/") for item in re.split(r"[,\s]+", raw) if item.strip()]
    return origins or None


def build_cors_options(origins: Optional[Iterable[str]] = None) -> dict:
    """
    Keyword arguments for Starlette's CORSMiddleware.

    * no origins configured -> loopback origins only;
    * explicit origins      -> exactly those origins;
    * "*"                   -> every origin (explicit opt-in), without credentials.

    Authentication uses the X-Session-Token header, not cookies, so
    ``allow_credentials`` is not needed for the API to work.
    """
    options = {
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }
    origin_list = list(origins) if origins else []
    if not origin_list:
        options["allow_origins"] = []
        options["allow_origin_regex"] = LOOPBACK_ORIGIN_REGEX
        options["allow_credentials"] = True
    elif "*" in origin_list:
        options["allow_origins"] = ["*"]
        options["allow_credentials"] = False
    else:
        options["allow_origins"] = origin_list
        options["allow_credentials"] = True
    return options
