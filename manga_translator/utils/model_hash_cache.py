"""
Cached SHA-256 verification of model files.

Model checkpoints can execute code when they are loaded, so a file that is
already present on disk should match the hash declared in ``_MODEL_MAPPING``
before it is used - not only right after it was downloaded.

Re-hashing multi-GB files on every start would be too slow, so a successful
verification is remembered per (absolute path, size, mtime) and the file is
hashed again only when one of those changes.

The cache is stored per user, OUTSIDE the models directory, so that a models
folder or portable package copied from elsewhere cannot bring its own
"already verified" records along.

Limitation: this detects replaced/corrupted files; it is not a defence against
someone who can already write to this user's profile (they could restore the
size and mtime of a modified file, or edit the cache itself).
"""

import hashlib
import json
import os
import sys
import tempfile
import threading
from typing import Dict, Optional, Tuple

CACHE_PATH_ENV = 'MANGA_TRANSLATOR_MODEL_HASH_CACHE'
CACHE_VERSION = 1
_HASH_CHUNK_SIZE = 1024 * 1024
# Only these small text files may differ from the declared hash by line endings.
_TEXT_SUFFIXES = ('.txt',)
_TEXT_SIZE_LIMIT = 16 * 1024 * 1024

_lock = threading.Lock()
# In-process memo so the JSON file is read at most once per cache path.
_loaded: Dict[str, dict] = {}


def get_cache_path() -> str:
    """Location of the per-user verification cache (override with MANGA_TRANSLATOR_MODEL_HASH_CACHE)."""
    override = os.environ.get(CACHE_PATH_ENV, '').strip()
    if override:
        return os.path.abspath(override)
    if sys.platform == 'win32':
        base = os.environ.get('LOCALAPPDATA') or os.path.join(os.path.expanduser('~'), 'AppData', 'Local')
    elif sys.platform == 'darwin':
        base = os.path.join(os.path.expanduser('~'), 'Library', 'Caches')
    else:
        base = os.environ.get('XDG_CACHE_HOME') or os.path.join(os.path.expanduser('~'), '.cache')
    return os.path.join(base, 'manga-translator-ui', 'model_hash_cache.json')


def _file_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _read_cache(cache_path: str) -> dict:
    if cache_path in _loaded:
        return _loaded[cache_path]
    files: dict = {}
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get('version') == CACHE_VERSION and isinstance(data.get('files'), dict):
            files = data['files']
    except (OSError, ValueError):
        files = {}
    _loaded[cache_path] = files
    return files


def _write_cache(cache_path: str, files: dict) -> None:
    """Best effort, atomic. A cache that cannot be written only costs a re-hash next time."""
    try:
        directory = os.path.dirname(cache_path)
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix='.model_hash_cache-', suffix='.tmp', dir=directory)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump({'version': CACHE_VERSION, 'files': files}, f, indent=2, sort_keys=True)
            os.replace(tmp_path, cache_path)
        except BaseException:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise
    except OSError:
        pass


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(_HASH_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text_with_lf_endings(path: str) -> Optional[str]:
    """
    SHA-256 of a small text file as if it used LF line endings, or None when
    the file is not a candidate.

    Dictionary files are plain text, and archives or checkouts made on Windows
    can store them with CRLF endings. The text is the same, so such a file is
    accepted when its LF form matches the declared hash. Model weights are
    never normalized.
    """
    if not path.lower().endswith(_TEXT_SUFFIXES):
        return None
    try:
        if os.path.getsize(path) > _TEXT_SIZE_LIMIT:
            return None
        with open(path, 'rb') as f:
            data = f.read()
    except OSError:
        return None
    if b'\r\n' not in data:
        return None
    return hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest()


def verify_file(path: str, expected_sha256: str) -> Tuple[bool, Optional[str], bool]:
    """
    Check that `path` has the SHA-256 `expected_sha256`.

    A text file that differs only by CRLF line endings also passes.

    Returns:
        (ok, actual_sha256, from_cache)
        ``actual_sha256`` is None when the result came from the cache.
    """
    expected = expected_sha256.strip().lower()
    stat_before = os.stat(path)
    key = _file_key(path)
    cache_path = get_cache_path()

    with _lock:
        entry = _read_cache(cache_path).get(key)
    if (
        isinstance(entry, dict)
        and entry.get('sha256') == expected
        and entry.get('size') == stat_before.st_size
        and entry.get('mtime_ns') == stat_before.st_mtime_ns
    ):
        return True, None, True

    actual = sha256_file(path).lower()
    if actual != expected and _sha256_text_with_lf_endings(path) == expected:
        actual = expected
    stat_after = os.stat(path)
    unchanged_while_hashing = (
        stat_after.st_size == stat_before.st_size and stat_after.st_mtime_ns == stat_before.st_mtime_ns
    )

    with _lock:
        files = _read_cache(cache_path)
        if actual == expected and unchanged_while_hashing:
            files[key] = {
                'sha256': actual,
                'size': stat_after.st_size,
                'mtime_ns': stat_after.st_mtime_ns,
            }
            _write_cache(cache_path, files)
        elif key in files:
            # Never keep a stale "verified" record for a file that no longer matches.
            del files[key]
            _write_cache(cache_path, files)

    return actual == expected, actual, False


def reset_memory_cache() -> None:
    """Forget the in-process copy of the cache (used by tests)."""
    with _lock:
        _loaded.clear()
