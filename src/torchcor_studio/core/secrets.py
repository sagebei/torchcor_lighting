"""Machine credentials (SSH passwords, key passphrases) kept in the OS keyring.

Secrets never touch the studio's JSON files. Each one is stored under the service
``"TorchCor Studio"`` with the key ``"<machine id>:password"`` or
``"<machine id>:passphrase"``.

Everything degrades gracefully: when no keyring backend is available (headless Linux
without Secret Service, a locked keychain, ...) secrets are kept in memory for the
lifetime of the process, so a password typed once still works until the app quits.
"""

from __future__ import annotations

import contextlib
import logging
import threading

log = logging.getLogger(__name__)

SERVICE = "TorchCor Studio"  # the app's first name: keeps saved passwords
PASSWORD = "password"
PASSPHRASE = "passphrase"

_memory: dict[str, str] = {}
_memory_lock = threading.Lock()


def _key(machine_id: str, kind: str) -> str:
    if kind not in (PASSWORD, PASSPHRASE):
        raise ValueError(f"unknown secret kind {kind!r}")
    return f"{machine_id}:{kind}"


def _backend():
    """The active keyring backend, or ``None`` when only the fail-safe one exists."""
    try:
        import keyring
        from keyring.backends import fail

        backend = keyring.get_keyring()
    except Exception:  # a broken keyring install must not break the app
        return None
    if isinstance(backend, fail.Keyring):
        return None
    return backend


def keyring_available() -> bool:
    """Whether secrets survive a restart (an OS keyring backend is usable)."""
    return _backend() is not None


def get_secret(machine_id: str, kind: str) -> str | None:
    """Look up a secret: this session's memory first, then the OS keyring."""
    key = _key(machine_id, kind)
    with _memory_lock:
        if key in _memory:
            return _memory[key]
    backend = _backend()
    if backend is None:
        return None
    try:
        value = backend.get_password(SERVICE, key)
    except Exception as exc:  # locked keychain, denied access, D-Bus errors ...
        log.info("keyring lookup failed for %s: %s", key, exc)
        return None
    if value is not None:
        remember(machine_id, kind, value)
    return value


def remember(machine_id: str, kind: str, value: str) -> None:
    """Keep a secret in memory for this session only (instant, no I/O)."""
    with _memory_lock:
        _memory[_key(machine_id, kind)] = value


def set_secret(machine_id: str, kind: str, value: str) -> bool:
    """Store a secret. Returns ``True`` if it was persisted in the OS keyring.

    The value is usable immediately even when persisting fails.
    """
    remember(machine_id, kind, value)
    backend = _backend()
    if backend is None:
        return False
    try:
        backend.set_password(SERVICE, _key(machine_id, kind), value)
    except Exception as exc:
        log.info("keyring write failed for %s: %s", machine_id, exc)
        return False
    return True


def delete_secret(machine_id: str, kind: str) -> None:
    """Forget a secret everywhere; missing secrets are ignored."""
    key = _key(machine_id, kind)
    with _memory_lock:
        _memory.pop(key, None)
    backend = _backend()
    if backend is None:
        return
    # PasswordDeleteError when absent; other backend errors are not worth surfacing
    with contextlib.suppress(Exception):
        backend.delete_password(SERVICE, key)


def delete_all(machine_id: str) -> None:
    """Forget every secret of a machine (call when the machine is removed)."""
    for kind in (PASSWORD, PASSPHRASE):
        delete_secret(machine_id, kind)


def get_password(machine_id: str) -> str | None:
    return get_secret(machine_id, PASSWORD)


def get_passphrase(machine_id: str) -> str | None:
    return get_secret(machine_id, PASSPHRASE)
