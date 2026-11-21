"""``~/.ssh/config``: import its hosts as machine profiles and resolve host aliases.

Only what the studio can honour is read: ``HostName``, ``User``, ``Port``,
``IdentityFile`` and ``ProxyJump``. Wildcard patterns (``Host *.lab``) are applied as
defaults when resolving a concrete name but are never offered as machines.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import paramiko

from torchcor_studio.core.models import AuthMethod, MachineKind, MachineProfile

log = logging.getLogger(__name__)

_HOST_LINE = re.compile(r"^\s*Host\s+(.+?)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class HostConfig:
    """The effective settings for one host name."""

    alias: str
    hostname: str
    user: str = ""
    port: int = 22
    identity_files: tuple[str, ...] = ()
    #: Comma-separated ``user@host:port`` hops ("" = connect directly).
    proxy_jump: str = ""


def default_path() -> Path:
    return Path.home() / ".ssh" / "config"


def load(path: Path | None = None) -> paramiko.SSHConfig | None:
    """Parse an SSH config file; ``None`` if it is missing or unreadable."""
    path = path or default_path()
    try:
        return paramiko.SSHConfig.from_path(str(path))
    except (OSError, paramiko.ConfigParseError, UnicodeDecodeError) as exc:
        if path.exists():
            log.info("could not read %s: %s", path, exc)
        return None


def _is_concrete(pattern: str) -> bool:
    return bool(pattern) and not any(ch in pattern for ch in "*?!")


def aliases(path: Path | None = None) -> list[str]:
    """Concrete ``Host`` names in file order (patterns and negations are skipped)."""
    path = path or default_path()
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    found: list[str] = []
    for line in text.splitlines():
        m = _HOST_LINE.match(line.split("#", 1)[0])
        if not m:
            continue
        for name in m.group(1).replace('"', " ").split():
            if _is_concrete(name) and name not in found:
                found.append(name)
    return found


def _resolve(config: paramiko.SSHConfig | None, alias: str) -> HostConfig:
    if config is None:
        return HostConfig(alias=alias, hostname=alias)
    try:
        data = config.lookup(alias)
    except Exception as exc:  # e.g. ``Match exec`` without the invoke package
        log.info("ssh config lookup failed for %s: %s", alias, exc)
        return HostConfig(alias=alias, hostname=alias)
    try:
        port = int(data.get("port", 22))
    except (TypeError, ValueError):
        port = 22
    files = tuple(os.path.expanduser(f) for f in data.get("identityfile", []) or [])
    jump = str(data.get("proxyjump", "") or "")
    if jump.lower() == "none":
        jump = ""
    return HostConfig(
        alias=alias,
        hostname=str(data.get("hostname", alias) or alias),
        user=str(data.get("user", "") or ""),
        port=port,
        identity_files=files,
        proxy_jump=jump,
    )


def lookup(alias: str, path: Path | None = None, config: paramiko.SSHConfig | None = None) -> HostConfig:
    """Effective settings for ``alias`` (an unknown name resolves to itself)."""
    return _resolve(config if config is not None else load(path), alias)


def format_hop(user: str, host: str, port: int) -> str:
    """``user@host:port`` with the defaults left out (IPv6 hosts are bracketed)."""
    shown = f"[{host}]" if ":" in host else host
    return f"{user + '@' if user else ''}{shown}{f':{port}' if port and port != 22 else ''}"


def parse_hop(spec: str) -> tuple[str, str, int]:
    """Split ``[user@]host[:port]`` (``[v6addr]:port`` supported) into its parts."""
    spec = spec.strip()
    if spec.startswith("ssh://"):
        spec = spec[len("ssh://"):]
    user, _, rest = spec.rpartition("@")
    port = 22
    if rest.startswith("["):
        host, _, tail = rest[1:].partition("]")
        if tail.startswith(":") and tail[1:].isdigit():
            port = int(tail[1:])
    else:
        host, sep, port_text = rest.partition(":")
        if sep and port_text.isdigit():
            port = int(port_text)
        elif sep:
            host = rest  # a bare IPv6 address without brackets
    return user, host, port


def resolve_jumps(proxy_jump: str, config: paramiko.SSHConfig | None) -> str:
    """Rewrite ``ProxyJump`` hops that are config aliases as explicit ``user@host:port``."""
    hops = []
    for hop in (h.strip() for h in proxy_jump.split(",")):
        if not hop:
            continue
        user, host, port = parse_hop(hop)
        resolved = _resolve(config, host)
        hops.append(format_hop(user or resolved.user, resolved.hostname,
                               port if port != 22 else resolved.port))
    return ",".join(hops)


def import_profiles(path: Path | None = None) -> list[MachineProfile]:
    """One candidate :class:`MachineProfile` per concrete ``Host`` in the SSH config."""
    path = path or default_path()
    config = load(path)
    if config is None:
        return []
    profiles: list[MachineProfile] = []
    seen: set[tuple[str, str, int]] = set()
    for alias in aliases(path):
        host = _resolve(config, alias)
        identity = next((f for f in host.identity_files if os.path.isfile(f)),
                        host.identity_files[0] if host.identity_files else "")
        key = (host.hostname, host.user, host.port)
        if key in seen:
            continue
        seen.add(key)
        profiles.append(MachineProfile(
            name=alias,
            kind=MachineKind.SSH,
            host=host.hostname,
            port=host.port,
            username=host.user,
            auth=AuthMethod.KEY if identity else AuthMethod.AGENT,
            key_path=identity,
            jump_host=resolve_jumps(host.proxy_jump, config) if host.proxy_jump else "",
        ))
    return profiles
