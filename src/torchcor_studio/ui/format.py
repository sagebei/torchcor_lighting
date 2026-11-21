"""Human-friendly formatting for times, durations, sizes and numbers."""

from __future__ import annotations

import math
import time
from datetime import datetime

from torchcor_studio.core.models import MIN_TORCHCOR


def relative_time(ts: float | None, now: float | None = None) -> str:
    if not ts:
        return ""
    delta = (now or time.time()) - ts
    if delta < 45:
        return "just now"
    if delta < 3600:
        return f"{round(delta / 60)}m ago"
    if delta < 86400:
        return f"{round(delta / 3600)}h ago"
    if delta < 7 * 86400:
        return f"{round(delta / 86400)}d ago"
    return datetime.fromtimestamp(ts).strftime("%b %d")


def timestamp(ts: float | None) -> str:
    if not ts:
        return "—"
    return datetime.fromtimestamp(ts).strftime("%b %d, %Y · %H:%M")


def duration(seconds: float | None) -> str:
    if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)):
        return ""
    s = round(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    if h < 48:
        return f"{h}h {m:02d}m"
    d, h = divmod(h, 24)
    return f"{d}d {h}h"


def size(num_bytes: float | None) -> str:
    if num_bytes is None:
        return ""
    n = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def number(value: float | int | str | None, digits: int = 3) -> str:
    """Compact, readable numbers: 1 234 567 → '1.23M', 0.000123 → '1.23e-4'."""
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer() and abs(value) < 1e6):
        v = int(value)
        if abs(v) >= 1_000_000:
            return f"{v / 1e6:.{digits - 1}f}M".replace(".00M", "M")
        return f"{v:,}"
    v = float(value)
    if math.isnan(v):
        return "—"
    if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e6):
        return f"{v:.{digits - 1}e}"
    return f"{v:.{digits}g}"


def megabytes(mb: float) -> str:
    return f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{mb:.0f} MB"


def plural(count: int, noun: str) -> str:
    """``plural(2, "GPU")`` → ``"2 GPUs"``."""
    return f"{count} {noun}{'' if count == 1 else 's'}"


def torch_summary(env) -> str:  # noqa: ANN001 - EnvironmentReport
    """``"2.7.1 · CUDA 12.8 · 2 GPUs"``, ``"2.5.1 · CPU only — can't use the 2 NVIDIA GPUs"``."""
    if not env.torch_version:
        return "Not installed" + (f" — the {plural(env.nvidia_gpus, 'NVIDIA GPU')} need it"
                                  if env.nvidia_gpus else "")
    if env.cuda_available:
        return f"{env.torch_version} · CUDA {env.cuda_version} · {plural(env.cuda_device_count, 'GPU')}"
    if env.nvidia_gpus:
        return f"{env.torch_version} · CPU only — can't use the {plural(env.nvidia_gpus, 'NVIDIA GPU')}"
    return f"{env.torch_version} · CPU only"


def torchcor_summary(env) -> str:  # noqa: ANN001 - EnvironmentReport
    """``"1.9.0"``, ``"0.1.6 — too old (needs 1.9 or newer)"`` or ``"Not installed"``."""
    if env.torchcor_version is None:
        return "Not installed"
    if env.torchcor_outdated:
        return f"{env.torchcor_version} — too old (needs {'.'.join(map(str, MIN_TORCHCOR))} or newer)"
    return env.torchcor_version
