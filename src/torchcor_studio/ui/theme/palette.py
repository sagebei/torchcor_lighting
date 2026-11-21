"""Design tokens: colour, type, spacing and radius.

The palette is derived from the TorchCor mark — ember orange and royal blue over a
navy-tinted neutral scale — and exists in a dark and a light variant with identical
token names, so every widget is written once against tokens, never raw colours.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from PySide6.QtGui import QColor


@dataclass(frozen=True)
class Palette:
    name: str
    dark: bool
    # surfaces, from the window outward
    bg0: str  # window
    bg1: str  # sidebar / chrome
    bg2: str  # cards
    bg3: str  # inputs / raised
    bg4: str  # hover
    line: str  # hairline borders
    line2: str  # strong borders / focus rings at rest
    # text
    text: str
    text2: str
    text3: str
    # brand & status
    accent: str
    accent_hover: str
    accent_pressed: str
    accent_text: str
    on_accent: str
    ember: str
    success: str
    warning: str
    danger: str
    info: str
    # status colours as small text on their own soft tint (darker in the light theme)
    success_text: str
    warning_text: str
    danger_text: str
    info_text: str
    ember_text: str
    # data visualisation
    plot_bg: str
    grid: str
    viewport_top: str
    viewport_bottom: str
    ecg_paper: str
    ecg_minor: str
    ecg_major: str
    ecg_trace: str
    series: tuple[str, ...]

    def qcolor(self, token: str, alpha: float | None = None) -> QColor:
        color = QColor(getattr(self, token))
        if alpha is not None:
            color.setAlphaF(alpha)
        return color

    def soft(self, token: str, alpha: float = 0.14) -> str:
        """``rgba()`` tint of a token for badges, selections and soft fills."""
        c = QColor(getattr(self, token))
        return f"rgba({c.red()}, {c.green()}, {c.blue()}, {alpha:.3f})"

    def state_color(self, state: str) -> str:
        return {
            "running": self.accent,
            "queued": self.text3,
            "completed": self.success,
            "failed": self.danger,
            "cancelled": self.text3,
            "lost": self.warning,
        }.get(state, self.text3)


DARK = Palette(
    name="dark",
    dark=True,
    bg0="#0B0E14",
    bg1="#0E121A",
    bg2="#131823",
    bg3="#182030",
    bg4="#1F2839",
    line="#222B3B",
    line2="#2E3a4F",
    text="#E7EBF3",
    text2="#A4AEC3",
    text3="#6F7A93",
    accent="#3D7BFF",
    accent_hover="#5A8FFF",
    accent_pressed="#2F67E0",
    accent_text="#8DB0FF",
    on_accent="#FFFFFF",
    ember="#FF7A2F",
    success="#34C77B",
    warning="#F5B03C",
    danger="#F2555A",
    info="#46B5F0",
    success_text="#34C77B",
    warning_text="#F5B03C",
    danger_text="#F2555A",
    info_text="#46B5F0",
    ember_text="#FF7A2F",
    plot_bg="#10151F",
    grid="#1D2533",
    viewport_top="#1A2232",
    viewport_bottom="#090C12",
    ecg_paper="#140F12",
    ecg_minor="#2A1A1F",
    ecg_major="#4A2630",
    ecg_trace="#F3F5FA",
    series=("#5A8FFF", "#FF8A47", "#3DD68C", "#E86BF2", "#F5C33C", "#4FD3E8", "#FF6B81", "#A68BFF"),
)

LIGHT = Palette(
    name="light",
    dark=False,
    bg0="#F4F6FA",
    bg1="#EDF0F6",
    bg2="#FFFFFF",
    bg3="#F7F9FC",
    bg4="#EBEFF6",
    line="#E1E6EF",
    line2="#CCD4E1",
    text="#0F1728",
    text2="#475467",
    text3="#667085",  # 4.6:1 on bg0, 5:1 on white
    accent="#2563EB",
    accent_hover="#3B76F6",
    accent_pressed="#1D4FD8",
    accent_text="#1D4FD8",
    on_accent="#FFFFFF",
    ember="#EE6A1C",
    success="#12A05C",
    warning="#C98A0B",
    danger="#D92D3A",
    info="#0B8BD0",
    success_text="#0B7A45",  # each at least 4.5:1 on its 14 % tint over white
    warning_text="#8F5F00",
    danger_text="#B42318",
    info_text="#05679C",
    ember_text="#B54708",
    plot_bg="#FFFFFF",
    grid="#EEF1F6",
    viewport_top="#F7F9FD",
    viewport_bottom="#DCE3EE",
    ecg_paper="#FFF7F7",
    ecg_minor="#F7DADD",
    ecg_major="#EBA9B1",
    ecg_trace="#111827",
    series=("#2563EB", "#EE6A1C", "#12A05C", "#C026D3", "#CA8A04", "#0891B2", "#E11D48", "#7C3AED"),
)


@dataclass(frozen=True)
class Metrics:
    """Spacing, radius and type scale (logical pixels)."""

    r_sm: int = 6
    r_md: int = 8
    r_lg: int = 12
    r_xl: int = 16
    fs_xs: int = 11
    fs_sm: int = 12
    fs_base: int = 13
    fs_md: int = 14
    fs_lg: int = 16
    fs_xl: int = 20
    fs_2xl: int = 26
    sidebar_width: int = 272
    control_h: int = 32

    def scaled(self, factor: float) -> Metrics:
        if factor == 1.0:
            return self
        return replace(
            self,
            **{k: max(1, round(v * factor)) for k, v in self.__dict__.items() if k.startswith("fs_")},
        )


METRICS = Metrics()

#: Font stacks — the platform UI font is used for text; these are fallbacks/mono.
MONO_FAMILIES = (
    "SF Mono", "Menlo", "Cascadia Mono", "Cascadia Code", "Consolas",
    "JetBrains Mono", "DejaVu Sans Mono", "Liberation Mono", "monospace",
)
