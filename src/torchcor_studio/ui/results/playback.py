"""Frame playback: a clock (:class:`PlaybackController`) and its transport bar (:class:`TimelineBar`).

The controller advances a fractional ``position`` in real time — ``fps`` data frames per
second at 1× — and ticks at most :attr:`PlaybackController.MAX_FPS` times a second, so
slow renders skip frames instead of slowing the clock. Views that interpolate between
frames (deformation) listen to :attr:`~PlaybackController.position_changed`; everything
else listens to :attr:`~PlaybackController.frame_changed`, emitted only when the
integer frame changes.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QElapsedTimer, QObject, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFontMetrics, QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import QAbstractButton, QHBoxLayout, QLabel, QSizePolicy, QSlider, QWidget

from torchcor_studio.ui import icons
from torchcor_studio.ui.theme import mono_font, theme
from torchcor_studio.ui.widgets import IconButton, SegmentedControl

__all__ = ["SPEEDS", "PlaybackController", "TimelineBar"]

#: Playback speed multipliers offered by the timeline.
SPEEDS = (0.5, 1.0, 2.0, 4.0)


class PlaybackController(QObject):
    """Real-time frame clock with play/pause, loop, speed and seeking."""

    frame_changed = Signal(int)
    position_changed = Signal(float)
    playing_changed = Signal(bool)

    #: Upper bound on ticks (and therefore renders) per second.
    MAX_FPS = 30

    def __init__(self, n_frames: int, fps: float = 24.0, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._n = max(1, int(n_frames))
        self._fps = max(0.1, float(fps))
        self._speed = 1.0
        self._loop = True
        self._pos = 0.0
        self._frame = 0
        self._clock = QElapsedTimer()
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(round(1000 / self.MAX_FPS))
        self._timer.timeout.connect(self._tick)

    # ------------------------------------------------------------------ state
    @property
    def n_frames(self) -> int:
        return self._n

    @property
    def frame(self) -> int:
        return self._frame

    @property
    def position(self) -> float:
        return self._pos

    @property
    def playing(self) -> bool:
        return self._timer.isActive()

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def loop(self) -> bool:
        return self._loop

    # ------------------------------------------------------------------ control
    def play(self) -> None:
        if self.playing or self._n < 2:
            return
        if not self._loop and self._pos >= self._n - 1:
            self._set_position(0.0)
        self._clock.start()
        self._timer.start()
        self.playing_changed.emit(True)

    def pause(self) -> None:
        if not self.playing:
            return
        self._timer.stop()
        self.playing_changed.emit(False)

    def toggle(self) -> None:
        self.pause() if self.playing else self.play()

    def seek(self, frame: int) -> None:
        """Jump to an integer frame (clamped)."""
        self._set_position(float(min(max(int(frame), 0), self._n - 1)))

    def seek_position(self, position: float) -> None:
        """Jump to a fractional frame position (clamped)."""
        self._set_position(min(max(float(position), 0.0), float(self._n - 1)))

    def step(self, delta: int) -> None:
        """Move ``delta`` frames, wrapping around when looping."""
        self.pause()
        target = self._frame + int(delta)
        if self._loop:
            target %= self._n
        self.seek(target)

    def set_speed(self, speed: float) -> None:
        self._speed = max(0.05, float(speed))

    def set_loop(self, loop: bool) -> None:
        self._loop = bool(loop)

    def set_fps(self, fps: float) -> None:
        self._fps = max(0.1, float(fps))

    # ------------------------------------------------------------------ internals
    def _tick(self) -> None:
        elapsed = min(self._clock.restart() / 1000.0, 0.25)  # no leaps after a stall
        pos = self._pos + elapsed * self._fps * self._speed
        if not self._loop and pos >= self._n - 1:
            self._set_position(float(self._n - 1))
            self.pause()
            return
        if pos >= self._n:  # the last frame holds for one frame time, then wrap
            pos %= self._n
        self._set_position(pos)

    def _set_position(self, pos: float) -> None:
        self._pos = pos
        frame = min(int(pos), self._n - 1)
        if frame != self._frame:
            self._frame = frame
            self.frame_changed.emit(frame)
        self.position_changed.emit(pos)


class _PlayButton(QAbstractButton):
    """The round, accent-filled play/pause button."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._playing = False
        self.setFixedSize(32, 32)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        theme().changed.connect(self._theme_changed)

    def _theme_changed(self, _palette=None) -> None:
        self.update()

    def set_playing(self, playing: bool) -> None:
        self._playing = playing
        self.setToolTip("Pause (Space)" if playing else "Play (Space)")
        self.setAccessibleName("Pause" if playing else "Play")
        self.update()

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(32, 32)

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = theme().palette
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor(p.accent_pressed if self.isDown() else p.accent_hover if self.underMouse() else p.accent)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(QRectF(self.rect()).adjusted(1, 1, -1, -1))
        size = 14
        glyph = icons.pixmap("pause" if self._playing else "play", p.on_accent, size, stroke=2.0)
        dx = 0 if self._playing else 1  # optical centring of the triangle
        painter.drawPixmap((self.width() - size) // 2 + dx, (self.height() - size) // 2, glyph)
        if self.hasFocus():
            ring = QColor(p.accent)
            ring.setAlphaF(0.45)
            painter.setPen(ring)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5))


class TimelineBar(QWidget):
    """Transport controls for a :class:`PlaybackController`.

    ``label_for(frame)`` formats the current time ("t = 123 ms", "λ = 0.75"). Space
    toggles playback anywhere inside ``shortcut_scope`` (default: the bar's parent).
    """

    def __init__(self, controller: PlaybackController, label_for: Callable[[int], str],
                 parent: QWidget | None = None, shortcut_scope: QWidget | None = None) -> None:
        super().__init__(parent)
        self.controller = controller
        self._label_for = label_for
        self._resume_after_drag = False
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(52)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 0, 14, 0)
        lay.setSpacing(6)
        self.play_button = _PlayButton()
        self.play_button.clicked.connect(controller.toggle)
        self.back_button = IconButton("chevron-left", "Previous frame", size=28,
                                      on_click=lambda: controller.step(-1))
        self.forward_button = IconButton("chevron-right", "Next frame", size=28,
                                         on_click=lambda: controller.step(1))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, controller.n_frames - 1)
        self.slider.setPageStep(max(1, controller.n_frames // 10))
        self.slider.setCursor(Qt.CursorShape.PointingHandCursor)
        self.slider.setAccessibleName("Timeline")
        self.slider.valueChanged.connect(self._slider_moved)
        self.slider.sliderPressed.connect(self._drag_started)
        self.slider.sliderReleased.connect(self._drag_finished)
        self.time_label = QLabel()
        self.time_label.setFont(mono_font(12))
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.time_label.setProperty("role", "muted")
        self.speed = SegmentedControl([(s, f"{s:g}×") for s in SPEEDS], value=controller.speed, compact=True)
        self.speed.setToolTip("Playback speed")
        self.speed.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.speed.changed.connect(controller.set_speed)
        self.loop_button = IconButton("repeat", "Loop", size=28, checkable=True)
        self.loop_button.setChecked(controller.loop)
        self.loop_button.toggled.connect(controller.set_loop)

        lay.addWidget(self.play_button)
        lay.addSpacing(2)
        lay.addWidget(self.back_button)
        lay.addWidget(self.forward_button)
        lay.addSpacing(8)
        lay.addWidget(self.slider, 1)
        lay.addSpacing(10)
        lay.addWidget(self.time_label)
        lay.addSpacing(10)
        lay.addWidget(self.speed)
        lay.addWidget(self.loop_button)

        controller.frame_changed.connect(self._frame_changed)
        controller.playing_changed.connect(self.play_button.set_playing)
        self.play_button.set_playing(controller.playing)
        self._fit_label()
        self._frame_changed(controller.frame)

        scope = shortcut_scope or parent or self
        self._shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), scope)
        self._shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._shortcut.activated.connect(self._space)

    # ------------------------------------------------------------------ slots
    def _space(self) -> None:
        if self.isVisible():
            self.controller.toggle()

    def _fit_label(self) -> None:
        fm = QFontMetrics(self.time_label.font())
        n = self.controller.n_frames
        samples = {0, n - 1, n // 2} | set(range(0, n, max(1, n // 16)))
        width = max(fm.horizontalAdvance(self._label_for(i)) for i in samples)
        self.time_label.setFixedWidth(width + 6)

    def _frame_changed(self, frame: int) -> None:
        if self.slider.value() != frame:
            self.slider.blockSignals(True)
            self.slider.setValue(frame)
            self.slider.blockSignals(False)
        self.time_label.setText(self._label_for(frame))

    def _slider_moved(self, value: int) -> None:
        self.controller.seek(value)

    def _drag_started(self) -> None:
        self._resume_after_drag = self.controller.playing
        self.controller.pause()

    def _drag_finished(self) -> None:
        if self._resume_after_drag:
            self.controller.play()
        self._resume_after_drag = False
