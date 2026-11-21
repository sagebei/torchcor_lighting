"""AppContext: machine sessions and telemetry."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.gui


def test_telemetry_from_a_stopped_monitor_is_dropped(themed, qtbot, monkeypatch) -> None:
    """Monitors stop on a background thread: a last reading may arrive after unwatch or
    shutdown (when the context can already be deleted) and must be ignored."""
    from torchcor_studio.core import gpu
    from torchcor_studio.core.settings import LOCAL_MACHINE_ID, Store
    from torchcor_studio.ui.context import AppContext

    callbacks = []

    class FakeMonitor:
        def __init__(self, machine, interval, callback) -> None:  # noqa: ANN001
            callbacks.append(callback)

        def start(self) -> None: ...

        def stop(self) -> None: ...

    monkeypatch.setattr(gpu, "GpuMonitor", FakeMonitor)
    ctx = AppContext(Store())
    sess = ctx.session(LOCAL_MACHINE_ID)
    received = []
    ctx.telemetry.connect(lambda mid, samples: received.append(samples))

    ctx._start_monitor(sess)
    first = callbacks[-1]
    first(["reading 1"])
    assert received == [["reading 1"]] and sess.samples == ["reading 1"]

    ctx._stop_monitor(sess)
    first(["late reading"])  # after unwatch: dropped
    ctx._start_monitor(sess)  # a new monitor replaces it
    first(["stale reading"])  # the old monitor's callback is ignored
    callbacks[-1](["reading 2"])
    assert received == [["reading 1"], ["reading 2"]]
    ctx.shutdown()
