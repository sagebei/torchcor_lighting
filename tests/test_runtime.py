"""The compute-side runtime: inspection, discovery, launch/supervise/cancel lifecycle."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

import pytest

from torchcor_studio.runtime import runtime_path
from torchcor_studio.runtime import tcs_runtime as rt

RUNTIME = str(runtime_path("tcs_runtime.py"))


def _run(*args: str, env: dict[str, str] | None = None) -> dict:
    out = subprocess.run([sys.executable, RUNTIME, *args], capture_output=True, text=True, timeout=120,
                         env=env)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


def _write_mesh(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "heart.pts").write_text("4\n0 0 0\n1000 0 0\n0 1000 0\n0 0 1000\n")
    (folder / "heart.elem").write_text("2\nTt 0 1 2 3 34\nTr 0 1 2 44\n")
    (folder / "heart.lon").write_text("1\n1 0 0\n1 0 0\n")
    (folder / "LV_sf.vtx").write_text("2\nextra\n0\n1\n")
    (folder / "LV_endo.surf").write_text("1\nTr 0 1 2\n")


def test_inspect_reads_mesh_summary(tmp_path: Path) -> None:
    _write_mesh(tmp_path / "mesh")
    info = _run("inspect", str(tmp_path / "mesh"))
    assert info["exists"] is True
    assert info["mesh_name"] == "heart"
    assert info["n_nodes"] == 4
    assert info["n_elems"] == 2
    assert info["elem_types"] == ["Tr", "Tt"]
    assert info["regions"] == [34, 44]
    assert info["fibre_columns"] == 3
    assert info["vtx"] == [{"name": "LV_sf.vtx", "count": 2}]
    assert info["surfaces"] == ["LV_endo.surf"]


def test_inspect_missing_folder(tmp_path: Path) -> None:
    info = _run("inspect", str(tmp_path / "nope"))
    assert info["exists"] is False


def test_probe_reports_interpreter() -> None:
    info = _run("probe")
    assert info["python"] == sys.executable
    assert info["python_version"].count(".") == 2
    assert info["cpu_count"] >= 1
    assert isinstance(info["nvidia_gpus"], int)  # 0 without an NVIDIA driver


def _make_run(root: Path, script: str, name: str = "run-a") -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / rt.SCRIPT).write_text(script)
    (run_dir / "tcs_runtime.py").write_text(Path(RUNTIME).read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / rt.MANIFEST).write_text(json.dumps({"id": name, "name": name, "kind": "ep",
                                                   "created_at": time.time()}))
    (run_dir / rt.STATUS).write_text(json.dumps({"state": "queued", "updated_at": time.time()}))
    return run_dir


def _wait_terminal(run_dir: Path, timeout: float = 60) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = json.loads((run_dir / rt.STATUS).read_text())
        if status.get("state") in ("completed", "failed", "cancelled"):
            return status
        time.sleep(0.2)
    raise AssertionError(f"run did not finish: {status}")


SCRIPT_OK = """
import json, sys, time
def mark(**kw):
    print("::tcs:: " + json.dumps(kw), flush=True)
mark(type="stage", id="solve", label="Solving")
for i in range(1, 6):
    print(f"t: {i * 10:.1f}/50 | Time elapsed: 0.1 | CG iter: 3", flush=True)
    time.sleep(0.05)
mark(type="metric", name="nodes", value=42)
mark(type="stage", id="done", label="Completed")
"""


def test_launch_supervise_completes_with_progress(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, SCRIPT_OK)
    info = _run("launch", str(run_dir))
    assert info["pid"] > 0
    status = _wait_terminal(run_dir)
    assert status["state"] == "completed", status
    assert status["exit_code"] == 0
    assert status["progress"] == 1.0
    log = (run_dir / rt.LOG).read_text()
    assert "t: 50.0/50" in log
    events = [json.loads(line) for line in (run_dir / rt.EVENTS).read_text().splitlines()]
    assert [e["type"] for e in events] == ["stage", "metric", "stage"]
    assert status["metrics"] == {"nodes": 42}


def test_the_message_describes_the_current_progress_only(tmp_path: Path) -> None:
    """A stage starts with no message, and a progress update without one clears the last:
    the line under the bar never shows what an earlier stage or update was doing."""
    progress = rt._Progress(str(tmp_path / rt.EVENTS))
    progress.feed("  load step  4: lambda = 1.000000  newton =   4")
    assert progress.message == "load step 4 · λ = 1.000"
    progress.feed(rt.MARKER + ' {"type": "stage", "id": "beats", "label": "Beating"}')
    assert progress.message == "" and progress.progress is None
    progress.feed(rt.MARKER + ' {"type": "progress", "value": 0.2, "message": "downloading 1 / 5 MB"}')
    progress.feed(rt.MARKER + ' {"type": "progress", "value": 0.4}')
    assert progress.message == "" and progress.progress == 0.4


def test_activated_path_puts_the_environment_first(tmp_path: Path) -> None:
    path = rt._activated_path(str(tmp_path))
    parts = path.split(os.pathsep)
    assert parts[0] == sysconfig.get_path("scripts")  # where `pip install ninja` puts ninja
    assert str(tmp_path) in parts
    assert rt._activated_path(path) == path  # nothing is added twice


def test_run_sees_its_environment_on_path(tmp_path: Path) -> None:
    """The app starts the interpreter by its path, never activated, from a bare PATH
    (a Finder-launched app has /usr/bin:/bin:...); run.py must still find the tools
    installed in its environment, such as ninja for TorchCor's compiled pre-pacing."""
    bare = tmp_path / "bin"
    bare.mkdir()
    script = "import os\nopen('path.txt', 'w').write(os.environ['PATH'])\n"
    run_dir = _make_run(tmp_path, script)
    _run("launch", str(run_dir), env={**os.environ, "PATH": str(bare)})
    assert _wait_terminal(run_dir)["state"] == "completed"
    parts = (run_dir / "path.txt").read_text().split(os.pathsep)
    assert parts[0] == sysconfig.get_path("scripts")
    assert str(bare) in parts


def test_failed_script_reports_exception(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, "raise RuntimeError('mesh folder is empty')\n")
    _run("launch", str(run_dir))
    status = _wait_terminal(run_dir)
    assert status["state"] == "failed"
    assert "RuntimeError: mesh folder is empty" in status["error"]


def test_cancel_stops_a_running_script(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, "import time\nprint('started', flush=True)\ntime.sleep(120)\n")
    _run("launch", str(run_dir))
    deadline = time.time() + 20
    while "started" not in (run_dir / rt.LOG).read_text(errors="ignore") if (run_dir / rt.LOG).exists() else True:
        assert time.time() < deadline
        time.sleep(0.1)
    _run("cancel", str(run_dir))
    status = _wait_terminal(run_dir, timeout=30)
    assert status["state"] == "cancelled"


def test_scan_finds_runs_and_marks_dead_ones_lost(tmp_path: Path) -> None:
    good = _make_run(tmp_path, SCRIPT_OK, "good")
    (good / rt.STATUS).write_text(json.dumps({"state": "completed", "updated_at": time.time()}))
    dead = _make_run(tmp_path, SCRIPT_OK, "dead")
    (dead / rt.STATUS).write_text(json.dumps({"state": "running", "pid": 999_999, "host": "",
                                              "updated_at": time.time() - 3600}))
    result = _run("scan", str(tmp_path))
    states = {Path(r["run_dir"]).name: r["status"]["state"] for r in result["runs"]}
    assert states == {"good": "completed", "dead": "lost"}


def test_remove_refuses_non_runs(tmp_path: Path) -> None:
    folder = tmp_path / "precious"
    folder.mkdir()
    out = subprocess.run([sys.executable, RUNTIME, "remove", str(folder)], capture_output=True, text=True)
    assert out.returncode != 0
    assert folder.exists()


@pytest.mark.parametrize(
    ("line", "progress"),
    [
        ("t: 25.0/100 | Time elapsed: 1 | CG iter: 9", 0.25),
        ("  load step  3: lambda = 0.600000  newton =   7  mixed constraint = 1e-6  min J = 0.9", 0.6),
        ("     V3:  CG iters =  1234   |res|/|b| = 1.2e-09", 1 / 9),
    ],
)
def test_progress_parser(tmp_path: Path, line: str, progress: float) -> None:
    parser = rt._Progress(str(tmp_path / "events.jsonl"))
    parser.feed(line)
    assert parser.progress == pytest.approx(progress)


def test_progress_parser_uses_declared_total(tmp_path: Path) -> None:
    parser = rt._Progress(str(tmp_path / "events.jsonl"))
    parser.feed('::tcs:: {"type": "total", "value": 800, "unit": "ms"}')
    parser.feed("t   200.0 ms | p (mmHg) LV   9.5 RV   5.9 LA   9.5 RA   5.9 | V (mL) LV 120.0")
    assert parser.progress == pytest.approx(0.25)


def test_preview_extracts_surface_and_sites(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    _write_mesh(tmp_path / "mesh")
    # the deployed layout: tcs_runtime.py and tcs_studio.py side by side
    for name in ("tcs_runtime.py", "tcs_studio.py"):
        (tmp_path / name).write_text(runtime_path(name).read_text(encoding="utf-8"), encoding="utf-8")
    out = subprocess.run([sys.executable, str(tmp_path / "tcs_runtime.py"), "preview",
                          str(tmp_path / "mesh"), str(tmp_path / "p.npz")],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    data = np.load(tmp_path / "p.npz")
    assert data["triangles"].shape[1] == 3 and len(data["points"]) >= 3
    assert list(data["vtx_names"]) == ["LV_sf.vtx"]
    assert data["vtx_points"].shape == (2, 3)


def test_queued_run_waits_for_its_dependency(tmp_path: Path) -> None:
    first = _make_run(tmp_path, "import time\ntime.sleep(2.5)\nprint('first done', flush=True)\n", "first")
    second = _make_run(tmp_path, "print('second ran', flush=True)\n", "second")
    manifest = json.loads((second / rt.MANIFEST).read_text())
    manifest["after"] = [str(first)]
    (second / rt.MANIFEST).write_text(json.dumps(manifest))
    _run("launch", str(first))
    _run("launch", str(second))
    time.sleep(1.0)
    waiting = json.loads((second / rt.STATUS).read_text())
    assert waiting["state"] == "queued" and waiting["stage"] == "waiting", waiting
    assert "first" in waiting["message"]
    assert _wait_terminal(first)["state"] == "completed"
    assert _wait_terminal(second)["state"] == "completed"
    assert (second / rt.LOG).read_text().strip() == "second ran"


def test_cancel_while_queued(tmp_path: Path) -> None:
    blocker = _make_run(tmp_path, "import time\ntime.sleep(30)\n", "blocker")
    queued = _make_run(tmp_path, "print('should not run')\n", "queued")
    manifest = json.loads((queued / rt.MANIFEST).read_text())
    manifest["after"] = [str(blocker)]
    (queued / rt.MANIFEST).write_text(json.dumps(manifest))
    _run("launch", str(blocker))
    _run("launch", str(queued))
    time.sleep(1.0)
    _run("cancel", str(queued))
    assert _wait_terminal(queued, timeout=20)["state"] == "cancelled"
    assert not (queued / rt.LOG).exists()
    _run("cancel", str(blocker))
    assert _wait_terminal(blocker, timeout=30)["state"] == "cancelled"


def test_stage_can_opt_out_of_parsed_progress(tmp_path: Path) -> None:
    parser = rt._Progress(str(tmp_path / "events.jsonl"))
    parser.feed('::tcs:: {"type": "stage", "id": "unload", "label": "Unloading", "auto": false}')
    parser.feed("  load step  9: lambda = 0.900000  newton =   7")
    assert parser.progress is None
    parser.feed('::tcs:: {"type": "progress", "value": 0.25, "message": "iteration 2"}')
    assert parser.progress == pytest.approx(0.25)
    parser.feed('::tcs:: {"type": "stage", "id": "load", "label": "Inflating"}')
    parser.feed("  load step  3: lambda = 0.300000  newton =   7")
    assert parser.progress == pytest.approx(0.3)


def test_unloading_iterations_become_the_message(tmp_path: Path) -> None:
    parser = rt._Progress(str(tmp_path / "events.jsonl"))
    parser.feed('::tcs:: {"type": "stage", "id": "unload", "label": "Unloading", "auto": false}')
    parser.feed("iteration 3: largest mismatch 0.0123, mean 0.004, lv 0.53%, step 1")
    assert parser.message == "iteration 3 · largest mismatch 0.0123"
    assert parser.progress is None
