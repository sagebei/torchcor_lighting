# TorchCor Lighting — architecture

TorchCor Lighting is a thin, fast client around TorchCor. It never imports TorchCor or
PyTorch itself: every simulation is a plain Python script that runs in a Python
environment on a *compute machine* (this computer or an SSH host), supervised by a tiny
stdlib-only agent. The desktop app writes the script, starts it, and reads back status,
logs and a compact visualisation bundle.

```
┌──────────────────────────── desktop app (PySide6) ─────────────────────────────┐
│ ui/        pages · sidebar · forms · results viewers · dialogs · theme          │
│   ▲ Qt signals                                                                   │
│ ui/context.py  AppContext — machine sessions, telemetry, history, run poller    │
│   ▲ worker threads (ui/tasks.run_async)                                          │
│ core/      Qt-free: machines (local/SSH) · gpu · simulations · runs · sweep     │
└──────────────────────────────────┬──────────────────────────────────────────────┘
                                   │ subprocess / SSH exec + SFTP
┌──────────────────────────── compute machine ───────────────────────────────────┐
│ runtime/tcs_runtime.py  probe · inspect · scan · launch → supervise · cancel    │
│ <output>/<run-id>/run.py   generated TorchCor script  (imports tcs_studio.py)   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Layers

| Package | Depends on | Responsibility |
| :--- | :--- | :--- |
| `core/` | stdlib, numpy, pydantic, paramiko | Machines, telemetry, simulation specs, run service, history, sweeps. **Never imports Qt**, so it is unit-tested headlessly and usable from scripts. |
| `runtime/` | stdlib (+ numpy/torch inside `tcs_studio`) | Files copied verbatim to compute machines and into every run folder. Must not import `torchcor_studio`. |
| `ui/` | PySide6, pyqtgraph, pyvista | Presentation. Blocking work goes through `ui.tasks.run_async`; colours come from theme tokens only. |

## Key contracts

* **`Machine`** (`core/machines/base.py`) — one interface for `LocalMachine` and
  `SSHMachine`: `run`, `stream`, file access, `runtime(command, …)`. Paths are strings in the
  target machine's convention; use `machine.join()`.
* **`SimulationSpec`** (`core/simulations/base.py`) — a pydantic parameter model whose
  defaults are a complete configuration, a `build_script(params, ctx) -> str`, demos
  (`Demo`: ready-to-run set-ups, each naming the data case it reads, such as
  `ep/ventricle`), panels, `check()` validation and the stage list for the live tracker.
  Field metadata (`core/simulations/fields.py::param`) drives the generated form, and
  `core/simulations/panels.py` groups it into panels.
* **Run folder protocol** (`runtime/tcs_runtime.py`): `manifest.json` (what was asked, written by
  the app) · `run.py` · `status.json` (state, progress, ETA — written by the supervisor) ·
  `run.log` · `events.jsonl` · `cancel.request` · `results/` · `viz/`. A run is just a folder,
  so it can be re-run with `python run.py`, survives app restarts and SSH drops, and is
  discovered again by `tcs_runtime.py scan`.
* **Progress markers** — scripts print `::tcs:: {json}` lines (`tcs_studio.stage/progress/total`);
  the supervisor also parses TorchCor's own output (`t: 12/500 |`, `load step N: lambda = …`).
  A stage that repeats inner solves (EM unloading) passes `stage(..., auto=False)` so those
  lines don't masquerade as overall progress.
* **Visualisation bundle** (`runtime/tcs_studio.py` docstring) — `viz/index.json` plus
  float16-packed `surface.npz`, `ep.npz`, `ecg.npz`, `mechanics.npz`, `em.npz`. Remote runs
  download only this folder; full results stay on the machine.
* **Queue** — `manifest["after"]` lists run folders that must finish first; the supervisor
  waits on the compute machine, which is how sweeps share GPUs safely.
* **Clocks** — a run's times (`status.json`) are written by its machine's clock, which may
  be off. The probe reports that clock, `Machine.clock_offset` keeps the difference, and
  run statuses (`runs.service.status_from_machine`) and file times are put on this
  computer's clock as they are read, so elapsed times, "lost contact" and "5 min ago" are
  right whatever the server's clock says.
* **Tabs** (`ui/workspace.py`, `ui/main_window.py`): the right panel has one tab per
  simulation, with its steps at the top (`pages.common.SIMULATION_STEPS`: Compute,
  Setup, Run, Result). A *draft* tab (`NewRunPage`: Compute, with the device, then
  Setup, with the parameter panels, the rail and the Run button) is of the kind chosen in
  the New menu (`ui/new_menu.py`: the sidebar's New, the strip's "+", Ctrl+N), which it
  keeps, and is named after it until renamed (`kinds.next_name`: EP, EP_1, …). It is
  listed in the sidebar as a
  `DraftEntry` (filed in a project by dragging it or with Move to project) and, once
  launched, is replaced in place by the run's `RunPage`. There the steps are the
  navigation: each shows its views as tabs (`run_page.VIEWS`), and Result opens by itself
  when the run completes. Runs opened from the sidebar
  reuse a single *preview* tab unless kept. Machines and Settings are single tabs. A
  simulation's name lives on its tab and its sidebar entry (both edit it in place). On
  macOS the window draws under its title bar (`ui/widgets/titlebar.py`: Qt's
  `ExpandedClientAreaHint`), and empty parts of the tab strip and the sidebar top move it.
* **Projects** (`core/runs/history.py`): sidebar folders. The list is kept locally, and a
  run's membership is saved in its manifest (`"project": {"id", "name"}`), so scans bring
  projects back. Deleted projects are remembered, so stale manifests on offline machines
  cannot recreate them.
* **Data folders** (`core/data_folders.py`): `~/torchcor_data/<kind>/<case>` (or the
  machine's input folder laid out the same way), listed once when a machine connects.
  New runs start their input in the folder of their kind, a demo points it at its case
  (`demo_data`: found, built in, downloaded by the run, missing, or not known until the
  machine connects), and `SimulationSpec.fit_input` then fills in what depends on the
  folder, such as EP stimuli defaulting to its first `.vtx`.
* **The Setup step** (`ui/pages/new_run/setup_step.py`): the kind as the page's title,
  then the parameters as panels beside a sticky rail (`StickyRow`): an outline of the panels with
  their issue counts, then three cards: the checks, the kind's demos (a list,
  `DemoPicker`) and the parameter sweep. On a narrow page the outline turns into chips
  above the panels and the sweep moves below them. Every kind's first panel ends with the
  output folder (`panels.OUTPUT`) and holds the input folder where the kind puts
  `panels.INPUT`; issues about the input folder point at `panels.INPUT`, so the folder
  field shows them. The form (`ui/forms`) resolves the kind's `PANELS` into blocks: a
  grid of fields sized by meaning (`FieldGrid`), group tiles, tables whose columns take
  the width their content needs, and the two folder slots. A collapsed panel is built
  when first opened, and demos, Reset, Undo, fits and duplicates update the form in
  place.
* **Choosing a Python** (`core/environment.py`) — `find_pythons` probes every interpreter
  on a machine (packages are located, not imported). It ranks them by
  `PythonCandidate.readiness`: TorchCor on a GPU first, then TorchCor on the CPU, then CUDA
  PyTorch without TorchCor, and so on. A TorchCor older than `models.MIN_TORCHCOR` counts as
  missing. `Machine.detect_python` takes the best one unless the profile pins one, and
  `better_python` drives the Environment card's "better match" hint.

## Adding a simulation kind

1. Create `core/simulations/<kind>.py` with a `Section` parameter model (every field via
   `param(...)` with a default, unit and help), a `build_script()` that emits a readable,
   self-contained script calling `studio.stage(...)` and a `studio.export_*`/`Bundle`, and
   `register(SimulationSpec(...))` with:
   * `DEMOS`: `Demo`s whose first one is the defaults, each with its data `case`;
   * `PANELS`: `Panel(title, icon, paths)` in physiology order, `INPUT` where the input
     folder goes and `OUTPUT` at the end of the first panel (a test checks that every
     parameter is in exactly one panel);
   * `input_use(params)`: is the input folder required, optional or unused?

   Layout hints stay in the field metadata: `group`, `short`, `span`, `summary`,
   `fixed`, `item`.
2. Add the module name to `_MODULES` in `core/simulations/base.py` and the kind to
   `SimulationKind`.
3. If it produces a new kind of result, add a dataset to the bundle spec and a view in
   `ui/results/`.

The form, validation display, script preview, history, sweeps and the run page work
without further changes.

## Testing on a GPU server

The fast suite runs headless anywhere (`pytest -m "not slow"`); `pytest -m slow` adds real,
small TorchCor runs on the CPU. Every kind was also run end to end through `RunService`
and the real window over SSH on a 2× RTX 3090 server:

| Run | Size | Wall time |
| :-- | :-- | :-- |
| EP, biventricle, five pacing sites (monodomain, ten Tusscher–Panfilov), 500 ms | 637k nodes, 2.9M tets | 2.4 min |
| EP, atrium (Courtemanche), 500 ms | 661k nodes, 1.3M triangles | 4.2 min |
| EP, biventricle, reaction–eikonal, 500 ms | 637k nodes | 3.9 min |
| 12-lead ECG, KCL torso 3 (incl. the one-off 122 MB dataset download) | 403k heart nodes | 6.7 min |
| Mechanics, Land problem 1 beam | 3k DOF | 16 s |
| Mechanics, idealised LV beating through one cycle (Aróstica B1) | 4.6k DOF, 400 steps | 79 s |

## Threading

The Qt main thread only draws. `run_async` runs I/O on an 8-thread pool and delivers
results back through a queued signal; callbacks owned by deleted widgets are dropped.
Python's cyclic garbage collector runs only on the GUI thread (`ui/gc.py`): a collection
started in a worker thread could otherwise destroy Qt objects off-thread and crash.
Telemetry uses one long-lived `nvidia-smi -lms` stream per machine, ref-counted by the
pages that display it.

## Qt layouts

To move a widget within a layout, use `removeWidget` and then add it again. Never use a
bare `takeAt`: PySide does not own the item `takeAt` returns, so the item stays alive, and
Qt keeps it as the widget's layout item. The new item's cached height-for-width then never
refreshes, and a card whose content grows stays clipped. `StickyRow` and
`ResponsiveGrid` do this, and `FieldGrid` never takes a widget out: it skips hidden ones.
