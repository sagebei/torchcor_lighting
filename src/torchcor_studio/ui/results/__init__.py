"""Results visualisation: one view per bundle kind (EP, ECG, mechanics, electromechanics).

    from torchcor_studio.ui.results import create_results_view

    view = create_results_view(run_dir / "viz", parent)   # loads in the background
    view.set_active(False)   # while its tab is hidden: pauses playback and rendering
    view.release()           # before deleting it: frees the VTK render window

The heavy modules (VTK, pyqtgraph) are imported only when a view is built.
"""

from torchcor_studio.ui.results.container import ResultsView, create_results_view

__all__ = ["ResultsView", "create_results_view"]
