# ================================================
#               Expose entry points
# ================================================

# Import warnings configuration first (must be before cartopy import)
import toad._warnings  # noqa: F401
from toad._version import __version__
from toad.clustering import compute_clusters
from toad.core import TOAD
from toad.plotting import MapStyle, Plotter
from toad.shifts import compute_shifts

__all__ = [
    "TOAD",
    "compute_shifts",
    "compute_clusters",
    "Plotter",
    "MapStyle",
    "__version__",
]

# ================================================
