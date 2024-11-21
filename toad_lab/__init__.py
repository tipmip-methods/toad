import logging
import xarray as xr
from typing import Union, Callable
import os

from . import shifts_detection
from . import clustering
from . import postprocessing
from .utils import deprecated
from _version import __version__


# ================================================
#               Expose entry points
# ================================================

from .core import TOAD
from .shifts_detection import compute_shifts
from .clustering import compute_clusters
from .visualisation import TOADPlotter

__all__ = ["TOAD", "compute_shifts", "compute_clusters", "TOADPlotter"]

# ================================================


