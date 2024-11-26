import xarray as xr
import numpy as np


class Preprocess:
    def __init__(self, toad):
        self.td = toad

    def apply_xmip_conventions(self):
        """Apply XMIP conventions to the data."""
        raise NotImplementedError("This function is not yet implemented.")
    