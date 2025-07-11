import pytest
import numpy as np
import xarray as xr
from toad import TOAD

"""
The following tests are written to cover cases that are not yet covered by other tests.

Things to test:
- TOAD.init: check if names of coordinates are renamed correctly
"""

def test_init():
    """
    - test 1: check if names "longitude" and "latitude" are renamed to "lon" and "lat"
    """
    # test 1
    # - setup
    data_arr = np.array([0], dtype=float)
    time_arr = np.arange(len(data_arr))
    xr_data = xr.Dataset(
            {"data": (["latitude", "longitude", "time"], data_arr.reshape(1, 1, -1))},
            coords={"time": time_arr, "latitude": [0], "longitude": [0]},
        )

    # - call function
    td = TOAD(xr_data)

    # - check names
    assert "lat" in td.data.coords
    assert "lon" in td.data.coords
    assert "latitude" not in td.data.coords
    assert "longitude" not in td.data.coords