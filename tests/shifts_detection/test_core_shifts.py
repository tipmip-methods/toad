import pytest
import numpy as np
import xarray as xr
from toad import TOAD
from toad.shifts_detection.methods import ASDETECT

"""
- TOAD.compute_shifts: make sure results can be extrated directly
"""

@pytest.fixture
def toad_instance():
    """
    Minimal TOAD instance for testing purposes.
    """
    data_arr = np.array([0], dtype=float)
    time_arr = np.arange(len(data_arr))
    xr_data = xr.Dataset(
            {"data": (["latitude", "longitude", "time"], data_arr.reshape(1, 1, -1))},
            coords={"time": time_arr, "latitude": [0], "longitude": [0]},
        )
    return TOAD(xr_data)

def test_compute_shifts(toad_instance):
    """
    - test 1: check if computed shifts can extracted directly and are not written to the dataset
    """

    # test 1
    # - setup
    td = toad_instance
    shifts = td.compute_shifts(
        "data",
        ASDETECT(),
        return_results_directly=True
        )

    # - check if shifts are extracted correctly
    assert isinstance(shifts, xr.DataArray)
    assert "data_dts" in shifts.name
    assert "lat" in shifts.coords
    assert "lon" in shifts.coords
    assert "time" in shifts.coords

    # - check if the data is not written to the dataset
    assert "data_dts" not in td.data.variables
