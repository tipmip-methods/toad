import pytest
import numpy as np
import xarray as xr
from toad import TOAD
from sklearn.cluster import HDBSCAN

"""
- TOAD.compute_clusters: make sure results can be extrated directly
"""

@pytest.fixture
def toad_instance():
    """
    Minimal TOAD instance with shift values for testing purposes.
    """
    data_arr = np.array([[0],[0],[0],[0],[0]], dtype=float)     # minimum of 5 cells required for HDBSCAN
    data_dts_arr = np.array([[0],[0],[0],[0],[0]], dtype=float)
    time_arr = np.arange(len(data_arr))
    xr_data = xr.Dataset(
            {"data": (["x", "y", "time"], data_arr.reshape(1, 1, -1)),
             "data_dts": (["x", "y", "time"], data_dts_arr.reshape(1, 1, -1))},
            coords={"time": time_arr, "x": [0], "y": [0]},
        )
    return TOAD(xr_data)

def test_compute_clusters(toad_instance):
    """
    - test 1: check if computed clusters can extracted directly and are not written to the dataset
    """

    # test 1
    # - setup
    td = toad_instance
    cluster = td.compute_clusters(
                var="data",                     # toad will find computed shifts for this variable
                method=HDBSCAN(),
                return_results_directly=True,   # <-- this is the key part of the test
    )

    # - check if shifts are extracted correctly
    assert isinstance(cluster, xr.DataArray)
    assert "data_cluster" in cluster.name
    assert "x" in cluster.coords
    assert "y" in cluster.coords
    assert "time" in cluster.coords

    # - check if the data is not written to the dataset
    assert "data_cluster" not in td.data.variables