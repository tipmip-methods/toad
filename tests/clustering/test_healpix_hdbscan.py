import gc

import numpy as np
import pytest
from sklearn.cluster import HDBSCAN  # type: ignore

from toad import TOAD


@pytest.fixture(autouse=True)
def cleanup_memory():
    """Clean up memory after each test. Important otherwise get bus errors on some machines."""
    yield
    gc.collect()


def test_healpix_hdbscan():
    """Test the HealPix HDBSCAN clustering method.

    Verifies that clustering runs correctly on coarsened synth_data with HealPix
    regridding. Cluster count has ±1 tolerance for HDBSCAN variability.

    Note:
        A RuntimeWarning about numpy.ndarray size is suppressed in pytest.ini
        (known HDBSCAN/numpy issue: https://github.com/scikit-learn-contrib/hdbscan/issues/457)
    """
    td = TOAD("tutorials/test_data/synth_data.nc")
    td.data = td.data.coarsen(lat=2, lon=2, boundary="trim").reduce(np.mean)

    td.compute_clusters(
        shift_threshold=0.5,
        method=HDBSCAN(min_cluster_size=25),
        overwrite=True,
        shift_selection="all",
    )

    N_clusters = len(td.get_cluster_ids(td.base_vars[0], exclude_noise=True))
    assert abs(N_clusters - 2) <= 1, f"Expected 2±1 clusters, got {N_clusters}"
