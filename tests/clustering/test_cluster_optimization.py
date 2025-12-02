import numpy as np
from sklearn.cluster import HDBSCAN  # type: ignore

from toad import TOAD


def test_cluster_optimization():
    """Test the cluster optimization."""

    # Setup
    td = TOAD("tutorials/test_data/global_mean_summer_tas.nc")
    td.data = td.data.coarsen(lat=3, lon=3, boundary="trim").reduce(np.mean)

    # Drop any cluster vars
    td.data = td.data.drop_vars(td.cluster_vars)

    td.compute_clusters(
        optimize=True,
        optimization_params={
            "min_cluster_size": (5, 15),
            "shift_threshold": 0.75,
            "time_scale_factor": (0.5, 2.0),
        },
        method=HDBSCAN,
        shift_selection="local",
        n_trials=10,
    )

    # Since a short optimization like this won't always converge to the same result, we just check that a new cluster label was added.
    assert len(td.cluster_vars) == 1
