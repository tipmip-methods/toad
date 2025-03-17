import pytest
import numpy as np
from toad import TOAD
from toad.shifts_detection.methods import ASDETECT
from toad.shifts_detection.methods.asdetect import centered_segmentation


@pytest.fixture
def test_params():
    """Fixture providing parameters for the ASDETECT test.

    Returns:
        dict: A dictionary containing:
            - lat (int): Latitude coarsening factor.
            - lon (int): Longitude coarsening factor.
            - time (int): Time coarsening factor.
            - expected_mean (float): Expected mean of the shifts.
            - expected_std (float): Expected standard deviation of the shifts.
    """
    return {
        "lat": 10,
        "lon": 10,
        "time": 3,
        "expected_mean": 0.0038361712,
        "expected_std": 0.19410655,
    }


@pytest.fixture
def toad_instance():
    return TOAD("../tutorials/test_data/global_mean_summer_tas.nc")


def test_asdetect(test_params, toad_instance):
    """Test the ASDETECT shift detection method.

    This test verifies the computation of shifts using the ASDETECT method
    after coarsening the data based on specified latitude, longitude, and
    time parameters. It checks that the computed mean and standard deviation
    of the shifts match the expected results.

    Args:
        test_params (dict): Parameters for the test.
        toad_instance (TOAD): Instance of TOAD containing the data.
    """
    # Setup
    td = toad_instance
    td.data = td.data.coarsen(
        lat=test_params["lat"],
        lon=test_params["lon"],
        time=test_params["time"],
        boundary="trim",
    ).mean()

    # Function to test
    td.compute_shifts("tas", ASDETECT(), overwrite=True)

    shifts = td.get_shifts("tas")
    mean = shifts.mean().values
    std = shifts.std().values

    # Verifcation
    np.testing.assert_allclose(mean, test_params["expected_mean"], rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(std, test_params["expected_std"], rtol=1e-3, atol=1e-3)


def test_centered_segmentation():
    """
    Test simple case examples without truncation.

    **Example**

    >>> tsanalysis.asdetect.centered_segmentation(l_tot=10, l_seg=2)
    array([ 0, 2, 4, 6, 8, 10])
    # -> segmented list: [[0,1],[2,3],[4,5],[6,7],[8,9]]
    """

    # Setup
    l_tot = 10
    l_seg = 2

    # Function to test
    out = centered_segmentation(l_tot,l_seg)

    # Verifcation
    assert isinstance(out,(np.ndarray))                 # output data type
    assert np.array_equal(out,np.array([0,2,4,6,8,10])) # output value
