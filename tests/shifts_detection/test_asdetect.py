import pytest
import numpy as np
import xarray as xr
from toad import TOAD
from toad.shifts import ASDETECT

@pytest.fixture
def test_params_centered():
    """Fixture providing parameters for the ASDETECT test in segmentation mode "centered".

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
        "expected_mean": 0.0045629148,
        "expected_std": 0.2109171152,
    }

@pytest.fixture
def test_params_fast_correction():
    """Fixture providing parameters for the ASDETECT test in segmentation mode "fast_correction".

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
        "expected_mean": 0.010336183,
        "expected_std": 0.2058115,
    }

@pytest.fixture
def test_params_fine_correction():
    """Fixture providing parameters for the ASDETECT test in segmentation mode "fine_correction".

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
        "expected_mean": 0.017693562,
        "expected_std": 0.20061477,
    }

@pytest.fixture
def test_params_full_correction():
    """Fixture providing parameters for the ASDETECT test in segmentation mode "full_correction".

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
        "expected_mean": 0.016856669,
        "expected_std": 0.18138373,
    }


@pytest.fixture
def toad_instance():
    return TOAD("tutorials/test_data/global_mean_summer_tas.nc")

def test_asdetect_centered(test_params_centered, toad_instance):
    """Test the ASDETECT shift detection method in segmentation mode "centered".

    Two tests are performed:
    1. A simple test with a known dataset to verify the correctness of the
       shift detection method.
    2. A test using a more realistic dataset to ensure the method works as expected
       in practical scenarios.

    Args:
        test_params (dict): Parameters for the test.
        toad_instance (TOAD): Instance of TOAD containing the data.

    - Test 1 - additional notes:
        Advantage of this test is that it can be easier to trace if
        there is anything wrong with the mathematics of the implementation.
        The given example can be computed by hand.

        >> The example here is:
        data = [0, 0, 0, 1, 1, 1]       <- clear shift between index 2 and 3
        l_min = 2                       <- minimum segmentation length
        l_max = 3                       <- maximum segmentation length
            -> this leads to two segmentation steps of the data:
            i)  data_seg1 = [[0,0],[0,1],[1,1]]
                -> gradients_seg1 = [0, 1 ,0]
                -> median_seg1 = 0, MAD_seg1 = 0
                -> detection_ts = [0, 0, 1, 1, 0, 0]
            ii) data_seg2 = [[0,0,0],[1,1,1]]
                -> gradients_seg2 = [0, 0]
                -> median_seg2 = 0, MAD_seg2 = 0
                -> detection_ts = [0, 0, 1, 1, 0, 0]    <- nothing changed
            The shift values are divided by the number of segment lenghts:
            -> shifts = [0, 0, 0.5, 0.5, 0, 0]

    - Test 2 - additional notes:
        This test verifies the computation of shifts using the ASDETECT method
        after coarsening the data based on specified latitude, longitude, and
        time parameters. It checks that the computed mean and standard deviation
        of the shifts match the expected results.
    """

    ##########################################
    # test 1

    # - setup
    data_arr = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    time_arr = np.arange(len(data_arr))
    td = TOAD(
        xr.Dataset(
            {"data": (["lat", "lon", "time"], data_arr.reshape(1, 1, -1))},
            coords={"time": time_arr, "lat": [0], "lon": [0]},
        )
    )

    # - call function
    td.compute_shifts("data", ASDETECT(lmin=2, lmax=3, segmentation="centered"), overwrite=True)
    shifts = td.get_shifts("data").data[0, 0]

    # - compare results
    assert np.array_equal(shifts, np.array([0, 0, 0.5, 0.5, 0, 0], dtype=float))

    ##########################################
    # test 2

    # - setup
    td = toad_instance
    td.data = td.data.coarsen(
        lat=test_params_centered["lat"],
        lon=test_params_centered["lon"],
        time=test_params_centered["time"],
        boundary="trim",
    ).mean()

    # - call function
    td.compute_shifts("tas", ASDETECT(segmentation="centered"), overwrite=True)
    shifts = td.get_shifts("tas")
    mean = shifts.mean().values
    std = shifts.std().values

    # - compare results
    np.testing.assert_allclose(mean, test_params_centered["expected_mean"], rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(std, test_params_centered["expected_std"], rtol=1e-5, atol=1e-8)


def test_asdetect_fast_correction(test_params_fast_correction, toad_instance):
    """Test the ASDETECT shift detection method in segmentation mode "fast_correction".

    Same idea as test 2 from test_asdetect_centered.
    """

    # - setup
    td = toad_instance
    td.data = td.data.coarsen(
        lat=test_params_fast_correction["lat"],
        lon=test_params_fast_correction["lon"],
        time=test_params_fast_correction["time"],
        boundary="trim",
    ).mean()

    # - call function
    td.compute_shifts("tas", ASDETECT(segmentation="fast_correction"), overwrite=True)
    shifts = td.get_shifts("tas")
    mean = shifts.mean().values
    std = shifts.std().values

    # - compare results
    np.testing.assert_allclose(mean, test_params_fast_correction["expected_mean"], rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(std, test_params_fast_correction["expected_std"], rtol=1e-5, atol=1e-8)


def test_asdetect_fine_correction(test_params_fine_correction, toad_instance):
    """Test the ASDETECT shift detection method in segmentation mode "fine_correction".

    Same idea as test 2 from test_asdetect_centered.
    """

    # - setup
    td = toad_instance
    td.data = td.data.coarsen(
        lat=test_params_fine_correction["lat"],
        lon=test_params_fine_correction["lon"],
        time=test_params_fine_correction["time"],
        boundary="trim",
    ).mean()

    # - call function
    td.compute_shifts("tas", ASDETECT(segmentation="fine_correction"), overwrite=True)
    shifts = td.get_shifts("tas")
    mean = shifts.mean().values
    std = shifts.std().values

    # - compare results
    np.testing.assert_allclose(mean, test_params_fine_correction["expected_mean"], rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(std, test_params_fine_correction["expected_std"], rtol=1e-5, atol=1e-8)

def test_asdetect_full_correction(test_params_full_correction, toad_instance):
    """Test the ASDETECT shift detection method in segmentation mode "full_correction".

    Same idea as test 2 from test_asdetect_centered.
    """

    # - setup
    td = toad_instance
    td.data = td.data.coarsen(
        lat=test_params_full_correction["lat"],
        lon=test_params_full_correction["lon"],
        time=test_params_full_correction["time"],
        boundary="trim",
    ).mean()

    # - call function
    td.compute_shifts("tas", ASDETECT(segmentation="full_correction"), overwrite=True)
    shifts = td.get_shifts("tas")
    mean = shifts.mean().values
    std = shifts.std().values

    # - compare results
    np.testing.assert_allclose(mean, test_params_full_correction["expected_mean"], rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(std, test_params_full_correction["expected_std"], rtol=1e-3, atol=1e-5)
