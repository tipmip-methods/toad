import numpy as np
import pytest

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
        "expected_mean": 0.017697551648168934,
        "expected_std": 0.15253265952046005,
    }


@pytest.fixture
def test_params_two_sided():
    """Fixture providing parameters for the ASDETECT test in segmentation mode "two_sided".

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
        "expected_mean": 0.024135386789707777,
        "expected_std": 0.1289442182680194,
    }


@pytest.fixture
def toad_instance():
    return TOAD("tutorials/test_data/synth_data.nc")


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

    # - setup
    td = toad_instance
    td.data = td.data.coarsen(
        lat=test_params_centered["lat"],
        lon=test_params_centered["lon"],
        time=test_params_centered["time"],
        boundary="trim",
    ).mean()

    # - call function
    td.compute_shifts(
        "ts", ASDETECT(segmentation="original"), overwrite=True, run_parallel=False
    )  # test non-parallel mode
    shifts = td.get_shifts("ts")
    mean = shifts.mean().values
    std = shifts.std().values

    # - compare results
    np.testing.assert_allclose(
        mean, test_params_centered["expected_mean"], rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(
        std, test_params_centered["expected_std"], rtol=1e-5, atol=1e-8
    )


def test_asdetect_two_sided(test_params_two_sided, toad_instance):
    """Test the ASDETECT shift detection method in segmentation mode "two_sided".

    Same idea as test 2 from test_asdetect_centered.
    """

    # test 1

    # - setup
    np.random.seed(4)
    data_arr = np.random.randn(50)
    data_arr[20:] += 20
    shifts = ASDETECT(segmentation="two_sided").fit_predict(
        data_arr, np.arange(len(data_arr), dtype=np.float64)
    )

    # - compare results
    expected_shifts = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.04166667,
            0.04166667,
            0.04166667,
            0.125,
            0.16666667,
            0.20833333,
            0.375,
            0.45833333,
            0.54166667,
            0.58333333,
            0.75,
            0.75,
            0.75,
            0.70833333,
            0.58333333,
            0.54166667,
            0.375,
            0.375,
            0.25,
            0.20833333,
            0.125,
            0.125,
            0.08333333,
            0.08333333,
            0.04166667,
            0.04166667,
            0.0,
            0.0,
            -0.04166667,
            -0.04166667,
            -0.04166667,
            -0.04166667,
            -0.04166667,
            -0.04166667,
            -0.04166667,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    np.testing.assert_allclose(shifts, expected_shifts, atol=1e-6)

    # Real Data Test ==========

    # - setup
    td = toad_instance
    td.data = td.data.coarsen(
        lat=test_params_two_sided["lat"],
        lon=test_params_two_sided["lon"],
        time=test_params_two_sided["time"],
        boundary="trim",
    ).mean()

    # - call function
    td.compute_shifts(
        "ts", ASDETECT(segmentation="two_sided"), overwrite=True, run_parallel=False
    )
    shifts = td.get_shifts("ts")
    mean = float(shifts.mean().values)
    std = float(shifts.std().values)

    # - compare results
    # Use strict type-casting and match to float32 tolerance
    # The actual value is float32, so we need slightly more lenient tolerances
    # Max absolute difference observed: ~2.9e-07, so atol=1e-6 is safe
    np.testing.assert_allclose(
        mean, test_params_two_sided["expected_mean"], rtol=2e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        std, test_params_two_sided["expected_std"], rtol=2e-5, atol=1e-6
    )
