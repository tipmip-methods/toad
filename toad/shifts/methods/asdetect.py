"""ASDETECT method for shifts detection.

Contains the ASDETECT algorithm with associated helper functions.

Created: January 22, ? (Sina)
Refactored: Nov, 2024 (Jakob)
"""

import logging
from typing import Optional
from numpy.typing import NDArray

import numpy as np
from numba import njit
from numpy.linalg import lstsq

from .base import ShiftsMethod


class ASDETECT(ShiftsMethod):
    """Detect abrupt shifts in a time series using gradient-based analysis related to [Boulton+Lenton2019]_.

    Steps:
        1. Divide the time series into (overlapping) segments of size `l`.
        2. Perform linear regression within each segment to calculate gradients.
        3. Identify significant gradients exceeding ±3 Median Absolute Deviations (MAD) from the median gradient.
        4. Update a detection array by adding +1 for significant positive gradients and -1 for significant negative gradients in each each segment.
        5. Iterate over multiple window sizes (`l`), updating the detection array at each step.
        6. Normalize the detection array by dividing by the number of window sizes used.

    Note: ASDETECT does not work with NaN values so it will return a detection time series of all zeros if the input time series contains NaN values.

    Args:
        lmin: The minimum segment length for detection. Defaults to 5.
        lmax: (Optional) The maximum segment length for detection. If not specified, it defaults to one-third of the size of the time dimension.
        timescale: (Optional) A tuple specifying the minimum and maximum time window sizes for shift detection, in the same units as the input time axis (e.g. years, days etc). If provided, this will be used instead of lmin/lmax to determine the window sizes. The tuple values can be None to use the default bounds (5 timesteps for minimum, 1/3 of series length for maximum).
        segmentation: (Optional) The segmentation method to use. Options are "centered" (classic), "fast_correction" (removes bias), "fine_correction" (removes bias + smoother) and "full_correction" (perfectly smooth, but very slow). Defaults to "fast_correction".
        ignore_nan_warnings: (Optional) If True, timeseries containing NaN values will be ignored, i.e. a detection time series of all zeros will be returned. If False, an error will be raised.
    """

    # minimum allowed segment length
    LMIN_MIN = 5

    def __init__(
        self,
        lmin: int = LMIN_MIN,
        lmax: Optional[int] = None,
        timescale: Optional[tuple[Optional[float], Optional[float]]] = None,
        segmentation: str = "fast_correction",
        ignore_nan_warnings: bool = False,
    ):
        self.lmin = lmin
        self.lmax = lmax
        self.timescale = timescale
        self.segmentation = segmentation
        self.ignore_nan_warnings = ignore_nan_warnings
        self._converted_timescale = False

        assert timescale is None or (
            isinstance(timescale, tuple)
            and len(timescale) == 2
            and (timescale[0] is not None or timescale[1] is not None)
            and (timescale[0] is None or isinstance(timescale[0], (float, int)))
            and (timescale[1] is None or isinstance(timescale[1], (float, int)))
            and (
                timescale[0] is None
                or timescale[1] is None
                or timescale[1] > timescale[0]
            )
        ), (
            f"Timescale must be a tuple of two numbers with the second number larger than the first, e.g. (20, 30). One of the numbers can be None. If the first is None, e.g. (None, 30), {self.LMIN_MIN} is used as default. If the second is None, e.g., (20, None), 1/3 the length of the time series is used as default."
        )

    def _get_segment_lengths(
            self,
            times_1d: NDArray[np.float64],
        ) -> tuple[int, int]:
        """Get the final lmin and lmax values, handling timescale conversion if needed.

        Args:
            times_1d: 1D array of times

        Returns:
            tuple: (lmin, lmax) as integers
        """

        # compute max lmax
        lmax_max = len(times_1d) // 3

        # Start with direct parameters
        lmin = self.lmin
        lmax = self.lmax if self.lmax is not None else lmax_max

        # Override with timescale if provided
        if self.timescale is not None:
            dt = np.diff(times_1d)[0]

            # Convert timescale to timesteps
            if self.timescale[0] is not None:
                lmin = int(self.timescale[0] / dt)

            if self.timescale[1] is not None:
                lmax = int(self.timescale[1] / dt)

            logging.getLogger("TOAD").debug(
                f"for dt={dt:.2f} -> (lmin={lmin}, lmax={lmax})"
            )

        # If the inferred lmin is too small, throw error [Boulton+Lenton2019 do not set strict lower bound]
        if (
            lmin < self.LMIN_MIN
            and self.timescale is not None
            and self.timescale[0] is not None
        ):
            raise ValueError(
                f"The temporal resolution is too low to detect shifts at timescales of {self.timescale[0]} (units of time). We recommend using a minimum timescale of {(self.LMIN_MIN * dt)} (units of time)."
            )

        # If the inferred lmax is too large, warn the user and overwrite it [Boulton+Lenton2019 sets a strict upper bound]
        if (
            lmax > lmax_max
            and self.timescale is not None
            and self.timescale[1] is not None
        ):
            logging.getLogger("TOAD").warning(
                f"The time series is not long enough for detecting shifts at timescales of {self.timescale[1]} (units of time). A maximum upper bound of {((lmax_max) * dt)} (units of time) has been imposed. This corresponds to 1/3 the length of the time series."
            )
            lmax = lmax_max

        # if user manually set lmax too high
        if lmax > lmax_max:
            logging.getLogger("TOAD").warning(
                f"lmax cannot be larger than 1/3 the length of the time series. Setting lmax to {lmax_max}."
            )
            lmax = lmax_max

        self._converted_timescale = True
        return lmin, lmax

    def fit_predict(
        self,
        values_1d: NDArray[np.float64],
        times_1d: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute the detection time series for each grid cell in the 3D data array.

        Args:
            values_1d: 1D array of values
            times_1d: 1D array of times

        Returns:
            A 1D array of the same length as `values_1d`, where each value represents the abrupt shift score for a grid cell at a specific time. The score ranges from -1 to 1:
                - `1` indicates that all tested segment lengths detected a significant positive gradient (i.e. exceeding 3 MAD of the median gradient),
                - `-1` indicates that all tested segment lengths detected a significant negative gradient.
                - Values between -1 and 1 indicate the proportion of segment lengths detecting a significant gradient at that time point.
        """

        # infer lmin/lmax from timescale, if provided,
        if self.timescale is not None and not self._converted_timescale:
            # overwrite self.lmin/self.lmax because they are saved to attrs in TOAD.
            self.lmin, self.lmax = self._get_segment_lengths(times_1d)

        shifts = construct_detection_ts(
            values_1d=values_1d,
            times_1d=times_1d,
            lmin=self.lmin,
            lmax=self.lmax,
            segmentation=self.segmentation,
            ignore_nan_warnings=self.ignore_nan_warnings,
        )

        return shifts


# 1D time series analysis of abrupt shifts =====================================
@njit
def construct_detection_ts(
    values_1d: NDArray[np.float64],
    times_1d: NDArray[np.float64],
    lmin: int = 5,
    lmax: Optional[int] = None,
    segmentation: str = "fast_correction",
    ignore_nan_warnings: bool = False,
) -> NDArray[np.float64]:
    """Construct a detection time series (asdetect algorithm).

    Following [Boulton+Lenton2019]_, the time series (ts) is divided into
    segments of length l, for each of which the gradient is computed. Segments
    with gradients > 3 MAD of the gradients distribution are marked. Averaging
    over many segmentation choices (i.e. values of l) results in a detection
    time series that indicates the points of largest relative gradients.

    >> Args:
        values_1d:
            Time series, shape (n,)
        times_1d:
            Times, shape (n,), same length as values_1d
        lmin:
            Smallest segment length, default = 5
        lmax:
            Largest segment length, default = n/3
        segmentation:
            Segmentation method to use. Options are "centered" (classic), "fast_correction" (removes bias), "fine_correction" (removes bias + smoother) and
            "full_correction" (perfectly smooth, but very slow). Defaults to "fast_correction".
        ignore_nan_warnings:
            If True, timeseries containing NaN values will be ignored, i.e. a detection time series of all zeros will be returned. If False, an error will be raised.

    >> Returns:
        - Abrupt shift score time series, shape (n,)
    """

    n_tot = len(values_1d)
    detection_ts = np.zeros_like(values_1d)

    # default to have at least three gradients (needed for grad distribution)
    if lmax is None:
        lmax = int(n_tot / 3)

    assert lmin < lmax, "lmin must be smaller than lmax"

    if not ignore_nan_warnings:
        assert not np.isnan(values_1d).any(), (
            "Input time series contains NaN values. Please remove them before running the detector."
        )
    else:
        # return zeros if timeseries contains nan values
        if np.isnan(values_1d).any():
            return detection_ts
        
    # chosen overlap for different segmentation methods    
    if segmentation == "fast_correction":
        overlap = 0.347                     # <- optimal value found analytically
    elif segmentation == "full_correction":
        overlap = 1.0                       # <- perfect sliding window
    elif segmentation == "fine_correction":
        overlap = 0
    elif segmentation == "centered":
        overlap = 0
    else:
        raise ValueError(
            f"Segmentation method '{segmentation}' not recognized. Choose another method."
        )
    

    # allocate space for normalization counter
    counter = np.zeros_like(values_1d, dtype=np.int32)

    for length in range(lmin, lmax + 1):
        # derive step size for sliding window from overlap
        if overlap == 1:
            sliding_step = 1
        else:
            sliding_step = length - int(overlap * length)

        detection_ts, counter = update_detection_ts(
            detection_ts,
            counter,
            values_1d,
            times_1d,
            length,
            sliding_step,
            segmentation,
        )

    # normalize detection time series by number of segment lengths used
    detection_ts /= counter

    return detection_ts

@njit
def update_detection_ts(
    detection_ts: NDArray[np.float64],
    counter: NDArray[np.int32],
    values_1d: NDArray[np.float64],
    times_1d: NDArray[np.float64],
    segment_length: int,
    sliding_step: int,
    segmentation: str,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Update the detection time series for a given segment length.

    Args:
        detection_ts: Current detection time series
        counter: Current normalization counter
        values_1d: 1D array of values (e.g., temperature, pressure, etc.)
        times_1d: 1D array of time points corresponding to the values
        segment_length: Length of the segments to divide the time series into
        sliding_step: Step size for sliding window
    Returns:
        Updated detection time series and normalization counter
    """

    n_tot = len(values_1d)
    n_segs = (n_tot - segment_length) // sliding_step + 1   # number of segments

    if segmentation == "fine_correction":
        gradients, ix0_arr = get_gradients_double(
            n_tot,
            n_segs,
            values_1d,
            times_1d,
            segment_length,
        )
    else:
        gradients, ix0_arr = get_gradients(
            n_tot,
            n_segs,
            values_1d,
            times_1d,
            segment_length,
            sliding_step,
        )

    # Note: numba-compatible versions of median absolute deviation (mad) and median
    grad_MAD = mad(gradients)       # median absolute deviation of the gradients
    grad_MEAN = median(gradients)   # median of the gradients

    # for each segment, check whether its gradient is larger than the
    # threshold. if yes, update the detection time series accordingly.
    # i1/i2 are the first/last index of a segment
    # - Create a mask for segments that exceed the threshold
    detection_mask = (
        np.abs(gradients - grad_MEAN) > 3 * grad_MAD
    )  # -> boolean mask; whether the gradient is significant
    sign_mask = np.sign(
        gradients - grad_MEAN
    )  # -> sign of the gradient (positive or negative)

    # - Update detection time series
    for i, shift_detected in enumerate(detection_mask):
        i1 = i % n_segs * sliding_step + ix0_arr[i]
        i2 = i1 + segment_length
        counter[i1:i2] += 1  # update counter
        if shift_detected:
            detection_ts[i1:i2] += sign_mask[i]

    return detection_ts, counter

@njit
def get_gradients(
    n_tot: int,
    n_segs: int,
    values_1d: NDArray[np.float64],
    times_1d: NDArray[np.float64],
    segment_length: int,
    sliding_step: int,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """
    Compute the gradients for each segment in the time series.

    Uses centered segmentation just like [Boulton+Lenton2019]_ with the addition of some
    overlap of the segments to reduce bias.

    Args:
        n_tot: Total number of time points
        n_segs: Number of segments
        values_1d: 1D array of values (e.g., temperature, pressure, etc.)
        times_1d: 1D array of time points corresponding to the values.
        segment_length: Length of the segments to divide the time series into.
        sliding_step: Step size for sliding window.
    Returns:
        gradients: 1D array of gradients for each segment
        ix0_arr: 1D array of starting indices for each segment
    """

    res = n_tot - segment_length - sliding_step * (n_segs - 1)      # number of residual values
    ix0 = res // 2                                          # starting index for sliding window
    ix0_arr = np.full(n_segs, ix0)

    gradients = compute_gradients(
        ix0,
        values_1d,
        times_1d,
        segment_length,
        sliding_step,
    )

    return gradients, ix0_arr

@njit
def get_gradients_double(
    n_tot: int,
    n_segs: int,
    values_1d: NDArray[np.float64],
    times_1d: NDArray[np.float64],
    segment_length: int,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """
    Compute the gradients by segmenting the time series in both forward and
    backward direction, to reduce bias.

    Args:
        n_tot: Total number of time points
        n_segs: Number of segments
        values_1d: 1D array of values (e.g., temperature, pressure, etc.)
        times_1d: 1D array of time points corresponding to the values
        segment_length: Length of the segments to divide the time series into.
        sliding_step: Step size for sliding window.
    Returns:
        gradients: 1D array of gradients for each segment
        ix0_arr: 1D array of starting indices for each segment
    """
    
    # forward iteration:
    ix0_fwd = 0  # starting index for sliding window
    gradients_fwd = compute_gradients(
        ix0_fwd,
        values_1d,
        times_1d,
        segment_length,
        sliding_step = segment_length,          # as there is no overlap
    )
    # backward iteration:
    ix0_bwd = n_tot - segment_length - segment_length * (n_segs - 1)      # starting index for sliding window
    gradients_bwd = compute_gradients(
        ix0_bwd,
        values_1d,
        times_1d,
        segment_length,
        sliding_step = segment_length,          # as there is no overlap
    )
    # append both gradients
    gradients = np.concatenate((gradients_fwd, gradients_bwd))
    ix0_arr = np.concatenate((
        np.full(len(gradients_fwd), ix0_fwd),
        np.full(len(gradients_bwd), ix0_bwd)
    ))

    return gradients, ix0_arr


@njit
def compute_gradients(
    ix0: int,
    values_1d: NDArray[np.float64],
    times_1d: NDArray[np.float64],
    segment_length: int,
    sliding_step: int,
) -> NDArray[np.float64]:
    """
    Slides the window of length `length` over the time series and computes the gradients
    for each position.

    Using a linear fit (1st degree polynomial).
    The function returns an array of gradients, one for each window position.

    Args:
        ix0: Starting index for the sliding window (usually 0)
        values_1d: 1D array of values (e.g., temperature, pressure, etc.)
        times_1d: 1D array of time points corresponding to the values
        length: Length of the sliding window
        sliding_step: Step size for sliding the window.

    Returns:
        gradients: 1D array of gradients for each segment
    """
    n_segs = (len(values_1d) - segment_length) // sliding_step + 1
    gradients = np.empty(n_segs)

    for i in range(n_segs):
        i1 = i * sliding_step + ix0
        i2 = i1 + segment_length
        tseg = times_1d[i1:i2]
        aseg = values_1d[i1:i2]

        gradients[i] = polyfit(tseg, aseg, 1)[0]

    return gradients


@njit
def polyfit(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    deg: int,
) -> NDArray[np.float64]:
    """Least squares polynomial fit.

    This function is a stripped-down version of numpy.polyfit, to make it
    compatible with numba. It computes the least squares polynomial fit for
    the given data points (x, y) of degree deg. The function returns the
    coefficients of the polynomial.

    Args:
        x: 1D array of x-coordinates (independent variable)
        y: 1D or 2D array of y-coordinates (dependent variable)
        deg: Degree of the polynomial to fit (0 <= deg <= 1 for linear fit, 2 for quadratic, etc.)

    Returns:
        Array of polynomial coefficients

    Raises:
        ValueError: If deg < 0
        TypeError: If x is not 1D, x is empty, y is not 1D/2D, or x and y have different lengths
    """

    order = int(deg) + 1
    x = np.asarray(x) + 0.0
    y = np.asarray(y) + 0.0

    # check arguments.
    if deg < 0:
        raise ValueError("expected deg >= 0")
    if x.ndim != 1:
        raise TypeError("expected 1D vector for x")
    if x.size == 0:
        raise TypeError("expected non-empty vector for x")
    if y.ndim < 1 or y.ndim > 2:
        raise TypeError("expected 1D or 2D array for y")
    if x.shape[0] != y.shape[0]:
        raise TypeError("expected x and y to have same length")

    # set rcond, machine precision
    rcond = len(x) * np.finfo(x.dtype).eps

    # set up least squares equation for powers of x
    lhs = np.vander(x, order)
    rhs = y

    # scale lhs to improve condition number and solve
    scale = np.sqrt((lhs * lhs).sum(axis=0))
    lhs /= scale
    coefficients = lstsq(lhs, rhs, rcond)[0]    # type: ignore
    coefficients = (coefficients.T / scale).T   # type: ignore # broadcast scale coefficients

    return coefficients                         # type: ignore


@njit
def mad(
    x: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Numba-compatible median-absolute-deviation function.

    Computes the median absolute deviation of the input array x.

    Args:
        x: 1D array of values (e.g., gradients)

    Returns:
        The median absolute deviation of the input array x
    """
    med = median(x)
    abs_dev = np.abs(x - med)
    return median(abs_dev)


@njit
def median(
    x: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Numba-compatible median function.

    Computes the median of the input array x.

    Args:
        x: 1D array of values (e.g., gradients)

    Returns:
        The median of the input array x
    """

    x_sorted = np.sort(x.copy())
    n = len(x_sorted)
    if n % 2 == 0:
        return np.array(0.5 * (x_sorted[n // 2 - 1] + x_sorted[n // 2]))
    else:
        return np.array(x_sorted[n // 2])
