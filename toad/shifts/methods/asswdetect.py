"""
asdetect method for shifts detection.

Contains the asdetect algorithm with associated helper functions.

Crated: January 22, ? (Sina)
Refactored: Nov, 2024 (Jakob)
"""

import numpy as np
from typing import Optional
from numba import njit

from numpy.linalg import lstsq

from .base import ShiftsMethod


class ASSWDETECT(ShiftsMethod):
    """
    Detect abrupt shifts in a time series using gradient-based analysis by [Boulton+Lenton2019]_. The algorithm is modified to use a
    sliding window (SW) approach.

    Steps:
    1. Given a segment of size `l`, slide it over the time series.
    2. Perform linear regression within each segment position to calculate gradients.
    3. Identify significant gradients exceeding ±3 Median Absolute Deviations (MAD) from the median gradient.
    4. Update a detection array by adding +1 for significant positive gradients and -1 for significant negative gradients in each each segment.
    5. Iterate over multiple window sizes (`l`), updating the detection array at each step.
    6. Normalize the detection array by dividing by the number of maximum calls of a value.

    Args:
        lmin: (Optional) The minimum segment length for detection. Defaults to 5.
        lmax: (Optional) The maximum segment length for detection. If not
            specified, it defaults to one-third of the size of the time dimension.
        sliding_step: (Optional) The step size for sliding the window. Defaults to 1.
    """

    def __init__(self, lmin=5, lmax=None, overlap=0):
        self.lmin = lmin
        self.lmax = lmax
        self.overlap = overlap

    def fit_predict(
        self,
        values_1d: np.ndarray,
        times_1d: np.ndarray,
        return_norm: bool = False,
        return_gap: bool = False,
        verbose: bool = False,
    ) -> np.ndarray:
        """Compute the detection time series for each grid cell in the 3D data array.

        Args:
            - values_1d: 1D array of values
            - times_1d: 1D array of times

        Returns:
            - A 1D array of the same length as `values_1d`, where each value represents
            the abrupt shift score for a grid cell at a specific time. The score ranges from -1 to 1:
                - `1` indicates that all tested segment lengths detected a significant positive gradient (i.e. exceeding 3 MAD of the median gradient),
                - `-1` indicates that all tested segment lengths detected a significant negative gradient.
                - Values between -1 and 1 indicate the proportion of segment lengths detecting a significant gradient at that time point.
        """

        shifts = construct_detection_ts(
            values_1d=values_1d,
            times_1d=times_1d,
            lmin=self.lmin,
            lmax=self.lmax,
            overlap=self.overlap,
            return_norm=return_norm,
            return_gap=return_gap,
            verbose=verbose,
        )

        return shifts


# 1D time series analysis of abrupt shifts =====================================
@njit
def construct_detection_ts(
    values_1d: np.ndarray,
    times_1d: np.ndarray,
    lmin: int = 5,
    lmax: Optional[int] = None,
    overlap: float = 0,
    return_norm: bool = False,
    return_gap: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """Construct a detection time series (asdetect algorithm).

    Enhancing [Boulton+Lenton2019]_, a window of length l is slided over the
    time series (ts), for each of which positions the gradient is computed.
    Segments with gradients > 3 MAD of the gradients distribution are marked.
    Averaging over many segmentation choices (i.e. values of l) results in a
    detection time series that indicates the points of largest relative gradients.

    >> Args:
        values_1d:
            Time series, shape (n,)
        times_1d:
            Times, shape (n,), same length as values_1d
        lmin:
            Smallest segment length, default = 5
        lmax:
            Largest segment length, default = n/3
        overlap:
            Relative overlap between segments, default = 0
            -> overlap = 0: no overlap, acts like original asdetect
            -> overlap = 0.5: 50% of the segments overlap
            -> overlap = 1: maximum meaningful overlap, i.e. sliding window with step size 1

    >> Returns:
        - Abrupt shift score time series, shape (n,)
    """

    n_tot = len(values_1d)

    detection_ts = np.zeros_like(values_1d)

    if np.isnan(values_1d).any():
        # print("you tried evaluating a ts with nan entries")
        return detection_ts

    # default to have at least three gradients (needed for grad distribution)
    if lmax is None:
        lmax = int(n_tot / 3)

    counter = np.zeros_like(values_1d)  # to count how often a value is called -> needed for normalization
    gap_counter = np.zeros_like(values_1d)  # to count how often a value is at the edge of a segment

    for length in range(lmin, lmax + 1):
        # Note: numba-compatible version of data splitting and 1st degree polyfit
        if overlap == 1:
            sliding_step = 1  # step size for sliding window
        else:
            sliding_step = length - int(length*overlap)  # step size for sliding window
        if verbose: print(f"length: {length}, sliding_step: {sliding_step}")

        detection_ts, counter, gap_counter = update_detection_ts(
            detection_ts,
            counter,
            gap_counter,
            values_1d,
            times_1d,
            length,
            sliding_step,
        )

    # normalize the detection time series to one
    detection_ts /= counter

    if return_norm:
        return counter
    if return_gap:
        return gap_counter

    return detection_ts

@njit
def update_detection_ts(
    detection_ts: np.ndarray,
    counter: np.ndarray,
    gap_counter: np.ndarray,
    values_1d: np.ndarray,
    times_1d: np.ndarray,
    length: int,
    sliding_step: int,
) -> None:
    """
    Update the detection time series based on the computed gradients.

    >> Args:
        detection_ts:
            1D array of detection time series to be updated
        counter:
            1D array to count how often a value is called (for normalization)
        length:
            Length of the sliding window
        sliding_step:
            Step size for sliding the window
        ix0:
            Starting index for the sliding window (usually 0)
    """
    n_tot = len(values_1d)

    n_segs = (n_tot - length) // sliding_step + 1
    res = n_tot - length - sliding_step * (n_segs - 1)      # starting index for sliding window
    ix0 = res // 2

    gradients = compute_gradients(ix0, values_1d, times_1d, length, sliding_step)
    
    # Note: numba-compatible versions of median absolute deviation (mad) and median
    grad_MAD = mad(gradients)  # median absolute deviation of the gradients
    grad_MEAN = median(gradients)  # median of the gradients

    # for each segment, check whether its gradient is larger than the
    # threshold. if yes, update the detection time series accordingly.
    # i1/i2 are the first/last index of a segment
    # - Create a mask for segments that exceed the threshold
    detection_mask = (
        np.abs(gradients - grad_MEAN) > 3 * grad_MAD
    )  # boolean mask; wether the gradient is significant
    sign_mask = np.sign(
        gradients - grad_MEAN
    )  # sign of the gradient (positive or negative)

    # Update detection time series
    for i, shift_detected in enumerate(detection_mask):
        i1 = i * sliding_step + ix0
        i2 = i1 + length
        counter[i1:i2] += 1  # update counter
        gap_counter[i1] += 1  # update gap counter
        gap_counter[i2-1] += 1  # update gap counter
        if shift_detected:
            detection_ts[i1:i2] += sign_mask[i]

    return detection_ts, counter, gap_counter


@njit
def compute_gradients(
    ix0: int,
    values_1d: np.ndarray,
    times_1d: np.ndarray,
    length: int,
    sliding_step: int = 1,
) -> np.ndarray:
    """
    Slides the window of length `length` over the time series and computes the gradients
    for each position.

    Using a linear fit (1st degree polynomial).
    The function returns an array of gradients, one for each window position.

    >> Args:
        ix0:
            Starting index for the sliding window (usually 0)
        values_1d:
            1D array of values (e.g., temperature, pressure, etc.)
        times_1d:
            1D array of time points corresponding to the values
        length:
            Length of the sliding window
        sliding_step:
            Step size for sliding the window

    >> Returns:
        gradients:
            1D array of gradients for each segment
    """
    n_tot = len(values_1d)
    n_segs = (n_tot - length) // sliding_step + 1
    gradients = np.empty(n_segs)

    for i in range(n_segs):
        i1 = i * sliding_step + ix0
        i2 = i1 + length
        tseg = times_1d[i1:i2]
        aseg = values_1d[i1:i2]

        gradients[i] = polyfit(tseg, aseg, 1)[0]

    return gradients


@njit
def polyfit(
    x: np.ndarray,
    y: np.ndarray,
    deg: int,
) -> np.ndarray:
    """
    Least squares polynomial fit.

    This function is a stripped-down version of numpy.polyfit, to make it
    compatible with numba. It computes the least squares polynomial fit for
    the given data points (x, y) of degree deg. The function returns the
    coefficients of the polynomial.

    >> Args:
        x:
            1D array of x-coordinates (independent variable)
        y:
            1D or 2D array of y-coordinates (dependent variable)
        deg:
            Degree of the polynomial to fit
            (0 <= deg <= 1 for linear fit, 2 for quadratic, etc.)
    >> Returns:
        coefficients:
            array of polynomial coefficients
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
    coefficients = lstsq(lhs, rhs, rcond)[0]
    coefficients = (coefficients.T / scale).T  # broadcast scale coefficients

    return coefficients


@njit
def mad(
    x: np.ndarray,
) -> float:
    """
    Numba-compatible median-absolute-deviation function.

    Computes the median absolute deviation of the input array x.

    >> Args:
        x:
            1D array of values (e.g., gradients)

    >> Returns:
        The median absolute deviation of the input array x.
    """
    med = median(x)
    abs_dev = np.abs(x - med)
    return median(abs_dev)


@njit
def median(
    x: np.ndarray,
) -> float:
    """
    Numba-compatible median function.

    Computes the median of the input array x.

    >> Args:
        x:
            1D array of values (e.g., gradients)

    >> Returns:
        The median of the input array x.
    """

    x_sorted = np.sort(x.copy())
    n = len(x_sorted)
    if n % 2 == 0:
        return np.array(0.5 * (x_sorted[n // 2 - 1] + x_sorted[n // 2]))
    else:
        return np.array(x_sorted[n // 2])
