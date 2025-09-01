"""
asdetect method for shifts detection.

Contains the asdetect algorithm with associated helper functions.

Crated: January 22, ? (Sina)
Refactored: Nov, 2024 (Jakob)
"""

import numpy as np
from typing import Optional
from numba import njit, jit

from numpy.linalg import lstsq

from .base import ShiftsMethod

# NOTE: remove next line!
import matplotlib.pyplot as plt

###################################################################################################
#################################### Segmentation Methods #########################################
###################################################################################################

@njit
def centered_segmentation(n_tot, length):
    """
    Old segmentation -> as comparison.
    """

    n_seg = int(n_tot / length)  # number of segments
    rest = n_tot - n_seg * length  # uncovered points
    idx0 = int(rest / 2)  # first index of the first segment

    return idx0

#@jit
def dynamic_segmentation(n_tot, length):
    """
    NOTE: Not Working properly yet...

    Dynamically choses the starting index of the first segment for each
    segment lengths, according to the one making the gap-distribution as
    uniform as possible.
    Chi^2 function is used to compare the distribution to a perfect
    uniform distribution.
    """
    
    n_seg = int(n_tot / length)                         # number of segments
    rest = n_tot - n_seg * length                       # uncovered points
    if rest != 0:
        q_arr = []
        for idx0 in range(rest + 1):                     # try all possible starting indices for the first segment
            seg_idces = idx0 + length * np.arange(n_seg + 1)
            q_arr.append(_chi2(seg_idces))                  # use chi2 to evaluate the uniformity of the segmentation

        idx0 = np.argmin(q_arr)  # choose the starting index which leads to the most uniform distribution
    else:
        idx0 = 0

    return idx0

#@jit
def _chi2(hist):
    hist = np.asarray(hist, dtype=float)
    
    # Normalize to probability distribution
    p = hist / hist.sum()
    n = len(p)
    uniform = np.ones(n) / n
    
    return np.sum((p - uniform) ** 2 / uniform)

@njit
def truncated_segmentation(n_tot, length):
    """
    Performs the whole truncation at the beginning.
    """

    n_seg = int(n_tot / length)     # number of segments
    rest = n_tot - n_seg * length   # uncovered points
    idx0 = rest                     # first index of the first segment

    return idx0

def anti_centered_segmentation(n_tot, length):
    """
    Aligns the center of the time series with the center of a segment.
    -> will lead to more truncation at the ends.
    """

    ix_center_ts = int(n_tot / 2)
    ix_center_seg = int(length / 2)
    idx_center = ix_center_ts - ix_center_seg
    idx0 = idx_center%length

    return idx0


###################################################################################################
########################################### ASDETECT ##############################################
###################################################################################################

class ASDETECT(ShiftsMethod):
    """
    Detect abrupt shifts in a time series using gradient-based analysis by [Boulton+Lenton2019]_.

    Steps:
    1. Divide the time series into overlapping segments of size `l`.
    2. Perform linear regression within each segment to calculate gradients.
    3. Identify significant gradients exceeding ±3 Median Absolute Deviations (MAD) from the median gradient.
    4. Update a detection array by adding +1 for significant positive gradients and -1 for significant negative gradients in each each segment.
    5. Iterate over multiple window sizes (`l`), updating the detection array at each step.
    6. Normalize the detection array by dividing by the number of window sizes used.

    Args:
        lmin: (Optional) The minimum segment length for detection. Defaults to 5.
        lmax: (Optional) The maximum segment length for detection. If not
            specified, it defaults to one-third of the size of the time dimension.
    """

    def __init__(self, lmin=5, lmax=None):
        self.lmin = lmin
        self.lmax = lmax

    def fit_predict(
        self,
        values_1d: np.ndarray,
        times_1d: np.ndarray,
        ufunc = centered_segmentation,
        counter_plot = False,
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
            ufunc=ufunc,
            counter_plot=counter_plot,
        )

        return shifts


# 1D time series analysis of abrupt shifts =====================================
#@jit
def construct_detection_ts(
    values_1d: np.ndarray,
    times_1d: np.ndarray,
    lmin: int = 5,
    lmax: Optional[int] = None,
    ufunc = centered_segmentation,
    counter_plot = False,
) -> np.ndarray:
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


    # NOTE: remove next line!
    counter = np.zeros_like(detection_ts)
    for length in range(lmin, lmax + 1):
        n_seg = int(n_tot / length)  # number of segments
        #rest = n_tot - n_seg * length  # uncovered points
        idx0 = ufunc(n_tot, length)
        #idx0 = int(rest / 2)  # first index of the first segment
        seg_idces = idx0 + length * np.arange(n_seg + 1)  # first index of each segment
        seg_idces = seg_idces[seg_idces <= n_tot]  # ensure indices are within bounds
        # NOTE: remove next line!
        counter[seg_idces[0]:seg_idces[-1]] += 1

        # Note: numba-compatible version of data splitting and 1st degree polyfit
        gradients = compute_gradients(values_1d, times_1d, seg_idces)

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
            if shift_detected:
                i1, i2 = seg_idces[i], seg_idces[i + 1]
                detection_ts[i1:i2] += sign_mask[i]

    # normalize the detection time series to one
    detection_ts /= lmax - lmin + 1

    if counter_plot:
        plt.figure()
        plt.plot(counter, label='# of detection_ts[ix_t] calls')
        plt.hlines(y=lmax - lmin + 1, xmin=0, xmax=len(counter), color='r', linestyle='--', label='normalization')
        plt.legend()
        plt.xlabel('Time Index ix_t')
        plt.ylabel('Counter')
        plt.title(f'Number of times each time index is included in segments\nSegmentation Method: {ufunc.__name__}')
        plt.grid()
        plt.show()

    return detection_ts


@njit
def compute_gradients(
    values_1d: np.ndarray,
    times_1d: np.ndarray,
    seg_idces: np.ndarray,
) -> np.ndarray:
    """
    Compute the gradients of the segments defined by given indices.

    Loops over the segments defined by seg_idces and computes the gradient
    of the values_1d time series using a linear fit (1st degree polynomial).
    The function returns an array of gradients, one for each segment.

    >> Args:
        values_1d:
            1D array of values (e.g., temperature, pressure, etc.)
        times_1d:
            1D array of time points corresponding to the values
        seg_idces:
            1D array of segment indices defining the segments

    >> Returns:
        gradients:
            1D array of gradients for each segment
    """
    n_segs = len(seg_idces) - 1
    gradients = np.empty(n_segs)

    for i in range(n_segs):
        i1 = seg_idces[i]
        i2 = seg_idces[i + 1]
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

