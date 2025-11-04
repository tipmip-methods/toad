"""
Utility functions for shift selection. Used primarily in clustering.
"""

from typing import Literal

import numpy as np
import xarray as xr
from numba import njit


@njit(cache=True, fastmath=True)
def _peaks_local_for_ts(ts: np.ndarray, thr: float, eps: float = 1e-12):
    """Finds local peaks in segments of a time series where values exceed a threshold.

    For each segment where absolute values exceed the threshold, identifies the maximum
    absolute value peak. For plateaus (consecutive equal maximum values), selects the
    middle point as the peak. NaN values break segments.

    Args:
        ts: 1D numpy array containing the time series data.
        thr: Threshold value that peaks must exceed in absolute value.
        eps: Small value for floating point comparisons. Defaults to 1e-12.

    Returns:
        tuple:
            - idxs (np.ndarray): Array of indices where peaks were found.
            - sgns (np.ndarray): Array of signs (-1 for negative peaks, +1 for positive peaks)
              corresponding to each index.

    Note:
        This is a numba-optimized implementation that uses @njit for performance.
    """
    n = ts.size
    idxs = np.empty(n, dtype=np.int64)  # over-alloc; trimmed later
    sgns = np.empty(n, dtype=np.int8)
    k = 0
    i = 0

    while i < n:
        # Skip below-threshold or NaN
        while i < n:
            v = ts[i]
            if not np.isnan(v) and (abs(v) > thr):
                break
            i += 1
        if i >= n:
            break

        # Start of segment
        max_abs = abs(ts[i])
        plat_start = i
        plat_end = i
        i += 1

        # Walk segment
        while i < n:
            v = ts[i]
            if np.isnan(v):
                break
            av = abs(v)
            if not (av > thr):
                break

            if av > max_abs + eps:
                max_abs = av
                plat_start = i
                plat_end = i
            elif abs(av - max_abs) <= eps:
                plat_end = i
            i += 1

        # Middle of the segment's max plateau
        max_idx = plat_start + (plat_end - plat_start) // 2
        # Verify peak exceeds threshold (safety check, should always be true)
        if max_abs > thr:
            idxs[k] = max_idx
            sgns[k] = np.int8(-1 if np.signbit(ts[max_idx]) else 1)
            k += 1

    return idxs[:k], sgns[:k]


@njit(cache=True, fastmath=True)
def _peak_global_for_ts(ts: np.ndarray, thr: float, eps: float = 1e-12):
    """Finds the global peak in a time series using middle-of-plateau tie rule.

    Performs a single pass through the time series to find the global maximum absolute value
    peak that exceeds the threshold. For plateaus (consecutive equal maximum values), the
    middle point is selected as the peak. NaN values break plateaus.

    Args:
        ts: 1D numpy array containing the time series data
        thr: Threshold value that peaks must exceed in absolute value
        eps: Small value for floating point comparisons. Defaults to 1e-12.

    Returns:
        tuple:
            - idx (np.int64): Index of the peak, or -1 if no peak passes threshold
            - sgn (np.int8): Sign of the peak (-1 for negative, +1 for positive, 0 if no peak)

    Note:
        This is a numba-optimized implementation that uses @njit for performance.
    """
    n = ts.size
    max_abs = -1.0
    have_max = False
    plat_start = 0
    plat_end = -1
    in_equal_run = False  # are we currently extending a contiguous max plateau?

    for i in range(n):
        v = ts[i]
        if np.isnan(v):
            in_equal_run = False
            continue
        av = abs(v)

        if av > max_abs + eps:
            max_abs = av
            have_max = True
            plat_start = i
            plat_end = i
            in_equal_run = True
        elif have_max and abs(av - max_abs) <= eps:
            # extend only if contiguous with current max plateau
            if in_equal_run:
                plat_end = i
            in_equal_run = True
        else:
            in_equal_run = False

    if (not have_max) or (max_abs <= thr):
        return np.int64(-1), np.int8(0)

    mid = plat_start + (plat_end - plat_start) // 2
    return np.int64(mid), np.int8(-1 if np.signbit(ts[mid]) else 1)


@njit(cache=True, fastmath=True)
def _compute_local_mask_TP(dts_TP: np.ndarray, thr: float, out_TP: np.ndarray):
    """Computes local peak mask for time series data.

    For each time series in dts_TP, identifies local peaks within segments where values exceed the threshold.
    Peaks are marked in out_TP as -1 for negative peaks and +1 for positive peaks. For plateaus (consecutive
    equal maximum values), only the middle point is marked as a peak.

    Args:
        dts_TP: Input array of shape (T, P) containing P time series of length T.
        thr: Threshold value that peaks must exceed in absolute value.
        out_TP: Output array of shape (T, P) that will be modified in-place.
            Values will be in {-1, 0, +1} indicating peak signs.

    Note:
        This is a numba-optimized implementation that modifies out_TP in-place.
        The @njit decorator compiles this function to machine code.
    """
    T, P = dts_TP.shape
    for p in range(P):
        ts = dts_TP[:, p]
        idxs, sgns = _peaks_local_for_ts(ts, thr)
        for m in range(idxs.size):
            out_TP[idxs[m], p] = sgns[m]


@njit(cache=True, fastmath=True)
def _compute_global_mask_TP(dts_TP: np.ndarray, thr: float, out_TP: np.ndarray):
    """Computes global peak mask for time series data.

    For each time series in dts_TP, finds the global peak and marks it in out_TP.
    A peak is marked with -1 for negative peaks or +1 for positive peaks that exceed
    the threshold. Only the middle point of the maximum plateau is marked.

    Args:
        dts_TP: Input array of shape (T, P) containing P time series of length T.
        thr: Threshold value that peaks must exceed in absolute value.
        out_TP: Output array of shape (T, P) that will be modified in-place.
            Values will be in {-1, 0, +1} indicating peak signs.

    Note:
        This is a numba-optimized implementation that modifies out_TP in-place.
        The @njit decorator compiles this function to machine code.
    """
    T, P = dts_TP.shape
    for p in range(P):
        ts = dts_TP[:, p]
        idx, sgn = _peak_global_for_ts(ts, thr)
        if idx >= 0:
            out_TP[idx, p] = sgn


def _compute_dts_peak_sign_mask(
    shifts: xr.DataArray,
    time_dim: str,
    shift_threshold: float = 0.8,
    shift_selection: Literal["local", "global"] = "local",
) -> xr.DataArray:
    """Computes a dense mask indicating peak signs in the shifts data.

    Creates an int8 mask with values in {-1, 0, +1} marking peaks in the shifts data.
    For local selection, marks the middle of max-|value| plateaus within each |shifts|>threshold segment.
    For global selection, marks the middle of the global max-|value| plateau (only if max > threshold).
    NaN values break segments/plateaus.

    Args:
        shifts: Input DataArray containing the shifts data.
        time_dim: Name of the time dimension.
        shift_threshold: Threshold value for detecting peaks. Defaults to 0.8.
        shift_selection: Selection method, either "local" or "global". Defaults to "local".

    Returns:
        xr.DataArray: Mask array with same dimensions as input, containing values:
            -1: Negative peak
            0: No peak
            +1: Positive peak

    Raises:
        ValueError: If shift_selection is not "local" or "global".

    Notes:
        - Works with float32 or float64 input without dtype casting
        - Optimized to avoid unnecessary transposes and extra passes
    """
    if shift_selection not in ("local", "global"):
        raise ValueError('shift_selection must be "local" or "global"')

    # Put time first (view), then flatten space -> (T, P) WITHOUT transposing to (P, T)
    space_dims = tuple(d for d in shifts.dims if d != time_dim)
    da_t_first = shifts.transpose(time_dim, *space_dims)

    # Use .data to avoid an eager copy if it's already a NumPy array; no dtype cast here
    vals = np.asarray(da_t_first.data)  # shape: (T, *space_shape), float32 or float64
    T = vals.shape[0]
    space_shape = vals.shape[1:]
    P = int(np.prod(space_shape)) if space_shape else 1

    dts_TP = vals.reshape(T, P)  # view if possible, minimal overhead
    out_TP = np.zeros((T, P), dtype=np.int8)  # dense mask (touches once)

    if shift_selection == "local":
        _compute_local_mask_TP(dts_TP, float(shift_threshold), out_TP)
    else:
        _compute_global_mask_TP(dts_TP, float(shift_threshold), out_TP)

    # Back to xarray with original dims
    out = out_TP.reshape((T, *space_shape))
    out_da_t_first = xr.DataArray(
        out,
        coords=da_t_first.coords,
        dims=(time_dim, *space_dims),
        name=shifts.name,
    )
    return out_da_t_first.transpose(*shifts.dims)
