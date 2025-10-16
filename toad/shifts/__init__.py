"""
Shifts module for TOAD.

This module provides functionality for detecting abrupt shifts in climate data. The main function
`compute_shifts` takes a dataset and a method for detecting abrupt shifts and returns the detected
shifts as an xarray object.

Currently implemented methods:
- ASDETECT: Implementation of the [Boulton+Lenton2019]_
"""

import logging
from typing import Union

import numpy as np
import xarray as xr

from toad._version import __version__
from toad.utils import _attrs, get_unique_variable_name

from .methods.asdetect import ASDETECT
from .methods.base import ShiftsMethod

# Currently implemented methods:
# - ASDETECT: Implementation of the [Boulton+Lenton2019]_ algorithm for detecting abrupt shifts

# Expose all methods here
__all__ = ["ASDETECT", "compute_shifts", "ShiftsMethod"]

logger = logging.getLogger("TOAD")


def compute_shifts(
    data: xr.Dataset,
    var: str,
    method: ShiftsMethod,
    time_dim: str = "time",
    output_label_suffix: str = "",
    overwrite: bool = False,
    merge_input: bool = True,
) -> Union[xr.Dataset, xr.DataArray]:
    """Apply an abrupt shift detection algorithm to a dataset along the specified temporal dimension.

    Args:
        data: Dataset containing the variable to analyze
        var: Name of the variable in the dataset to analyze for abrupt shifts
        method: The abrupt shift detection algorithm to use. Choose from predefined method objects
            in toad.shifts or create your own following the base class in toad.shifts.methods.base
        time_dim: Name of the dimension along which the time-series analysis is performed.
            Defaults to "time".
        output_label_suffix: A suffix to add to the output label. Defaults to "".
        overwrite: Whether to overwrite existing variable. Defaults to False.
        merge_input: Whether to merge results into input dataset (True) or return separately (False)

    Returns:
        Union[xr.Dataset, xr.DataArray]: If merge_input is True, returns an xarray.Dataset containing
            the original data and the detected shifts. If merge_input is False, returns an
            xarray.DataArray containing the detected shifts.

    Raises:
        ValueError: If data is invalid or required parameters are missing
    """

    """
    Overview of the shifts detection process:
    1. Input validation
    2. Apply the detector
    3. Rename the output variable
    4. Save details as attributes
    5. Merge the detected shifts with the original data or return the detected shifts

    The input data is not modified.
    """

    # check if var is in data
    data_array = data.get(var)
    if data_array is None:
        raise ValueError(f"variable {var} not found in dataset!")

    # check that data_array is not empty
    if data_array.size == 0:
        raise ValueError(f"data array for variable {var} is empty!")

    # check that data_array is 3-dimensional
    if data_array.ndim != 3:
        raise ValueError("data must be 3-dimensional")

    # check that time dim consists of ints or floats
    if not (
        np.issubdtype(data_array[time_dim].dtype, np.integer)
        or np.issubdtype(data_array[time_dim].dtype, np.floating)
    ):
        raise ValueError("time dimension must consist of integers or floats.")

    # Check if the output_label is already in the data
    output_label = f"{var}_dts{output_label_suffix}"
    if merge_input and not overwrite:
        output_label = get_unique_variable_name(output_label, data, logger)
    elif overwrite and output_label in data:
        data = data.drop_vars(output_label)

    # Apply the detector
    logger.debug(f"Applying detector {method.__class__.__name__} to {var}")

    # Create a mask for non-constant, non-NaN cells
    # Create separate masks for constant values and NaN values
    constant_mask = data_array.min(dim=time_dim) == data_array.max(dim=time_dim)
    nan_mask = data_array.isnull().all(dim=time_dim)

    # Combine masks to get valid cells (not constant and not all NaN)
    valid_mask = ~(constant_mask | nan_mask)

    # Initialize output array with NaN values
    shifts = xr.full_like(data_array, fill_value=np.nan)

    # Apply detector only to valid cells
    valid_shifts = xr.apply_ufunc(
        method.fit_predict,
        data_array.where(valid_mask),
        kwargs=dict(times_1d=data_array[time_dim].values),
        input_core_dims=[[time_dim]],
        output_core_dims=[[time_dim]],
        vectorize=True,
    ).transpose(*data_array.dims)

    # Update only the valid cells in the output array
    shifts = shifts.where(~valid_mask, valid_shifts)

    # Rename the output variable
    shifts = shifts.rename(output_label)

    # Save method params
    method_params = {
        f"method_{param}": str(value)
        for param, value in dict(sorted(vars(method).items())).items()
        if value is not None and not param.startswith("_")
    }

    # Save details as attributes
    shifts.attrs.update(
        {
            _attrs.TIME_DIM: time_dim,
            _attrs.METHOD_NAME: method.__class__.__name__,
            _attrs.TOAD_VERSION: __version__,
            _attrs.BASE_VARIABLE: var,
            _attrs.VARIABLE_TYPE: _attrs.TYPE_SHIFT,
            **method_params,
        }
    )

    n_constants = int((constant_mask).sum().values)
    n_nans = int((nan_mask).sum().values)
    n_true = int(valid_mask.sum().values)
    left_behind_ratio = float(100 * (1 - n_true / valid_mask.size))
    min_shift = float(shifts.min().values)
    mean_shift = float(shifts.mean().values)
    max_shift = float(shifts.max().values)

    logger.info(
        f"New shifts variable \033[1m{output_label}\033[0m: min/mean/max={min_shift:.3f}/{mean_shift:.3f}/{max_shift:.3f} using {n_true} grid cells. "
        f"Left behind {left_behind_ratio:.1f}% grid cells: {int(n_nans)} NaN, {int(n_constants)} constant."
    )

    # 6. Merge the detected shifts with the original data
    if merge_input:
        return xr.merge(
            [data, shifts], combine_attrs="override", compat="override"
        )  # xr.dataset
    else:
        return shifts  # xr.dataarray
