import logging
from typing import Union
import xarray as xr
from toad._version import __version__
import numpy as np
from toad.utils import get_unique_variable_name, attrs

from .methods.base import ShiftsMethod
from .methods.asdetect import ASDETECT

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

    >> Args:
        var:
            Name of the variable in the dataset to analyze for abrupt shifts.
        method:
            The abrupt shift detection algorithm to use. Choose from predefined method objects in toad.shifts or create your own following the base class in toad.shifts.methods.base
        time_dim:
            Name of the dimension along which the time-series analysis is performed. Defaults to "time".
        output_label_suffix:
            A suffix to add to the output label. Defaults to "".
        overwrite:
            Whether to overwrite existing variable. Defaults to False.
        merge_input:
            Whether to merge results into input dataset (True) or return separately (False)

    >> Returns:
        - xr.Dataset: If `merge_input` is `True`, returns an `xarray.Dataset` containing the original data and the detected shifts.
        - xr.DataArray: If `merge_input` is `False`, returns an `xarray.DataArray` containing the detected shifts.

    >> Raises:
        ValueError:
            If data is invalid or required parameters are missing
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
    logger.info(f"Applying detector {method.__class__.__name__} to {var}")
    shifts = xr.apply_ufunc(
        method.fit_predict,
        data_array,
        kwargs=dict(times_1d=data_array[time_dim].values),
        input_core_dims=[[time_dim]],
        output_core_dims=[[time_dim]],
        vectorize=True,
    ).transpose(*data_array.dims)

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
            attrs.TIME_DIM: time_dim,
            attrs.METHOD_NAME: method.__class__.__name__,
            attrs.TOAD_VERSION: __version__,
            attrs.BASE_VARIABLE: var,
            attrs.VARIABLE_TYPE: attrs.TYPE_SHIFT,
            **method_params,
        }
    )

    # 6. Merge the detected shifts with the original data
    if merge_input:
        return xr.merge(
            [data, shifts], combine_attrs="override", compat="override"
        )  # xr.dataset
    else:
        return shifts  # xr.dataarray
