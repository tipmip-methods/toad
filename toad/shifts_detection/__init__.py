import logging
from typing import Union
import xarray as xr
from toad._version import __version__
import numpy as np
from dask.diagnostics import ProgressBar

from toad.shifts_detection.methods.base import ShiftsMethod

logger = logging.getLogger("TOAD")


def compute_shifts(
    data: xr.Dataset,
    var: str,
    method: ShiftsMethod,
    time_dim: str = "time",
    output_label_suffix: str = "",
    overwrite: bool = False,
    merge_input: bool = True,
    chunk_size: int = None,
) -> Union[xr.Dataset, xr.DataArray]:
    """Apply an abrupt shift detection algorithm to a dataset along the specified temporal dimension.

    >> Args:
        var:
            Name of the variable in the dataset to analyze for abrupt shifts.
        method:
            The abrupt shift detection algorithm to use. Choose from predefined method objects in toad.shifts_detection.methods or create your own following the base class in toad.shifts_detection.methods.base
        time_dim:
            Name of the dimension along which the time-series analysis is performed. Defaults to "time".
        output_label_suffix:
            A suffix to add to the output label. Defaults to "".
        overwrite:
            Whether to overwrite existing variable. Defaults to False.
        merge_input:
            Whether to merge results into input dataset (True) or return separately (False)
        chunk_size:
            Size of the chunks to use for parallel processing. If None, the data will be chunked into one big chunk. Defaults to None.

    >> Returns:
        - xr.Dataset: If `merge_input` is `True`, returns an `xarray.Dataset` containing the original data and the detected shifts.
        - xr.DataArray: If `merge_input` is `False`, returns an `xarray.DataArray` containing the detected shifts.

    >> Raises:
        ValueError:
            If data is invalid or required parameters are missing
    """ 

    # set chunk size
    if chunk_size is None:
        data = data.chunk(data.sizes)           # one big chunk
    else:
        spatial_indices = list(data.dims)       # get all default dimensions/indices of the data object
        spatial_indices.remove(time_dim)        # remove time dimension from the list, only spatial dimensions should be chunked
        data = data.chunk({dim: chunk_size for dim in spatial_indices})     # chunk the spatial dimensions
            

    # 1. Set output label
    output_label = f"{var}_dts{output_label_suffix}"

    # Check if the output_label is already in the data
    if output_label in data and merge_input:
        if overwrite:
            data = data.drop_vars(output_label)
        else:
            logger.warning(
                f"{output_label} already exists. Please pass overwrite=True to overwrite it."
            )
            return data

    # 2. Get var from data
    logger.info(f"extracting variable {var} from Dataset")
    data_array = data.get(var)

    # check if var is in data
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

    # Save method params (to be consistent with clustering structure.)
    method_params = {
        f"method_{param}": str(value)
        for param, value in dict(sorted(vars(method).items())).items()
        if value is not None
    }

    # 3. Apply the detector
    # dask lazy function
    logger.info(f"applying detector {method} to data")
    shifts = xr.apply_ufunc(
        method.fit_predict,
        data_array,
        kwargs=dict(times_1d=data_array[time_dim].values),
        input_core_dims=[[time_dim]],
        output_core_dims=[[time_dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32],
    ).transpose(*data_array.dims)

    # dask.compute() to trigger the computation - use progress bar
    with ProgressBar():
        shifts = shifts.compute()

    # 4. Rename the output variable
    shifts = shifts.rename(output_label)

    # 5. Save details as attributes
    shifts.attrs.update(
        {
            "time_dim": time_dim,
            "method_name": method.__class__.__name__,
        }
    )

    # Add method params as separate attributes
    for param, value in dict(sorted(vars(method).items())).items():
        if value is not None:
            shifts.attrs[f"method_{param}"] = str(value)

    # Add saved params as attributes
    shifts.attrs.update(method_params)

    # add git version
    shifts.attrs["toad_version"] = __version__

    # 6. Merge the detected shifts with the original data
    if merge_input:
        return xr.merge([data, shifts], combine_attrs="override")  # xr.dataset
    else:
        return shifts  # xr.dataarray
