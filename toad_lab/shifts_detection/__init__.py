
import logging
from typing import Union
import xarray as xr
from _version import __version__

from toad_lab.method_dictionary import shifts_methods


logger = logging.getLogger("TOAD")

def compute_shifts(
        data: xr.Dataset,
        var: str,
        temporal_dim: str = "time",
        method: Union [str, callable] = "asdetect",
        output_label: str = None,
        overwrite: bool = False,
        merge_input = True,
        **method_kwargs
    ) -> xr.Dataset :
    """
    Apply an abrupt shift detection algorithm to a dataset along the specified temporal dimension.

    This function detects abrupt shifts in the specified variable of a dataset using a chosen detection 
    algorithm. It processes the variable along the temporal dimension and returns an updated dataset 
    with the detected shifts and associated metadata.

    :param data: Dataset with two spatial and one temporal dimension. Must be an `xarray.Dataset`.
    :type data: xr.Dataset
    :param var: Name of the variable in the dataset to analyze for abrupt shifts.
    :type var: str
    :param temporal_dim: Dimension along which the one-dimensional time-series analysis for abrupt shifts is executed. 
                         Typically the time axis but could also represent another forcing axis. Default is "time".
    :type temporal_dim: str, optional
    :param method: Abrupt shift detection algorithm to use. Can be a string referring to a predefined method 
                   (e.g., "asdetect") or a custom callable. Custom callables must have the signature 
                   `def custom_detector(data: xr.DataArray, temporal_dim: str, **method_kwargs) -> xr.DataArray`.
    :type method: Union[str, callable]
    :param output_label: Name of the variable in the dataset to store the shift detection results. 
                         Defaults to `{var}_dts` if not provided.
    :type output_label: str, optional
    :param overwrite: Whether to overwrite an existing variable in the dataset with the same name as `output_label`. 
                      Default is `False`.
    :type overwrite: bool
    :param merge_input: Whether to merge the detected shifts with the original data. If False, only the detected shifts 
                        are returned. Default is `True`.
    :type merge_input: bool
    :param method_kwargs: Additional keyword arguments specific to the detection algorithm.
    :type method_kwargs: dict, optional
    :return: An xarray.Dataset containing the following variables:
                - `{var}`: Original variable data.
                - `{output_label}`: Nonzero values denote detected abrupt shifts, with the values corresponding 
                  to their magnitude.
             The returned dataset also includes attributes detailing the detection method used.
    :rtype: xr.Dataset

    :raises AssertionError: If `data` is not an `xarray.Dataset`, if the dataset does not have three dimensions, 
                             or if `var` is not a valid variable in the dataset.
    :raises ValueError: If `method` is invalid, or if `output_label` conflicts with an existing variable and 
                        `overwrite` is `False`.

    Notes:
    - Predefined methods are stored in the `detection_methods` dictionary and can be referenced by name.
    - If a custom detection algorithm is used, it must adhere to the specified callable signature.
    - The detected shifts are stored in the dataset under `output_label`, with metadata describing the method used.
    """
    
    # 1. Get the shifts method
    if callable(method):
        detector = method
    elif type(method) == str:
        logging.info(f'looking up detector {method}')
        detector = shifts_methods[method]
    else:
        raise ValueError('method must be a string or a callable') 

    # 2. Set output label
    default_name = f'{var}_dts'
    output_label = output_label or default_name
    if output_label in data and merge_input:
        if overwrite:
            logging.warning(f'overwriting variable {output_label} in data')
            data = data.drop_vars(output_label)
        else:
            raise ValueError(f'data already contains a variable named {output_label}. Please specify a different output_label or set overwrite=True or set merge_input=False')

    # 3. Get var from data
    logging.info(f'extracting variable {var} from Dataset')
    data_array = data.get(var) 
    assert data_array.ndim == 3, 'data must be 3-dimensional!'

    # 4. Apply the detector
    logging.info(f'applying detector {method} to data')
    shifts, method_details = detector(
        data=data_array, 
        temporal_dim=temporal_dim,
        **method_kwargs
    )

    # 5. Rename the output variable
    shifts = shifts.rename(output_label)

    # 6. Save details as attributes
    shifts.attrs.update({
        f'method': method_details,
        f'_git_version': __version__
    })

    # 7. Merge the detected shifts with the original data
    if merge_input:
        return xr.merge([data, shifts], combine_attrs="override")
    else:
        return shifts
