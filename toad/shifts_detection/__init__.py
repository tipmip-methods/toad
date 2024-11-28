import logging
from typing import Union, Optional
import xarray as xr
from _version import __version__
import numpy as np

from toad.shifts_detection.methods.base import ShiftsMethod

logger = logging.getLogger("TOAD")

def compute_shifts(
        data: xr.Dataset,
        var: str,
        method: ShiftsMethod,
        time_dim: str = "time",
        output_label: Optional[str] = None,
        overwrite: bool = False,
        merge_input: bool = True,
    ) -> Union[xr.Dataset, xr.DataArray]:
    """Implementation of shift detection logic.
    
    Internal function called by TOAD.compute_shifts(). 
    See that method's documentation for usage details.
    
    Additional params:
        merge_input: Whether to merge results into input dataset (True) or return separately (False)
    """
    
    # 1. Set output label
    default_name = f'{var}_dts'
    output_label = output_label or default_name
    if output_label in data and merge_input:
        if overwrite:
            logger.warning(f'Overwriting variable {output_label}')
            data = data.drop_vars(output_label)
        else:
            logger.warning(f'{output_label} already exists. Please pass overwrite=True to overwrite it.')
            return data

    # 2. Get var from data
    logger.info(f'extracting variable {var} from Dataset')
    data_array = data.get(var) 

    # check if var is in data
    if data_array is None:
        raise ValueError(f'variable {var} not found in dataset!')

    # check that data_array is not empty
    if data_array.size == 0:
        raise ValueError(f'data array for variable {var} is empty!')

    # check that data_array is 3-dimensional
    if data_array.ndim != 3:
        raise ValueError('data must be 3-dimensional!')
    # check that time dim consists of ints or floats
    if not (np.issubdtype(data_array[time_dim].dtype, np.integer) or np.issubdtype(data_array[time_dim].dtype, np.floating)):
        raise ValueError('time dimension must consist of integers or floats.')

    # Save method params (to be consistent with clustering structure.)
    method_params = {
        f'method_{param}': str(value) 
        for param, value in dict(sorted(vars(method).items())).items() 
        if value is not None
    }

    # 3. Apply the detector
    logger.info(f'applying detector {method} to data')
    shifts = method.fit_predict(dataarray=data_array, time_dim=time_dim)
    
    # 4. Rename the output variable
    shifts = shifts.rename(output_label)

    # 5. Save details as attributes
    shifts.attrs.update({
        'time_dim': time_dim,
        'method': method.__class__.__name__,
    })

    # Add method params as separate attributes
    for param, value in dict(sorted(vars(method).items())).items():
        if value is not None:
            shifts.attrs[f'method_{param}'] = str(value)


    # Add saved params as attributes
    shifts.attrs.update(method_params)

    # add git version
    shifts.attrs['toad_version'] = __version__

    # 6. Merge the detected shifts with the original data
    if merge_input:
        return xr.merge([data, shifts], combine_attrs="override") # xr.dataset
    else:
        return shifts # xr.dataarray
