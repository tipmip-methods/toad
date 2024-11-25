
import logging
from typing import Union
import xarray as xr
from _version import __version__

from toad_lab.shifts_detection.methods.base import ShiftsMethod

logger = logging.getLogger("TOAD")

def compute_shifts(
        data: xr.Dataset,
        var: str,
        temporal_dim: str = "time",
        method: ShiftsMethod = None,
        output_label: str = None,
        overwrite: bool = False,
        merge_input = True,
    ) -> xr.Dataset :
    """
    Main abrupt shift detection coordination function. Called from the TOAD.compute_shifts method. Ref that docstring for more info.
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
    assert type(data) == xr.Dataset, 'data must be an xr.DataSet!'
    logging.info(f'extracting variable {var} from Dataset')
    data_array = data.get(var) 
    assert data_array.ndim == 3, 'data must be 3-dimensional!'

    # 3. Apply the detector
    logging.info(f'applying detector {method} to data')
    shifts, method_details = method.apply(dataarray=data_array, temporal_dim=temporal_dim)
    
    # 4. Rename the output variable
    shifts = shifts.rename(output_label)

    # 5. Save details as attributes
    shifts.attrs.update({
        'method': method_details,
        '_git_version': __version__
    })

    # 6. Merge the detected shifts with the original data
    if merge_input:
        return xr.merge([data, shifts], combine_attrs="override")
    else:
        return shifts
