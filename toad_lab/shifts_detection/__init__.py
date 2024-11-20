
import logging
from typing import Union
import xarray as xr
from .method_dictionary import detection_methods
from _version import __version__
from ..utils import deprecated

logger = logging.getLogger("TOAD")

def compute_shifts(
        data: xr.Dataset,
        var: str,
        temporal_dim: str = "time",
        method: Union [str, callable] = "asdetect",
        output_label: str = None,
        overwrite: bool = False,
        **method_kwargs
    ) -> xr.Dataset :
    """Map an abrupt shift detection algorithm to the dataset in the temporal dimension.

    :param data:                Data with two spatial and one temporal dimension. If `data` is an xr.Dataset, `var` needs to be provided.
    :type data:                 xr.Dataset or xr.DataArray
    :param temporal_dim:        Specifies the dimension along which the one-dimensional time-series analysis for abrupt shifts is executed. Usually the time axis but could also be the forcing.
    :type temporal_dim:         str
    :param method:              Reference an in-built shifts algorithm such as 'asdetect' or pass your own function. If a custom function is provided, it should have the signature `def custom_detector(data: xr.DataArray, temporal_dim: str, **method_kwargs) -> xr.DataArray`.
    :type method:               str or callable
    :param var:                 Must be used in combination with `data` being an xr.Dataset. Since the algorithms work on xr.DataArrays, it is needed to specify here which variable to extract from the xr.Dataset.
    :type var:                  str, optional
    :param keep_other_vars:     Can be provided if `data` is an xr.Dataset. If True, the resulting xr.DataArray is appended to the xr.Dataset. Defaults to False, such that the xr.Dataset variables which are not analysed (i.e. all others than `var`) are discarded from the resulting xr.Dataset.
    :type keep_other_vars:      bool, optional
    :param method_kwargs:       Kwargs that need to be specifically passed to the analysing algorithm.
    :type method_kwargs:        dict, optional
    :return:                    Dataset with (at least) these variables of same dimensions and lengths:
                                    * `var` : original variable data,
                                    * `as_var` : Nonzero values denote an AS with the value corresponding to its magnitude,

                                The attributes are
                                    * `as_detection_method` : details on the used as detection method
                                    
                                If `keep_other_vars` is True, then these results are complemented by the unprocessed variables and attributes of the original `data`.
    :rtype:                     xr.Dataset


    **See also**

    toad.tsanalysis : Collection of abrupt shift detection algorithms 
    toad.clustering: Clustering algorithms using the results of the detection
    """
    
    # 1. Get the shifts method
    if callable(method):
        detector = method
    elif type(method) == str:
        logging.info(f'looking up detector {method}')
        detector = detection_methods[method]
    else:
        raise ValueError('method must be a string or a callable') 

    # 2. Set output label
    default_name = f'{var}_dts'
    output_label = output_label or default_name
    if output_label in data:
        if overwrite:
            logging.warning(f'overwriting variable {output_label} in data')
            data = data.drop_vars(output_label)
        else:
            raise ValueError(f'data already contains a variable named {output_label}. Please specify a different output_label or pass overwrite=True')

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
        f'{output_label}_git_version': __version__,
        f'{output_label}_shifts_method': method_details
    })
    
    return shifts
