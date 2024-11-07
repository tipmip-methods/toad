
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
    
    # Get the shifts method
    if callable(method):
        detector = method
    elif type(method) == str:
        logging.info(f'looking up detector {method}')
        detector = detection_methods[method]
    else:
        raise ValueError('method must be a string or a callable') 

    logging.info(f'extracting variable {var} from Dataset')
    data_array = data.get(var) 

    assert data_array.ndim == 3, 'data must be 3-dimensional!'

    # Application of a detector results in an xr.DataArray with 
    # coords = (<temporal_dim>, 2 spatial dimensions)
    # variables
    #   var (<temporal_dim>, SD1, SD2)
    #   as_var (<temporal_dim>, SD1, SD2)
    #   as_types_var (<temporal_dim>, SD1, SD2)
    logging.info(f'applying detector {method} to data')
    data_array_dts = detector(
        data=data_array, 
        temporal_dim=temporal_dim,
        **method_kwargs
    )

    # Save gitversion to dataset
    data_array_dts.attrs[f'{var}_git_detect'] = __version__
    
    return data_array_dts


@deprecated("detect is deprecated. Please use compute_shifts instead.")
def detect(
        data: Union[xr.Dataset, xr.DataArray],
        temporal_dim: str,
        method: str,
        var: str = None,
        keep_other_vars : bool = False, 
        **method_kwargs
    ) -> xr.Dataset :
    
    data_array = data.get(var) 
    data_array_dts = compute_shifts(data, var, temporal_dim, method, keep_other_vars, **method_kwargs)

    # If True, dataset_with_as is merged into data. Else, only return dataarray
    # with its dts together as one dataset.
    if keep_other_vars:
        # assert type(data) == xr.Dataset, 'Using keep_other_vars requires type(data) == xr.DataSet!'
        logging.info(f'merging new variable {var}_dts into dataset')
        dataset_with_as = xr.merge([data, data_array_dts])
    else:
        logging.info(f'merging {var} and {var}_dts')
        dataset_with_as = xr.merge([data_array , data_array_dts])
        dataset_with_as.attrs = []
    return dataset_with_as


