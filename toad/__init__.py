import logging
from typing import Union
import xarray as xr

from .tsanalysis import asdetect

try:
    with open('_VERSION', 'r') as verfile:
        _gitversion = verfile.read()
except:
    _gitversion = 'not_specified'

# Each new abrupt shift detection method needs to register the function which
# maps the analysis to xr.DataArray 
_detection_methods = {
    'asdetect': asdetect.detect
} 

def detect(
        data: Union[xr.Dataset, xr.DataArray],
        temporal_dim: str,
        method: str,
        var: str = None,
        keep_other_vars : bool = False, 
        method_kwargs={}
    ) -> xr.Dataset :
    """Map an abrupt shift detection algorithm to the dataset in the temporal
    dimension.

    Parameters
    ----------
    data : xr.Dataset or xr.DataArray
        Data with two spatial and one temporal dimension. If `data` is an
        xr.Dataset, `var` needs to be provided.
    temporal_dim : str
        Specifies the dimension along which the one-dimensional time-series
        analysis for abrupt shifts is executed. Usually the time axis but could
        also be the forcing.
    method : {'asdetect'} 
        One-dimensional time-series analysis algorithm to use.
    var : str, optional
        Must be used in combination with `data` being an xr.Dataset. Since the
        algorithms work on xr.DataArrays, it is needed to specify here which
        variable to extract from the xr.Dataset.
    keep_other_vars : bool, optional
        Can be provided if `data` is an xr.Dataset. If True, the resulting
        xr.DataArray is appended to the xr.Dataset. Defaults to False, such that
        the xr.Dataset variables which are not analysed (i.e. all others than
        `var`) are discarded from the resulting xr.Dataset.
    method_kwargs : dict, optional
        Kwargs that need to be specifically passed to the analysing algorithm.

    Returns
    -------
    dataset_with_as : xr.Dataset
        Dataset with (at least) these variables of same dimensions and lengths: 
            * `var` : original variable data, 
            * `as_var` : Nonzero values denote an AS with the value
              corresponding to its magnitude,
        The attributes are
            * `as_detection_method` : details on the used as detection method
        If `keep_other_vars` is True, then these results are complemented by the
        unprocessed variables and attributes of the original `data`.

    See also
    --------
    toad.tsanalysis : Collection of abrupt shift detection algorithms 
    toad.clustering: Clustering algorithms using the results of the detection

    """
    logging.info(f'looking up detector {method}')
    detector = _detection_methods[method]

    if var:
        assert type(data) == xr.Dataset, \
                 'Using var requires type(data) == xr.DataSet!'
        logging.info(f'extracting variable {var} from Dataset')
        data_array = data.get(var) 
    else:
        data_array = data

    assert data_array.ndim == 3, 'data must be 3-dimensional!'

    # Application of a detector results in an xr.DataArray with 
    # coords = (<temporal_dim>, 2 spatial dimensions)
    # variables
    #   var (<temporal_dim>, SD1, SD2)
    #   as_var (<temporal_dim>, SD1, SD2)
    #   as_types_var (<temporal_dim>, SD1, SD2)
    logging.info(f'applying detector {method} to data')
    dataset_with_as = detector(
        data=data_array, 
        temporal_dim=temporal_dim,
        **method_kwargs
    )

    # Save gitversion to dataset
    dataset_with_as.attrs[f'git_detect_{var}'] = _gitversion

    # If True, dataset_with_as is merged into data.
    if keep_other_vars:
        assert type(data) == xr.Dataset, \
                'Using keep_other_vars requires type(data) == xr.DataSet!'
        logging.info(f'merging new variables as_{var} and as_types_{var}')
        dataset_with_as = xr.merge(
            [data , dataset_with_as], combine_attrs='no_conflicts')
        pass

    return dataset_with_as

# TODO: implement
def cluster():
    print('clustering')
    ...