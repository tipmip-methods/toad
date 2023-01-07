import logging
from typing import Union
import xarray as xr
from xarray.core import dataset

from .tsanalysis import asdetect
from .clustering import dbscan

from _version import __version__

# Each new abrupt shift detection method needs to register the function which
# maps the analysis to xr.DataArray 
_detection_methods = {
    'asdetect': asdetect.detect
} 

# Each new clustering detection procedure needs to register the function which
# maps the analysis to xr.DataArray 
_clustering_methods = {
    'dbscan': dbscan.cluster
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
    data_array_dts = detector(
        data=data_array, 
        temporal_dim=temporal_dim,
        **method_kwargs
    )

    # Save gitversion to dataset
    data_array_dts.attrs[f'{var}_git_detect'] = __version__

    # If True, dataset_with_as is merged into data. Else, only return dataarray
    # with its dts together as one dataset.
    if keep_other_vars:
        assert type(data) == xr.Dataset, \
                'Using keep_other_vars requires type(data) == xr.DataSet!'
        logging.info(f'merging new variable {var}_dts into dataset')
        dataset_with_as = xr.merge(
            [data, data_array_dts])
    else:
        logging.info(f'merging {var} and {var}_dts')
        dataset_with_as = xr.merge(
            [data_array , data_array_dts])
        dataset_with_as.attrs = []

    return dataset_with_as

def cluster(
        data: xr.Dataset,
        var : str,
        method : str,
        method_kwargs = {}
    ) -> xr.Dataset:
    """
    """
    assert type(data) == xr.Dataset, 'data must be an xr.DataSet!'
    assert data.get(var).ndim == 3, 'data must be 3-dimensional!'
    assert f'{var}_dts' in list(data.data_vars.keys()), \
                                f'data lacks detection time series {var}_dts'

    logging.info(f'looking up clusterer {method}')
    clusterer = _clustering_methods[method]

    logging.info(f'applying clusterer {method} to data')
    dataset_with_clusterlabels = clusterer(
        data=data, 
        var=var,
        **method_kwargs
    )

    # Save gitversion to dataset
    dataset_with_clusterlabels.attrs[f'{var}_git_cluster'] = __version__

    return dataset_with_clusterlabels