import logging
from typing import Union
import numpy as np
import xarray as xr
from xarray.core import dataset

from .tsanalysis import asdetect
from .clustering import dbscan
from .clustering.cluster import Clustering
from .utils import infer_dims

#from _version import __version__

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

# Adapted for multivariate TOAD
def detect(
        data: Union[xr.Dataset, xr.DataArray],
        temporal_dim: str,
        method: str,
        vars: Union[str, List[str]] = None,
        keep_other_vars: bool = False, 
        method_kwargs={}
    ) -> xr.Dataset:
    """Map an abrupt shift detection algorithm to the dataset in the temporal
    dimension.

    Parameters
    ----------
    data : xr.Dataset or xr.DataArray
        Data with two spatial and one temporal dimension. If `data` is an
        xr.Dataset, `vars` needs to be provided.
    temporal_dim : str
        Specifies the dimension along which the one-dimensional time-series
        analysis for abrupt shifts is executed. Usually the time axis but could
        also be the forcing.
    method : {'asdetect'} 
        One-dimensional time-series analysis algorithm to use.
    vars : str or list of str, optional
        Must be used in combination with `data` being an xr.Dataset. Since the
        algorithms work on xr.DataArrays, it is needed to specify here which
        variable(s) to extract from the xr.Dataset. Can be a single variable
        name or a list of variable names.
    keep_other_vars : bool, optional
        Can be provided if `data` is an xr.Dataset. If True, the resulting
        xr.DataArray is appended to the xr.Dataset. Defaults to False, such that
        the xr.Dataset variables which are not analyzed (i.e. all others than
        `vars`) are discarded from the resulting xr.Dataset.
    method_kwargs : dict, optional
        Kwargs that need to be specifically passed to the analyzing algorithm.

    Returns
    -------
    dataset_with_as : xr.Dataset
        Dataset with (at least) these variables of same dimensions and lengths: 
            * `vars` : original variable data, 
            * `as_vars` : Nonzero values denote an AS with the value
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

    # Ensure vars is a list
    if isinstance(vars, str):
        vars = [vars]

    results = []
    
    # Iterate over the variables to process each one
    for var in vars:
        if var:
            assert type(data) == xr.Dataset, \
                     'Using vars requires type(data) == xr.Dataset!'
            logging.info(f'extracting variable {var} from Dataset')
            data_array = data.get(var)
        else:
            assert type(data) == xr.DataArray, 'Please provide vars or an xr.DataArray!'
            data_array = data

        assert data_array.ndim == 3, 'data must be 3-dimensional!'

        logging.info(f'applying detector {method} to data variable {var}')
        data_array_dts = detector(
            data=data_array, 
            temporal_dim=temporal_dim,
            **method_kwargs
        )

        # Rename the detection result to avoid overwriting when merging
        data_array_dts = data_array_dts.rename(f'{var}_dts')
        results.append(data_array_dts)

    # Merge results into a dataset
    dataset_with_as = xr.merge([data] + results) if keep_other_vars else xr.merge(results)
    
    # Clean up attributes if not keeping other variables
    if not keep_other_vars:
        dataset_with_as.attrs = []

    return dataset_with_as


# ORIGINAL
# def detect(
#         data: Union[xr.Dataset, xr.DataArray],
#         temporal_dim: str,
#         method: str,
#         var: str = None,
#         keep_other_vars : bool = False, 
#         method_kwargs={}
#     ) -> xr.Dataset :
#     """Map an abrupt shift detection algorithm to the dataset in the temporal
#     dimension.

#     Parameters
#     ----------
#     data : xr.Dataset or xr.DataArray
#         Data with two spatial and one temporal dimension. If `data` is an
#         xr.Dataset, `var` needs to be provided.
#     temporal_dim : str
#         Specifies the dimension along which the one-dimensional time-series
#         analysis for abrupt shifts is executed. Usually the time axis but could
#         also be the forcing.
#     method : {'asdetect'} 
#         One-dimensional time-series analysis algorithm to use.
#     var : str, optional
#         Must be used in combination with `data` being an xr.Dataset. Since the
#         algorithms work on xr.DataArrays, it is needed to specify here which
#         variable to extract from the xr.Dataset.
#     keep_other_vars : bool, optional
#         Can be provided if `data` is an xr.Dataset. If True, the resulting
#         xr.DataArray is appended to the xr.Dataset. Defaults to False, such that
#         the xr.Dataset variables which are not analysed (i.e. all others than
#         `var`) are discarded from the resulting xr.Dataset.
#     method_kwargs : dict, optional
#         Kwargs that need to be specifically passed to the analysing algorithm.

#     Returns
#     -------
#     dataset_with_as : xr.Dataset
#         Dataset with (at least) these variables of same dimensions and lengths: 
#             * `var` : original variable data, 
#             * `as_var` : Nonzero values denote an AS with the value
#               corresponding to its magnitude,
#         The attributes are
#             * `as_detection_method` : details on the used as detection method
#         If `keep_other_vars` is True, then these results are complemented by the
#         unprocessed variables and attributes of the original `data`.

#     See also
#     --------
#     toad.tsanalysis : Collection of abrupt shift detection algorithms 
#     toad.clustering: Clustering algorithms using the results of the detection

#     """
#     logging.info(f'looking up detector {method}')
#     detector = _detection_methods[method]

#     if var:
#         assert type(data) == xr.Dataset, \
#                  'Using var requires type(data) == xr.DataSet!'
#         logging.info(f'extracting variable {var} from Dataset')
#         data_array = data.get(var) 
#     else:
#         assert type(data) == xr.DataArray, 'Please provide var or an xr.DataArray!'
#         data_array = data

#     assert data_array.ndim == 3, 'data must be 3-dimensional!'

#     # Application of a detector results in an xr.DataArray with 
#     # coords = (<temporal_dim>, 2 spatial dimensions)
#     # variables
#     #   var (<temporal_dim>, SD1, SD2)
#     #   as_var (<temporal_dim>, SD1, SD2)
#     #   as_types_var (<temporal_dim>, SD1, SD2)
#     logging.info(f'applying detector {method} to data')
#     data_array_dts = detector(
#         data=data_array, 
#         temporal_dim=temporal_dim,
#         **method_kwargs
#     )

#     # Save gitversion to dataset
#     #data_array_dts.attrs[f'{var}_git_detect'] = __version__

#     # If True, dataset_with_as is merged into data. Else, only return dataarray
#     # with its dts together as one dataset.
#     if keep_other_vars:
#         assert type(data) == xr.Dataset, \
#                 'Using keep_other_vars requires type(data) == xr.DataSet!'
#         logging.info(f'merging new variable {var}_dts into dataset')
#         dataset_with_as = xr.merge(
#             [data, data_array_dts])
#     else:
#         logging.info(f'merging {var} and {var}_dts')
#         dataset_with_as = xr.merge(
#             [data_array , data_array_dts])
#         dataset_with_as.attrs = []

#     return dataset_with_as

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
    #dataset_with_clusterlabels.attrs[f'{var}_git_cluster'] = __version__

    return dataset_with_clusterlabels    

@xr.register_dataarray_accessor("toad")
class ToadAccessor:

    def __init__(self, xarr_da):
        self._da = xarr_da

    def timeseries(
                self, 
                clustering,
                cluster_lbl,
                masking = 'simple',
                how=('aggr',)  # mean, median, std, perc, per_gridcell
            ):

        da = clustering._apply_mask_to(self._da, cluster_lbl, masking=masking)
        tdim, sdims = infer_dims(self._da)

        if type(how)== str:
            how = (how,)

        if 'normalised' in how:
            if masking=='simple':
                print('Warning: normalised currently does not work with simple masking')
            initial_da =  da.isel({f'{tdim}':0})
            da = da / initial_da
            da = da.where(np.isfinite(da))
    
        if 'mean' in how:
            timeseries = da.mean(dim=sdims, skipna=True)
        elif 'median' in how:
            timeseries = da.median(dim=sdims, skipna=True)
        elif 'aggr' in how:
            timeseries = da.sum(dim=sdims, skipna=True)
        elif 'std' in how:
            timeseries = da.std(dim=sdims, skipna=True)
        elif 'perc' in how:
            # takes the (first) numeric value to be found in how 
            pval = [arg for arg in how if type(arg)==float][0]
            timeseries = da.quantile(pval, dim=sdims, skipna=True)
        elif 'per_gridcell' in how:
            timeseries = da.stack(cell_xy=sdims).transpose().dropna(dim='cell_xy', how='all')
        else:
            raise ValueError('how needs to be one of mean, median, aggr, std, perc, per_gridcell')

        return timeseries

    def compute_score(
                    self, 
                    clustering,
                    cluster_lbl, 
                    how='mean'
                ):

        tdim, _ = infer_dims(self._da)  
        xvals = self._da.__getattr__(tdim).values
        yvals = self.timeseries(clustering=clustering, cluster_lbl=cluster_lbl, masking='spatial', how=how).values
        (a,b) , res, _, _, _ = np.polyfit(xvals, yvals, 1, full=True)
        
        _score = res[0] 
        _score_fit = b + a*xvals

        return _score, _score_fit

    def score(
            self, 
            clustering,
            cluster_lbl, 
            how='mean'
        ):
        return self.compute_score(clustering, cluster_lbl, how)[0]