import logging
from typing import Union
import numpy as np
import xarray as xr
from xarray.core import dataset

from .tsanalysis import asdetect
from .clustering import dbscan
from .clustering.cluster import Clustering
from .utils import infer_dims

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
        assert type(data) == xr.DataArray, 'Please provide var or an xr.DataArray!'
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

@xr.register_dataarray_accessor("toad")
class ToadAccessor:

    def __init__(self, xarr_da):
        self._da = xarr_da

    def _apply_clustering(self, clustering, regions):

        if clustering:
            print(regions)
            return clustering(self._da, regions)
        else:
            return self._da
        
    def timeseries(
                self, 
                clustering=None, 
                regions=None,
                how=('aggr',), 
                temporal_dim=None
            ):   

        if regions: assert clustering, 'region requires also clustering argument' 
        da = self._apply_clustering(clustering, regions)
      
        tdim, sdims = infer_dims(self._da, tdim=temporal_dim)
        if type(how)== str:
            how = (how,)

        if 'normalised' in how:
            if regions==False:
                print('Warning: normalised currently does not work with regions')
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
        
        # if 'normalised' in how:
        #     if regions==False:
        #         print('Warning: normalised currently does not work with regions')

        #     # if regions==False:
        #     #     da1 = self._apply_clustering(clustering, regions=True)
        #     #     initial_value1 = da1.
        #     #     self._da.isel({f'{tdim}':0})

        #     # # if initial_value==np.nan : print('Warning, no initial value for this time series')
        #     # else:
        #     initial_value = timeseries.isel({f'{tdim}':0})
        #     timeseries = timeseries / initial_value

        return timeseries
    
    def compute_score(self, how='mean'):

        tdim, _ = infer_dims(self._da)  
        xvals = self._da.__getattr__(tdim).values
        yvals = self.timeseries(how=how).values
        (a,b) , res, _, _, _ = np.polyfit(xvals, yvals, 1, full=True)
        
        _score = res[0] 
        _score_fit = b + a*xvals

        return _score, _score_fit

    def score(self, how='mean'):
        return self.compute_score(how=how)[0]

# attempt to use .toad for detection + clustering

   # def _infer_dims(self, tdim=None):

    #     # spatial dims are all non-temporal dims
    #     if tdim:
    #         sdims = list(self._da.dims)
    #         assert tdim in self._da.dims, f"provided temporal dim '{tdim}' is not in the dimensions of the dataset!"
    #         sdims.remove(tdim)
    #         print(f"inferring spatial dims {sdims} given temporal dim '{tdim}'")
    #         return (tdim, sdims)
    #     # check if one of the standard combinations in present and auto-infer
    #     else:
    #         for pair in [('x','y'),('lat','lon'),('latitude','longitude')]:
    #             if all(i in list(self._da.dims) for i in pair):
    #                 sdims = pair
    #                 tdim = list(self._da.dims)
    #                 for sd in sdims:
    #                     tdim.remove(sd)

    #                 print(f"auto-detecting: spatial dims {sdims}, temporal dim '{tdim[0]}'")
    #                 return (tdim[0], sdims)

    # def spatial_mask(self, clustering=None):
    #     return self._apply_clustering(clustering, regions=True)


    # def _has_dts(self, var_name=None, strict=True):
    #     """Check existence of a detection time series.

    #     Default: Check if there is one variable named "*_dts".
        
    #     var_name: Name of variable to check for. Optional, if not provided then
    #     returns True if there is one detection time series in the dataset

    #     strict: If set, then returns True iff there is exactly one dts. 

    #     """
    #     # check if detection time series exists for that variable
    #     if var_name:
    #         if (var_name+'_dts') in list(self._obj.keys()):
    #             return True 
    #     # must include exactly one detection time series
    #     elif ''.join(list(self._obj.keys())).count('dts')==1:
    #         return True
    #     # strict determines output when there are more than one dts 
    #     elif ''.join(list(self._obj.keys())).count('dts')>1:
    #         print('Warning: More than one detection time series.')
    #         return False if strict else True
    #     else:
    #         return False

    # def _has_clusters(self, var_name=None, strict=True):
    #     """Check existence of cluster labels.

    #     Default: Check if there is one variable named "*_cluster".
        
    #     var_name: Name of variable to check for. Optional, if not provided then
    #     returns True iff there is exactly one cluster label set

    #     """
    #     # check if cluster labels exists for that variable
    #     if var_name:
    #         if (var_name+'_cluster') in list(self._obj.keys()):
    #             return True 
    #     # must include exactly one cluster label set
    #     elif ''.join(list(self._obj.keys())).count('cluster')==1:
    #         return True
    #     # strict determines output when there is more than one label set 
    #     elif ''.join(list(self._obj.keys())).count('cluster')>1:
    #         print('Warning: More than one set of cluster labels.')
    #         return False if strict else True
    #     else:
    #         return False

    # def detect(
    #     self,
    #     temporal_dim: str,
    #     method: str,
    #     var: str = None,
    #     keep_other_vars : bool = False, 
    #     method_kwargs={}
    # ):
    #     """ Provide toad.detect function as accessor
        
    #     I.e. allows to use

    #         ds1 = ds.toad.detect(*args, **kwargs)
        
    #     alternatively to

    #         ds1 = toad.detect(ds, *args, **kwargs)

    #     """
    #     return detect(
    #         data = self._obj,
    #         temporal_dim = temporal_dim,
    #         method = method,
    #         var = var,
    #         keep_other_vars = keep_other_vars, 
    #         method_kwargs = method_kwargs
    #     )
    

    # def cluster(
    #     self,
    #     var : str,
    #     method : str,
    #     method_kwargs = {}
    # ):
    #     """ Provide toad.detect function as accessor
        
    #     I.e. allows to use

    #         ds2 = ds1.toad.cluster(*args, **kwargs)
        
    #     alternatively to

    #         ds2 = toad.cluster(ds1, *args, **kwargs)

    #     """
    #     assert self._has_dts(var_name=var), ' '
    #     return cluster(
    #         data = self._obj,
    #         method = method,
    #         var = var,
    #         method_kwargs = method_kwargs
    #     )
        

    # # def __call__(self, idx=None):
    # #     pass