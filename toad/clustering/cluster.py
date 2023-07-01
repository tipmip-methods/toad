import numpy as np
import xarray as xr
from ..utils import infer_dims

class ClusterSelection():
    """
    Collection of points in 3D (space x space x time|forcing) space that allows
    joint operation, like mean, plot etc.
    """

    def __init__(self, mask, temporal_dim):
        self.mask = mask
        self.temporal_dim = temporal_dim
        self.spatial_mask = self.mask.any(dim=self.temporal_dim)
        _, (self.sdimA, self.sdimB) = infer_dims(self.mask, tdim=self.temporal_dim)


        # Coordinate properties of the cluster selection
        self.dimT = xr.where( self.mask, self.mask.__getattr__(self.temporal_dim), np.nan)
        self.meanT = self.dimT.mean().values
        self.stdT = self.dimT.mean().values
        self.medianT = self.dimT.median().values

        self.dimA = xr.where( self.mask, self.mask.__getattr__(self.sdimA), np.nan)
        self.meanA = self.dimA.mean().values
        self.stdA = self.dimA.mean().values
        self.medianA = self.dimA.median().values

        self.dimB = xr.where( self.mask, self.mask.__getattr__(self.sdimB), np.nan)
        self.meanB = self.dimB.mean().values
        self.stdB = self.dimB.mean().values
        self.medianB = self.dimB.median().values

    def percT(self, percentile=0.02):
        return self.dimT.quantile(percentile).values

    def percA(self, percentile=0.02):
        return self.dimA.quantile(percentile).values
    
    def percB(self, percentile=0.02):
        return self.dimB.quantile(percentile).values

    def __call__(self, xarr_obj, regions=True):
        """ Just apply mask to other xarray object"""
        if regions:
            return xr.where( self.spatial_mask, xarr_obj, np.nan)
        else:
            return xr.where( self.mask, xarr_obj, np.nan)

    # def _non_temporal_dims(self, xarr_obj):
    #     sdims = list(xarr_obj.keys())
    #     sdims.remove(self.temporal_dim)
    #     return sdims

    # def aggr_timeseries(self, xarr_obj):
    #     sdims = self._non_temporal_dims
    #     timeseries = self.__call__(xarr_obj).any(dim=self.temporal_dim)
    #     return timeseries.sum(dim=sdims)
   
    # @property
    # def naggr_timeseries(self):
    #     return self.aggr_timeseries / self.aggr_timeseries.isel(GMST=0)
    
    # @property
    # def mean_timeseries(self):
    #     return self.timeseries.mean(dim=['x','y'], skipna=True)

    # @property
    # def std_timeseries(self):
    #     return self.timeseries.std(dim=['x','y'], skipna=True)

    # @property
    # def mean_normalised_timeseries(self):
    #     return self.normalised_timeseries.mean(dim=['x','y'], skipna=True)

    # @property
    # def std_normalised_timeseries(self):
    #     return self.normalised_timeseries.std(dim=['x','y'], skipna=True)

    # def quantile_timeseries(self, quantile):
    #     return self.timeseries.quantile(quantile, dim=['x','y'], skipna=True)
    
    # def quantile_normalised_timeseries(self, quantile):
    #     return self.normalised_timeseries.quantile(quantile, dim=['x','y'], skipna=True)

    # @property
    # def median_timeseries(self):
    #     return self.quantile_timeseries(0.5)

    # @property
    # def median_normalised_timeseries(self):
    #     return self.quantile_normalised_timeseries(0.5)
    
    # @property
    # def all_timeseries(self):
    #     return self.timeseries.stack(xy=('x', 'y')).transpose().dropna(dim="xy", how='all')

    # @property
    # def all_normalised_timeseries(self):
    #     return self.all_timeseries / self.all_timeseries.isel(GMST=0)
    
    # def calculate_scores(self):
    #     (a,b) , res, _, _, _ = np.polyfit(self.GMST, self.median_normalised_timeseries.values, 1, full=True)
    #     self.median_score = res[0] 
    #     self.median_score_fit = b + a*self.GMST

    #     (a,b) , res, _, _, _ = np.polyfit(self.GMST, self.mean_normalised_timeseries.values, 1, full=True)
    #     self.mean_score = res[0] 
    #     self.mean_score_fit = b + a*self.GMST
class Clustering:
    """
    Handler of Cluster selection instances
    """
    def __init__(self, clustered_ds, temporal_dim, var):
        assert type(clustered_ds) == xr.Dataset, 'needs xr.DataSet!'
        self._source = clustered_ds
        self._temporal_dim = temporal_dim
        self._var = var
        self._cluster_labels = self._source.get(self._var+'_cluster')
        self._shape = self._cluster_labels.shape

    def select(self, cluster_lbl):
        """ Return Cluster Selection """
        if type(cluster_lbl) is not list: cluster_lbl = [ cluster_lbl ]
        _mask = self._cluster_labels.isin(cluster_lbl)
        return ClusterSelection(_mask, self._temporal_dim)

#   def __init__(self, cluster_idx, quantile=0.05):
#         self.index = int(cluster_idx)
#         self.mask = (thk_clusters == cluster_idx)
#         self.spatial_mask = self.mask.any(dim='GMST')
#         self._compute_cluster_properties(quantile)

#     def _compute_cluster_properties(self, quantile):
#         cluster_times = xr.where(self.mask, thk.GMST, np.nan)
#         self.mean_t = cluster_times.mean().values
#         self.std_t = cluster_times.std().values
#         self.q1_t, self.median_t, self.q2_t = cluster_times.quantile([quantile, 0.5, 1-quantile]).values

#         cluster_x = xr.where(self.mask, thk.x, np.nan)
#         self.mean_x = cluster_x.mean().values
#         self.std_x = cluster_x.std().values
#         self.q1_x, self.median_x, self.q2_x = cluster_x.quantile([quantile, 0.5, 1-quantile]).values

#         cluster_y = xr.where(self.mask, thk.y, np.nan)
#         self.mean_y = cluster_y.mean().values
#         self.std_y = cluster_y.std().values
#         self.q1_y, self.median_y, self.q2_y = cluster_y.quantile([quantile, 0.5, 1-quantile]).values



# class ClusteredObj:

#     def __init__(self, xarr_dataarray, cluster):
#         self.spatial_mask = cluster.spatial_mask
        
#         # general properties of the cluster
#         self.mean_t = cluster.mean_t
#         self.std_t = cluster.std_t
#         self.q1_t = cluster.q1_t
#         self.median_t = cluster.median_t 
#         self.q2_t = cluster.q2_t

#         self.mean_x = cluster.mean_x
#         self.std_t = cluster.std_x

#         self.mean_y = cluster.mean_y
#         self.std_y = cluster.std_y

#         self.GMST = xarr_dataarray.GMST
#         self.timeseries = xarr_dataarray.where(self.spatial_mask)        
#         self.normalised_timeseries = self.timeseries / self.timeseries.isel(GMST=0)
#         self.normalised_timeseries = self.normalised_timeseries.where(
#                                             np.isfinite(self.normalised_timeseries),
#                                             other=np.nan)
    
#         self.calculate_scores()

#     @property
#     def aggr_timeseries(self):
#         return self.timeseries.sum(dim=['x','y'])
   
#     @property
#     def naggr_timeseries(self):
#         return self.aggr_timeseries / self.aggr_timeseries.isel(GMST=0)
    
#     @property
#     def mean_timeseries(self):
#         return self.timeseries.mean(dim=['x','y'], skipna=True)

#     @property
#     def std_timeseries(self):
#         return self.timeseries.std(dim=['x','y'], skipna=True)

#     @property
#     def mean_normalised_timeseries(self):
#         return self.normalised_timeseries.mean(dim=['x','y'], skipna=True)

#     @property
#     def std_normalised_timeseries(self):
#         return self.normalised_timeseries.std(dim=['x','y'], skipna=True)

#     def quantile_timeseries(self, quantile):
#         return self.timeseries.quantile(quantile, dim=['x','y'], skipna=True)
    
#     def quantile_normalised_timeseries(self, quantile):
#         return self.normalised_timeseries.quantile(quantile, dim=['x','y'], skipna=True)

#     @property
#     def median_timeseries(self):
#         return self.quantile_timeseries(0.5)

#     @property
#     def median_normalised_timeseries(self):
#         return self.quantile_normalised_timeseries(0.5)
    
#     @property
#     def all_timeseries(self):
#         return self.timeseries.stack(xy=('x', 'y')).transpose().dropna(dim="xy", how='all')

#     @property
#     def all_normalised_timeseries(self):
#         return self.all_timeseries / self.all_timeseries.isel(GMST=0)
    
#     def calculate_scores(self):
#         (a,b) , res, _, _, _ = np.polyfit(self.GMST, self.median_normalised_timeseries.values, 1, full=True)
#         self.median_score = res[0] 
#         self.median_score_fit = b + a*self.GMST

#         (a,b) , res, _, _, _ = np.polyfit(self.GMST, self.mean_normalised_timeseries.values, 1, full=True)
#         self.mean_score = res[0] 
#         self.mean_score_fit = b + a*self.GMST

#     def plot(self, ax, how=('nmean'), **kwargs):
#         how = how if type(how)==tuple else (how,)

#         if 'aggr' in how:
#             self.aggr_timeseries.plot(ax=ax, **kwargs)
#         if 'naggr' in how:
#             self.naggr_timeseries.plot(ax=ax, **kwargs)
#         if 'mean' in how:
#             self.mean_timeseries.plot(ax=ax, **kwargs)
#         if 'std' in how:
#             self.std_timeseries.plot(ax=ax, **kwargs)
#         if 'nmean' in how:
#             self.mean_normalised_timeseries.plot(ax=ax, **kwargs)
#         if 'median' in how:
#             self.median_timeseries.plot(ax=ax, **kwargs)
#         if 'nmedian' in how:
#             self.median_normalised_timeseries.plot(ax=ax, **kwargs)
#         if 'nstd' in how:
#             self.std_normalised_timeseries.plot(ax=ax, **kwargs)
#         if 'band' in how:
#             ax.fill_between(
#                 self.timeseries.GMST.values,
#                 self.mean_timeseries-self.std_timeseries.values,
#                 self.mean_timeseries+self.std_timeseries.values,
#                 alpha=0.2, **kwargs
#             )
#             (self.mean_timeseries-self.std_timeseries).plot(ax=ax, **kwargs)
#             (self.mean_timeseries+self.std_timeseries).plot(ax=ax, **kwargs)

#         if 'nband' in how:
#             ax.fill_between(
#                 self.timeseries.GMST.values,
#                 self.mean_normalised_timeseries-self.std_normalised_timeseries,
#                 self.mean_normalised_timeseries+self.std_normalised_timeseries,
#                 alpha=0.2, **kwargs
#             )
#             (self.mean_normalised_timeseries-self.std_normalised_timeseries).plot(ax=ax, **kwargs)
#             (self.mean_normalised_timeseries+self.std_normalised_timeseries).plot(ax=ax, **kwargs)

#         if 'nqband' in how:
#             quantile = how[1]
#             ax.fill_between(
#                 self.timeseries.GMST.values,
#                 self.quantile_normalised_timeseries(quantile).values,
#                 self.quantile_normalised_timeseries(1-quantile).values,
#                 alpha=0.2, **kwargs
#             )
#             self.quantile_normalised_timeseries(quantile).plot(ax=ax, **kwargs)
#             self.quantile_normalised_timeseries(1-quantile).plot(ax=ax, **kwargs)

#         if 'qband' in how:
#             quantile = how[1]
#             ax.fill_between(
#                 self.timeseries.GMST.values,
#                 self.quantile_timeseries(quantile).values,
#                 self.quantile_timeseries(1-quantile).values,
#                 alpha=0.2, **kwargs
#             )
#             self.quantile_timeseries(quantile).plot(ax=ax, **kwargs)
#             self.quantile_timeseries(1-quantile).plot(ax=ax, **kwargs)

#         if 'all' in how:
#             self.all_timeseries.plot.line(x='GMST', ax=ax, **kwargs)
#             ax.get_legend().remove()
#         if 'nall' in how:
#             self.all_normalised_timeseries.plot.line(x='GMST', ax=ax, **kwargs)
#             ax.get_legend().remove()

# class Cluster():
#     def __init__(self, cluster_idx, quantile=0.05):
#         self.index = int(cluster_idx)
#         self.mask = (thk_clusters == cluster_idx)
#         self.spatial_mask = self.mask.any(dim='GMST')
#         self._compute_cluster_properties(quantile)

#     def _compute_cluster_properties(self, quantile):
#         cluster_times = xr.where(self.mask, thk.GMST, np.nan)
#         self.mean_t = cluster_times.mean().values
#         self.std_t = cluster_times.std().values
#         self.q1_t, self.median_t, self.q2_t = cluster_times.quantile([quantile, 0.5, 1-quantile]).values

#         cluster_x = xr.where(self.mask, thk.x, np.nan)
#         self.mean_x = cluster_x.mean().values
#         self.std_x = cluster_x.std().values
#         self.q1_x, self.median_x, self.q2_x = cluster_x.quantile([quantile, 0.5, 1-quantile]).values

#         cluster_y = xr.where(self.mask, thk.y, np.nan)
#         self.mean_y = cluster_y.mean().values
#         self.std_y = cluster_y.std().values
#         self.q1_y, self.median_y, self.q2_y = cluster_y.quantile([quantile, 0.5, 1-quantile]).values


#     def __call__(self, xarr_dataarray, quantile=0.05):
#         assert type(xarr_dataarray) == xr.DataArray, 'needs xr.DataArray!'
#         cluster_object = ClusteredObj(xarr_dataarray, self)
#         return cluster_object