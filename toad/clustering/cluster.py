import numpy as np
import xarray as xr
from ..utils import infer_dims

class ClusterSelection():
    """
    Collection of points in 3D (space x space x time|forcing) space that allows
    joint operation, like mean, plot etc.

        clustering_selection = myclustering.select([0,1])
        clustering_selection.spatial_mask.plot()
        print(clustering_selection.meanA)
        clustering_selection(xr_ds).toad.score()

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

class Clustering:
    """
    Handler of Cluster selection instances.

        myclustering = toad.Clustering(
                            clustered_ds = ds_with_clusters, 
                            temporal_dim = 'GMST', 
                            var = 'thk'
                        )
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