import numpy as np
import xarray as xr
from ..utils import infer_dims

class Clustering():

    def __init__(
            self,
            cluster_label_da,
            temporal_dim=None,
    ):
        """ Handle clusterings to allow simplified operation.

        cluster_label_ds: dataarray with cluster label variable masking: how the

        cluster labels should be processed
            simple: apply the 3D mask to a 3D dataarray 
            spatial: reduce in the temporal dimension
            strict: same as spactial, but create new cluster labels for regions
            that lie in the spatial overlap of multiple clusters 
            
        temporal_dim: Dimension in which the abrupt shifts have been detected. 
        Automatically inferred if not provided.

        """
        self.tdim, self.sdims = infer_dims(cluster_label_da, tdim=temporal_dim)
        self._cluster_labels = cluster_label_da

    def _apply_mask_to(
            self,
            xarr_obj,
            cluster_lbl,
            masking='simple' # spatial, strict
    ):
        """ Apply mask to an xarray object.

        Could directly be used as_
            clustering = Clustering(clustered_ds, masking='spatial)
            other_ds_clustered = clustering._apply_mask_to(other_ds, [0,2,3])
            other_ds_clustered.mean()

        But usually will be wrapped in toad accessor, allowing
            other_ds.toad.timeseries(
                clustering = Clustering(clustered_ds),
                cluster_lbl = [0,2,3]
                masking='spatial',
                how=('mean')
            )
        
        """
        if type(cluster_lbl) is not list: cluster_lbl = [ cluster_lbl ]

        if masking=='simple':
            _mask = self.simple_mask(cluster_lbl)
        elif masking=='spatial':
            _mask = self.spatial_mask(cluster_lbl)
        else:
            raise ValueError('masking must be either simple or spatial')
        return xarr_obj.where(_mask)

    def simple_mask(self, cluster_lbl, exclusive=False):
        if exclusive:
            pass #todo
        else:
            return self._cluster_labels.isin(cluster_lbl)

    def spatial_mask(self, cluster_lbl):
        return self.simple_mask(cluster_lbl).any(dim=self.tdim)

    def tprops(
            self, 
            cluster_lbl, 
            how=('mean',) # median, std, perc, dist
        ):
        if type(how)== str:
            how = (how,)
    
        # spatial mask does not make sense for t-properties (would always be the
        # same)
        mask = self.simple_mask(cluster_lbl)
        dimT = xr.where( mask, mask.__getattr__(self.tdim), np.nan)

        if 'mean' in how:
            return dimT.mean().values
        elif 'median' in how:
            return dimT.median().values
        elif 'std' in how:
            return dimT.std().values
        elif 'perc' in how:
            # takes the (first) numeric value to be found in how 
            try:
                pval = [arg for arg in how if type(arg)==float][0]
            except IndexError:
                raise ValueError("using perc needs additional numerical arg specifying which percentile, like how=('perc',0.2)") from None
            return dimT.quantile(pval, skipna=True)
        elif 'dist' in how:
            return dimT

    def sprops(
            self, 
            cluster_lbl,
            masking = 'spatial',
            how=('mean',) # median, std, perc, dist
        ):
        if type(how)== str:
            how = (how,)
    
        if masking=='spatial':
            mask = self.spatial_mask(cluster_lbl)
        elif masking=='simple': 
            mask = self.simple_mask(cluster_lbl)
        dimA = xr.where( mask, mask.__getattr__(self.sdims[0]), np.nan)
        dimB = xr.where( mask, mask.__getattr__(self.sdims[1]), np.nan)

        if 'mean' in how:
            return dimA.mean().values, dimB.mean().values
        elif 'median' in how:
            return dimA.median().values, dimB.median().values
        elif 'std' in how:
            return dimA.std().values, dimB.std().values
        elif 'perc' in how:
            # takes the (first) numeric value to be found in how 
            try:
                pval = [arg for arg in how if type(arg)==float][0]
                return dimA.quantile(pval, skipna=True).values, dimB.quantile(pval, skipna=True).values
            except IndexError:
                raise ValueError("using perc needs additional numerical arg specifying which percentile, like how=('perc',0.2)") from None
        elif 'dist' in how:
            return dimA, dimB

    def __call__(
            self,
            xarr_obj,
            cluster_lbl = None,
    ):
        self._apply_mask(cluster_lbl, xarr_obj)
        #


        #     def __init__(self, mask, temporal_dim):
#         self.mask = mask
#         self.temporal_dim = temporal_dim
#         self.spatial_mask = self.mask.any(dim=self.temporal_dim)
#         _, (self.sdimA, self.sdimB) = infer_dims(self.mask, tdim=self.temporal_dim)


#         # Coordinate properties of the cluster selection
#         self.dimT = xr.where( self.mask, self.mask.__getattr__(self.temporal_dim), np.nan)
#         self.meanT = self.dimT.mean().values
#         self.stdT = self.dimT.mean().values
#         self.medianT = self.dimT.median().values

#         self.dimA = xr.where( self.mask, self.mask.__getattr__(self.sdimA), np.nan)
#         self.meanA = self.dimA.mean().values
#         self.stdA = self.dimA.mean().values
#         self.medianA = self.dimA.median().values

#         self.dimB = xr.where( self.mask, self.mask.__getattr__(self.sdimB), np.nan)
#         self.meanB = self.dimB.mean().values
#         self.stdB = self.dimB.mean().values
#         self.medianB = self.dimB.median().values

#     def percT(self, percentile=0.02):
#         return self.dimT.quantile(percentile).values

#     def percA(self, percentile=0.02):
#         return self.dimA.quantile(percentile).values
    
#     def percB(self, percentile=0.02):
#         return self.dimB.quantile(percentile).values

#     def __call__(self, xarr_obj, regions=True):
#         """ Just apply mask to other xarray object"""
#         if regions:
#             return xr.where( self.spatial_mask, xarr_obj, np.nan)
#         else:
#             return xr.where( self.mask, xarr_obj, np.nan)