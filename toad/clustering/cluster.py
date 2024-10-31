import numpy as np
import xarray as xr
from ..utils import infer_dims

class Clustering():
    """ Handle clusterings to allow simplified operation.

    :param cluster_label_ds:    dataarray with cluster label variable, cluster labels should be processed:

                                    * simple: apply the 3D mask to a 3D dataarray
                                    * spatial: reduce in the temporal dimension
                                    * strict: same as spactial, but create new cluster labels for regions that lie in the spatial overlap of multiple clusters

    :type cluster_label_ds:     xarray.DataArray
    :param temporal_dim:        Dimension in which the abrupt shifts have been detected. Automatically inferred if not provided.
    :type temporal_dim:         str, optional
    """

    def __init__(
            self,
            cluster_label_da,
            temporal_dim=None,
            ):
        self.tdim, self.sdims = infer_dims(cluster_label_da, tdim=temporal_dim)
        self._cluster_labels = cluster_label_da

    def _apply_mask_to(
            self,
            xarr_obj,
            cluster_lbl,
            masking='simple' # spatial, strict
            ):
        """ Apply mask to an xarray object.

        :param xarr_obj:        xarray object to apply the mask to
        :type xarr_obj:         xarray.DataArray
        :param cluster_lbl:     cluster label to apply the mask for
        :type cluster_lbl:      int, list
        :param masking:         type of masking to apply

                                    * simple: apply the 3D mask to a 3D dataarray 
                                    * spatial: reduce in the temporal dimension
                                    * strict: same as spactial, but create new cluster labels for regions that lie in the spatial overlap of multiple clusters

        :type masking:          str, optional

        **Examples**

        Could directly be used as
            >>> clustering = Clustering(clustered_ds, masking='spatial)
            >>> other_ds_clustered = clustering._apply_mask_to(other_ds, [0,2,3])
            >>> other_ds_clustered.mean()

        But usually will be wrapped in toad accessor, allowing
            >>> other_ds.toad.timeseries(
            >>>     clustering = Clustering(clustered_ds),
            >>>     cluster_lbl = [0,2,3]
            >>>     masking='spatial',
            >>>     how=('mean')
            >>>     )
        
        """
        if type(cluster_lbl) is not list: cluster_lbl = [ cluster_lbl ]

        if masking=='simple':
            _mask = self.simple_mask(cluster_lbl)
        elif masking=='spatial':
            _mask = self.spatial_mask(cluster_lbl)
        elif masking=='always_in_cluster':
            _mask = self.always_in_cluster_mask(cluster_lbl)
        else:
            raise ValueError('masking must be either simple or spatial or always_in_cluster')
        return xarr_obj.where(_mask)

    def simple_mask(self, cluster_lbl):
        """ Create a simple mask for a cluster label by simply applying the 3D mask to the 3D dataarray.

        :param cluster_lbl:     cluster label to apply the mask for
        :type cluster_lbl:      int, list
        
        """
        return self._cluster_labels.isin(cluster_lbl)

    def spatial_mask(self, cluster_lbl):
        """ Create a spatial mask for a cluster label by reducing the 3D mask in the temporal dimension.
        
        :param cluster_lbl:     cluster label to apply the mask for
        :type cluster_lbl:      int, list

        """
        return self.simple_mask(cluster_lbl).any(dim=self.tdim)

    def always_in_cluster_mask(self, cluster_lbl):
        """ Create a mask for cells that always have the same cluster label (such as completely unclustered cells by passing -1)"""
        return (self._cluster_labels == cluster_lbl).all(dim=self.tdim)

    def tprops(
            self, 
            cluster_lbl, 
            how=('mean',) # median, std, perc, dist
            ):
        """ Calculate temporal properties of a cluster label.
        
        :param cluster_lbl:     cluster label to apply the mask for
        :type cluster_lbl:      int, list
        :param how:             how to calculate the temporal properties
        :type how:              tuple
        :return:                temporal properties of the cluster label
        :rtype:                 ...

        """
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
            try:
                # takes the (first) numeric value to be found in how 
                pval = [arg for arg in how if type(arg)==float][0]
                return dimT.quantile(pval, skipna=True)
            except IndexError:
                raise TypeError("using perc needs additional numerical arg specifying which percentile, like how=('perc',0.2)") from None
        elif 'dist' in how:
            return dimT

    def sprops(
            self, 
            cluster_lbl,
            masking = 'spatial',
            how=('mean',) # median, std, perc, dist
            ):
        """ Calculate spatial properties of a cluster label.

        :param cluster_lbl:     cluster label to apply the mask for
        :type cluster_lbl:      int, list
        :param masking:         type of masking to apply

                                    * simple: apply the 3D mask to a 3D dataarray
                                    * spatial: reduce in the temporal dimension
                                    * strict: same as spactial, but create new cluster labels for regions that lie in the spatial overlap of multiple clusters 
        
        :type masking:          str, optional
        :param how:             how to calculate the spatial properties
        :type how:              tuple
        :return:                spatial properties of the cluster label
        :rtype:                 ...

        """
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
            try:
                # takes the (first) numeric value to be found in how 
                pval = [arg for arg in how if type(arg)==float][0]
                return dimA.quantile(pval, skipna=True).values, dimB.quantile(pval, skipna=True).values
            except IndexError:
                raise TypeError("using perc needs additional numerical arg specifying which percentile, like how=('perc',0.2)") from None
        elif 'dist' in how:
            return dimA, dimB

    def __call__(
            self,
            xarr_obj,
            cluster_lbl = None,
            ):
        """ Apply mask to an xarray object.

        :param xarr_obj:        xarray object to apply the mask to
        :type xarr_obj:         xarray.DataArray
        :param cluster_lbl:     cluster label to apply the mask for
        :type cluster_lbl:      int, list
        """
        return self._apply_mask_to(xarr_obj,cluster_lbl)