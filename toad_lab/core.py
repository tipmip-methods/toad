import logging
import xarray as xr
import numpy as np
from typing import Union, Callable
import os

from toad_lab import shifts_detection, clustering, postprocessing, visualisation, preprocessing
from toad_lab.utils import infer_dims
import toad_lab.clustering.methods
import toad_lab.shifts_detection.methods
from _version import __version__


class TOAD:
    def __init__(self, 
                 data: Union[xr.Dataset, str],
                 log_level = "WARNING"
        ):
        
        # load data from path if string
        if type(data) is str:
            if not os.path.exists(data):
                raise ValueError(f"File {data} does not exist.")
            self.data = xr.open_dataset(data)
            self.data.attrs['title'] = os.path.basename(data).split('.')[0] # store path as title for saving toad file later
        elif type(data) is xr.Dataset or type(data) is xr.DataArray:
            self.data = data  # Original data
        
        # Initialize the logger for the TOAD object
        self.logger = logging.getLogger("TOAD")
        self.logger.propagate = False  # Prevent propagation to the root logger :: i.e. prevents dupliate messages
        self.set_log_level(log_level) 


    # # ======================================================================
    # #               Module functions
    # # ======================================================================
    def preprocess(self):
        return preprocessing.Preprocess(self)
    
    def stats(self):
        return postprocessing.Stats(self)
    
    def plotter(self):
        return visualisation.TOADPlotter(self)
    

    # # ======================================================================
    # #               SET functions
    # # ======================================================================


    def set_log_level(self, level):
        """Sets the logging level for the TOAD logger.
        
        Used like this:         
            logger.debug("This is a debug message.")
            logger.info("This is an info message.")
            logger.warning("This is a warning message.")
            logger.error("This is an error message.")
            logger.critical("This is a critical message.")

            In sub-modules get logger like this:
            logger = logging.getLogger("TOAD")
        """
        level = level.upper()
        if level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError("Invalid log level. Choose from 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'")

        self.logger.setLevel(getattr(logging, level))
        
        # Only add a handler if there are no handlers yet (to avoid duplicate messages)
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s: %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logging level set to {level}")
    

    # # ======================================================================
    # #               COMPUTE functions
    # # ======================================================================

    def compute_shifts(
        self,
        var: str,
        temporal_dim: str = "time", 
        method: shifts_detection.ShiftsMethod = toad_lab.shifts_detection.methods.default_shifts_method,
        output_label: str = None,
        overwrite: bool = False,
        merge_input = True,
    ) -> xr.Dataset :
        """Apply an abrupt shift detection algorithm to a dataset along the specified temporal dimension.

        Args:
            var: Name of the variable in the dataset to analyze for abrupt shifts.
            temporal_dim: Dimension along which the time-series analysis is performed. Defaults to "time".
            method: The abrupt shift detection algorithm to use. Defaults to "asdetect".
            output_label: Name of the variable to store results. Defaults to {var}_dts.
            overwrite: Whether to overwrite existing variable. Defaults to False.
            merge_input: Whether to merge shifts with original data. Defaults to True.

        Returns:
            xr.Dataset: Original data with detected shifts if merge_input=True, otherwise just shifts.

        Raises:
            AssertionError: If invalid dataset or variable.
            ValueError: If invalid method or output_label conflicts.
        """
        results = shifts_detection.compute_shifts(data=self.data, var=var, temporal_dim=temporal_dim, method=method, output_label=output_label, overwrite=overwrite, merge_input=merge_input)
        if merge_input:
            self.data = results
        else:
            return results


    def compute_clusters(
        self,
        var : str,
        var_dts: str = None,
        min_abruptness: float = None,
        method : clustering.ClusteringMethod = toad_lab.clustering.methods.default_clustering_method,
        var_func: Callable[[float], bool] = None,
        dts_func: Callable[[float], bool] = None,
        scaler: str = 'StandardScaler',
        output_label: str = None,
        overwrite: bool = False,
        merge_input: bool = True,
        transpose_output: bool = False,
    ) -> xr.Dataset:
        """Apply a clustering algorithm to the dataset along the temporal dimension.

        Args:
            var: Name of the variable in the dataset to cluster.
            var_dts: Name of the variable containing precomputed shifts. Defaults to {var}_dts.
            min_abruptness: Minimum threshold for abruptness to filter shifts. Required if dts_func not provided.
            method: The clustering method to use. Choose from predefined method objects in toad_lab.clustering.methods.
                Defaults to euclidian HDBSCAN with min_cluster_size=25.
            var_func: A callable used to filter the primary variable before clustering. Defaults to None.
            dts_func: A callable used to filter the shifts before clustering. Defaults to None.
            scaler: The scaling method to apply to the data before clustering. Defaults to 'StandardScaler'.
            output_label: Name of the variable to store clustering results. Defaults to {var}_cluster.
            overwrite: Whether to overwrite existing variable. Defaults to False.
            merge_input: Whether to merge clustering results with original data. Defaults to True.
            transpose_output: Whether to transpose the output array. Defaults to False.

        Returns:
            xr.Dataset:  If `merge_input` is `False`, returns an `xarray.DataArray` containing cluster labels for the data 
            points. Otherwise, the clustering results are merged into the original dataset, and the function 
            returns `None`.

        Raises:
            AssertionError: If invalid dataset, dimensions, or missing required parameters.
            ValueError: If var_dts not found, invalid method, or output_label conflicts.

        Notes:
            - Both var_func and dts_func filters must pass for data points to be included
            - Scaling is automatically applied based on scaler parameter
            - transpose_output helps with datasets requiring specific axis arrangement
        """
        result = clustering.compute_clusters(self.data, var, var_dts, min_abruptness, method, var_func, dts_func, scaler, output_label, overwrite, merge_input, transpose_output)
        if merge_input:
            self.data = result
        else:
            return result
        

    # # ======================================================================
    # #               GET functions (postprocessing)
    # # ======================================================================
    def get_cluster_counts(self, var, sort=False):
        """
        Calculate the number of cells in each cluster for a specified variable.

        Each cell may belong to multiple clusters over time. This function computes the number
        of unique cells in each cluster and allows optional sorting of the results.

        Parameters
        ----------
        var : str
            The name of the variable for which cluster counts are computed.
            Requires the dataset to have a corresponding `"{var}_cluster"` key.
        sort : bool, optional
            If `True`, the resulting dictionary is sorted in descending order
            by the number of cells in each cluster. Defaults to `False`.

        Returns
        -------
        dict
            A dictionary where keys are cluster IDs (as integers) and values are the
            number of unique cells in each cluster.

        Raises
        ------
        ValueError
            If cluster information for the specified variable is not found in the dataset.

        Notes
        -----
        - The function counts the number of unique spatial cells that are part of each cluster,
        regardless of the number of time steps they appear in the cluster.
        - For the unclustered data (typically represented by cluster ID `-1`), the function uses
        a special masking strategy to ensure accurate counts.

        TODO: I think this returns the number of cells that are part of the cluster in 
        both space and time, so if the same cell is part of the cluster for several time
        steps, it adds up. Verify this. 

        """
        if self.data.get(f"{var}_cluster") is None:
            raise ValueError(f"No clusters computed for {var} yet.")
        
        counts = {}
        for cluster_id in np.unique(self.data[f"{var}_cluster"]):
            timeseries_data = self.timeseries(
                self.data,
                clustering=Clustering(self.data[f"{var}_cluster"]),
                cluster_lbl=[cluster_id],
                masking='always_in_cluster' if cluster_id == -1 else 'spatial', # the spatial mask returns cells that at any point is in cluster_id, so for -1 you would get all cells. Therefore, we need another mask for unclustered cells (i.e. -1).
                how=('per_gridcell') # get time series for each grid cell
            )
            counts[int(cluster_id)] = len(timeseries_data.cell_xy)

        if sort:
            return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
        else:
            return counts


    def get_shifts(self, var):
        """
        Return the shifts dataset for further analysis.
        """
        if self.data.get(f"{var}_dts") is None:
            raise ValueError(f"No shifts computed for {var} yet.")
        return self.data[f"{var}_dts"]


    def get_clusters(self, var):
        """
        Return the clusters dataset for further analysis.
        """
        if self.data.get(f"{var}_cluster") is None:
            raise ValueError(f"No clusters computed for {var} yet.")
        return self.data[f"{var}_cluster"]


    def get_timeseries_in_cluster(self, var, cluster_id):
        clusters = self.get_clusters(var)
        timeseries_data = self.timeseries(
            self.data,
            clustering=Clustering(clusters),
            cluster_lbl=[cluster_id],
            masking='always_in_cluster' if cluster_id == -1 else 'spatial', # the spatial mask returns cells that at any point is in cluster_id, so for -1 you would get all cells. Therefore, we need another mask for unclustered cells (i.e. -1).
            how="per_gridcell" # get time series for each grid cell
        )
        return [timeseries_data.isel(cell_xy=j) for j in range(len(timeseries_data.cell_xy))]


    def get_timeseries_in_cluster_aggregate(self, var, cluster_id, cluster_label=None, how="mean"):
        clusters = self.data[cluster_label] if cluster_label else self.get_clusters(var)
        return self.timeseries(
            self.data,
            clustering=Clustering(clusters),
            cluster_lbl=[cluster_id],
            masking='always_in_cluster' if cluster_id == -1 else 'spatial', # the spatial mask returns cells that at any point is in cluster_id, so for -1 you would get all cells. Therefore, we need another mask for unclustered cells (i.e. -1).
            how=how
        )

    def get_largest_cluster_ids(self, var):
        """
        Return list of cluster ids sorted by the number of cells in each cluster.
        """
        counts = self.get_cluster_counts(var, sort=True)
        return list(counts.keys())


    def cluster_persistence_fraction(self, var, cluster_id):
        """
        Calculate the persistence fraction of cells in a given cluster over time.
        
        This function computes the fraction of grid cells that belong to a specified 
        cluster at each time step, relative to the total number of cells that have 
        ever belonged to that cluster. The "persistence fraction" represents how 
        many cells are part of the cluster at each moment, based on cells that 
        have been in the cluster at any point in time.

        Parameters:
        ----------
        var : str
            Name of the variable representing the cluster data. This variable will 
            be used to access the corresponding cluster field (e.g., "thk_cluster").
            
        cluster_id : int
            The identifier for the cluster of interest. If `cluster_id == -1`, the function 
            computes the fraction for unclustered cells using the 'always_in_cluster' mask. 
            Otherwise, it uses the 'spatial' mask to compute for the selected cluster.
        
        Returns:
        -------
        np.ndarray
            A 1D array of length equal to the number of time steps, where each element 
            represents the fraction of cells that are part of the cluster at that specific 
            time step. If there are no cells in the cluster, an array of zeros is returned.
        """
        cluster_var = f"{var}_cluster"
        timeseries_data = self.timeseries(
            self.data,
            clustering=Clustering(self.data[cluster_var]),
            cluster_lbl=[cluster_id],
            masking='always_in_cluster' if cluster_id == -1 else 'spatial', # the spatial mask returns cells that at any point is in cluster_id, so for -1 you would get all cells. Therefore, we need another mask for unclustered cells (i.e. -1).
            how=('per_gridcell') # get time series for each grid cell
        )
            
        # Number of cells in the dataset
        num_cells = len(timeseries_data.cell_xy)
        
        # Preallocate cluster matrix
        cluster_matrix = np.full((num_cells, len(timeseries_data.time)), -1)
        
        for i in np.arange(num_cells):
            cluster_matrix[i, :] = timeseries_data.isel(cell_xy=i)[cluster_var].values

        # Calculate fraction of cells in the cluster at each time step
        num_cells_in_cluster = (cluster_matrix == cluster_id).sum(axis=0)
        
        # Avoid division by zero if no cells are in the cluster
        if num_cells == 0:
            return np.zeros(len(timeseries_data.time))

        return num_cells_in_cluster / num_cells


    def timeseries(
            self,
            dataframe, 
            clustering,
            cluster_lbl,
            masking = 'simple',
            how=('aggr',)  # mean, median, std, perc, per_gridcell
        ):
        """Extracts the time series of a cluster label.
        
        :param clustering:      Clustering object
        :type clustering:       toad.clustering.cluster.Clustering
        :param cluster_lbl:     Cluster label to extract the time series from.
        :type cluster_lbl:      int, list
        :param masking:         Type of masking to apply.
                                    * simple: apply the 3D mask to a 3D dataarray 
                                    * spatial: reduce in the temporal dimension
                                    * strict: same as spactial, but create new cluster labels for regions that lie in the spatial overlap of multiple clusters
        :type masking:          str, optional
        :param how:             How to aggregate the time series.
                                    * mean: mean value
                                    * median: median value
                                    * aggr: sum of values
                                    * std: standard deviation
                                    * perc: percentile value, eg. how=('perc',0.9)
                                    * per_gridcell: time series for each grid cell
        :type how:              str, tuple
        :return:                Time series of the cluster label.
        :rtype:                 xr.DataArray

        TODO: consider renaming this.
        TODO: consider making this private 
        TODO: consider spliting this into two functions, one for getting individual time series and one for aggreating them
        """
        da = clustering._apply_mask_to(dataframe, cluster_lbl, masking=masking)
        tdim, sdims = infer_dims(dataframe)

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
            assert len([arg for arg in how if type(arg)==float])==1, "using perc needs additional numerical arg specifying which percentile, like how=('perc',0.2)"
            pval = [arg for arg in how if type(arg)==float][0]
            timeseries = da.quantile(pval, dim=sdims, skipna=True)
        elif 'per_gridcell' in how:
            timeseries = da.stack(cell_xy=sdims).transpose().dropna(dim='cell_xy', how='all')
        else:
            raise ValueError('how needs to be one of mean, median, aggr, std, perc, per_gridcell')

        return timeseries

    # end of TOAD object


class Clustering():
    """ Handle clusterings to allow simplified operation.

    :param cluster_label_ds:    dataarray with cluster label variable, cluster labels should be processed:

                                    * simple: apply the 3D mask to a 3D dataarray
                                    * spatial: reduce in the temporal dimension
                                    * strict: same as spactial, but create new cluster labels for regions that lie in the spatial overlap of multiple clusters

    :type cluster_label_ds:     xarray.DataArray
    :param temporal_dim:        Dimension in which the abrupt shifts have been detected. Automatically inferred if not provided.
    :type temporal_dim:         str, optional

    TODO: Consider whether to merge this with the TOAD object
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

        
        TODO: rewrite docstring
        TODO: verify this works and move to postprocessing/stats.py
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

        TODO: rewrite docstring
        TODO: verify this works and move to postprocessing/stats.py

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

    # End of Clustering object
