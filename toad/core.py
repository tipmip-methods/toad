import logging
import xarray as xr
import numpy as np
from typing import List, Union, Callable, Optional, Literal
import os
from sklearn.base import ClusterMixin

from toad import shifts_detection, clustering, postprocessing, visualisation, preprocessing
from toad.utils import infer_dims
from toad._version import __version__
<<<<<<< HEAD
from toad.utils import get_space_dims, is_equal_to, contains_value, deprecated
=======
>>>>>>> 01b5596 (Moved _version.py inside toad package)


class TOAD:
    """
    Main object for interacting with TOAD.
    TOAD (Tippping and Other Abrupt events Detector) is a framework for detecting and clustering spatio-temporal patterns in spatio-temporal data.

<<<<<<< HEAD
    >> Args:
        data : (xr.Dataset or str)
            The input data. If a string, it is interpreted as a path to a netCDF file.
        log_level : (str)
            The logging level. Choose from 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'. Defaults to 'WARNING'.
=======
    Args:
        data (xr.Dataset or str): The input data. If a string, it is interpreted as a path to a netCDF file.
        log_level (str): The logging level. Choose from 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'. Defaults to 'WARNING'.
>>>>>>> c6fc662 (Docstring and type fixes)
    """
    
    data: xr.Dataset
    def __init__(self, 
                 data: Union[xr.Dataset, str],
                 time_dim: str = "time",
                 log_level = "WARNING"
        ):
        
        # load data from path if string
        if isinstance(data, str):
            if not os.path.exists(data):
                raise ValueError(f"File {data} does not exist.")
            self.data = xr.open_dataset(data)
            self.data.attrs['title'] = os.path.basename(data).split('.')[0] # store path as title for saving toad file later
        elif isinstance(data, (xr.Dataset, xr.DataArray)):
            self.data = data  # Original data
        
        self.time_dim = time_dim
        self.space_dims = get_space_dims(self.data, self.time_dim)

        # Initialize the logger for the TOAD object
        self.logger = logging.getLogger("TOAD")
        self.logger.propagate = False  # Prevent propagation to the root logger :: i.e. prevents dupliate messages
        self.set_log_level(log_level) 


    # # ======================================================================
    # #               Module functions
    # # ======================================================================
    def preprocess(self):
        """ Access preprocessing methods. """
        return preprocessing.Preprocess(self)
    
<<<<<<< HEAD

    def cluster_stats(self, var):
        """ Access cluster statistical methods. 
        
        >> Args:
            var : (str)
                Base variable name (e.g. 'temperature', will look for 'temperature_cluster') or custom cluster variable name.
        
        >> Returns:
            toad.postprocessing.cluster_stats.ClusterStats: ClusterStats object
        """
        return postprocessing.ClusterStats(self, var)

=======
    def stats(self):
        """ Access statistical methods. """
        return postprocessing.Stats(self)
>>>>>>> c6fc662 (Docstring and type fixes)
    
    def aggregation(self):
        """ Access aggregation methods. """
        return postprocessing.Aggregation(self)
    
    def plotter(self):
        """ Access plotting methods. """
        return visualisation.TOADPlotter(self)
    

    # # ======================================================================
    # #               SET functions
    # # ======================================================================


    def set_log_level(self, level):
        """Sets the logging level for the TOAD logger.

<<<<<<< HEAD
        >> Args:
            level:
                The logging level to set. Choose from 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
        
        >> Examples: 
=======
        Args:
            level: The logging level to set. Choose from 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
        
        Examples: 
>>>>>>> c6fc662 (Docstring and type fixes)
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
        method: shifts_detection.ShiftsMethod,
        time_dim: str = "time", 
        output_label_suffix: str = "",
        overwrite: bool = False,
        return_results_directly: bool = False,
    ) -> Union[xr.DataArray, None]:
        """Apply an abrupt shift detection algorithm to a dataset along the specified temporal dimension.

<<<<<<< HEAD
        >> Args:
            var:
                Name of the variable in the dataset to analyze for abrupt shifts.
            method:
                The abrupt shift detection algorithm to use. Choose from predefined method objects in toad.shifts_detection.methods or create your own following the base class in toad.shifts_detection.methods.base
            time_dim:
                Name of the dimension along which the time-series analysis is performed. Defaults to "time".
            output_label_suffix:
                A suffix to add to the output label. Defaults to "".
            overwrite:
                Whether to overwrite existing variable. Defaults to False.
            return_results_directly:
                Whether to return the detected shifts directly or merge into the original dataset. Defaults to False.

        >> Returns:
            - If `return_results_directly` is `True`, returns an `xarray.DataArray` containing the detected shifts. 
            - If `return_results_directly` is `False`, the detected shifts are merged into the original dataset, and the function returns `None`.

        >> Raises:
            ValueError:
                If data is invalid or required parameters are missing
=======
        Args:
            var: Name of the variable in the dataset to analyze for abrupt shifts.
            method: The abrupt shift detection algorithm to use. Choose from predefined method objects in toad.shifts_detection.methods or create your own following the base class in toad.shifts_detection.methods.base
            time_dim: Name of the dimension along which the time-series analysis is performed. Defaults to "time".
            output_label_suffix: A suffix to add to the output label. Defaults to "".
            overwrite: Whether to overwrite existing variable. Defaults to False.
            return_results_directly: Whether to return the detected shifts directly or merge into the original dataset. Defaults to False.

        Returns:
            - If `return_results_directly` is `True`, returns an `xarray.DataArray` containing the detected shifts. 
            - If `return_results_directly` is `False`, the detected shifts are merged into the original dataset, and the function returns `None`.

        Raises:
            ValueError: If data is invalid or required parameters are missing
>>>>>>> c6fc662 (Docstring and type fixes)
        """
        results = shifts_detection.compute_shifts(
            data=self.data, 
            var=var, 
            time_dim=time_dim, 
            method=method, 
            output_label_suffix=output_label_suffix, 
            overwrite=overwrite, 
            merge_input=not return_results_directly
        )
        if return_results_directly and isinstance(results, xr.DataArray):
            return results
        elif isinstance(results, xr.Dataset):
            self.data = results 
            return None


    def compute_clusters(
        self,
        var : str,
        method : ClusterMixin,
        shifts_filter_func: Callable[[float], bool],
        var_filter_func: Optional[Callable[[float], bool]] = None,
        shifts_label: Optional[str] = None,
<<<<<<< HEAD
        scaler: Optional[str] = 'StandardScaler',
=======
        scaler: str = 'StandardScaler',
>>>>>>> 341e8af ([Minor breaking changes] Enhancements to Cluster and Shifts Variable Handling)
        output_label_suffix: str = "",
        overwrite: bool = False,
        return_results_directly: bool = False,
<<<<<<< HEAD
        sort_by_size: bool = True,
    ) -> Union[xr.DataArray, None]:
        """Apply clustering to a dataset's temporal shifts using a sklearn-compatible clustering algorithm. 

        >> Args:
            var:
                Name of the variable in the dataset to cluster.
            method:
                The clustering method to use. Choose methods from `sklearn.cluster` or create your by inheriting from `sklearn.base.ClusterMixin`.
            shifts_filter_func:
                A callable used to filter the shifts before clustering, such as `lambda x: np.abs(x)>0.8`. 
            var_filter_func:
                A callable used to filter the primary variable before clustering. Defaults to None.
            shifts_label:
                Name of the variable containing precomputed shifts. Defaults to {var}_dts.
            scaler:
                The scaling method to apply to the data before clustering. Choose between 'StandardScaler', 'MinMaxScaler' and None. Defaults to 'StandardScaler'.
            output_label_suffix:
                A suffix to add to the output label. Defaults to "".
            overwrite:
                Whether to overwrite existing variable. Defaults to False.
            return_results_directly:
                Whether to return the clustering results directly or merge into the original dataset. Defaults to False.
            sort_by_size:
                Whether to reorder clusters by size. Defaults to True.

        >> Returns:
            If `return_results_directly` is `True`, returns an `xarray.DataArray` containing cluster labels for the data 
            points. Otherwise, the clustering results are merged into the original dataset, and the function returns `None`.

        >> Raises:
=======
    ) -> Union[xr.DataArray, None]:
        """Apply clustering to a dataset's temporal shifts using a sklearn-compatible clustering algorithm. 

        Args:
            var: Name of the variable in the dataset to cluster.
            method: The clustering method to use. Choose methods from `sklearn.cluster` or create your by inheriting from `sklearn.base.ClusterMixin`.
            shifts_filter_func: A callable used to filter the shifts before clustering, such as `lambda x: np.abs(x)>0.8`. 
            var_filter_func: A callable used to filter the primary variable before clustering. Defaults to None.
            shifts_label: Name of the variable containing precomputed shifts. Defaults to {var}_dts.
            scaler: The scaling method to apply to the data before clustering. Choose between 'StandardScaler', 'MinMaxScaler' and None. Defaults to 'StandardScaler'.
            output_label_suffix: A suffix to add to the output label. Defaults to "".
            overwrite: Whether to overwrite existing variable. Defaults to False.
            return_results_directly: Whether to return the clustering results directly or merge into the original dataset. Defaults to False.

        Returns:
            If `return_results_directly` is `True`, returns an `xarray.DataArray` containing cluster labels for the data 
            points. Otherwise, the clustering results are merged into the original dataset, and the function returns `None`.

        Raises:
>>>>>>> c6fc662 (Docstring and type fixes)
            ValueError: If data is invalid or required parameters are missing

        """
        results = clustering.compute_clusters(
            data=self.data,
            var=var,
            method=method,
            shifts_filter_func=shifts_filter_func,
            var_filter_func=var_filter_func,
            shifts_label=shifts_label,
            scaler=scaler,
            output_label_suffix=output_label_suffix,
            overwrite=overwrite,
            merge_input=not return_results_directly,
<<<<<<< HEAD
            sort_by_size=sort_by_size,
=======
>>>>>>> c6fc662 (Docstring and type fixes)
        )

        if return_results_directly and isinstance(results, xr.DataArray):
            return results 
        elif isinstance(results, xr.Dataset):
            self.data = results 
            return None
        

    # # ======================================================================
    # #               GET functions (postprocessing)
    # # ======================================================================
<<<<<<< HEAD
<<<<<<< HEAD
    def get_shifts(self, var, label_suffix: str = "") -> xr.DataArray:
        """
        Get shifts xr.DataArray for the specified variable.

        >> Args:
            var : (str)
                Base variable name (e.g. 'temperature', will look for 'temperature_cluster') or custom cluster variable name.
            label_suffix : (str)
                If you added a suffix to the shifts variable, help the function find it. Defaults to "".

        >> Returns:
            xarray.DataArray:
            The shifts xr.DataArray for the specified variable.

        >> Raises:
            ValueError:
                Failed to find valid shifts xr.DataArray for the given var. Note: An xr.DataArray is only considered a shifts label if it contains _dts in its name.
        """
=======
    def get_shifts(self, var) -> xr.DataArray:
=======
    def get_shifts(self, var, label_suffix: str = "") -> xr.DataArray:
>>>>>>> 341e8af ([Minor breaking changes] Enhancements to Cluster and Shifts Variable Handling)
        """
        Get shifts xr.DataArray for the specified variable.

        Args:
            var (str): Either the name of the variable for which shifts have been computed (say temperature) or the name of the custom shifts variable.
            label_suffix (str): If you added a suffix to the shifts variable, help the function find it. Defaults to "".

        Returns:
            xarray.DataArray: The shifts xr.DataArray for the specified variable.

        Raises:
            ValueError: Failed to find valid shifts xr.DataArray for the given var. Note: An xr.DataArray is only considered a shifts label if it contains _dts in its name.
        """
        
        # Check if the variable is a shifts variable
        v = f'{var}{label_suffix}'
        if v in self.data and "_dts" in v:
            return self.data[v]

        # Infer the default shifts variable name
        shifts_var = f'{var}_dts{label_suffix}'
        if shifts_var in self.data:
            return self.data[shifts_var]

        # Tell the user about alternative shifts variables
        all_shift_vars = [v for v in self.data.data_vars if '_dts' in v]
        raise ValueError((f"No shifts variable found for {var} or {shifts_var}. Please first run compute_shifts()." \
            f" Or did you mean to use any of these?: {', '.join(all_shift_vars)}" if all_shift_vars else ""))
        

    def get_clusters(self, var, label_suffix: str = "") -> xr.DataArray:
        """
        Get cluster xr.DataArray for the specified variable.

        Args:
            var (str): Either the name of the variable for which clusters have been computed (say temperature) or the name of the custom cluster variable.
            label_suffix (str): If you added a suffix to the cluster variable, help the function find it. Defaults to "".

        Returns:
            xarray.DataArray: The clusters xr.DataArray for the specified variable.

        Raises:
            ValueError: Failed to find valid cluster xr.DataArray for the given var. An xr.DataArray is only considered a cluster label if it contains _cluster in its name.
        """
        
        # Check if the variable is a cluster variable
        v = f'{var}{label_suffix}'
        if v in self.data and "_cluster" in v:
            return self.data[v]

        # Infer the default cluster variable name
        cluster_var = f'{var}_cluster{label_suffix}'
        if cluster_var in self.data:
            return self.data[cluster_var]

        # Tell the user about alternative cluster variables
        alt_cluster_vars = [v for v in self.data.data_vars if '_cluster' in v]
        raise ValueError((f"No cluster variable found for {var} or {cluster_var}. Please first run compute_clusters()." \
            f" Or did you mean to use any of these?: {', '.join(alt_cluster_vars)}" if alt_cluster_vars else ""))
        

    def get_cluster_counts(self, var, sort=False):
        """Calculate the number of cells (in space and time) in each cluster for a specified variable.

        Each cell may belong to multiple clusters over time. This function computes the number
        of unique cells in each cluster and allows optional sorting of the results.

        Args:
            var: The name of the variable for which cluster counts are computed.
                Requires the dataset to have a corresponding "{var}_cluster" key.
            sort: If True, the resulting dictionary is sorted in descending order
                by the number of cells in each cluster. Defaults to False.

        Returns:
            dict: A dictionary where keys are cluster IDs (as integers) and values are the
                number of unique cells in each cluster.

        Raises:
            ValueError: If cluster information for the specified variable is not found in the dataset.

        Notes:
            - The function counts the number of unique spatial cells that are part of each cluster,
              regardless of the number of time steps they appear in the cluster. i: verify this.
        """

        # TODO: I think this actually returns the number of cells that are part of the cluster in 
        # both space and time, so if the same cell is part of the cluster for several time
        # steps, it adds up. Verify this. 

<<<<<<< HEAD
        if self.data.get(f"{var}_cluster") is None:
            raise ValueError(f"No clusters computed for {var} yet.")
>>>>>>> c6fc662 (Docstring and type fixes)
        
        # Check if the variable is a shifts variable
        v = f'{var}{label_suffix}'
        if v in self.data and "_dts" in v:
            return self.data[v]
=======
        clusters = self.get_clusters(var)
        counts = {}
        for cluster_id in np.unique(clusters):
            timeseries_data = self.timeseries(
                self.data,
                clustering=Clustering(clusters),
                cluster_lbl=[cluster_id],
                masking='always_in_cluster' if cluster_id == -1 else 'spatial', # the spatial mask returns cells that at any point is in cluster_id, so for -1 you would get all cells. Therefore, we need another mask for unclustered cells (i.e. -1).
                how=('per_gridcell') # get time series for each grid cell
            )
            counts[int(cluster_id)] = len(timeseries_data.cell_xy)
>>>>>>> 341e8af ([Minor breaking changes] Enhancements to Cluster and Shifts Variable Handling)

        # Infer the default shifts variable name
        shifts_var = f'{var}_dts{label_suffix}'
        if shifts_var in self.data:
            return self.data[shifts_var]

        # Tell the user about alternative shifts variables
        all_shift_vars: List[str] = [str(data_var) for data_var in self.data.data_vars if '_dts' in str(data_var)]
        raise ValueError((f"No shifts variable found for {var} or {shifts_var}. Please first run compute_shifts()." \
            f" Or did you mean to use any of these?: {', '.join(all_shift_vars)}" if all_shift_vars else ""))
        

    def get_clusters(self, var, label_suffix: str = "") -> xr.DataArray:
        """
        Get cluster xr.DataArray for the specified variable.

        >> Args:
            var : (str)
                Base variable name (e.g. 'temperature', will look for 'temperature_cluster') or custom cluster variable name.
            label_suffix : (str)
                If you added a suffix to the cluster variable, help the function find it. Defaults to "".

        >> Returns:
            xarray.DataArray:
                The clusters xr.DataArray for the specified variable.

        >> Raises:
            ValueError:
                Failed to find valid cluster xr.DataArray for the given var. An xr.DataArray is only considered a cluster label if it contains _cluster in its name.
        """
        
        # Check if the variable is a cluster variable
        v = f'{var}{label_suffix}'
        if v in self.data and "_cluster" in v:
            return self.data[v]

        # Infer the default cluster variable name
        cluster_var = f'{var}_cluster{label_suffix}'
        if cluster_var in self.data:
            return self.data[cluster_var]

        # Tell the user about alternative cluster variables
        alt_cluster_vars: List[str] = [str(data_var) for data_var in self.data.data_vars if '_cluster' in str(data_var)]
        raise ValueError((f"No cluster variable found for {var} or {cluster_var}. Please first run compute_clusters()." \
            f" Or did you mean to use any of these?: {', '.join(alt_cluster_vars)}" if alt_cluster_vars else ""))
        


    def get_cluster_counts(self, var):
        """Returns sorted dictionary with number of cells in both space and time for each cluster.
        
        >> Args:
            var : (str)
                Base variable name (e.g. 'temperature', will look for 'temperature_cluster') or custom cluster variable name.
        
        >> Returns:
            dict: {cluster_id: count}
        """
        counts = {}
        for cluster_id in self.get_clusters(var).cluster_ids:
            count = self.get_cluster_mask(var, cluster_id).sum()
            counts[int(cluster_id)] = int(count)
        
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))


    def get_cluster_ids(self, var):
        """
<<<<<<< HEAD
        Return list of cluster ids sorted by total number of cells in each cluster.

        >> Args:
            var : (str)
                Base variable name (e.g. 'temperature', will look for 'temperature_cluster') or custom cluster variable name.

        >> Returns:
=======
        Return list of cluster ids, optionally sorted by the number of cells in each cluster.

        Args:
            var: Name of the variable in the dataset to get cluster ids for.
            sort: If True, the cluster ids are sorted by the number of cells in each cluster. Defaults to False.

        Returns:
>>>>>>> c6fc662 (Docstring and type fixes)
            list: A list of cluster ids.
        """
        return np.array(list(self.get_cluster_counts(var).keys()))
    

<<<<<<< HEAD
    def get_active_clusters_count_per_timestep(self, var):
        """Get number of active clusters for each timestep.
        
        >> Args:
            var : (str)
                Base variable name (e.g. 'temperature', will look for 'temperature_cluster') or custom cluster variable name.
=======
    def get_cluster_cell_data(self, var, cluster_id):
        """ Returns a list of xr.datasets for each cell that at one point in time is 
        in the speicified cluster. Except if cluster_id=-1, the method will return 
        cells that are always in -1, i.e. cells that remain unclustered throughout time.
        
        Args:
            var: Name of the variable in the dataset to get cluster cell data for.
            cluster_id: The cluster id to get cell data for.

        Returns:
            list: A list of xr.datasets for each cell that at one point in time is in the specified cluster.
        """
>>>>>>> c6fc662 (Docstring and type fixes)

        >> Returns:
            xr.DataArray: Number of active clusters for each timestep.
        """
        clusters = self.get_clusters(var)
<<<<<<< HEAD
        return xr.DataArray(
            [len(np.unique(clusters.sel(**{self.time_dim: t}))) for t in self.data[self.time_dim]],
            coords={self.time_dim: self.data[self.time_dim]},
            dims=[self.time_dim]
        ).rename(f'Number of active clusters for {var}')


    def get_cluster_mask(self, var:str, cluster_id:Union[int,List[int]]) -> xr.DataArray:
        """ Returns a 3D boolean mask (time x space x space) indicating which points belong to the specified cluster(s). 

        >> Args:
            var : (str)
                Base variable name (e.g. 'temperature', will look for 'temperature_cluster') or custom cluster variable name.
            cluster_id : (int or list)
                cluster id(s) to apply the mask for
        >> Returns:
            xr.DataArray: Mask for the cluster label
        """
        clusters = self.get_clusters(var)
        return clusters.isin(cluster_id)
=======
        timeseries_data = self.timeseries(
            self.data,
            clustering=Clustering(clusters),
            cluster_lbl=[cluster_id],
            masking='always_in_cluster' if cluster_id == -1 else 'spatial', # the spatial mask returns cells that at any point is in cluster_id, so for -1 you would get all cells. Therefore, we need another mask for unclustered cells (i.e. -1).
            how="per_gridcell" # get time series for each grid cell
        )
        return [timeseries_data.isel(cell_xy=j) for j in range(len(timeseries_data.cell_xy))]
    
    def timeseries(
            self,
            dataframe, 
            clustering,
            cluster_lbl,
            masking = 'simple',
            how=('aggr',)  # mean, median, std, perc, per_gridcell
        ):
        """Extracts the time series of a cluster label.
        
        Args:
            clustering (toad.core.Clustering): Clustering object of type toad.core.Clustering
            cluster_lbl (int or list): Cluster label to extract the time series from. Can be int or list.
            masking (str): Type of masking to apply. Options:
                * simple: apply the 3D mask to a 3D dataarray 
                * spatial: reduce in the temporal dimension
                * strict: same as spatial, but create new cluster labels for regions 
                  that lie in the spatial overlap of multiple clusters
            how (str or tuple): How to aggregate the time series. Options:
                * mean: mean value
                * median: median value
                * aggr: sum of values
                * std: standard deviation
                * perc: percentile value, eg. how=('perc',0.9)
                * per_gridcell: time series for each grid cell

        Returns:
            xr.DataArray: Time series of the cluster label.
        """
        # TODO: Decide whether to merge this with the TOAD object

        da = clustering._apply_mask_to(dataframe, cluster_lbl, masking=masking)
        tdim, sdims = infer_dims(dataframe) # TODO: this will only work if space dims are (x, y), or (lat, lon), otherwise it crashes and tells you to pass in time dim
>>>>>>> c6fc662 (Docstring and type fixes)

    def apply_cluster_mask(self, var: str, apply_to_var: str, cluster_id: int) -> xr.DataArray:
        """Apply the cluster mask to a variable
        
        >> Args:
            var : (str)
                Base variable name (e.g. 'temperature', will look for 'temperature_cluster') or custom cluster variable name.
            apply_to_var: 
                The variable to apply the mask to
            cluster_id:
                The cluster id to apply the mask for
        
        >> Returns:
            xr.DataArray: The masked variable
        """
        mask = self.get_cluster_mask(var, cluster_id)
        return self.data[apply_to_var].where(mask)


    def get_spatial_cluster_mask(self, var:str, cluster_id:Union[int,List[int]]) -> xr.DataArray:
        """ Returns a 2D boolean mask indicating which grid cells belonged to the specified cluster at any point in time.

        I.e. a grid cell is True if it belonged to the specified cluster at any point in time during the entire timeseries.

        >> Args:
            var : (str)
                Base variable name (e.g. 'temperature', will look for 'temperature_cluster') or custom cluster variable name.
            cluster_id : (int or list)
                cluster id to apply the mask for

        >> Returns:
            xr.DataArray: Mask for the cluster id

        """
        
        # Notify user of better masking for cluster_id = -1
        if contains_value(cluster_id, -1):
            self.logger.info("Hint: If you want to get the mask for unclustered cells, use get_permanent_unclustered_mask() instead.")

        return self.get_cluster_mask(var, cluster_id).any(dim=self.time_dim)


    def apply_spatial_cluster_mask(self, var: str, apply_to_var: str, cluster_id: int) -> xr.DataArray:
        """Apply the spatial cluster mask to a variable
        
        >> Args:
            var : (str)
                Base variable name (e.g. 'temperature', will look for 'temperature_cluster') or custom cluster variable name.
            apply_to_var:
                The variable to apply the mask to
            cluster_id : (int)
                The cluster id to apply the mask for
        
        >> Returns:
            xr.DataArray: All data (regardless of cluster) masked by the spatial extend of the specified cluster.
        """
        mask = self.get_spatial_cluster_mask(var, cluster_id)
        return self.data[apply_to_var].where(mask)


    def apply_temporal_cluster_mask(self, var: str, apply_to_var: str, cluster_id: int) -> xr.DataArray:
        """Apply the temporal cluster mask to a variable
        
        >> Args:
            var : (str)
                Base variable name (e.g. 'temperature', will look for 'temperature_cluster') or custom cluster variable name.
            apply_to_var:
                The variable to apply the mask to
            cluster_id : (int)
                The cluster id to apply the mask for
        
        >> Returns:
            xr.DataArray: All data (regardless of cluster) masked by the temporal extend of the specified cluster.
        """
        mask = self.get_temporal_cluster_mask(var, cluster_id)
        return self.data[apply_to_var].where(mask)


    def get_permanent_cluster_mask(self, var:str, cluster_id:int) -> xr.DataArray:
        """ Create a mask for cells that always have the same cluster label (such as completely unclustered cells by passing -1)
        
        >> Args:
            var : (str)
                Base variable name (e.g. 'temperature', will look for 'temperature_cluster') or custom cluster variable name.
            cluster_id : (int)
                The cluster id

        >> Returns:
            xr.DataArray: Boolean mask where True indicates cells that always belonged to the specified cluster.
        """
        clusters = self.get_clusters(var)
        return (clusters == cluster_id).all(dim=self.time_dim)


    def get_permanent_unclustered_mask(self, var:str) -> xr.DataArray:
        """ Create the spatial mask for cells that are always unclustered (i.e. -1)
        
        >> Args:
            var : (str)
                Base variable name (e.g. 'temperature', will look for 'temperature_cluster') or custom cluster variable name.
            
        >> Returns:
            xr.DataArray: Boolean mask where True indicates cells that were never clustered (always had value -1).
        """
        return self.get_permanent_cluster_mask(var, -1)


    def get_cluster_temporal_density(self, var:str, cluster_id:int) -> xr.DataArray:
        """Calculate the temporal density of a cluster at each grid cell.
        
        >> Args:
            var : (str)
                Base variable name (e.g. 'temperature', will look for 'temperature_cluster') or custom cluster variable name.
            cluster_id : (int)
                The cluster id to calculate density for.
            
        >> Returns:
            xr.DataArray: 2D spatial array where each grid cell contains a fraction (0-1) representing the proportion of timesteps that cell belonged to the specified cluster.
        """
        density = self.get_cluster_mask(var, cluster_id).mean(dim=self.time_dim)
        density = density.rename(f'{density.name}_temporal_density')
        return density


    def get_cluster_spatial_density(self, var:str, cluster_id:int) -> xr.DataArray:
        """Calculate the spatial density of a cluster across all grid cells.
        
        >> Args:
            var:
                Base variable name (e.g. 'temperature', will look for 'temperature_cluster') or custom cluster variable name.
            cluster_id:
                The cluster id to calculate density for.
            
        >> Returns:
            xr.DataArray: 1D timeseries containing the fraction (0-1) of grid cells that belonged to the specified cluster at each timestep.
        """
        density = self.get_cluster_mask(var, cluster_id).mean(dim=self.space_dims)
        density = density.rename(f'{density.name}_spatial_density')
        return density


    def get_temporal_cluster_mask(self, var:str, cluster_id:int) -> xr.DataArray:
        """Calculate a temporal footprint indicating cluster presence at each timestep.
        
        For each timestep, returns a boolean mask indicating whether any grid cell belonged 
        to the specified cluster. This is useful for determining when a cluster was active,
        regardless of its spatial extent.

        >> Args:
            var:
                Base variable name (e.g. 'temperature', will look for 'temperature_cluster') 
                or custom cluster variable name.
            cluster_id:
                The cluster ID to calculate the temporal footprint for.

        >> Returns:
            xr.DataArray: Boolean array with True indicating timesteps where the cluster existed
                somewhere in the spatial domain.
        """
        footprint = self.get_cluster_mask(var, cluster_id).any(dim=self.space_dims)
        footprint = footprint.rename(f'{footprint.name}_temporal_footprint')
        return footprint


    def get_total_spatial_temporal_density(self, var: str) -> xr.DataArray:
        """Calculate the fraction of all grid cells that belong to any cluster at each timestep.
        
        For each timestep, calculates what fraction of all grid cells belong to any cluster,
        excluding noise points (cluster ID -1). This gives a measure of how much of the spatial
        domain is covered by clusters at each point in time.

        >> Args:
            var:
                Base variable name (e.g. 'temperature', will look for 'temperature_cluster') 
                or custom cluster variable name.

        >> Returns:
            xr.DataArray: Fraction (0-1) of grid cells belonging to any cluster at each timestep.
                The output is named '{var}_total_cluster_temporal_density'.
        """
        # Get the mask for all clusters except -1
        non_noise_mask = self.get_cluster_mask(var, -1) == 0
        # Calculate the mean over the spatial dimensions
        density = non_noise_mask.mean(dim=self.space_dims)
        # Rename the result for clarity
        density = density.rename(f'{var}_total_cluster_temporal_density')
        return density
    

    def get_cluster_data(self, var: str, cluster_id: Union[int, List[int]]) -> xr.Dataset:
        """Get raw data for specified cluster(s) with mask applied.

        >> Args:
            var:
                Base variable name (e.g. 'temperature', will look for 'temperature_cluster') or custom cluster variable name.
            cluster_id:
                Single cluster ID or list of cluster IDs
        
        >> Returns:
            xr.Dataset: Full dataset masked by the cluster id

        >> Note: 
            - If cluster_id == -1, returns the unclustered mask.
            - If cluster_id is a list, returns the union of the masks for each cluster id.
        """

        # use the unclustered mask if cluster_id == -1
        if is_equal_to(cluster_id, -1): # checks if cluster_id is a scalar and equals -1
            mask = self.get_permanent_unclustered_mask(var)
        else:
            mask = self.get_cluster_mask(var, cluster_id)

        return self.data.where(mask)


<<<<<<< HEAD
    def _aggregate_spatial(
        self, 
        data: xr.DataArray,
        method: str = "raw",
        percentile: Optional[float] = None
    ) -> xr.DataArray:
        """Aggregate data across spatial dimensions.

        >> Args:
            data:
                Data to aggregate
            method:
                Aggregation method:
                - "mean": Average across space
                - "median": Median across space  
                - "sum": Sum across space
                - "std": Standard deviation across space
                - "percentile": Percentile across space (requires percentile arg)
                - "raw": Return data for each grid cell separately (default).
            percentile:
                Percentile value between 0-1 when using percentile aggregation
=======
class Clustering():
    """ Handle clusterings to allow simplified operation.
    
    Args:
        cluster_label_ds (xr.DataArray): Dataarray with cluster label variable. 
            Cluster labels should be processed:
            * simple: apply the 3D mask to a 3D dataarray
            * spatial: reduce in the temporal dimension
            * strict: same as spatial, but create new cluster labels for regions 
              that lie in the spatial overlap of multiple clusters
        time_dim (str): Dimension in which the abrupt shifts have been detected. 
            Automatically inferred if not provided.

    """
    # TODO: Decide whether to merge this with the TOAD object

    def __init__(
            self,
            cluster_label_da,
            time_dim=None,
            ):
        self.tdim, self.sdims = infer_dims(cluster_label_da, tdim=time_dim)
        self._cluster_labels = cluster_label_da

    def _apply_mask_to(
            self,
            xarr_obj,
            cluster_lbl,
            masking='simple' # spatial, strict
            ):
        """ Apply mask to an xr.DataArray.

        Args:
            xarr_obj (xr.DataArray): xarray object to apply the mask to
            cluster_lbl (int or list): cluster label to apply the mask for
            masking (str, optional): type of masking to apply. Options:
                * simple: apply the 3D mask to a 3D dataarray
                * spatial: reduce in the temporal dimension 
                * strict: same as spatial, but create new cluster labels for regions that lie in the spatial overlap of multiple clusters

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
>>>>>>> c6fc662 (Docstring and type fixes)
        
        >> Returns:
            xr.DataArray: Aggregated data. If method="raw", includes cell_xy dimension.
        """
        if method == "mean":
            return data.mean(dim=self.space_dims)
        elif method == "median": 
            return data.median(dim=self.space_dims)
        elif method == "sum":
            return data.sum(dim=self.space_dims)
        elif method == "std":
            return data.std(dim=self.space_dims)
        elif method == "percentile":
            if percentile is None:
                raise ValueError("percentile argument required for percentile aggregation")
            return data.quantile(percentile, dim=self.space_dims)
        elif method == "raw":
            result = data.stack(cell_xy=self.space_dims).transpose()
            return result.dropna(dim="cell_xy", how="all")
        else:
            raise ValueError(f"Unknown aggregation method: {method}")


<<<<<<< HEAD
    def get_cluster_timeseries(
        self, 
        var: str, 
        cluster_id: Union[int, List[int]],
        cluster_var: Optional[str] = None,
        aggregation: Literal["raw", "mean", "sum", "std", "median", "percentile"] = "raw",
        percentile: Optional[float] = None,
        normalize: Optional[Literal["first", "max", "last"]] = None,
        keep_full_timeseries: bool = True
    ) -> xr.DataArray:
        """Get time series for cluster, optionally aggregated across space.
        
        >> Args:
            var:
                Variable name to extract time series from
            cluster_var: 
                Variable name to extract cluster ids from. Default to None and is attemped to be inferred from var.
            cluster_id: 
                Single cluster ID or list of cluster IDs
            aggregation: 
                How to aggregate spatial data:
                - "mean": Average across space
                - "median": Median across space  
                - "sum": Sum across space
                - "std": Standard deviation across space
                - "percentile": Percentile across space (requires percentile arg)
                - "raw": Return data for each grid cell separately
            percentile: 
                Percentile value between 0-1 when using percentile aggregation
            normalize:
                - "first": Normalize by the first non-zero, non-nan timestep
                - "max": Normalize by the maximum value
                - "last": Normalize by the last non-zero, non-nan timestep
                - "none": Do not normalize
=======
        Args:
            cluster_lbl (int or list): cluster label to apply the mask for

        Returns:
            xr.DataArray: Mask for the cluster label
        """
        return self._cluster_labels.isin(cluster_lbl)
>>>>>>> c6fc662 (Docstring and type fixes)

            keep_full_timeseries: 
                If True, returns full time series of cluster cells. If False, only returns time series of cells when they were in the cluster. Defaults to True.

        >> Returns:
            xr.DataArray: Time series as xarray DataArray. If aggregation="raw", includes cell_xy dimension.
        """
        cluster_var = cluster_var if cluster_var else var
        
<<<<<<< HEAD
        if keep_full_timeseries:
            # Handle unclustered case (-1)
            if is_equal_to(cluster_id, -1):
                mask = self.get_permanent_unclustered_mask(cluster_var)
            else:
                mask = self.get_spatial_cluster_mask(cluster_var, cluster_id)
        else:
            # Original behavior - only keep timesteps where cells are in cluster
            mask = self.get_cluster_mask(cluster_var, cluster_id)
=======
        Args:
            cluster_lbl (int or list): cluster label to apply the mask for

        Returns:
            xr.DataArray: Mask for the cluster label
>>>>>>> c6fc662 (Docstring and type fixes)

        # Apply mask
        data = self.data[var].where(mask)

        # First aggregate spatially
        data = self._aggregate_spatial(data, aggregation, percentile)
        
<<<<<<< HEAD
        # Normalise
        if normalize:
            if normalize == "first":
                filtered = data.where(data != 0).dropna(dim=self.time_dim)
                scalar = filtered.isel({self.time_dim: 0}) if len(filtered[self.time_dim]) > 0 else np.nan # get first non-zero, non-nan timestep if exists
            elif normalize == "max":
                scalar = float(data.max())
            elif normalize == "last":
                filtered = data.where(data != 0).dropna(dim=self.time_dim)
                scalar = filtered.isel({self.time_dim: -1}) if len(filtered[self.time_dim]) > 0 else np.nan # get last non-zero, non-nan timestep if exists
            else:
                raise ValueError(f"Unknown normalization method: {normalize}")
        
            if scalar == 0 or np.isnan(scalar) or scalar is None:
                self.logger.error(f"Failed to normalise by {normalize} = {scalar}")
            else:
                normalized = data / scalar
                data = normalized.where(np.isfinite(normalized))

        return data
=======
        Args:
            cluster_lbl (int or list): cluster label to apply the mask for
            how (str or tuple): how to calculate the temporal properties

        Returns:
            Temporal properties of the cluster label
        """
        # TODO: verify this works and move to postprocessing/stats.py
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

        Args:
            cluster_lbl (int or list): Cluster label to apply the mask for.
            masking (str, optional): Type of masking to apply:
                * simple: apply the 3D mask to a 3D dataarray
                * spatial: reduce in the temporal dimension  
                * strict: same as spatial, but create new cluster labels for regions that lie in the spatial overlap of multiple clusters
            how (tuple): How to calculate the spatial properties.

        Returns:
            Spatial properties of the cluster label.

        """
        # TODO: verify this works and move to postprocessing/stats.py
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

        Args:
            xarr_obj (xarray.DataArray): xarray object to apply the mask to.
            cluster_lbl (int, list): cluster label to apply the mask for.

        Returns:
            xarray.DataArray: Masked xarray object.
        """
        return self._apply_mask_to(xarr_obj,cluster_lbl)

    # End of Clustering object
>>>>>>> c6fc662 (Docstring and type fixes)
