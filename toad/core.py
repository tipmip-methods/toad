import logging
import xarray as xr
import numpy as np
from typing import List, Union, Optional, Literal
import os
from sklearn.base import ClusterMixin
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
)

from toad import (
    shifts_detection,
    clustering,
    postprocessing,
    visualisation,
    preprocessing,
)
from toad.utils import get_space_dims, is_equal_to, contains_value
from toad.regridding.base import BaseRegridder
from toad.regridding import HealPixRegridder


class TOAD:
    """
    Main object for interacting with TOAD.
    TOAD (Tippping and Other Abrupt events Detector) is a framework for detecting and clustering spatio-temporal patterns in spatio-temporal data.

    >> Args:
        data : (xr.Dataset or str)
            The input data. If a string, it is interpreted as a path to a netCDF file.
        log_level : (str)
            The logging level. Choose from 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'. Defaults to 'WARNING'.
    """

    data: xr.Dataset

    def __init__(
        self, data: Union[xr.Dataset, str], time_dim: str = "time", log_level="WARNING"
    ):
        # load data from path if string
        if isinstance(data, str):
            if not os.path.exists(data):
                raise ValueError(f"File {data} does not exist.")
            self.data = xr.open_dataset(data)
            self.data.attrs["title"] = os.path.basename(data).split(".")[
                0
            ]  # store path as title for saving toad file later
        elif isinstance(data, (xr.Dataset, xr.DataArray)):
            self.data = data  # Original data

        # rename longitude and latitude to lon and lat
        if "longitude" in self.data.dims:
            self.data = self.data.rename({"longitude": "lon"})
            logging.info("Renamed longitude to lon")
        if "latitude" in self.data.dims:
            self.data = self.data.rename({"latitude": "lat"})
            logging.info("Renamed latitude to lat")

        # TODO: check that self.space_dims returns two values only, if not, raise eror and tell user to specify space_dims manually (new param in TOAD init)
        # TODO: Check that the time_dim exists, otherwise raise error and tell user to specify time_dim manually (param in TOAD init)
        # TODO: warn user if their variables contain _dts or _cluster, as variables with such names have special meaning in TOAD and may be overwritten

        # Save time dim for later
        self.time_dim = time_dim

        # Initialize the logger for the TOAD object
        self.logger = logging.getLogger("TOAD")
        self.logger.propagate = False  # Prevent propagation to the root logger :: i.e. prevents dupliate messages
        self.set_log_level(log_level)

    # # ======================================================================
    # #               Module functions
    # # ======================================================================
    def preprocess(self) -> preprocessing.Preprocess:
        """Access preprocessing methods."""
        return preprocessing.Preprocess(self)

    def cluster_stats(self, var: str) -> postprocessing.ClusterStats:
        """Access cluster statistical methods.


        >> Args:
            var : (str)
                Base variable name (e.g. 'temperature', will look for 'temperature_cluster') or custom cluster variable name.

        >> Returns:
            toad.postprocessing.cluster_stats.ClusterStats: ClusterStats object
        """
        return postprocessing.ClusterStats(self, var)

    def aggregation(self) -> postprocessing.Aggregation:
        """Access aggregation methods."""
        return postprocessing.Aggregation(self)

    def plotter(
        self, config: Optional[visualisation.PlotConfig] = None
    ) -> visualisation.TOADPlotter:
        """Access plotting methods."""
        return visualisation.TOADPlotter(self, config=config)

    # # ======================================================================
    # #               SET functions
    # # ======================================================================

    def set_log_level(self, level):
        """Sets the logging level for the TOAD logger.

        >> Args:
            level:
                The logging level to set. Choose from 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.

        >> Examples:
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
            raise ValueError(
                "Invalid log level. Choose from 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'"
            )

        self.logger.setLevel(getattr(logging, level))

        # Only add a handler if there are no handlers yet (to avoid duplicate messages)
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter("%(levelname)s: %(message)s")
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
        output_label_suffix: str = "",
        overwrite: bool = False,
        return_results_directly: bool = False,
    ) -> Union[xr.DataArray, None]:
        """Apply an abrupt shift detection algorithm to a dataset along the specified temporal dimension.

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
        """
        results = shifts_detection.compute_shifts(
            data=self.data,
            var=var,
            time_dim=self.time_dim,
            method=method,
            output_label_suffix=output_label_suffix,
            overwrite=overwrite,
            merge_input=not return_results_directly,
        )
        if return_results_directly and isinstance(results, xr.DataArray):
            return results
        elif isinstance(results, xr.Dataset):
            self.data = results
            return None

    def compute_clusters(
        self,
        var: str,
        method: ClusterMixin,
        shift_threshold: float = 0.8,
        shift_sign: str = "absolute",
        shifts_label: Optional[str] = None,
        scaler: Optional[
            Union[StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler]
        ] = StandardScaler(),
        time_scale_factor: Optional[float] = None,
        regridder: Optional[BaseRegridder] = None,
        output_label_suffix: str = "",
        overwrite: bool = False,
        return_results_directly: bool = False,
        sort_by_size: bool = True,
    ) -> Union[xr.DataArray, None]:
        """Apply clustering to a dataset's temporal shifts using a sklearn-compatible clustering algorithm.

        >> Args:
            var:
                Name of the variable in the dataset to cluster.
            method:
                The clustering method to use. Choose methods from `sklearn.cluster` or create your by inheriting from `sklearn.base.ClusterMixin`.
            shift_threshold:
                The threshold for the shift magnitude. Defaults to 0.8.
            shift_sign:
                The sign of the shift. Options are "absolute", "positive", "negative". Defaults to "absolute".
            shifts_label:
                Name of the variable containing precomputed shifts. Defaults to {var}_dts.
            scaler:
                The scaling method to apply to the data before clustering. StandardScaler(), MinMaxScaler(), RobustScaler() and MaxAbsScaler() from sklearn.preprocessing are supported. Defaults to StandardScaler().
            time_scale_factor:
                The factor to scale the time values by. Defaults to None.
            regridder:
                The regridding method to use from `toad.clustering.regridding`. 
                Defaults to None. If None and coordinates are lat/lon, a HealPixRegridder will be created automatically.
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
            ValueError: If data is invalid or required parameters are missing

        >> Notes:
            For global datasets, use `toad.regridding.HealPixRegridder` to ensure equal spacing between data points and prevent biased clustering at high latitudes.

        """
        results = clustering.compute_clusters(
            data=self.data,
            var=var,
            method=method,
            shift_threshold=shift_threshold,
            shift_sign=shift_sign,
            shifts_label=shifts_label,
            time_dim=self.time_dim,
            space_dims=self.space_dims,
            scaler=scaler,
            time_scale_factor=time_scale_factor,
            regridder=regridder,
            output_label_suffix=output_label_suffix,
            overwrite=overwrite,
            merge_input=not return_results_directly,
            sort_by_size=sort_by_size,
        )

        if return_results_directly and isinstance(results, xr.DataArray):
            return results
        elif isinstance(results, xr.Dataset):
            self.data = results
            return None

    # # ======================================================================
    # #               GET functions (postprocessing)
    # # ======================================================================

    @property
    def space_dims(self):
        return get_space_dims(self.data, self.time_dim)

    @property
    def base_vars(self) -> np.ndarray:
        return np.array(
            [
                x
                for x in list(self.data.data_vars.keys())
                if "_dts" not in x and "_cluster" not in x
            ]
        )

    @property
    def shift_vars(self) -> np.ndarray:
        return np.array([x for x in list(self.data.data_vars.keys()) if "_dts" in x])

    @property
    def cluster_vars(self) -> np.ndarray:
        return np.array(
            [x for x in list(self.data.data_vars.keys()) if "_cluster" in x]
        )

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

        # Check if the variable is a shifts variable
        v = f"{var}{label_suffix}"
        if v in self.data and "_dts" in v:
            return self.data[v]

        # Infer the default shifts variable name
        shifts_var = f"{var}_dts{label_suffix}"
        if shifts_var in self.data:
            return self.data[shifts_var]

        # Tell the user about alternative shifts variables
        all_shift_vars: List[str] = [
            str(data_var) for data_var in self.data.data_vars if "_dts" in str(data_var)
        ]
        raise ValueError(
            (
                f"No shifts variable found for {var} or {shifts_var}. Please first run compute_shifts()."
                f" Or did you mean to use any of these?: {', '.join(all_shift_vars)}"
                if all_shift_vars
                else ""
            )
        )

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
        v = f"{var}{label_suffix}"
        if v in self.data and "_cluster" in v:
            return self.data[v]

        # Infer the default cluster variable name
        cluster_var = f"{var}_cluster{label_suffix}"
        if cluster_var in self.data:
            return self.data[cluster_var]

        # Tell the user about alternative cluster variables
        alt_cluster_vars: List[str] = [
            str(data_var)
            for data_var in self.data.data_vars
            if "_cluster" in str(data_var)
        ]
        raise ValueError(
            (
                f"No cluster variable found for {var} or {cluster_var}. Please first run compute_clusters()."
                f" Or did you mean to use any of these?: {', '.join(alt_cluster_vars)}"
                if alt_cluster_vars
                else ""
            )
        )

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
        Return list of cluster ids sorted by total number of cells in each cluster.

        >> Args:
            var : (str)
                Base variable name (e.g. 'temperature', will look for 'temperature_cluster') or custom cluster variable name.

        >> Returns:
            list: A list of cluster ids.
        """
        return np.array(list(self.get_cluster_counts(var).keys()))

    def get_active_clusters_count_per_timestep(self, var):
        """Get number of active clusters for each timestep.

        >> Args:
            var : (str)
                Base variable name (e.g. 'temperature', will look for 'temperature_cluster') or custom cluster variable name.

        >> Returns:
            xr.DataArray: Number of active clusters for each timestep.
        """
        clusters = self.get_clusters(var)
        return xr.DataArray(
            [
                len(np.unique(clusters.sel(**{self.time_dim: t})))
                for t in self.data[self.time_dim]
            ],
            coords={self.time_dim: self.data[self.time_dim]},
            dims=[self.time_dim],
        ).rename(f"Number of active clusters for {var}")

    def get_cluster_mask(
        self, var: str, cluster_id: Union[int, List[int]]
    ) -> xr.DataArray:
        """Returns a 3D boolean mask (time x space x space) indicating which points belong to the specified cluster(s).

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

    def apply_cluster_mask(
        self, var: str, apply_to_var: str, cluster_id: int
    ) -> xr.DataArray:
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

    def get_spatial_cluster_mask(  # TODO rename to get_cluster_mask_spatial
        self, var: str, cluster_id: Union[int, List[int]]
    ) -> xr.DataArray:
        """Returns a 2D boolean mask indicating which grid cells belonged to the specified cluster at any point in time.

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
            self.logger.info(
                "Hint: If you want to get the mask for unclustered cells, use get_permanent_unclustered_mask() instead."
            )

        return self.get_cluster_mask(var, cluster_id).any(dim=self.time_dim)

    def apply_spatial_cluster_mask(
        self, var: str, apply_to_var: str, cluster_id: int
    ) -> xr.DataArray:
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

    def apply_temporal_cluster_mask(
        self, var: str, apply_to_var: str, cluster_id: int
    ) -> xr.DataArray:
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

    def get_permanent_cluster_mask(self, var: str, cluster_id: int) -> xr.DataArray:
        """Create a mask for cells that always have the same cluster label (such as completely unclustered cells by passing -1)

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

    def get_permanent_unclustered_mask(self, var: str) -> xr.DataArray:
        """Create the spatial mask for cells that are always unclustered (i.e. -1)

        >> Args:
            var : (str)
                Base variable name (e.g. 'temperature', will look for 'temperature_cluster') or custom cluster variable name.

        >> Returns:
            xr.DataArray: Boolean mask where True indicates cells that were never clustered (always had value -1).
        """
        return self.get_permanent_cluster_mask(var, -1)

    def get_cluster_temporal_density(self, var: str, cluster_id: int) -> xr.DataArray:
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
        density = density.rename(f"{density.name}_temporal_density")
        return density

    def get_cluster_spatial_density(self, var: str, cluster_id: int) -> xr.DataArray:
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
        density = density.rename(f"{density.name}_spatial_density")
        return density

    def get_temporal_cluster_mask(
        self, var: str, cluster_id: int
    ) -> xr.DataArray:  # TODO rename to get_cluster_mask_temporal
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
        footprint = footprint.rename(f"{footprint.name}_temporal_footprint")
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
        density = density.rename(f"{var}_total_cluster_temporal_density")
        return density

    def get_cluster_data(
        self, var: str, cluster_id: Union[int, List[int]]
    ) -> xr.Dataset:
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
        if is_equal_to(
            cluster_id, -1
        ):  # checks if cluster_id is a scalar and equals -1
            mask = self.get_permanent_unclustered_mask(var)
        else:
            mask = self.get_cluster_mask(var, cluster_id)

        return self.data.where(mask)

    def _aggregate_spatial(
        self,
        data: xr.DataArray,
        method: str = "raw",
        percentile: Optional[float] = None,
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
                raise ValueError(
                    "percentile argument required for percentile aggregation"
                )
            return data.quantile(percentile, dim=self.space_dims)
        elif method == "raw":
            result = data.stack(cell_xy=self.space_dims).transpose()
            return result.dropna(dim="cell_xy", how="all")
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def get_cluster_timeseries(
        self,
        var: str,
        cluster_id: Union[int, List[int]], # TODO: rename to cluster_ids ? 
        cluster_var: Optional[str] = None,
        aggregation: Literal[
            "raw", "mean", "sum", "std", "median", "percentile"
        ] = "raw",
        percentile: Optional[float] = None,
        normalize: Optional[Literal["first", "max", "last"]] = None,
        keep_full_timeseries: bool = True,
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

            keep_full_timeseries:
                If True, returns full time series of cluster cells. If False, only returns time series of cells when they were in the cluster. Defaults to True.

        >> Returns:
        """
        cluster_var = cluster_var if cluster_var else var

        if keep_full_timeseries:
            # Handle unclustered case (-1)
            if is_equal_to(cluster_id, -1):
                mask = self.get_permanent_unclustered_mask(cluster_var)
            else:
                mask = self.get_spatial_cluster_mask(cluster_var, cluster_id)
        else:
            # Original behavior - only keep timesteps where cells are in cluster
            mask = self.get_cluster_mask(cluster_var, cluster_id)

        # Apply mask
        data = self.data[var].where(mask)

        # First aggregate spatially
        data = self._aggregate_spatial(data, aggregation, percentile)

        # Normalise
        if normalize:
            if normalize == "first":
                filtered = data.where(data != 0).dropna(dim=self.time_dim)
                # TODO: this crashes
                scalar = (
                    filtered.isel({self.time_dim: 0})
                    if len(filtered[self.time_dim]) > 0
                    else np.nan
                )  # get first non-zero, non-nan timestep if exists
            elif normalize == "max":
                scalar = float(data.max())
            elif normalize == "last":
                # TODO: this crashes
                filtered = data.where(data != 0).dropna(dim=self.time_dim)
                scalar = (
                    filtered.isel({self.time_dim: -1})
                    if len(filtered[self.time_dim]) > 0
                    else np.nan
                )  # get last non-zero, non-nan timestep if exists
            else:
                raise ValueError(f"Unknown normalization method: {normalize}")

            if scalar == 0 or np.isnan(scalar) or scalar is None:
                self.logger.error(f"Failed to normalise by {normalize} = {scalar}")
            else:
                normalized = data / scalar
                data = normalized.where(np.isfinite(normalized))

        return data

    # end of TOAD object


@xr.register_dataarray_accessor("toad")
class TOADAccessor:
    """Accessor for xarray DataArrays providing TOAD-specific functionality."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def to_timeseries(self):
        """Convert spatial data to timeseries format by stacking spatial dimensions.

        Returns:
            DataArray with dimensions [time, cell_xy] suitable for timeseries plotting.

        Example:
            >>> data.toad.to_timeseries().plot.line(x="time", add_legend=False, color='k', alpha=0.1);
        """
        td = TOAD(self._obj)
        # Get all dims except time dim
        non_time_dims = [d for d in self._obj.dims if d != td.time_dim]

        return (
            self._obj.stack(cell_xy=non_time_dims)
            .transpose("cell_xy", td.time_dim)
            .dropna(dim="cell_xy", how="all")
        )
