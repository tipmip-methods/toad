import logging
import os
from collections.abc import Callable
from typing import List, Literal, Optional, Union

import numpy as np
import optuna
import sklearn.cluster
import xarray as xr
from sklearn.base import ClusterMixin
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from toad import (
    clustering,
    postprocessing,
    preprocessing,
    shifts,
    visualisation,
)
from toad.clustering.optimising import (
    default_hdbscan_optimisation_params,
)
from toad.regridding.base import BaseRegridder
from toad.utils import (
    _attrs,
    contains_value,
    detect_latlon_names,
    get_space_dims,
    is_equal_to,
)


class TOAD:
    """Main object for interacting with TOAD.

    TOAD (Tippping and Other Abrupt events Detector) is a framework for detecting and clustering spatio-temporal patterns in spatio-temporal data.

    Args:
        data: The input data. Can be either an xarray Dataset or a path to a netCDF file.
        time_dim: The name of the time dimension. Defaults to 'time'.
        log_level: The logging level. Choose from 'DEBUG', 'INFO', 'WARNING', 'ERROR',
            'CRITICAL'. Defaults to 'INFO'.
        engine: The engine to use to open the netCDF file. Defaults to 'netcdf4'.

    Raises:
        ValueError: If the input file path does not exist or if data dimensions are not 3D.
    """

    data: xr.Dataset

    def __init__(
        self,
        data: Union[xr.Dataset, str],
        time_dim: str = "time",
        log_level: str = "INFO",
        engine: str = "netcdf4",
    ):
        # load data from path if string
        if isinstance(data, str):
            if not os.path.exists(data):
                raise ValueError(f"File {data} does not exist.")
            self.data = xr.open_dataset(data, engine=engine)
            self.data.attrs["title"] = os.path.basename(data).split(".")[
                0
            ]  # store path as title for saving toad file later
            self.path = data  # store path
        elif isinstance(data, (xr.DataArray)):
            self.data = data.to_dataset()  # convert to dataset if data is a DataArray
        elif isinstance(data, (xr.Dataset)):
            self.data = data  # Original data

        # Initialize the logger for the TOAD object
        self.logger = logging.getLogger("TOAD")
        self.logger.propagate = False  # Prevent propagation to the root logger :: i.e. prevents dupliate messages
        self.set_log_level(log_level)

        if len(self.data.dims) != 3:
            raise ValueError("Data must be 3-dimensional: time/forcing x space x space")

        # rename longitude and latitude to lon and lat
        if "longitude" in self.data.dims:
            self.data = self.data.rename({"longitude": "lon"})
            self.logger.info("Renamed longitude to lon")
        if "latitude" in self.data.dims:
            self.data = self.data.rename({"latitude": "lat"})
            self.logger.info("Renamed latitude to lat")

        lat, lon = detect_latlon_names(self.data)
        if (lat and lat not in self.data.dims) and (lon and lon not in self.data.dims):
            self.logger.info(
                "Found lat/lon coordinates (not dimensions). TOAD will use these for clustering and plotting instead of native dimensions. Drop lat/lon variables to use native coordinates."
            )

        # Save time dim for later
        self.time_dim = time_dim
        assert self.time_dim in self.data.dims, (
            f"Time dimension {self.time_dim} not found in data."
        )

    def _repr_html_(self):
        """Representation of the TOAD object in html with collapsible hierarchy."""
        # TODO: maybe show method params here?

        # Generate a unique instance ID
        import uuid

        instance_id = str(uuid.uuid4()).replace("-", "")

        # Get the xarray dataset HTML representation
        ds_repr = self.data._repr_html_()

        # Build hierarchy tree
        hierarchy = {}

        # Start with base variables
        for base_var in self.base_vars:
            hierarchy[base_var] = {"shifts": [], "clusters": []}

        # Add shift variables and their relationships
        for shift_var in self.shift_vars:
            shift_data = self.data[shift_var]
            base_var = shift_data.attrs.get(
                _attrs.BASE_VARIABLE, shift_var.split("_dts")[0]
            )

            if base_var not in hierarchy:
                hierarchy[base_var] = {"shifts": [], "clusters": []}

            hierarchy[base_var]["shifts"].append({"name": shift_var, "clusters": []})

        # Add cluster variables and their relationships
        for cluster_var in self.cluster_vars:
            cluster_data = self.data[cluster_var]
            base_var = cluster_data.attrs.get(_attrs.BASE_VARIABLE)
            shifts_var = cluster_data.attrs.get(_attrs.SHIFTS_VARIABLE)

            if base_var and base_var in hierarchy:
                if shifts_var:
                    # Find the specific shift variable and add cluster to it
                    for shift_info in hierarchy[base_var]["shifts"]:
                        if shift_info["name"] == shifts_var:
                            shift_info["clusters"].append(cluster_var)
                            break
                else:
                    # Fallback: add to base variable clusters
                    hierarchy[base_var]["clusters"].append(cluster_var)

        # Generate HTML for the hierarchy
        variable_table = ""
        if any(
            len(info["shifts"]) > 0 or len(info["clusters"]) > 0
            for info in hierarchy.values()
        ):
            hierarchy_html = []
            for base_var in sorted(hierarchy.keys()):
                info = hierarchy[base_var]

                # Count total derived variables
                shift_count = len(info["shifts"])
                cluster_count = sum(len(s["clusters"]) for s in info["shifts"]) + len(
                    info["clusters"]
                )

                if shift_count == 0 and cluster_count == 0:
                    continue  # Skip base variables with no derived variables

                # Base variable row
                base_id = f"{instance_id}_base_{base_var.replace('.', '_')}"
                hierarchy_html.append(f"""
                <div style="margin: 2px 0;">
                    <span onclick="toggleSection_{instance_id}('{base_id}')" style="cursor: pointer; user-select: none;">
                        <span id="{base_id}_arrow" style="font-family: monospace; font-weight: bold;">▶</span>
                        <span style="color: black; background-color: #A8D5FF; padding: 2px 4px; border-radius: 4px;">base var</span> {base_var}
                        <span style="opacity: 0.5; font-size: 0.85em;">
                            ({shift_count} shifts, {cluster_count} clusterings)
                        </span>
                    </span>
                    <div id="{base_id}_content" style="display: none; margin-left: 20px; margin-top: 5px;">
                """)

                # Shift variables
                for shift_info in info["shifts"]:
                    shift_var = shift_info["name"]
                    shift_clusters = shift_info["clusters"]
                    shift_id = f"{instance_id}_shift_{shift_var.replace('.', '_')}"

                    if shift_clusters:
                        # Shift variable with clusters (expandable)
                        hierarchy_html.append(f"""
                        <div style="margin: 4px 0;">
                            <span onclick="toggleSection_{instance_id}('{shift_id}')" style="cursor: pointer; user-select: none;">
                                <span id="{shift_id}_arrow" style="font-family: monospace; font-weight: bold;">▶</span>
                                <span style="color: black; background-color: #FFE0A3; padding: 2px 4px; border-radius: 4px;">shifts var</span> {shift_var} <span style="opacity: 0.5; font-size: 0.85em;">({len(shift_clusters)} clusterings)</span>
                            </span>
                            <div id="{shift_id}_content" style="display: none; margin-left: 20px; margin-top: 3px;">
                        """)

                        # Cluster variables under this shift
                        for cluster_var in shift_clusters:
                            n_clusters = self.data[cluster_var].attrs.get(
                                _attrs.CLUSTER_IDS
                            )
                            n_clusters = (
                                len(n_clusters) - 1 if n_clusters is not None else 0
                            )  # -1 to remove noise cluster
                            hierarchy_html.append(f"""
                            <div style="margin-left: 12px; padding: 2px 0px;">
                                <span style="color: black; background-color: #B8E6C1; padding: 2px 4px; border-radius: 4px;">cluster var</span> {cluster_var} <span style="opacity: 0.5; font-size: 0.85em;">({n_clusters} clusters)</span>
                            </div>
                            """)

                        hierarchy_html.append("</div></div>")
                    else:
                        # Shift variable without clusters
                        hierarchy_html.append(f"""
                        <div style="margin: 2px 0;">
                            <span style="font-family: monospace; font-weight: bold; opacity: 0;">▶</span>
                            <span style="color: black; background-color: #FFE0A3; padding: 2px 4px; border-radius: 4px;">shifts var</span> {shift_var}  <span style="opacity: 0.5; font-size: 0.85em;">({len(shift_clusters)} clusterings)</span>
                        </div>
                        """)
                hierarchy_html.append("</div></div>")

            variable_table = f"""
            <div style='margin: 10px 0px;'>
                <h4 style="margin: 5px 0; font-size: 1.1em;">Variable Hierarchy:</h4>
                <div style="font-family: monospace; font-size: 1.0em; border: 1px solid #ddd; padding: 10px; line-height: 1.4;">
                    {"".join(hierarchy_html)}
                </div>
            </div>
            
            <script>
            function toggleSection_{instance_id}(sectionId) {{
                const content = document.getElementById(sectionId + '_content');
                const arrow = document.getElementById(sectionId + '_arrow');
                
                if (content.style.display === 'none') {{
                    content.style.display = 'block';
                    arrow.textContent = '▼';
                }} else {{
                    content.style.display = 'none';
                    arrow.textContent = '▶';
                }}
            }}

            // Auto-expand logic
            function autoExpand_{instance_id}() {{
                let visibleClusterings = 0;
                const maxVisible = 10;
                
                // First count total clusterings in each base section
                const baseSections = document.querySelectorAll('[id^="{instance_id}_base_"][id$="_arrow"]');
                const sectionCounts = [];
                
                // Get counts for each base section
                baseSections.forEach(baseArrow => {{
                    const baseId = baseArrow.id.replace('_arrow', '');
                    const baseContent = document.getElementById(baseId + '_content');
                    const clusterCount = baseContent.querySelectorAll('[style*="background-color: lightgreen"]').length;
                    sectionCounts.push({{baseId, clusterCount}});
                }});
                
                // Sort sections by cluster count (ascending) to expand smaller sections first
                sectionCounts.sort((a, b) => a.clusterCount - b.clusterCount);
                
                // Expand sections until we hit the limit
                for (const {{baseId, clusterCount}} of sectionCounts) {{
                    if (visibleClusterings + clusterCount <= maxVisible) {{
                        const baseContent = document.getElementById(baseId + '_content');
                        const baseArrow = document.getElementById(baseId + '_arrow');
                        
                        // Expand base section
                        baseContent.style.display = 'block';
                        baseArrow.textContent = '▼';
                        
                        // Expand all shift sections within
                        const shiftSections = baseContent.querySelectorAll('[id^="{instance_id}_shift_"][id$="_arrow"]');
                        shiftSections.forEach(shiftArrow => {{
                            const shiftId = shiftArrow.id.replace('_arrow', '');
                            const shiftContent = document.getElementById(shiftId + '_content');
                            shiftArrow.textContent = '▼';
                            shiftContent.style.display = 'block';
                        }});
                        
                        visibleClusterings += clusterCount;
                    }}
                }}
            }}

            // Run auto-expand when the notebook cell is rendered
            autoExpand_{instance_id}();
            </script>
            """

        # Try to load and encode the TOAD logo
        logo_html = ""
        try:
            import base64
            import os

            current_dir = os.path.dirname(__file__)
            logo_path = os.path.abspath(
                os.path.join(
                    current_dir, "..", "docs", "source", "resources", "toad.png"
                )
            )

            if os.path.exists(logo_path):
                with open(logo_path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode()
                    logo_html = f'<img src="data:image/png;base64,{img_data}" style="height: 40px; margin-right: 10px; vertical-align: middle;">'
        except Exception:
            pass

        # Wrap everything in a TOAD container
        html = f"""
        <div style='padding: 12px'>
            <h2 style='margin-bottom: 0px; display: flex; align-items: center;'>{logo_html}TOAD Object</h2>
            {variable_table}
            <p style='font-size: 0.9em; margin: 16px 0;'>Hint: to access the xr.dataset call <code>td.data</code></p>
            {ds_repr}
        </div>
        """

        return html

    # # ======================================================================
    # #               Module functions
    # # ======================================================================
    def preprocess(self) -> preprocessing.Preprocess:
        """Access preprocessing methods."""
        return preprocessing.Preprocess(self)

    def cluster_stats(self, var: str) -> postprocessing.ClusterStats:
        """Access cluster statistical methods.

        Args:
            var: Base variable name (e.g. 'temperature', will look for 'temperature_cluster')
                or custom cluster variable name.

        Returns:
            ClusterStats object for analyzing cluster statistics.
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

    def set_log_level(self, level: str):
        """Sets the logging level for the TOAD logger.

        Sets the logging level and configures handlers for the TOAD logger instance.
        Available levels are 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.

        Examples:
            Used like this:
                >>> logger.debug("This is a debug message.")
                >>> logger.info("This is an info message.")
                >>> logger.warning("This is a warning message.")
                >>> logger.error("This is an error message.")
                >>> logger.critical("This is a critical message.")

            In sub-modules get logger like this:
                >>> logger = logging.getLogger("TOAD")

        Args:
            level: The logging level to set

        Raises:
            ValueError: If level is not one of the valid logging levels
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

        self.logger.debug(f"Logging level set to {level}")

    # # ======================================================================
    # #               COMPUTE functions
    # # ======================================================================

    def compute_shifts(
        self,
        var: str | None = None,
        method: shifts.ShiftsMethod = shifts.ASDETECT(),
        output_label_suffix: str = "",
        overwrite: bool = False,
        return_results_directly: bool = False,
    ) -> Union[xr.DataArray, None]:
        """Apply an abrupt shift detection algorithm to a dataset along the specified temporal dimension.

        Args:
            var: Name of the base variable to analyze for abrupt shifts. If None and only one base variable exists,
                that variable will be used automatically. If None and multiple base variables exist, raises a ValueError.
                Defaults to None.
            method: The abrupt shift detection algorithm to use. Choose from predefined method objects in `toad.shifts` (e.g., `ASDETECT`),
                or create your own by subclassing `ShiftsMethod` from `toad.shifts`. Defaults to `ASDETECT()`.
            output_label_suffix: A suffix to add to the output label. Defaults to `""`.
            overwrite: Whether to overwrite existing variable. Defaults to `False`.
            return_results_directly: Whether to return the detected shifts directly or merge into the original dataset. Defaults to `False`.

        Returns:
            If `return_results_directly` is True, returns an `xarray.DataArray` containing the detected shifts.
            If `return_results_directly` is False, the detected shifts are merged into the original dataset, and the function returns `None`.

        Raises:
            ValueError: If data is invalid or required parameters are missing
        """

        results = shifts.compute_shifts(
            data=self.data,
            var=self._get_base_var_if_none(var),
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
        var: str | None = None,
        method: ClusterMixin | type = sklearn.cluster.HDBSCAN(),
        shift_threshold: float = 0.8,
        shift_direction: Literal["both", "positive", "negative"] = "both",
        shift_selection: Literal["local", "global", "all"] = "local",
        scaler: StandardScaler
        | MinMaxScaler
        | RobustScaler
        | MaxAbsScaler
        | None = StandardScaler(),
        time_scale_factor: float = 1,
        regridder: BaseRegridder | None = None,
        output_label_suffix: str = "",
        output_label: str | None = None,
        overwrite: bool = False,
        sort_by_size: bool = True,
        # optimisation related params
        optimise: bool = False,
        optimisation_params: dict = default_hdbscan_optimisation_params,
        objective: Callable
        | Literal[
            "median_heaviside",
            "mean_heaviside",
            "mean_consistency",
            "mean_spatial_autocorrelation",
            "mean_nonlinearity",
            "combined_spatial_nonlinearity",
        ] = "combined_spatial_nonlinearity",
        n_trials: int = 50,
        direction: str = "maximize",
        log_level: int = optuna.logging.WARNING,
        show_progress_bar: bool = True,
    ):
        """Apply clustering to a dataset's temporal shifts using a sklearn-compatible clustering algorithm.

        Args:
            var: Name of the shifts variable to cluster, or name of the base variable whose shifts
                should be clustered. If None, TOAD will attempt to infer which shifts to use.
                A ValueError is raised if the shifts variable cannot be uniquely determined.
            method: The clustering method to use. Choose methods from sklearn.cluster or create
                your by inheriting from sklearn.base.ClusterMixin. Defaults to HDBSCAN().
            shift_threshold: The threshold for the shift magnitude. Defaults to 0.8.
            shift_direction: The sign of the shift. Options are "both", "positive", "negative". Defaults to "both".
            shift_selection: How shift values are selected for clustering. All options respect shift_threshold and shift_direction:
                "local": Finds peaks within individual shift episodes. Cluster only local maxima within each contiguous segment where abs(shift) > shift_threshold.
                "global": Finds the overall strongest shift per grid cell. Cluster only the single maximum shift value per grid cell where abs(shift) > shift_threshold.
                "all": Cluster all shift values that meet the threshold and direction criteria. Includes all data points above threshold, not just peaks.
                Defaults to "local".
            scaler: The scaling method to apply to the data before clustering. StandardScaler(),
                MinMaxScaler(), RobustScaler() and MaxAbsScaler() from sklearn.preprocessing are
                supported. Defaults to StandardScaler().
            time_scale_factor: The factor to scale the time values by. Defaults to 1.
            regridder: The regridding method to use from toad.clustering.regridding.
                Defaults to None. If None and coordinates are lat/lon, a HealPixRegridder will
                be created automatically.
            output_label_suffix: A suffix to add to the output label. Defaults to "".
            overwrite: Whether to overwrite existing variable. Defaults to False.
            return_results_directly: Whether to return the clustering results directly or merge
                into the original dataset. Defaults to False.
            sort_by_size: Whether to reorder clusters by size. Defaults to True.
            optimise: Whether to optimise the clustering parameters. Defaults to False.
            optimisation_params: Parameters for the optimisation. Defaults to default_hdbscan_optimisation_params.
            objective: The objective function to optimise. Defaults to combined_spatial_nonlinearity. Can be one of:
                - callable: Custom objective function taking (td, output_label) as arguments
                - "median_heaviside": Median heaviside score across clusters
                - "mean_heaviside": Mean heaviside score across clusters
                - "mean_consistency": Mean consistency score across clusters
                - "mean_spatial_autocorrelation": Mean spatial autocorrelation score
                - "mean_nonlinearity": Mean nonlinearity score across clusters
            n_trials: Number of trials to run for optimisation. Defaults to 50.
            direction: The direction of the optimisation. Defaults to "maximize".
            log_level: The log level for the optimisation. Defaults to optuna.logging.WARNING.
            show_progress_bar: Whether to show the progress bar for the optimisation. Defaults to True.

        Returns:
            None.

        Raises:
            ValueError: If data is invalid or required parameters are missing

        Notes:
            For global datasets, use toad.regridding.HealPixRegridder to ensure equal spacing
            between data points and prevent biased clustering at high latitudes.
        """
        results = clustering.compute_clusters(
            td=self,
            var=self._get_base_var_if_none(var),
            method=method,
            shift_threshold=shift_threshold,
            shift_selection=shift_selection,
            shift_direction=shift_direction,
            scaler=scaler,
            time_scale_factor=time_scale_factor,
            regridder=regridder,
            output_label_suffix=output_label_suffix,
            output_label=output_label,
            overwrite=overwrite,
            sort_by_size=sort_by_size,
            optimise=optimise,
            optimisation_params=optimisation_params,
            objective=objective,
            n_trials=n_trials,
            direction=direction,
            log_level=log_level,
            show_progress_bar=show_progress_bar,
        )

        self.data = results

    # # ======================================================================
    # #               netCDF functions
    # # ======================================================================

    def save(self, suffix: Optional[str] = None, path: Optional[str] = None):
        """Save the TOAD object to a netCDF file.

        Args:
            suffix: Optional string to append to filename before extension
            path: Optional path to save file to. If not provided, uses self.path

        Raises:
            ValueError: If neither path nor self.path is set
            ValueError: If using self.path without a suffix (to prevent overwriting)
        """
        if path is None and self.path is None:
            raise ValueError("Path to save TOAD dataset not set. Please provide path.")

        # Prevent overwriting when using self.path
        if path is None and self.path is not None and suffix is None:
            raise ValueError(
                "Please provide either a suffix to append to the original path or specify a new path."
            )

        # Use user-provided path if specified, otherwise use self.path
        save_path = path if path is not None else self.path

        if save_path == self.path:
            # Get original extension if using self.path
            original_ext = self.path.rsplit(".", 1)[1] if "." in self.path else "nc"
        else:
            # For user-provided path without extension, default to .nc
            original_ext = save_path.rsplit(".", 1)[1] if "." in save_path else "nc"

        if suffix:
            # Split path into base and add suffix before extension
            base = save_path.rsplit(".", 1)[0] if "." in save_path else save_path
            save_path = f"{base}_{suffix}.{original_ext}"
        elif "." not in save_path:
            # Add extension if path has none
            save_path = f"{save_path}.{original_ext}"

        # Apply compression =====
        try:
            # First clear any existing encoding
            for var in self.data.variables:
                self.data[var].encoding.clear()

            # Define compression settings
            compression_settings = {
                "zlib": True,
                "complevel": 1,
            }

            # Apply compression to both float and int data variables
            for var in self.data.data_vars:
                if np.issubdtype(self.data[var].dtype, np.number):
                    self.data[var].encoding.update(compression_settings)

                    if np.issubdtype(self.data[var].dtype, np.integer):
                        self.data[var].encoding.update(
                            {"_FillValue": None, "dtype": self.data[var].dtype}
                        )
        except Exception as e:
            self.logger.warning(
                f"Could not apply compression settings: {str(e)}. Proceeding with save without compression."
            )

        self.data.to_netcdf(save_path)

    # # ======================================================================
    # #               GET functions (postprocessing)
    # # ======================================================================

    @property
    def space_dims(self):
        return get_space_dims(self.data, self.time_dim)

    @property
    def base_vars(self) -> list[str]:
        """Gets the list of base variables in the dataset.

        Base variables are those that have not been derived from shift detection or
            clustering. A variable is considered a base variable if either:
                1. It has no 'variable_type' attribute, or
                2. Its 'variable_type' is neither 'shift' nor 'cluster'

        Returns:
            A list of strings containing the base variable names in the dataset.
        """
        return [
            str(x)
            for x in list(self.data.data_vars.keys())
            if self.data[x].attrs.get(_attrs.VARIABLE_TYPE)
            not in [_attrs.TYPE_SHIFT, _attrs.TYPE_CLUSTER]
        ]

    @property
    def shift_vars(self) -> list[str]:
        """Gets the list of shift variables in the dataset.

        Shift variables are those that have been derived from shift detection.
        A variable is considered a shift variable if it has a 'variable_type=_attrs.TYPE_SHIFT'
        attribute.

        Returns:
            A list of strings containing the shift variable names in the dataset.
        """
        return [
            str(x)
            for x in list(self.data.data_vars.keys())
            if self._is_shift_variable(x)
        ]

    @property
    def cluster_vars(self) -> list[str]:
        """Get the list of cluster variables in the dataset.

        Cluster variables are those that have been derived from clustering.
        A variable is considered a cluster variable if it has a 'variable_type="cluster"' attribute.

        Returns:
            list[str]: List of cluster variable names in the dataset
        """
        return [
            str(x)
            for x in list(self.data.data_vars.keys())
            if self._is_cluster_variable(x)
        ]

    def shift_vars_for_var(self, var: str) -> list[str]:
        """Get the shift variables for a given variable.

        Args:
            var: The variable to get shift variables for. Can be either:
                - A base variable (e.g. 'temperature')
                - A cluster variable (e.g. 'temperature_cluster')
                Cannot be a shift variable.

        Returns:
            List of shift variables associated with the given variable:
                - For base variables: Returns all shift variables that have this as their base variable
                - For cluster variables: Returns the shift variable used to create this cluster

        Raises:
            ValueError: If var is a shift variable, or if no shift variables are found.
        """
        # If variable is a cluster variable, get the shift variable from attrs
        if self._is_cluster_variable(var):
            shift_variable = self.data[var].attrs.get(_attrs.SHIFTS_VARIABLE)
            if shift_variable:
                if shift_variable in self.shift_vars:
                    return [shift_variable]
                else:
                    raise ValueError(
                        f"Shift variable {shift_variable} not found in shift variables."
                    )
            else:
                raise ValueError(f"No shift variable found for cluster variable {var}.")
        # If variable is a shift variable, raise error
        if self._is_shift_variable(var):
            raise ValueError(
                "This is a shift variable. Use this function to get shift variable of a cluster or base variable."
            )
        # Else, must be a base variable, get all shift variables for that base variable
        else:
            return [
                str(x)
                for x in self.shift_vars
                if self.data[x].attrs.get(_attrs.BASE_VARIABLE) == var
            ]

    def cluster_vars_for_var(self, var: str) -> list[str]:
        """Get the cluster variables for a given variable.

        Args:
            var: The variable to get cluster variables for. Can be either:
                - A base variable (e.g. 'temperature')
                - A shift variable (e.g. 'temperature_dts')
                Cannot be a cluster variable.

        Returns:
            List of cluster variables associated with the given variable:
                - For base variables: Returns cluster variables that have this as their base variable
                - For shift variables: Returns cluster variables that were derived from this shift variable

        Raises:
            ValueError: If var is a cluster variable. This function can only get cluster variables
                for base or shift variables.
        """
        if self._is_cluster_variable(var):
            raise ValueError(
                "This is a cluster variable. Use this function to get cluster variables of a base or shift variable."
            )
        elif self._is_shift_variable(var):
            return [
                str(x)
                for x in self.cluster_vars
                if self.data[x].attrs.get(_attrs.SHIFTS_VARIABLE) == var
            ]
        else:
            return [
                str(x)
                for x in self.cluster_vars
                if self.data[x].attrs.get(_attrs.BASE_VARIABLE) == var
            ]

    def get_base_var(self, var: str) -> Optional[str]:
        """Get the base variable for a given variable."""
        return self.data[var].attrs.get(_attrs.BASE_VARIABLE)

    def get_shifts(self, var, label_suffix: str = "") -> xr.DataArray:
        """Get shifts xr.DataArray for the specified variable.

        Args:
            var: Base variable name (e.g. 'temperature', will look for 'temperature_cluster')
                or custom cluster variable name.
            label_suffix: If you added a suffix to the shifts variable, help the function find it.
                Defaults to "".

        Returns:
            The shifts xr.DataArray for the specified variable.

        Raises:
            ValueError: Failed to find valid shifts xr.DataArray for the given var.
        """

        # Check if the variable is a shifts variable
        if self._is_shift_variable(var):
            return self.data[var]

        shift_vars = self.shift_vars_for_var(var)

        # Filter by label_suffix if provided
        if label_suffix:
            shift_vars = [s for s in shift_vars if s.endswith(label_suffix)]

        if len(shift_vars) > 1:
            raise ValueError(
                f"Multiple shift variables exist for {var}: {shift_vars}. Please specify which one to use"
            )
        elif len(shift_vars) == 0:
            raise ValueError(
                f"No shifts variable found for {var}. Please first run compute_shifts()."
            )
        else:
            return self.data[shift_vars[0]]

    def get_clusters(self, var: str) -> xr.DataArray:
        """Get cluster xr.DataArray for the specified variable.

        Args:
            var: Base variable name (e.g. 'temperature', will look for 'temperature_cluster')
                or custom cluster variable name.

        Returns:
            The clusters xr.DataArray for the specified variable.

        Raises:
            ValueError: Failed to find valid cluster xr.DataArray for the given var. An
                xr.DataArray is only considered a cluster label if it contains _cluster in
                its name.
        """
        # Check if the variable is a cluster variable
        if self._is_cluster_variable(var):
            return self.data[var]

        cluster_vars = self.cluster_vars_for_var(var)
        if len(cluster_vars) > 1:
            raise ValueError(
                f"Multiple cluster variables exist for {var}: {cluster_vars}. Please specify which one to use"
            )
        elif len(cluster_vars) == 0:
            raise ValueError(
                f"No cluster variables found for {var}. Please first run compute_clusters()."
            )
        else:
            return self.data[cluster_vars[0]]

    def get_cluster_counts(self, var: str, exclude_noise: bool = True) -> dict:
        """Returns sorted dictionary with number of cells in both space and time for each cluster.

        Args:
            var: Base variable name (e.g. 'temperature', will look for 'temperature_cluster')
                or custom cluster variable name.
            exclude_noise: Whether to exclude noise points (cluster ID -1). Defaults to True.

        Returns:
            Dictionary mapping cluster IDs to their total cell counts, sorted by count in
            descending order.
        """
        counts = {}
        for cluster_id in self.get_cluster_ids(var, exclude_noise):
            count = self.get_cluster_mask(var, cluster_id).sum()
            counts[int(cluster_id)] = int(count)

        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    def get_cluster_ids(self, var: str, exclude_noise: bool = True) -> np.ndarray:
        """Return list of cluster ids sorted by total number of cells in each cluster.

        Args:
            var: Base variable name (e.g. 'temperature', will look for 'temperature_cluster')
                or custom cluster variable name.
            exclude_noise: Whether to exclude noise points (cluster ID -1). Defaults to True.

        Returns:
            List of cluster ids.
        """
        cluster_ids = self.get_clusters(var).cluster_ids
        if exclude_noise:
            return np.array([id for id in cluster_ids if id != -1])
        else:
            return cluster_ids

    def get_active_clusters_count_per_timestep(self, var: str) -> xr.DataArray:
        """Get number of active clusters for each timestep.

        Args:
            var: Base variable name (e.g. 'temperature', will look for 'temperature_cluster')
                or custom cluster variable name.

        Returns:
            Number of active clusters for each timestep.
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

        Args:
            var: Base variable name (e.g. 'temperature', will look for 'temperature_cluster')
                or custom cluster variable name.
            cluster_id: Cluster id(s) to apply the mask for.

        Returns:
            Mask for the cluster label.
        """
        clusters = self.get_clusters(var)
        return clusters.isin(cluster_id)

    def apply_cluster_mask(
        self, var: str, apply_to_var: str, cluster_id: int
    ) -> xr.DataArray:
        """Apply the cluster mask to a variable.

        Args:
            var: Base variable name (e.g. 'temperature', will look for 'temperature_cluster')
                or custom cluster variable name.
            apply_to_var: The variable to apply the mask to.
            cluster_id: The cluster id to apply the mask for.

        Returns:
            The masked variable.
        """
        mask = self.get_cluster_mask(var, cluster_id)
        return self.data[apply_to_var].where(mask)

    def get_spatial_cluster_mask(  # TODO rename to get_cluster_mask_spatial
        self, var: str, cluster_id: Union[int, List[int]]
    ) -> xr.DataArray:
        """Returns a 2D boolean mask indicating which grid cells belonged to the specified cluster at any point in time.

        I.e. a grid cell is True if it belonged to the specified cluster at any point in time during the entire timeseries.

        Args:
            var: Base variable name (e.g. 'temperature', will look for 'temperature_cluster')
                or custom cluster variable name.
            cluster_id: Cluster id to apply the mask for.

        Returns:
            Mask for the cluster id.
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
        """Apply the spatial cluster mask to a variable.

        Args:
            var: Base variable name (e.g. 'temperature', will look for 'temperature_cluster')
                or custom cluster variable name.
            apply_to_var: The variable to apply the mask to.
            cluster_id: The cluster id to apply the mask for.

        Returns:
            All data (regardless of cluster) masked by the spatial extend of the specified cluster.
        """
        mask = self.get_spatial_cluster_mask(var, cluster_id)
        return self.data[apply_to_var].where(mask)

    def apply_temporal_cluster_mask(
        self, var: str, apply_to_var: str, cluster_id: int
    ) -> xr.DataArray:
        """Apply the temporal cluster mask to a variable.

        Args:
            var: Base variable name (e.g. 'temperature', will look for 'temperature_cluster')
                or custom cluster variable name.
            apply_to_var: The variable to apply the mask to.
            cluster_id: The cluster id to apply the mask for.

        Returns:
            All data (regardless of cluster) masked by the temporal extend of the specified cluster.
        """
        mask = self.get_temporal_cluster_mask(var, cluster_id)
        return self.data[apply_to_var].where(mask)

    def get_permanent_cluster_mask(self, var: str, cluster_id: int) -> xr.DataArray:
        """Create a mask for cells that always have the same cluster label (such as completely unclustered cells by passing -1).

        Args:
            var: Base variable name (e.g. 'temperature', will look for 'temperature_cluster')
                or custom cluster variable name.
            cluster_id: The cluster id.

        Returns:
            Boolean mask where True indicates cells that always belonged to the specified cluster.
        """
        clusters = self.get_clusters(var)
        return (clusters == cluster_id).all(dim=self.time_dim)

    def get_permanent_unclustered_mask(self, var: str) -> xr.DataArray:
        """Create the spatial mask for cells that are always unclustered (i.e. -1).

        Args:
            var: Base variable name (e.g. 'temperature', will look for 'temperature_cluster')
                or custom cluster variable name.

        Returns:
            Boolean mask where True indicates cells that were never clustered (always had value -1).
        """
        return self.get_permanent_cluster_mask(var, -1)

    def get_cluster_temporal_density(self, var: str, cluster_id: int) -> xr.DataArray:
        """Calculate the temporal density of a cluster at each grid cell.

        Args:
            var: Base variable name (e.g. 'temperature', will look for 'temperature_cluster')
                or custom cluster variable name.
            cluster_id: The cluster id to calculate density for.

        Returns:
            2D spatial array where each grid cell contains a fraction (0-1) representing the proportion of timesteps that cell belonged to the specified cluster.
        """
        density = self.get_cluster_mask(var, cluster_id).mean(dim=self.time_dim)
        density = density.rename(f"{density.name}_temporal_density")
        return density

    def get_cluster_spatial_density(self, var: str, cluster_id: int) -> xr.DataArray:
        """Calculate the spatial density of a cluster across all grid cells.

        Args:
            var: Base variable name (e.g. 'temperature', will look for 'temperature_cluster')
                or custom cluster variable name.
            cluster_id: The cluster id to calculate density for.

        Returns:
            1D timeseries containing the fraction (0-1) of grid cells that belonged to the specified cluster at each timestep.
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

        Args:
            var: Base variable name (e.g. 'temperature', will look for 'temperature_cluster')
                or custom cluster variable name.
            cluster_id: The cluster ID to calculate the temporal footprint for.

        Returns:
            Boolean array with True indicating timesteps where the cluster existed
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

        Args:
            var: Base variable name (e.g. 'temperature', will look for 'temperature_cluster')
                or custom cluster variable name.

        Returns:
            Fraction (0-1) of grid cells belonging to any cluster at each timestep.
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

        Args:
            var: Base variable name (e.g. 'temperature', will look for 'temperature_cluster')
                or custom cluster variable name.
            cluster_id: Single cluster ID or list of cluster IDs.

        Returns:
            Full dataset masked by the cluster id.

        Note:
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

    def _get_base_var_if_none(self, var: str | None) -> str:
        """Get the default base variable if none specified, or return the provided variable.

        Helper method to handle cases where a variable is optional and should default to the
        single base variable if one exists, or raise an error if multiple exist.

        Args:
            var: Optional variable name. If None, will attempt to use the single base variable.

        Returns:
            The variable name to use - either the provided var or the default base variable.

        Raises:
            ValueError: If var is None and multiple base variables exist.
        """
        if var is None:
            if len(self.base_vars) > 1:
                raise ValueError(
                    f"Multiple base variables exist: {self.base_vars}. Please specify which one to use."
                )
            else:
                return self.base_vars[0]
        else:
            return var

    def _is_shift_variable(self, var: str) -> bool:
        """Check if a variable is a shift variable."""
        return self.data[var].attrs.get(_attrs.VARIABLE_TYPE) == _attrs.TYPE_SHIFT

    def _is_cluster_variable(self, var: str) -> bool:
        """Check if a variable is a cluster variable."""
        return self.data[var].attrs.get(_attrs.VARIABLE_TYPE) == _attrs.TYPE_CLUSTER

    def _is_base_variable(self, var: str) -> bool:
        """Check if a variable is a base variable."""
        return self.data[var].attrs.get(_attrs.VARIABLE_TYPE) not in [
            _attrs.TYPE_SHIFT,
            _attrs.TYPE_CLUSTER,
        ]

    def _aggregate_spatial(
        self,
        data: xr.DataArray,
        method: str = "raw",
        percentile: Optional[float] = None,
    ) -> xr.DataArray:
        """Aggregate data across spatial dimensions.

        Args:
            data: Data to aggregate.
            method: Aggregation method:
                - "mean": Average across space
                - "median": Median across space
                - "sum": Sum across space
                - "std": Standard deviation across space
                - "percentile": Percentile across space (requires percentile arg)
                - "max": Maximum across space
                - "min": Minimum across space
                - "raw": Return data for each grid cell separately (default).
            percentile: Percentile value between 0-1 when using percentile aggregation.

        Returns:
            Aggregated data. If method="raw", includes cell_xy dimension.
        """
        if method == "mean":
            return data.mean(dim=self.space_dims)
        elif method == "median":
            return data.median(dim=self.space_dims)
        elif method == "sum":
            return data.sum(dim=self.space_dims)
        elif method == "std":
            return data.std(dim=self.space_dims)
        elif method == "max":
            return data.max(dim=self.space_dims)
        elif method == "min":
            return data.min(dim=self.space_dims)
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
        cluster_id: Union[int, List[int]],  # TODO: rename to cluster_ids ?
        cluster_var: Optional[str] = None,
        aggregation: Literal[
            "raw", "mean", "sum", "std", "median", "percentile", "max", "min"
        ] = "raw",
        percentile: Optional[float] = None,
        normalize: Optional[Literal["first", "max", "last"]] = None,
        keep_full_timeseries: bool = True,
    ) -> xr.DataArray:
        """Get time series for cluster, optionally aggregated across space.

        Args:
            var: Variable name to extract time series from.
            cluster_var: Variable name to extract cluster ids from. Default to None and is
                attempted to be inferred from var.
            cluster_id: Single cluster ID or list of cluster IDs.
            aggregation: How to aggregate spatial data:
                - "mean": Average across space
                - "median": Median across space
                - "sum": Sum across space
                - "std": Standard deviation across space
                - "percentile": Percentile across space (requires percentile arg)
                - "max": Maximum across space
                - "min": Minimum across space
                - "raw": Return data for each grid cell separately
            percentile: Percentile value between 0-1 when using percentile aggregation.
            normalize: How to normalize the data:
                - "first": Normalize by the first non-zero, non-nan timestep
                - "max": Normalize by the maximum value
                - "last": Normalize by the last non-zero, non-nan timestep
                - None: Do not normalize
            keep_full_timeseries: If True, returns full time series of cluster cells. If
                False, only returns time series of cells when they were in the cluster.

        Returns:
            The time series data for the specified cluster(s).
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

    def to_timeseries(self, time_dim: str = "time"):
        """Convert spatial data to timeseries format by stacking spatial dimensions.

        Args:
            time_dim: Name of the time dimension. Defaults to "time".

        Returns:
            DataArray with dimensions [time, cell_xy] suitable for timeseries plotting.

        Examples:
            >>> data.toad.to_timeseries().plot.line(x="time", add_legend=False, color='k', alpha=0.1);
        """

        # Check if time_dim is in dims
        if time_dim not in self._obj.dims:
            raise ValueError(
                f"Time dimension '{time_dim}' not found in data. Please specify a time dimension using the time_dim argument."
            )

        # Get all dims except time dim
        non_time_dims = [d for d in self._obj.dims if d != time_dim]

        return (
            self._obj.stack(cell_xy=non_time_dims)
            .transpose("cell_xy", time_dim)
            .dropna(dim="cell_xy", how="all")
        )
