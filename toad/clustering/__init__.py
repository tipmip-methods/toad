"""
Clustering module for TOAD (Temporal Offset Analysis and Detection).

This module provides functionality for clustering temporal shifts in climate data. The main function
`compute_clusters` takes temporal shift patterns and groups them into clusters using sklearn-compatible
clustering algorithms. The clustering is performed in both space and time dimensions, allowing
identification of regions with similar temporal shift behaviors.

The module supports various clustering methods from scikit-learn (e.g., HDBSCAN, DBSCAN, etc.) and
includes utilities for:
- Preprocessing data with different scaling methods
- Handling geographic coordinates and projections
- Converting between geodetic and cartesian coordinates
- Sorting clusters by size
- Preserving metadata and attributes in xarray objects
- Filtering shifts based on thresholds and directions
- Selecting between local and global shift patterns

The clustering results are returned as xarray objects with appropriate metadata and can be
visualized using TOAD's plotting utilities.
"""

import logging
from time import time as time_now
from typing import TYPE_CHECKING, Literal, Union

import numpy as np
import xarray as xr
from sklearn.base import ClusterMixin
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from toad._version import __version__
from toad.regridding import HealPixRegridder
from toad.regridding.base import BaseRegridder
from toad.utils import (
    _attrs,
    detect_latlon_names,
    get_unique_variable_name,
    reorder_space_dims,
)
from toad.utils.shift_selection_utils import _compute_dts_peak_sign_mask

logger = logging.getLogger("TOAD")

# to avoid circular import we use TYPE_CHECKING for importing TOAD obj
if TYPE_CHECKING:
    from toad.core import TOAD


def compute_clusters(
    td: "TOAD",
    var: str,
    method: ClusterMixin,
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
    overwrite: bool = False,
    merge_input: bool = True,
    sort_by_size: bool = True,
) -> Union[xr.Dataset, xr.DataArray]:
    """Apply clustering to a dataset's temporal shifts using a sklearn-compatible clustering algorithm.

    Args:
        td: TOAD object containing the data to cluster
        var: Name of the base variable or shifts variable to compute clusters for. If multiple shifts variables exist for the base variable, a ValueError is thrown, in which case you should specify the shifts variable name.
        method: The clustering method to use. Choose methods from `sklearn.cluster` or create your own by inheriting from `sklearn.base.ClusterMixin`.
        shift_threshold: The threshold for the shift magnitude. Defaults to 0.8.
        shift_direction: The direction of the shift. Options are "both", "positive", "negative". Defaults to "both".
        shift_selection: How shift values are selected for clustering. All options respect shift_threshold and shift_direction:
            - "local": Finds peaks within individual shift episodes. Cluster only local maxima within each contiguous segment where abs(shift) > shift_threshold.
            - "global": Finds the overall strongest shift per grid cell. Cluster only the single maximum shift value per grid cell where abs(shift) > shift_threshold.
            - "all": Cluster all shift values that meet the threshold and direction criteria. Includes all data points above threshold, not just peaks.
            Defaults to "local".
        scaler: The scaling method to apply to the data before clustering. StandardScaler(), MinMaxScaler(), RobustScaler() and MaxAbsScaler() from sklearn.preprocessing are supported. Defaults to StandardScaler().
        time_scale_factor: The factor to scale the time values by. Defaults to 1.
        regridder: The regridding method to use from `toad.clustering.regridding`. Defaults to None. If None and coordinates are lat/lon, a HealPixRegridder will be created automatically.
        output_label_suffix: A suffix to add to the output label. Defaults to "".
        overwrite: If True, overwrite existing variable of same name. If False, same name is used with an added number. Defaults to False.
        merge_input: Whether to merge the clustering results with the input dataset. Defaults to True.
        sort_by_size: Whether to reorder clusters by size. Defaults to True.

    Returns:
        If `merge_input` is `True`, returns an `xarray.Dataset` containing the original data and the clustering results.
        If `merge_input` is `False`, returns an `xarray.DataArray` containing the clustering results.

    Notes:
        For global datasets, use `toad.clustering.regridding.HealpyRegridder` to ensure equal spacing between data points and prevent biased clustering at high latitudes.
    """

    """
    Overview of the clustering process:
    1. Input Validation
        - Verify shifts variable exists in dataset (either directly or via base variable)
        - Check data has required 3 dimensions
        - Validate shift_threshold is positive
    2. Preprocessing
        - Generate output label with optional suffix
        - Check for existing results and handle overwrite based on parameters
        - Compute peak/sign mask based on shift_selection ("local"/"global")
        - Filter points based on shift_direction ("both"/"positive"/"negative")
        - Extract spatial and temporal coordinates
        - Apply optional regridding to standardize coordinates
        - Scale coordinates using sklearn preprocessing
        - Scale time values by time_scale_factor
        - Calculate weights from shift magnitudes
    3. Clustering
        - Store clustering parameters as metadata
        - Fit clustering model to coordinates using weights
        - Generate cluster labels for each point
    4. Postprocessing
        - Sort clusters by size if requested
        - Scatter labels back to xarray coordinates
        - Add clustering parameters as attributes
        - Optionally merge results with input dataset
        - Return Dataset or DataArray based on merge_input parameter
    """

    start_time = time_now()

    # if supplied variable is a shift variable, use that
    if td.data[var].attrs.get(_attrs.VARIABLE_TYPE) == _attrs.TYPE_SHIFT:
        shifts_variable = var
    else:
        # if supplied variable is a base variable, check if multiple shifts variables exist
        shift_vars = td.shift_vars_for_var(var)
        if len(shift_vars) > 1:
            raise ValueError(
                f"Multiple shifts variables exist for {var}: {shift_vars}. Please specify which one to use"
            )
        elif len(shift_vars) == 0:
            raise ValueError(
                f"No shifts found for base variable {var}. Please run compute_shifts() for var={var} first."
            )
        # use the first/only shift variable
        shifts_variable = shift_vars[0]

    if td.data[shifts_variable].ndim != 3:
        raise ValueError(
            "Shifts variable must be 3-dimensional"
        )  # TODO: make it work for 2D data

    # we add neg sign manually to detect negative shift
    if shift_threshold < 0:
        raise ValueError(f"shift_threshold must be positive, got {shift_threshold}")

    # Set output label (name of shifts_variable + _cluster + output_label_suffix) and check if already in data
    output_label = f"{shifts_variable}_cluster{output_label_suffix}"
    if merge_input and not overwrite:
        output_label = get_unique_variable_name(output_label, td.data, logger)
    elif overwrite and output_label in td.data:
        td.data = td.data.drop_vars(output_label)

    sh = td.data[shifts_variable]
    if shift_selection in ("local", "global"):
        mask_da = _compute_dts_peak_sign_mask(
            sh,
            td.time_dim,
            shift_threshold,
            shift_selection=shift_selection,
        )
        if shift_direction == "both":
            cond = mask_da != 0
        elif shift_direction == "positive":
            cond = mask_da > 0
        else:  # "negative"
            cond = mask_da < 0
    else:
        # shift_selection == "all": filter original magnitudes
        if shift_direction == "both":
            cond = np.abs(sh) > shift_threshold
        elif shift_direction == "positive":
            cond = sh > shift_threshold
        else:
            cond = sh < -shift_threshold

    # boolean → indices (tuple: (t_idx, *space_idx))
    cond_vals = np.asarray(cond.data)
    idx = np.nonzero(cond_vals)
    n_pts = idx[0].size

    if n_pts == 0:
        logger.warning(
            f'No gridcells left after applying shift_threshold={shift_threshold} and shift_direction="{shift_direction}"'
        )
        # create cluster variable with all -1
        clusters = xr.full_like(sh, -1).rename(output_label)
        preprocessing_time = 0.0
        clustering_time = 0.0
        cluster_labels = np.array([-1], dtype=int)
        coords = np.empty((0, 3))
        method_params = {}
        regridder_params = {}
    else:
        # Handle dimensions
        space_dims = td.space_dims
        space_dims = reorder_space_dims(space_dims)

        # Determine latitude/longitude names from dataset (e.g. lat, latitude, or None)
        lat_name, lon_name = detect_latlon_names(td.data)

        # check if dataset has lat/lon (as dims, or coords, or variables)
        has_latlon = lat_name is not None and lon_name is not None

        # Determine if this is a regular 1D lat/lon grid (i.e. dims are exactly lat, lon)
        is_latlon_dims = has_latlon and (space_dims == [lat_name, lon_name])

        # Build coordinates array (NumPy only, no DataFrame merges)
        # time coordinate
        time_vals = td.data[td.time_dim].values[idx[0]]
        time_numeric = _as_numeric_time(time_vals)

        if is_latlon_dims:
            # lat/lon are 1D dims: index directly
            lat_vals = td.data[lat_name].values[idx[1]]
            lon_vals = td.data[lon_name].values[idx[2]]
            coords = np.column_stack((time_numeric, lat_vals, lon_vals))
        elif has_latlon:
            # Irregular i/j grids: take 2D lat/lon variables aligned with space_dims
            lat_grid = td.data[lat_name].transpose(*space_dims).values
            lon_grid = td.data[lon_name].transpose(*space_dims).values
            lat_vals = lat_grid[tuple(idx[1:])]
            lon_vals = lon_grid[tuple(idx[1:])]
            coords = np.column_stack((time_numeric, lat_vals, lon_vals))
        else:
            # No lat/lon (as dims or coords or variables) → Fall back to using raw index dimensions (e.g., x/y or i/j)
            cols = [time_numeric]
            for d, i_idx in zip(space_dims, idx[1:]):
                vals_d = td.data[shifts_variable].coords[d].values
                cols.append(vals_d[i_idx])
            coords = np.column_stack(cols)

        # take absolute value of shifts as weights (at selected points)
        vals_sh = np.asarray(sh.data)[idx]
        weights = np.abs(vals_sh)

        # Create HealPixRegridder only for regular 1D lat/lon grids
        if regridder is None and is_latlon_dims:
            regridder = HealPixRegridder()

        # Regrid and scale
        if regridder:
            logger.debug(
                f"Regridding {shifts_variable} with {regridder.__class__.__name__}"
            )
            coords, weights = regridder.regrid(
                coords,
                weights,
                space_dims_size=(
                    td.data.sizes[td.space_dims[0]],
                    td.data.sizes[td.space_dims[1]],
                ),
            )

        # Convert to Cartesian (time, x, y, z) coordinates when lat/lon are available
        if has_latlon:
            coords = geodetic_to_cartesian(
                time=coords[:, 0], lat=coords[:, 1], lon=coords[:, 2]
            )

        # Scale coordinates using sklearn preprocessing
        if scaler:
            coords = scaler.fit_transform(coords)

        # Scale time values by scaler value
        if time_scale_factor != 1:
            coords[:, 0] = coords[:, 0] * time_scale_factor

        # Save method params before clustering (because they might change during clustering)
        method_params = {
            f"method_{param}": str(value)
            for param, value in dict(sorted(vars(method).items())).items()
            if value is not None and not param.startswith("_")
        }

        # Save regridder params
        regridder_params = {}
        if regridder:
            regridder_params["regridder_name"] = regridder.__class__.__name__
            regridder_params.update(
                {
                    f"regridder_{param}": str(value)
                    for param, value in dict(sorted(vars(regridder).items())).items()
                    if value is not None and isinstance(value, (int, float, str))
                }
            )

        # Measure preprocessing time
        preprocessing_time = time_now() - start_time

        logger.debug(
            f"Applying clusterer {method.__class__.__name__} to {shifts_variable}"
        )

        cluster_start = time_now()
        try:
            cluster_labels = np.array(method.fit_predict(coords, weights))
        except ValueError as e:
            if "min_samples" in str(e) and "must be at most" in str(e):
                logger.warning(
                    f"Clustering failed due to insufficient data points. Returning no clusters. Error: {e}"
                )
                cluster_labels = np.full(len(coords), -1)
            else:
                raise e
        clustering_time = time_now() - cluster_start

        # Regrid back
        if regridder:
            cluster_labels = regridder.regrid_clusters_back(
                cluster_labels
            )  # regridder holds the original coordinates

        # Sort cluster labels by size (After regridding because regridding
        # may change the number of members in each cluster)
        cluster_labels = (
            sorted_cluster_labels(cluster_labels) if sort_by_size else cluster_labels
        )

        # Scatter labels back into xarray without DataFrame
        clusters = xr.full_like(sh, -1).rename(output_label)
        clusters.data = clusters.data.astype(np.int32, copy=False)
        clusters.data[idx] = np.asarray(cluster_labels, dtype=np.int32)

        # Transpose if dimensions don't match (shouldn't be needed but keep)
        if clusters.dims != td.data[shifts_variable].dims:
            clusters = clusters.transpose(*td.data[shifts_variable].dims)

    # Get base variable from shifts attrs
    base_variable = td.data[shifts_variable].attrs.get(_attrs.BASE_VARIABLE)
    base_variable = base_variable if base_variable else "Unknown"

    # Save details as attributes (single update block)
    clusters.attrs.update(
        {
            _attrs.CLUSTER_IDS: np.unique(cluster_labels).astype(int),
            _attrs.SHIFT_THRESHOLD: shift_threshold,
            _attrs.SHIFT_SELECTION: shift_selection,
            _attrs.SHIFT_DIRECTION: shift_direction,
            _attrs.SCALER: scaler.__class__.__name__ if scaler else "None",
            _attrs.TIME_SCALE_FACTOR: time_scale_factor,
            _attrs.N_DATA_POINTS: n_pts,
            _attrs.METHOD_NAME: method.__class__.__name__,
            _attrs.RUNTIME_PREPROCESSING: float(preprocessing_time),
            _attrs.RUNTIME_CLUSTERING: float(clustering_time),
            _attrs.RUNTIME_TOTAL: float(preprocessing_time + clustering_time),
            _attrs.TOAD_VERSION: __version__,
            _attrs.BASE_VARIABLE: base_variable,
            _attrs.SHIFTS_VARIABLE: shifts_variable,
            _attrs.VARIABLE_TYPE: _attrs.TYPE_CLUSTER,
            **method_params,
            **regridder_params,
        }
    )

    logger.info(_format_cluster_summary(output_label, cluster_labels, n_pts))

    # Merge the cluster labels back into the original data
    return (
        xr.merge([td.data, clusters], combine_attrs="override", compat="override")
        if merge_input
        else clusters
    )


def _format_cluster_summary(
    output_label: str, cluster_labels: np.ndarray, n_points_used: int
) -> str:
    """
    Produce a concise summary:
      - name of the new variable (output_label)
      - number of identified clusters (excluding -1)
      - number of data points used (after filtering)
      - percentage of points labeled as noise (-1)
    """
    n = int(n_points_used)
    if n == 0:
        return f"{output_label}: Identified 0 CLUSTERS in 0 points"

    labels = np.asarray(cluster_labels)
    noise = int(np.count_nonzero(labels == -1))
    pct_noise = 100.0 * noise / n
    n_clusters = int(np.unique(labels[labels != -1]).size)

    # nice, compact, and informative
    clusters_text = f"{n_clusters} {'cluster' if n_clusters == 1 else 'clusters'}"
    return (
        f"New cluster variable \033[1m{output_label}\033[0m: Identified \033[1m{clusters_text}\033[0m in {n:,} pts; "
        f"Left behind {pct_noise:.1f}% as noise"
        f" ({noise:,} pts)."
    )


def _as_numeric_time(t: np.ndarray) -> np.ndarray:
    """Convert time values to float for sklearn (seconds since epoch if datetime64)."""
    t = np.asarray(t)
    if np.issubdtype(t.dtype, np.datetime64):
        return t.astype("datetime64[ns]").astype("int64") / 1e9
    return t.astype(float, copy=False)


def sorted_cluster_labels(cluster_labels: np.ndarray) -> np.ndarray:
    """Sort clusters by size (largest cluster -> 0, second largest -> 1, etc., keeping -1 for noise)"""
    # Get unique labels and counts, excluding -1
    unique_labels, counts = np.unique(
        cluster_labels[cluster_labels != -1], return_counts=True
    )

    # Sort by counts in descending order
    sorted_indices = np.argsort(counts)[::-1]
    sorted_unique_labels = unique_labels[sorted_indices]

    # Create mapping from old labels to new labels (0 to n-1)
    label_mapping = {old: new for new, old in enumerate(sorted_unique_labels)}
    label_mapping[-1] = -1  # Keep -1 for noise points

    # Apply mapping to all labels
    return np.array([label_mapping[label] for label in cluster_labels])


def geodetic_to_cartesian(time, lat, lon, height=0) -> np.ndarray:
    """Converts geodetic coordinates to Cartesian coordinates.

    Transforms geodetic coordinates (time, latitude, longitude, optional height) into
    Cartesian coordinates (time, x, y, z) using the WGS84 ellipsoid model.

    Args:
        time: Array of timestamps.
        lat: Array of latitudes in degrees.
        lon: Array of longitudes in degrees.
        height: Optional array of heights above ellipsoid in km. Defaults to 0.

    Returns:
        np.ndarray: Array of shape (n, 4) containing [time, x, y, z] coordinates,
            where x, y, z are in km from the Earth's center.
    """
    # WGS84 parameters
    a = 6378.137  # semi-major axis (km)
    b = 6356.752  # semi-minor axis (km)
    e2 = 1 - (b**2 / a**2)  # eccentricity squared

    # Convert latitude and longitude to radians
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    # Radius of curvature in the prime vertical
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad) ** 2)

    # Cartesian coordinates
    x = (N + height) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + height) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (b**2 / a**2 * N + height) * np.sin(lat_rad)

    return np.column_stack((time, x, y, z))  # Shape: (n, 4)
