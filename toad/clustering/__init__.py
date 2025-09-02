import logging
import xarray as xr
import numpy as np
from toad._version import __version__
from typing import TYPE_CHECKING, Optional, Union
from toad.utils import (
    get_space_dims,
    reorder_space_dims,
    detect_latlon_names,
    get_unique_variable_name,
    attrs,
)

from sklearn.base import ClusterMixin
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    MaxAbsScaler,
)
from toad.regridding.base import BaseRegridder
from toad.regridding import HealPixRegridder

logger = logging.getLogger("TOAD")

# to avoid circular import we use TYPE_CHECKING for importing TOAD obj
if TYPE_CHECKING:
    from toad.core import TOAD


def compute_clusters(
    td: "TOAD",
    var: str,
    method: ClusterMixin,
    shift_threshold: float = 0.8,
    shift_sign: str = "absolute",  # TODO: rename to shift_direction
    time_dim: str = "time",
    space_dims: Optional[list[str]] = None,
    scaler: Optional[
        Union[StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler]
    ] = StandardScaler(),
    time_scale_factor: float = 1,
    regridder: Optional[BaseRegridder] = None,
    output_label_suffix: str = "",
    overwrite: bool = False,
    merge_input: bool = True,
    sort_by_size: bool = True,
) -> Union[xr.Dataset, xr.DataArray]:
    """Apply clustering to a dataset's temporal shifts using a sklearn-compatible clustering algorithm.

    >> Args:
        var:
            Name of the base variable or shifts variable to compute clusters for. If multiple shifts variables exist for the base variable, a ValueError is throw, in which case you should specify the shifts variable name.
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
            The factor to scale the time values by. Defaults to 1.
        regridder:
            The regridding method to use from `toad.clustering.regridding`.
            Defaults to None. If None and coordinates are lat/lon, a HealPixRegridder will be created automatically.
        output_label_suffix:
            A suffix to add to the output label. Defaults to "".
        overwrite:
            If True, overwrite existing variable of same name. If False, same name is used with an added number. Defaults to False.
        merge_input:
            Whether to merge the clustering results with the input dataset. Defaults to True.
        sort_by_size:
            Whether to reorder clusters by size. Defaults to True.

    >> Returns:
        xr.Dataset:
            If `merge_input` is `True`, returns an `xarray.Dataset` containing the original data and the clustering results.
        xr.DataArray:
            If `merge_input` is `False`, returns an `xarray.DataArray` containing the clustering results.

    >> Raises:
        ValueError:
            If data is invalid or required parameters are missing

    >> Notes:
        For global datasets, use `toad.clustering.regridding.HealpyRegridder` to ensure equal spacing between data points and prevent biased clustering at high latitudes.

    """

    """
    Overview of the clustering process:
    1. Input Validation
        - Verify shifts variable exists in dataset
        - Check data has required 3 dimensions
        - Validate dimension names and ordering
    2. Preprocessing
        - Generate output label with optional suffix
        - Check for existing results and handle overwrite
        - Convert xarray Dataset/DataArray to pandas DataFrame
        - Apply user-defined filtering on variables and shifts
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
        - Convert cluster labels to xarray format
        - Add clustering parameters as attributes
        - Optionally merge results with input dataset
        - Return Dataset or DataArray based on merge_input
    """

    # if supplied variable is a shift variable, use that
    if td.data[var].attrs.get(attrs.VARIABLE_TYPE) == attrs.TYPE_SHIFT:
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

    # Extract and filter data for clustering
    df_dts = td.data[shifts_variable].to_dataframe().reset_index()

    def shifts_filter_func(x):
        """Filter shifts based on sign and threshold."""
        if shift_sign == "absolute":
            return np.abs(x) > shift_threshold
        elif shift_sign == "positive":
            return x > shift_threshold
        elif shift_sign == "negative":
            return x < -shift_threshold
        else:
            raise ValueError(
                f"shift_sign must be 'absolute', 'positive', or 'negative', got {shift_sign}"
            )

    # apply filter
    filtered_df = df_dts[df_dts[shifts_variable].apply(shifts_filter_func)]

    # return empty clusters if no data points left
    if filtered_df.empty:
        logger.warning(
            f"No gridcells left after applying shift threshold {shift_threshold} and shift sign {shift_sign}"
        )

        clusters = td.data[shifts_variable].copy().rename(output_label)
        clusters.data[:] = -1
        clusters.attrs = {}

        # Save details as attributes
        clusters.attrs.update(
            {
                attrs.CLUSTER_IDS: [],
                attrs.SHIFT_THRESHOLD: shift_threshold,
                attrs.SHIFT_SIGN: shift_sign,
                attrs.N_DATA_POINTS: 0,
                attrs.TOAD_VERSION: __version__,
            }
        )

        # Merge the cluster labels back into the original data
        return (
            xr.merge([td.data, clusters], combine_attrs="override")
            if merge_input
            else clusters
        )

    # Handle dimensions
    space_dims = space_dims if space_dims else get_space_dims(td.data, time_dim)
    space_dims = reorder_space_dims(space_dims)
    index_dims = [time_dim] + space_dims

    # Determine latitude/longitude names from dataset (e.g. lat, latitude, or None)
    lat_name, lon_name = detect_latlon_names(td.data)

    # check if dataset has lat/lon (as dims, or coords, or variables)
    has_latlon = lat_name is not None and lon_name is not None

    # Determine if this is a regular 1D lat/lon grid (i.e. dims are exactly lat, lon)
    is_latlon_dims = space_dims == [lat_name, lon_name]

    # Build coordinates array
    if is_latlon_dims:
        # lat/lon already present as columns in filtered_df
        coord_cols = [time_dim, lat_name, lon_name]
        coords = filtered_df[coord_cols].to_numpy()
    elif has_latlon:
        # Irregular i/j grids: join 2D lat/lon onto the filtered rows keyed by spatial dims
        lat_df = td.data[lat_name].to_dataframe(name=lat_name).reset_index()
        lon_df = td.data[lon_name].to_dataframe(name=lon_name).reset_index()
        latlon_df = lat_df.merge(lon_df, on=space_dims, how="inner")
        filtered_df_ext = filtered_df.merge(latlon_df, on=space_dims, how="left")
        coord_cols = [time_dim, lat_name, lon_name]
        coords = filtered_df_ext[coord_cols].to_numpy()
    else:
        # No lat/lon (as dims or coords or variables) → Fall back to using raw index dimensions (e.g., x/y or i/j)
        coords = filtered_df[index_dims].to_numpy()

    # take absolute value of shifts as weights
    weights = np.abs(filtered_df[shifts_variable].to_numpy())

    # Create HealPixRegridder only for regular 1D lat/lon grids
    if regridder is None and is_latlon_dims:
        regridder = HealPixRegridder()

    # Regrid and scale
    if regridder:
        logger.info(f"Regridding {shifts_variable} with {regridder.__class__.__name__}")
        coords, weights = regridder.regrid(coords, weights)

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
                if value is not None
                and type(value)
                in [
                    int,
                    float,
                    str,
                ]  # regridder has other params (such as coords and df) which we don't want to save
            }
        )

    # Perform clustering
    logger.info(f"Applying clusterer {method.__class__.__name__} to {shifts_variable}")
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

    # Convert back to xarray DataArray
    df_dims = df_dts[index_dims].copy()
    df_dims[output_label] = -1
    df_dims.loc[filtered_df.index, output_label] = cluster_labels
    clusters = df_dims.set_index(index_dims).to_xarray()[output_label]

    # Transpose if dimensions don't match
    if clusters.dims != td.data[shifts_variable].dims:
        clusters = clusters.transpose(*td.data[shifts_variable].dims)

    # Get base variable from shifts attrs
    base_variable = td.data[shifts_variable].attrs.get(attrs.BASE_VARIABLE)
    base_variable = base_variable if base_variable else "Unknown"

    # Save details as attributes
    clusters.attrs.update(
        {
            attrs.CLUSTER_IDS: np.unique(cluster_labels).astype(int),
            attrs.SHIFT_THRESHOLD: shift_threshold,
            attrs.SHIFT_SIGN: shift_sign,
            attrs.SCALER: scaler.__class__.__name__,
            attrs.TIME_SCALE_FACTOR: time_scale_factor,
            attrs.N_DATA_POINTS: len(coords),
            attrs.METHOD_NAME: method.__class__.__name__,
            attrs.TOAD_VERSION: __version__,
            attrs.BASE_VARIABLE: base_variable,
            attrs.SHIFTS_VARIABLE: shifts_variable,
            attrs.VARIABLE_TYPE: attrs.TYPE_CLUSTER,
            **method_params,
            **regridder_params,
        }
    )

    logger.info(f"Detected {len(np.unique(cluster_labels)) - 1} clusters")

    # Merge the cluster labels back into the original data
    return (
        xr.merge([td.data, clusters], combine_attrs="override")
        if merge_input
        else clusters
    )


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
    """Convert geodetic coordinates (time, lat, lon) to Cartesian coordinates (time, x, y, z)"""

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
