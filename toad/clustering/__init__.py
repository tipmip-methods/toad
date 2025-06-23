import logging
import xarray as xr
import numpy as np
from toad._version import __version__
from typing import Optional, Union
from toad.utils import get_space_dims, reorder_space_dims

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


def compute_clusters(
    data: xr.Dataset,
    var: str,
    method: ClusterMixin,
    shift_threshold: float = 0.8,
    shift_sign: str = "absolute",
    shifts_label: Optional[str] = None,
    time_dim: str = "time",
    space_dims: Optional[list[str]] = None,
    scaler: Optional[
        Union[StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler]
    ] = StandardScaler(),
    time_scale_factor: Optional[float] = None,
    regridder: Optional[BaseRegridder] = None,
    output_label_suffix: str = "",
    overwrite: bool = False,
    merge_input: bool = True,
    sort_by_size: bool = True,
) -> Union[xr.Dataset, xr.DataArray]:
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

    # Check shifts var
    all_vars = list(data.data_vars.keys())
    shifts_label = (
        shifts_label if shifts_label else f"{var}_dts"
    )  # default to {var}_dts

    if shifts_label not in all_vars:
        raise ValueError(
            f'Shifts not found at {shifts_label}. Please run shifts on {var} first, or provide a custom "shifts_label"'
        )
    if data[shifts_label].ndim != 3:
        raise ValueError("data must be 3-dimensional")  # TODO: make it work for 2D data

    # we add neg sign manually to detect negative shift
    if shift_threshold < 0:
        raise ValueError(f"shift_threshold must be positive, got {shift_threshold}")

    # Set output label and check if already in data
    output_label = f"{var}_cluster{output_label_suffix}"
    if output_label in data and merge_input:
        if overwrite:
            data = data.drop_vars(output_label)
        else:
            logger.warning(
                f"{output_label} already exists. Please pass overwrite=True to overwrite it."
            )
            return data

    # Extract and filter data for clustering
    df_data = {
        "var": data[var].to_dataframe().reset_index(),
        "dts": data[shifts_label].to_dataframe().reset_index(),
    }

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

    filtered_df = df_data["dts"][df_data["dts"][shifts_label].apply(shifts_filter_func)]
    if filtered_df.empty:
        raise ValueError(
            f"No gridcells left after applying shift threshold {shift_threshold} and shift sign {shift_sign}"
        )

    # Handle dimensions
    space_dims = space_dims if space_dims else get_space_dims(data, time_dim)
    space_dims = reorder_space_dims(space_dims)
    isLatLon = space_dims == ["lat", "lon"]
    dims = [time_dim] + space_dims

    # Get coordinates and weights
    coords = (
        filtered_df[dims].to_numpy()
    )  # convert to numpy, usually in this order [time, lat, lon] or [time, x, y]
    weights = np.abs(
        filtered_df[shifts_label].to_numpy()
    )  # take absolute value of shifts as weights

    # Create HealPixRegridder if regridder is None and coordinates are lat/lon
    if regridder is None and isLatLon:
        regridder = HealPixRegridder()
        logger.info("Created default HealPixRegridder for lat/lon coordinates")

    # HealPixRegridder is only supported for lat/lon coordinates
    if isinstance(regridder, HealPixRegridder) and not isLatLon:
        logger.info(
            "HealPixRegridder is only supported for lat/lon coordinates. Ignoring regridder."
        )
        regridder = None

    # Regrid and scale
    if regridder:
        coords, weights = regridder.regrid(coords, weights)

    # If lat/lon, convert to Cartesian (time, x, y, z) coordinates
    if isLatLon:
        coords = geodetic_to_cartesian(
            time=coords[:, 0], lat=coords[:, 1], lon=coords[:, 2]
        )
    
    # Scale coordinates using sklearn preprocessing
    if scaler:
        coords = scaler.fit_transform(coords)

    # Scale time values by scaler value
    if time_scale_factor:
        coords[:, 0] = coords[:, 0] * time_scale_factor
    
    # Save method params before clustering (because they might change during clustering)
    method_params = {
        f"method_{param}": str(value)
        for param, value in dict(sorted(vars(method).items())).items()
        if value is not None
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
    logger.info(f"Applying clustering method {method}")
    cluster_labels = np.array(method.fit_predict(coords, weights))

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
    df_dims = (
        data[dims].to_dataframe().reset_index()
    )  # create a pandas df with original dims
    df_dims[output_label] = -1  # Initialize cluster column with -1
    df_dims.loc[filtered_df.index, output_label] = (
        cluster_labels  # Assign cluster labels to the dataframe
    )
    clusters = df_dims.set_index(
        dims
    ).to_xarray()  # Convert dataframe to xarray (DataSet)
    clusters = clusters[output_label]

    # Transpose if dimensions don't match
    if clusters.sizes.keys() != data[dims].sizes.keys():
        print("transposing")
        clusters = clusters.transpose(*data[dims].sizes.keys())

    # Save details as attributes
    clusters.attrs.update(
        {
            "cluster_ids": np.unique(cluster_labels).astype(int),
            "shift_threshold": shift_threshold,
            "shift_sign": shift_sign,
            "scaler": scaler,
            "method_name": method.__class__.__name__,
            "toad_version": __version__,
            **method_params,
            **regridder_params,
        }
    )

    # Merge the cluster labels back into the original data
    return (
        xr.merge([data, clusters], combine_attrs="override")
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
