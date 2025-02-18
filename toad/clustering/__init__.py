import logging
import xarray as xr
import numpy as np
from typing import Callable
from toad._version import __version__
import inspect
from typing import Optional, Union
from toad.utils import get_space_dims, reorder_space_dims

from sklearn.base import ClusterMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from toad.regridding.base import BaseRegridder


logger = logging.getLogger("TOAD")

def compute_clusters(
        data: xr.Dataset,
        var : str,
        method : ClusterMixin,
        shifts_filter_func: Callable = lambda _: True,  # empty filtering function as default
        var_filter_func: Callable = lambda _: True,     # empty filtering function as default
        shifts_label: Optional[str] = None,
        time_dim: str = "time",
        space_dims: Optional[list[str]] = None,
        scaler: Optional[Union[StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler]] = StandardScaler(),
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
            shifts_filter_func:
                A callable used to filter the shifts before clustering, such as `lambda x: np.abs(x)>0.8`. Defaults to a filter that keeps all values.
            var_filter_func:
                A callable used to filter the primary variable before clustering. Defaults to a filter that keeps all values.
            shifts_label:
                Name of the variable containing precomputed shifts. Defaults to {var}_dts.
            scaler:
                The scaling method to apply to the data before clustering. StandardScaler(), MinMaxScaler(), RobustScaler() and MaxAbsScaler() from sklearn.preprocessing are supported. Defaults to StandardScaler().
            regridder:
                The regridding method to use from `toad.clustering.regridding`. When provided, filtered data points are regridded and transformed from lat/lon to x/y/z coordinates for clustering using Euclidean distance. Defaults to None.
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
    
    '''
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
    '''

    # Check shifts var
    all_vars = list(data.data_vars.keys())
    shifts_label = shifts_label if shifts_label else f'{var}_dts'  # default to {var}_dts
    
    if shifts_label not in all_vars:
        raise ValueError(f'Shifts not found at {shifts_label}. Please run shifts on {var} first, or provide a custom "shifts_label"')
    if data[shifts_label].ndim != 3: 
        raise ValueError('data must be 3-dimensional') # TODO: make it work for 2D data

    # Set output label and check if already in data
    output_label = f'{var}_cluster{output_label_suffix}'
    if output_label in data and merge_input:
        if overwrite:
            data = data.drop_vars(output_label)
        else:
            logger.warning(f'{output_label} already exists. Please pass overwrite=True to overwrite it.')
            return data

    # Extract and filter data for clustering
    df_data = {
        'var': data[var].to_dataframe().reset_index(),
        'dts': data[shifts_label].to_dataframe().reset_index()
    }
    mask = (
        df_data['var'][var].apply(var_filter_func) & 
        df_data['dts'][shifts_label].apply(shifts_filter_func)
    )
    filtered_df = df_data['dts'][mask]
    if filtered_df.empty:
        raise ValueError('No data left after filtering.')

    # Handle dimensions
    space_dims = space_dims if space_dims else get_space_dims(data, time_dim)
    space_dims = reorder_space_dims(space_dims)
    dims = [time_dim] + space_dims
    
    # Get coordinates and weights
    coords = filtered_df[dims].to_numpy() # convert to numpy, usually in this order [time, lat, lon] or [time, x, y]
    weights = np.abs(filtered_df[shifts_label].to_numpy()) # take absolute value of shifts as weights
    
    # Regrid and scale
    if regridder:
        coords, weights = regridder.regrid(coords, weights)
        coords = geodetic_to_cartesian(time=coords[:, 0], lat=coords[:, 1], lon=coords[:, 2])
    if scaler:
        coords = scaler.fit_transform(coords)
        
    # Save method params before clustering (because they might change during clustering)
    method_params = {
        f'method_{param}': str(value) 
        for param, value in dict(sorted(vars(method).items())).items() 
        if value is not None
    }

    # Save regridder params
    regridder_params = {}
    if(regridder):
        regridder_params["regridder_name"] = regridder.__class__.__name__
        regridder_params.update({
            f'regridder_{param}': str(value) 
            for param, value in dict(sorted(vars(regridder).items())).items() 
            if value is not None and type(value) in [int, float, str] # regridder has other params (such as coords and df) which we don't want to save
        })

    # Perform clustering
    logger.info(f'Applying clustering method {method}')
    cluster_labels = np.array(method.fit_predict(coords, weights))

    # Sort cluster labels by size
    cluster_labels = sorted_cluster_labels(cluster_labels) if sort_by_size else cluster_labels

    # Regrid back
    if regridder:
        cluster_labels = regridder.regrid_clusters_back(cluster_labels) # regridder holds the original coordinates    

    # Convert back to xarray DataArray
    df_dims = data[dims].to_dataframe().reset_index()       # create a pandas df with original dims
    df_dims[output_label] = -1                              # Initialize cluster column with -1
    df_dims.loc[filtered_df.index, output_label] = cluster_labels # Assign cluster labels to the dataframe
    clusters = df_dims.set_index(dims).to_xarray()          # Convert dataframe to xarray (DataSet)
    clusters = clusters[output_label]    
    
    # Transpose if dimensions don't match
    if clusters.sizes.keys() != data[dims].sizes.keys():
        print("transposing")
        clusters = clusters.transpose(*data[dims].sizes.keys())

    # Save details as attributes
    clusters.attrs.update({
        f'cluster_ids': np.unique(cluster_labels).astype(int),
        f"var_filter_func": inspect.getsource(var_filter_func) if var_filter_func else "None",
        f"shifts_filter_func": inspect.getsource(shifts_filter_func) if shifts_filter_func else "None",
        f"scaler": scaler,
        f'method_name': method.__class__.__name__,
        'toad_version': __version__,
        **method_params,
        **regridder_params,
    })

    # Merge the cluster labels back into the original data
    return xr.merge([data, clusters], combine_attrs="override") if merge_input else clusters


def sorted_cluster_labels(cluster_labels: np.ndarray) -> np.ndarray:
    """Sort clusters by size (largest cluster -> 0, second largest -> 1, etc., keeping -1 for noise)
    """
    unique_labels, counts = np.unique(cluster_labels[cluster_labels != -1], return_counts=True) # ignore -1
    label_mapping = dict(zip(unique_labels[np.argsort(counts)[::-1]], range(len(unique_labels))))
    return np.array([label_mapping.get(label, -1) for label in cluster_labels])


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
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
    
    # Cartesian coordinates
    x = (N + height) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + height) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (b**2 / a**2 * N + height) * np.sin(lat_rad)
    
    return np.column_stack((time, x, y, z)) # Shape: (n, 4)
