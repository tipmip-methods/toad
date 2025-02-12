import logging
import xarray as xr
import numpy as np
from typing import Callable
from toad._version import __version__
import inspect
from typing import Optional, Union

from sklearn.base import ClusterMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

from toad.clustering.prepare_data import prepare_dataframe

logger = logging.getLogger("TOAD")

def compute_clusters(
        data: xr.Dataset,
        var : str,
        method : ClusterMixin,
        shifts_filter_func: Callable = lambda _: True,  # empty filtering function as default
        var_filter_func: Callable = lambda _: True,     # empty filtering function as default
        shifts_label: Optional[str] = None,
        time_dim: str = "time",
        space_dims: list[str] = ["lon", "lat"],
        scaler: Optional[Union[StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler]] = StandardScaler(),
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

    """
    
    # Check shifts var
    all_vars = list(data.data_vars.keys())
    shifts_label = shifts_label if shifts_label else f'{var}_dts'  # default to {var}_dts
    
    if shifts_label not in all_vars:
        raise ValueError(f'Shifts not found at {shifts_label}. Please run shifts on {var} first, or provide a custom "shifts_label"')
    if data[shifts_label].ndim != 3:
        raise ValueError('data must be 3-dimensional')
    
    # 1. Preprocessing ======================================================

    # Set output label
    output_label = f'{var}_cluster{output_label_suffix}'

    # Check if the output_label is already in the data
    if output_label in data and merge_input:
        if overwrite:
            data = data.drop_vars(output_label)
        else:
            logger.warning(f'{output_label} already exists. Please pass overwrite=True to overwrite it.')
            return data

    # Convert to pandas dataframe
    df_data = {
        'var': data[var].to_dataframe().reset_index(),
        'dts': data[shifts_label].to_dataframe().reset_index()
    }

    # filter df with var_filter_func and shifts_filter_func
    mask = (
        df_data['var'][var].apply(var_filter_func) & 
        df_data['dts'][shifts_label].apply(shifts_filter_func)
    )
    filtered_df = df_data['dts'][mask]
    if filtered_df.empty:
        raise ValueError('No data left after filtering.')

    # Get coordinates and scale them if needed
    dims = [time_dim] + space_dims
    coords = filtered_df[dims].to_numpy()
    coords = scaler.fit_transform(coords) if scaler else coords

    # Compute importance weights as the absolute values of the dts variable
    weights = np.abs(filtered_df[shifts_label].to_numpy())

    # Save method params before clustering (because they might change during clustering)
    method_params = {
        f'method_{param}': str(value) 
        for param, value in dict(sorted(vars(method).items())).items() 
        if value is not None
    }

    # 2. Perform clustering ==================================================
    logger.info(f'Applying clustering method {method}')
    cluster_labels = np.array(method.fit_predict(coords, weights))
    
    # 3. Postprocessing =====================================================

    # Rename cluster labels to reflect size
    cluster_labels = sorted_cluster_labels(cluster_labels) if sort_by_size else cluster_labels

    # Convert back to xarray DataArray
    df_dims = data[dims].to_dataframe().reset_index()       # create a pandas df with original dims
    df_dims[output_label] = -1                              # Initialize cluster column with -1
    df_dims.loc[filtered_df.index, output_label] = cluster_labels # Assign cluster labels to the dataframe
    clusters = df_dims.set_index(dims).to_xarray()          # Convert dataframe to xarray (DataSet)
    clusters = clusters[output_label]                       # select only cluster labels

    # 6. Save details as attributes
    clusters.attrs.update({
        f'cluster_ids': np.unique(cluster_labels).astype(int),
        f"var_filter_func": inspect.getsource(var_filter_func) if var_filter_func else "None",
        f"shifts_filter_func": inspect.getsource(shifts_filter_func) if shifts_filter_func else "None",
        f"scaler": scaler,
        f'method_name': method.__class__.__name__,
    })
    
    # Add saved params as attributes
    clusters.attrs.update(method_params)

    # add git version
    clusters.attrs['toad_version'] = __version__

    # 7. Merge the cluster labels back into the original data
    if merge_input:
        return xr.merge([data, clusters], combine_attrs="override") # xr.dataset
    else:
        return clusters # xr.dataarray


def sorted_cluster_labels(cluster_labels: np.ndarray) -> np.ndarray:
    """Sort clusters by size (largest cluster -> 0, second largest -> 1, etc., keeping -1 for noise)
    """
    unique_labels, counts = np.unique(cluster_labels[cluster_labels != -1], return_counts=True) # ignore -1
    label_mapping = dict(zip(unique_labels[np.argsort(counts)[::-1]], range(len(unique_labels))))
    return np.array([label_mapping.get(label, -1) for label in cluster_labels])

