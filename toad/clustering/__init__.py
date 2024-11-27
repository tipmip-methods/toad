import logging
import xarray as xr
import numpy as np
from typing import Callable
from _version import __version__
import inspect

from toad.clustering.prepare_data import prepare_dataframe
from toad.clustering.methods.base import ClusteringMethod


logger = logging.getLogger("TOAD")

def compute_clusters(
        data: xr.Dataset,
        var : str,
        method : ClusteringMethod,
        shifts_filter_func: Callable[[float], bool],
        var_filter_func: Callable[[float], bool] = None,
        shifts_label: str = None,
        scaler: str = 'StandardScaler',
        output_label: str = None,
        overwrite: bool = False,
        merge_input: bool = True,
        transpose_output: bool = False,
    ) -> xr.Dataset:
    """
    Main clustering coordination function. Called from the TOAD.compute_clusters method. Ref that docstring for more info.

    TODO: (1) Fix: should also return auxillary coordinates. For now only returns coords in dims. 
    """

    # Check shifts var
    all_vars = list(data.data_vars.keys())
    shifts_label = shifts_label if shifts_label else f'{var}_dts'  # default to {var}_dts
    if shifts_label not in all_vars:
        raise ValueError(f'Shifts not found at {shifts_label}. Please run shifts on {var} first, or provide a custom "shifts_label"')
    if data[shifts_label].ndim != 3:
        raise ValueError('data must be 3-dimensional!')
    
    # 1. Check if the output_label is already in the data
    default_name = f'{var}_cluster'
    output_label = output_label or default_name
    if output_label in data and merge_input:
        if overwrite:
            logger.warning(f'Overwriting variable {output_label}')
            data = data.drop_vars(output_label)
        else:
            logger.warning(f'{output_label} already exists. Please pass overwrite=True to overwrite it.')
            return data

    # 2. Preprocessing

    # Prepare the data for clustering
    # filtered_data is a pandas df that contains the indeces of the data that passed the filters (var_func and dts_func)
    filtered_data, dims, importance_weights, scaled_coords = prepare_dataframe(
        data, var, shifts_label, var_filter_func, shifts_filter_func, scaler
    )

    # 3. Perform clustering
    logger.info(f'Applying clustering method {method}')
    clusters, method_params = method.apply(coords=scaled_coords, weights=importance_weights)

    if transpose_output:
        clusters = clusters.transpose()

    # 5. Convert back to xarray DataArray
    df_dims = data[dims].to_dataframe().reset_index()       # create a pandas df with original dims
    df_dims[output_label] = -1                              # Initialize cluster column with -1
    df_dims.loc[filtered_data.index, output_label] = clusters # Assign cluster labels to the dataframe
    cluster_labels = df_dims.set_index(dims).to_xarray()    # Convert dataframe to xarray (DataSet)
    cluster_labels = cluster_labels[output_label]           # select only cluster labels
    
    # 6. Save details as attributes
    cluster_labels.attrs.update({
        f'clusters': np.unique(clusters).astype(int),
        f"var_filter_func": inspect.getsource(var_filter_func) if var_filter_func else "None",
        f"shifts_filter_func": inspect.getsource(shifts_filter_func) if shifts_filter_func else "None",
        f"scaler": scaler,
        f'method': method.__class__.__name__,
    })

    # Add method params as separate attributes
    for param, value in method_params.items():
        cluster_labels.attrs[f'method_param_{param}'] = str(value) if value is not None else ''

    # add git version
    cluster_labels.attrs['git_version'] = __version__

    # 7. Merge the cluster labels back into the original data
    if merge_input:
        return xr.merge([data, cluster_labels], combine_attrs="override")
    else:
        return cluster_labels


