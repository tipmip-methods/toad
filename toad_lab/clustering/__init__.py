import logging
import xarray as xr
import numpy as np
from typing import Callable
from _version import __version__

from toad_lab.clustering.prepare_data import prepare_dataframe
from toad_lab.clustering.methods.base import ClusteringMethod


logger = logging.getLogger("TOAD")

def compute_clusters(
        data: xr.Dataset,
        var : str,
        var_dts: str = None,
        min_abruptness: float = None,
        method : ClusteringMethod = None,
        var_func: Callable[[float], bool] = None,
        dts_func: Callable[[float], bool] = None,
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

    assert type(data) == xr.Dataset, 'data must be an xr.DataSet!'
    assert data.get(var).ndim == 3, 'data must be 3-dimensional!'
    assert min_abruptness is not None or dts_func is not None, 'either min_abruptness or dts_func must be provided'

    # Check shifts var
    all_vars = list(data.data_vars.keys())
    var_dts = var_dts if var_dts else f'{var}_dts'  # default to {var}_dts
    assert var_dts in all_vars, f'Please run shifts on {var} first, or provide a custom "shifts" variable'

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
    # Set a default abruptness filter if no custom dts_func provided
    dts_func = dts_func if dts_func else lambda x: np.abs(x) > min_abruptness

    # Prepare the data for clustering
    # filtered_data is a pandas df that contains the indeces of the data that passed the filters (var_func and dts_func)
    filtered_data, dims, importance_weights, scaled_coords = prepare_dataframe(
        data, var, var_dts, var_func, dts_func, scaler
    )

    # 3. Perform clustering
    logger.info(f'Applying clustering method {method}')
    clusters, method_details = method.apply(coords=scaled_coords, weights=importance_weights)

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
        f'method': f'{method_details} with {scaler} and min_abruptness={min_abruptness}',
        f'_git_version': __version__
    })

    # 7. Merge the cluster labels back into the original data
    if merge_input:
        return xr.merge([data, cluster_labels], combine_attrs="override")
    else:
        return cluster_labels


