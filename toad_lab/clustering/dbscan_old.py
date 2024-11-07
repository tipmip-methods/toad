"""dbscan module

Uses the scipy dbscan clustering algorithm and a taylored preprocessing
pipeline.

October 22
"""

import xarray as xr
import numpy as np
from typing import Callable

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import DBSCAN

def construct_dataframe(
    data : xr.Dataset,
    var : str,
    var_func : Callable[[float], bool] = None,
    dts_func : Callable[[float], bool] = None,
    ):
    """Construct a dataframe from the data and apply the given functions.
    
    :param data:        xarray dataset
    :type data:         xr.Dataset
    :param var:         ...
    :type var:          str
    :param var_func:    ...
    :type var_func:     Callable[[float], bool]
    :param dts_func:    ...
    :type dts_func:     Callable[[float], bool]
    :return:            ...
    :rtype:             ...
    """
    df_var = data.get(var).to_dataframe().reset_index()
    df_dts = data.get(f'{var}_dts').to_dataframe().reset_index()

    if not var_func:
        var_func = np.vectorize(lambda x: True)
    if not dts_func:
        dts_func = np.vectorize(lambda x: True)

    var_mask = var_func(df_var.get(var))
    dts_mask = dts_func(df_dts.get(f'{var}_dts'))

    df = df_dts.loc[var_mask & dts_mask]

    return df_var, df_dts, df

def prepare_dataframe(
    data: xr.Dataset,
    var: str, 
    var_func : Callable[[float], bool] = None,
    dts_func : Callable[[float], bool] = None,
    scaler : str = 'StandardScaler'
    ):

    """Prepare the dataframe for clustering.

    :param data:        xarray dataset
    :type data:         xr.Dataset
    :param var:         ...
    :type var:          str
    :param var_func:    ...
    :type var_func:     Callable[[float], bool]
    :param dts_func:    ...
    :type dts_func:     Callable[[float], bool]
    :param scaler:      ...
    :type scaler:       str
    :return:            ...
    :rtype:             ...
    """
    # Data preparation: Transform into a dataframe and rescale the coordinates 
    df_var, df_dts, df = construct_dataframe(data, var, var_func, dts_func)
    dims = list(data.sizes.keys())
    coords = df[dims]
    vals = np.abs(df[[f'{var}_dts']])

    if scaler == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler == 'MinMaxScaler':
        scaler = MinMaxScaler() 
    
    scaled_coords = scaler.fit_transform(coords)

    return df_var, df_dts, df, dims, vals.values.flatten(), scaled_coords

# Main function called from outside ============================================
def cluster(
    data: xr.DataArray,
    var : str,
    eps : float,
    min_samples : int,
    min_abruptness : float,
    var_func : Callable[[float], bool] = None,
    dts_func : Callable[[float], bool] = None,
    scaler : str = 'StandardScaler'
    ):
    """Cluster the data using the DBSCAN algorithm.

    :param data:        xarray dataset
    :type data:         xr.Dataset
    :param var:         ...
    :type var:          str
    :param eps:         ...
    :type eps:          float
    :param min_samples: ...
    :type min_samples:  int
    :param var_func:    ...
    :type var_func:     Callable[[float], bool]
    :param dts_func:    ...
    :type dts_func:     Callable[[float], bool]
    :param scaler:      ...
    :type scaler:       str
    """

    # Define method details for logging
    method_details = f'dbscan (eps={eps}, min_samples={min_samples}, {scaler})'

    # Define default dts_func if not provided and min_abruptness is specified
    if dts_func is None and min_abruptness:
        dts_func = lambda x : np.abs(x) > min_abruptness

    # Prepare the dataframe for clustering
    df_var, df_dts, df, dims, weights, scaled_coords = prepare_dataframe(data, var, var_func, dts_func, scaler)

    # Fit clusters with DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit_predict(scaled_coords, sample_weight=weights)
    
    # Get the labels from the DBSCAN clustering
    lbl_dbscan = dbscan.labels_.astype(float)
    
    # Initialize cluster column with -1
    df_var[f'{var}_cluster'] = -1
    
    # Assign cluster labels to the dataframe
    df_var.loc[df.index, f'{var}_cluster'] = lbl_dbscan
    
    # Convert the dataframe back to xarray dataset
    clusters = df_var.set_index(dims).to_xarray()
    
    # Add identified cluster labels to dataset attributes
    clusters.attrs[f'{var}_clusters'] = np.unique(lbl_dbscan)

    # Add clustering method details to dataset attributes
    clusters.attrs[f'{var}_clustering_method'] = method_details

    # Return the dataset with cluster labels
    return clusters[[f'{var}_cluster']]