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


# Main function called from outside ============================================
def cluster(
    data: xr.DataArray,
    var : str,
    eps : float,
    min_samples : int,
    var_func : Callable[[float], bool] = None,
    dts_func : Callable[[float], bool] = None,
    scaler : str = 'StandardScaler'
):
    method_details = f'dbscan (eps={eps}, min_samples={min_samples}, {scaler})'
    print(var)

    # Data preparation: Transform into a dataframe and rescale the coordinates 
    df_var, df_dts, df = construct_dataframe(data, var, var_func, dts_func)
    dims = list(data.dims.keys())
    coords = df[dims]
    vals = df[[f'{var}_dts']]

    print(df_dts)

    if scaler == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler == 'MinMaxScaler':
        scaler = MinMaxScaler() 
    
    scaled_coords = scaler.fit_transform(coords)

    # Clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    y_pred = dbscan.fit_predict(
                        scaled_coords, 
                        sample_weight=vals.values.flatten()
                    )
    lbl_dbscan = dbscan.labels_.astype(float)
    labels = np.unique(lbl_dbscan)

    # Writing to dataset
    df_var[[f'{var}_cluster']] = -1
    df_var[[f'{var}_dts']] = df_dts[[f'{var}_dts']]
    df_var.loc[df.index, f'{var}_cluster'] = lbl_dbscan

    dataset_with_clusterlabels = df_var.set_index(dims).to_xarray()
    dataset_with_clusterlabels.attrs[f'{var}_clusters'] = labels

    dataset_with_clusterlabels.attrs[f'{var}_clustering_method'] = method_details

    return dataset_with_clusterlabels