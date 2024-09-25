"""dbscan module

Uses the scipy dbscan clustering algorithm and a taylored preprocessing
pipeline.

October 22
"""

import xarray as xr
import numpy as np
from typing import Callable, List

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import DBSCAN

# ADAPTED FOR MULTIVARIATE ANALYSIS:

def construct_dataframe(
    # This function takes an xarray.Dataset 
    # and a list of variables to extract and process into Pandas dataframes, ready for further analysis
    data: xr.Dataset,
    vars: List[str],  # List of variables
    var_funcs: Callable[[float], bool] = None, # Single function or list of functions to filter/mask variable data
    dts_funcs: Callable[[float], bool] = None, # Single function or list of functions to filter/mask detection time series
    driver_var: str = None,  # Driver variable name
    driver_dts_func: Callable[[float], bool] = None,  # Masking function for driver variable
):
    df_vars = []
    df_dts_vars = []
    combined_var_mask = None # initialise this variable
    combined_dts_mask = None # initialise this variable 

    for idx, var in enumerate(vars): # for each variable
        df_var = data.get(var).to_dataframe().reset_index() # Convert time series to pandas
        df_dts = data.get(f'{var}_dts').to_dataframe().reset_index() # convert detection series to pandas

        # Apply a mask to both the time series data and the detection series using the functions in var_funcs and dts_funcs. 
        # If these functions are not provided, a default function is used, which passes all values (i.e., no filtering).

        # Handle the case where var_funcs is a single function, or a list of functions
        if callable(var_funcs):
            var_func = var_funcs  # Single function for all variables
        elif var_funcs and len(var_funcs) > idx:
            var_func = var_funcs[idx]  # Different function per variable
        else:
            var_func = np.vectorize(lambda x: True)  # Default mask: no filtering

        # Handle the case where dts_funcs is a single function, or a list of functions
        if callable(dts_funcs):
            dts_func = dts_funcs  # Single function for all variables
        elif dts_funcs and len(dts_funcs) > idx:
            dts_func = dts_funcs[idx]  # Different function per variable
        else:
            dts_func = np.vectorize(lambda x: True)  # Default mask: no filtering

        var_mask = var_func(df_var.get(var))
        dts_mask = dts_func(df_dts.get(f'{var}_dts'))

         # Combine the var_mask using AND logic: retain point only if all variables pass var_mask
        if combined_var_mask is None:
            combined_var_mask = var_mask  # Initialize combined_var_mask
        else:
            combined_var_mask &= var_mask  # AND across all variables' var_mask

        # Combine the dts_mask using OR logic: retain point if at least one variable's detection series passes
        if combined_dts_mask is None:
            combined_dts_mask = dts_mask  # Initialize combined_dts_mask
        else:
            combined_dts_mask |= dts_mask  # OR across all variables' dts_mask

        # Append to initialized lists (keeping the unmasked data for each variable)
        df_vars.append(df_var)
        df_dts_vars.append(df_dts)


    # Apply the driver variable mask if provided
    if driver_var and driver_dts_func:
        driver_dts_df = data.get(f'{driver_var}_dts').to_dataframe().reset_index()  # Convert driver variable's dts to pandas
        driver_dts_mask = driver_dts_func(driver_dts_df[f'{driver_var}_dts'])  # Apply driver_dts_func to the driver dts

        # Apply driver_mask to the combined var_mask and dts_mask
        combined_var_mask &= driver_dts_mask
        combined_dts_mask &= driver_dts_mask

    # Final combined mask: only retain points where all var_mask conditions are satisfied and at least one dts_mask is True
    combined_mask = combined_var_mask & combined_dts_mask

    # Apply the combined mask across all variables
    df_dts_vars_masked = [df_dts.loc[combined_mask] for df_dts in df_dts_vars]

    return df_vars, df_dts_vars, df_dts_vars_masked #Unmasked vars, unmasked dts, masked dts

def prepare_dataframe(
    data: xr.Dataset,
    vars: List[str], 
    var_funcs: Callable[[float], bool] = None,
    dts_funcs: Callable[[float], bool] = None,
    driver_var: str = None,  # Driver variable name
    driver_dts_func: Callable[[float], bool] = None,  # Masking function for driver variable
    scaler: str = 'StandardScaler'
):
    # Construct dataframes for all variables
    df_vars, df_dts_vars, df_dts_vars_masked = construct_dataframe(data, vars, var_funcs, dts_funcs, driver_var, driver_dts_func)

    # Collect all dimensions (common across all variables)
    dims = list(data.dims.keys())

    # Collect all coordinates and values across variables
    combined_coords = []
    combined_vals = []
    for i in range(len(vars)): # loop over all variables
        df = df_dts_vars_masked[i] # get the variable 
        coords = df[dims]
        vals = np.abs(df[[f'{vars[i]}_dts']])
        combined_coords.append(coords)
        combined_vals.append(vals.values.flatten())

    # Stack coordinates and values from all variables in one large array for clustering 
    stacked_coords = np.hstack(combined_coords)
    stacked_vals = np.column_stack(combined_vals)  # Stack values column-wise to form the multivariate feature space

    if scaler == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler == 'MinMaxScaler':
        scaler = MinMaxScaler() 

    scaled_coords = scaler.fit_transform(stacked_coords) # perform scaling (normalisation)
    scaled_vals = scaler.fit_transform(stacked_vals)  # Scale the combined values for clustering

    return df_vars, df_dts_vars, df_dts_vars_masked, dims, scaled_vals, scaled_coords

def cluster(
    data: xr.Dataset,  # Use xr.Dataset instead of xr.DataArray
    vars: List[str],
    eps: float, # The maximum distance between two points for them to be considered as neighbors in DBSCAN.
    min_samples: int, # The minimum number of samples in a neighborhood for a point to be considered a core point in DBSCAN.
    var_funcs: Callable[[float], bool] = None, # masking functions.
    dts_funcs: Callable[[float], bool] = None,
    driver_var: str = None,  # Driver variable name
    driver_dts_func: Callable[[float], bool] = None,  # Masking function for driver variable
    scaler: str = 'StandardScaler'
):
    method_details = f'dbscan (eps={eps}, min_samples={min_samples}, {scaler})'

    # Prepare the combined dataframe with multiple variables
    df_vars, df_dts_vars, df_dts_vars_masked, dims, scaled_vals, scaled_coords = prepare_dataframe(
        data, vars, var_funcs, dts_funcs, driver_var, driver_dts_func, scaler
    )

    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    # Call fit_predict on the scaled coordinates to assign cluster labels (y_pred). 
    # DBSCAN returns -1 for noise points and a unique integer for each cluster.
    y_pred = dbscan.fit_predict(np.hstack([scaled_coords, scaled_vals]))  # Combine coords and values for clustering
    lbl_dbscan = dbscan.labels_.astype(float)
    labels = np.unique(lbl_dbscan)

    datasets = [] # Empty list to store all xr.datasets for each variable

    # Writing to dataset (for each variable, store clustering results as attribute)
    for i, var in enumerate(vars):
        # Create a new column in the dataframe ({var}_cluster) to store the cluster labels.
        df_vars[i][[f'{var}_cluster']] = -1 # Initialise with -1 for unclustered points
        # keep detection series
        df_vars[i][[f'{var}_dts']] = df_dts_vars[i][[f'{var}_dts']]
        # Store the cluster labels
        df_vars[i].loc[df_dts_vars_masked[i].index, f'{var}_cluster'] = lbl_dbscan
        # Transfer to xarray dataset
        df_var_xarray = df_vars[i].set_index(dims).to_xarray()
        # Append to datasets 
        datasets.append(df_var_xarray)

    # Combine the different variable datasets 
    cluster_dataset = xr.merge(datasets)
    
    # The cluster labels and method details (DBSCAN parameters) are also saved as attributes in the resulting dataset.
    cluster_dataset.attrs[f'{vars}_clusters'] = labels
    cluster_dataset.attrs[f'{vars}_clustering_method'] = method_details

    return cluster_dataset 

## ORIGINAL CODE FOR UNIVARIATE ANALYSIS BELOW:
# def construct_dataframe(
#     data : xr.Dataset,
#     var : str,
#     var_func : Callable[[float], bool] = None,
#     dts_func : Callable[[float], bool] = None,
# ):
#     df_var = data.get(var).to_dataframe().reset_index()
#     df_dts = data.get(f'{var}_dts').to_dataframe().reset_index()

#     if not var_func:
#         var_func = np.vectorize(lambda x: True)
#     if not dts_func:
#         dts_func = np.vectorize(lambda x: True)

#     var_mask = var_func(df_var.get(var))
#     dts_mask = dts_func(df_dts.get(f'{var}_dts'))

#     df = df_dts.loc[var_mask & dts_mask]

#     return df_var, df_dts, df

# def prepare_dataframe(
#     data: xr.Dataset,
#     var: str, 
#     var_func : Callable[[float], bool] = None,
#     dts_func : Callable[[float], bool] = None,
#     scaler : str = 'StandardScaler'
# ):
#     # Data preparation: Transform into a dataframe and rescale the coordinates 
#     df_var, df_dts, df = construct_dataframe(data, var, var_func, dts_func)
#     dims = list(data.dims.keys())
#     coords = df[dims]
#     vals = np.abs(df[[f'{var}_dts']])

#     if scaler == 'StandardScaler':
#         scaler = StandardScaler()
#     elif scaler == 'MinMaxScaler':
#         scaler = MinMaxScaler() 
    
#     scaled_coords = scaler.fit_transform(coords)

#     return df_var, df_dts, df, dims, vals.values.flatten(), scaled_coords

# # Main function called from outside ============================================
# def cluster(
#     data: xr.DataArray,
#     var : str,
#     eps : float,
#     min_samples : int,
#     var_func : Callable[[float], bool] = None,
#     dts_func : Callable[[float], bool] = None,
#     scaler : str = 'StandardScaler'
# ):
#     method_details = f'dbscan (eps={eps}, min_samples={min_samples}, {scaler})'

#     df_var, df_dts, df, dims, weights, scaled_coords = prepare_dataframe(data, var, var_func, dts_func, scaler)

#     # Clustering
#     dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#     y_pred = dbscan.fit_predict(
#                         scaled_coords, 
#                         sample_weight=weights
#                     )
#     lbl_dbscan = dbscan.labels_.astype(float)
#     labels = np.unique(lbl_dbscan)

#     # Writing to dataset
#     df_var[[f'{var}_cluster']] = -1
#     df_var[[f'{var}_dts']] = df_dts[[f'{var}_dts']]
#     df_var.loc[df.index, f'{var}_cluster'] = lbl_dbscan

#     dataset_with_clusterlabels = df_var.set_index(dims).to_xarray()
#     dataset_with_clusterlabels.attrs[f'{var}_clusters'] = labels

#     dataset_with_clusterlabels.attrs[f'{var}_clustering_method'] = method_details

#     return dataset_with_clusterlabels