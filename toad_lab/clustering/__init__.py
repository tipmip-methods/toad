import logging
import xarray as xr
import numpy as np
from typing import Union
from typing import Callable
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from _version import __version__

from .method_dictionary import clustering_methods
from ..utils import deprecated

logger = logging.getLogger("TOAD")

def compute_clusters(
        data: xr.Dataset,
        var : str,
        var_dts: str = None,
        min_abruptness: float = None,
        method : Union [str, callable] = "hdbscan",
        var_func: Callable[[float], bool] = None,
        dts_func: Callable[[float], bool] = None,
        scaler: str = 'StandardScaler',
        output_label: str = None,
        overwrite: bool = False,
        merge_input: bool = True,
        transpose_output: bool = False,
        **method_kwargs
    ) -> xr.Dataset:
    """
    Apply a clustering algorithm to the dataset along the temporal dimension.

    This function clusters data points in the temporal dimension using a specified clustering algorithm 
    (default is HDBSCAN). It filters the data using specified conditions on the primary variable and its 
    computed shifts before performing clustering. Results are returned as a new xarray.DataArray containing 
    the cluster labels.

    :param data: Data with two spatial and one temporal dimension. Must be an `xarray.Dataset`.
    :type data: xr.Dataset
    :param var: The name of the variable in the dataset to cluster.
    :type var: str
    :param var_dts: The name of the variable containing precomputed shifts (e.g., `{var}_dts`). Defaults to `{var}_dts` if not provided.
    :type var_dts: str, optional
    :param min_abruptness: Minimum threshold for abruptness to filter shifts. Must be provided if `dts_func` is not supplied.
    :type min_abruptness: float, optional
    :param method: The clustering algorithm to use, either a string referring to a predefined method or a callable. 
                   Default is "hdbscan".
    :type method: Union[str, callable]
    :param var_func: A callable to filter the primary variable before clustering. Defaults to `None`.
    :type var_func: Callable[[float], bool], optional
    :param dts_func: A callable to filter the shifts before clustering. Defaults to `None`.
    :type dts_func: Callable[[float], bool], optional
    :param scaler: The scaling method to apply to the data before clustering. Default is 'StandardScaler'.
    :type scaler: str
    :param output_label: The name of the variable in the dataset to store the clustering results. 
                         Defaults to `{var}_cluster` if not provided.
    :type output_label: str, optional
    :param overwrite: Whether to overwrite the existing variable in the dataset if `output_label` already exists. 
                      Default is `False`.
    :type overwrite: bool
    :param merge_input: Whether to merge the clustering results back into the original dataset. Default is `True`.
    :type merge_input: bool
    :param transpose_output: Whether to transpose the output DataArray. Sometimes needed... Default is `False`.
    :type transpose_output: bool
    :param method_kwargs: Additional keyword arguments specific to the clustering algorithm.
    :type method_kwargs: dict, optional
    :return: An xarray.DataArray containing cluster labels for the data points.
    :rtype: xr.DataArray

    :raises AssertionError: If `data` is not an `xarray.Dataset`, if `data` does not have three dimensions, 
                             or if neither `min_abruptness` nor `dts_func` is provided.
    :raises ValueError: If `var_dts` is not found in the dataset, if `method` is invalid, or if `output_label`
                        conflicts with an existing variable and `overwrite` is `False`.

    Notes:
    - The `method` can either be a string referring to a predefined clustering method or a custom callable. 
      Predefined methods are stored in the `clustering_methods` dictionary.
    - If both `var_func` and `dts_func` are provided, data points must pass both filters to be included in clustering.
    - The function automatically applies scaling to the data based on the specified `scaler`.

    TODO: (1) Fix: should also return auxillary coordinates. For now only returns coords in dims. 
    TODO: (2) coordinates are sometimes flipped in the output. 
    TODO: (3) Find out why transpose_output is needed for antarctica data. Related to (2)? 
    """
    assert type(data) == xr.Dataset, 'data must be an xr.DataSet!'
    assert data.get(var).ndim == 3, 'data must be 3-dimensional!'
    assert min_abruptness is not None or dts_func is not None, 'either min_abruptness or dts_func must be provided'

    # Check shifts var
    all_vars = list(data.data_vars.keys())
    var_dts = var_dts if var_dts else f'{var}_dts'  # default to {var}_dts
    assert var_dts in all_vars, f'Please run shifts on {var} first, or provide a custom "shifts" variable'

    # 1. Get the clustering method
    if callable(method):
        clusterer = method
    elif type(method) == str:
        logger.info(f'looking up clusterer {method}')
        clusterer = clustering_methods[method]
    else:
        raise ValueError('method must be a string or a callable') 

    # 2. Check if the output_label is already in the data
    default_name = f'{var}_cluster'
    output_label = output_label or default_name
    if output_label in data and merge_input:
        if overwrite:
            logger.warning(f'Overwriting variable {output_label}')
            data = data.drop_vars(output_label)
        else:
            logger.warning(f'{output_label} already exists. Please pass overwrite=True to overwrite it.')
            return data

    # 3. Preprocessing
    # Set a default abruptness filter if no custom dts_func provided
    dts_func = dts_func if dts_func else lambda x: np.abs(x) > min_abruptness

    # Prepare the data for clustering
    # filtered_data is a pandas df that contains the indeces of the data that passed the filters (var_func and dts_func)
    filtered_data, dims, importance_weights, scaled_coords = prepare_dataframe(
        data, var, var_dts, var_func, dts_func, scaler
    )

    # 4. Perform clustering
    logger.info(f'applying clusterer {method} to data')
    clusters, method_details = clusterer(
        coords=scaled_coords, 
        weights=importance_weights,
        **method_kwargs
    )

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


def prepare_dataframe(
        data: xr.Dataset,
        var: str, 
        var_dts: str,
        var_func: Callable[[float], bool] = None,
        dts_func: Callable[[float], bool] = None,
        scaler: str = 'StandardScaler'
    ):
    """Prepare data for clustering by filtering, extracting coordinates, and scaling.

    This function converts the specified variables from an xarray Dataset to Pandas
    DataFrames, applies optional filtering functions and scales the coordinates 
    for clustering. It also calculates importance weights based on the detection
    time series (dts) variable.

    Args:
        data (xr.Dataset): The input xarray Dataset containing the variable to be 
            clustered and its detection time series (dts).
        var (str): The name of the variable in the Dataset to be clustered.
        var_func (Callable[[float], bool], optional): A function to filter the 
            `var` values. Defaults to a function that keeps all values.
        dts_func (Callable[[float], bool], optional): A function to filter the 
            `dts` values. Defaults to a function that filters values based on 
            `min_abruptness` in the calling function.
        scaler (str, optional): The scaler to use for normalizing coordinates. 
            Options are 'StandardScaler' or 'MinMaxScaler'. Defaults to 'StandardScaler'.

    Returns:
        tuple: A tuple containing:
            - filtered_data_pandas (pd.DataFrame): A Pandas DataFrame of the 
              filtered data, including the original coordinates and the `dts` variable.
            - dims (list): A list of dimension names from the original xarray Dataset.
            - importance_weights (np.ndarray): A 1D NumPy array of the absolute values 
              of the `dts` variable, used as sample weights for clustering.
            - scaled_coords (np.ndarray): A 2D NumPy array of the scaled coordinates 
              for clustering.

    Raises:
        ValueError: If no data remains after filtering.
    """
    
    # Convert the specified variables to Pandas DataFrames
    var_data = data[var].to_dataframe().reset_index()
    dts_data = data[var_dts].to_dataframe().reset_index()

    # Apply filtering functions, defaulting to keeping all values if not provided
    var_func = var_func if var_func else lambda x: True
    dts_func = dts_func if dts_func else lambda x: True

    # Use vectorized filtering to create masks
    var_mask = np.vectorize(var_func)(var_data[var])
    dts_mask = np.vectorize(dts_func)(dts_data[var_dts])
    filtered_data_pandas = dts_data.loc[var_mask & dts_mask]

    # throw error if no data left
    if filtered_data_pandas.empty:
        raise ValueError('No data left after filtering.')

    # Extract dimension names and coordinates
    # dims = list(data.sizes.keys())
    dims = list(data[var].dims) # take var dims instead of dataset dims, as they may not be the same.
    coords = filtered_data_pandas[dims].to_numpy()

    # Choose the scaler and scale the coordinates
    scaler_instance = StandardScaler() if scaler == 'StandardScaler' else MinMaxScaler()
    scaled_coords = scaler_instance.fit_transform(coords)

    # Compute importance weights as the absolute values of the dts variable
    importance_weights = np.abs(filtered_data_pandas[var_dts].to_numpy())

    return filtered_data_pandas, dims, importance_weights, scaled_coords