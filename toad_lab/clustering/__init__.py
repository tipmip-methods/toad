import logging
import xarray as xr
import numpy as np
from typing import Union
from typing import Callable
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from _version import __version__

from .method_dictionary import clustering_methods
from ..utils import deprecated

def compute_clusters(
        data: xr.Dataset,
        var : str,
        min_abruptness: float = None,
        method : Union [str, callable] = "hdbscan",
        var_func: Callable[[float], bool] = None,
        dts_func: Callable[[float], bool] = None,
        scaler: str = 'StandardScaler',
        output_label: str = None,
        **method_kwargs
    ) -> xr.Dataset:
    """
    Map a clustering algorithm to the dataset in the temporal dimension.

    :param data:            Data with two spatial and one temporal dimension.
    :type data:             xr.Dataset
    :param var:             Variable to cluster.
    :type var:              str
    :param method:          Clustering algorithm to use.
    :type method:           str
    :param method_kwargs:   Kwargs that need to be specifically passed to the clustering algorithm.
    :type method_kwargs:    dict, optional

    TODO: Fix: should also return auxillary coordinates. For now only returns coords in dims. 
    """
    assert type(data) == xr.Dataset, 'data must be an xr.DataSet!'
    assert data.get(var).ndim == 3, 'data must be 3-dimensional!'
    assert f'{var}_dts' in list(data.data_vars.keys()), f'data lacks detection time series {var}_dts'
    assert min_abruptness is not None or dts_func is not None, 'either min_abruptness or dts_func must be provided'

    # 1. Get the clustering method
    if callable(method):
        clusterer = method
    elif type(method) == str:
        logging.info(f'looking up clusterer {method}')
        clusterer = clustering_methods[method]
    else:
        raise ValueError('method must be a string or a callable') 

    # 2. Preprocessing
    # Set a default abruptness filter if no custom dts_func provided
    dts_func = dts_func if dts_func else lambda x: np.abs(x) > min_abruptness

    # Prepare the data for clustering
    filtered_data, dims, importance_weights, scaled_coords = prepare_dataframe(
        data, var, var_func, dts_func, scaler
    )

    # 3. Perform clustering
    logging.info(f'applying clusterer {method} to data')
    clusters, method_details = clusterer(
        coords=scaled_coords, 
        weights=importance_weights,
        **method_kwargs
    )

    # 4. Convert back to xarray DataArray
    output_label = output_label if output_label else f'{var}_cluster'
    df_dims = data[dims].to_dataframe().reset_index()       # create a pandas df with original dims
    df_dims[output_label] = -1                              # Initialize cluster column with -1
    df_dims.loc[filtered_data.index, output_label] = clusters # Assign cluster labels to the dataframe
    cluster_labels = df_dims.set_index(dims).to_xarray()    # Convert dataframe to xarray DataArray

    # 5. Save details as attributes
    # Save gitversion to dataset : TODO add this to _clustering_method attr? 
    cluster_labels.attrs[f'{var}_git_cluster'] = __version__

    # Convert to xarray DataArray and add attributes
    # make this attr dynamic if data already has a clustering method
    cluster_labels.attrs.update({
        f'{var}_clusters': np.unique(clusters),
        f'{var}_clustering_method': f'{method_details} {scaler}'
    })

    return cluster_labels[output_label]


def prepare_dataframe(
        data: xr.Dataset,
        var: str, 
        var_func: Callable[[float], bool] = None,
        dts_func: Callable[[float], bool] = None,
        scaler: str = 'StandardScaler'
    ):
    """Prepare data for clustering by filtering, extracting coordinates, and scaling.

    This function converts the specified variables from an xarray Dataset to Pandas
    DataFrames, applies optional filtering functions, and scales the coordinates 
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
    dts_data = data[f'{var}_dts'].to_dataframe().reset_index()

    # Apply filtering functions, defaulting to keeping all values if not provided
    var_func = var_func if var_func else lambda x: True
    dts_func = dts_func if dts_func else lambda x: True

    # Use vectorized filtering to create masks
    var_mask = np.vectorize(var_func)(var_data[var])
    dts_mask = np.vectorize(dts_func)(dts_data[f'{var}_dts'])
    filtered_data_pandas = dts_data.loc[var_mask & dts_mask]

    # throw error if no data left
    if filtered_data_pandas.empty:
        raise ValueError('No data left after filtering.')

    # Extract dimension names and coordinates
    dims = list(data.sizes.keys())
    coords = filtered_data_pandas[dims].to_numpy()

    # Choose the scaler and scale the coordinates
    scaler_instance = StandardScaler() if scaler == 'StandardScaler' else MinMaxScaler()
    scaled_coords = scaler_instance.fit_transform(coords)

    # Compute importance weights as the absolute values of the dts variable
    importance_weights = np.abs(filtered_data_pandas[f'{var}_dts'].to_numpy())

    return filtered_data_pandas, dims, importance_weights, scaled_coords


@deprecated("cluster is deprecated. Please use compute_clusters instead.")
def cluster(
        data: xr.Dataset,
        var : str,
        method : str = None,
        method_func : callable = None,
        method_kwargs = {}
    ) -> xr.Dataset:
    return compute_clusters(data, var, method, method_func, method_kwargs)
    