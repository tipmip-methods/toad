from typing import Callable
import xarray as xr
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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