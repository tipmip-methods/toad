from typing import Callable
import xarray as xr
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Optional
import pandas as pd
<<<<<<< HEAD
<<<<<<< HEAD
from typing import Union
=======
>>>>>>> c6fc662 (Docstring and type fixes)
=======
from typing import Union
>>>>>>> 404d1af (Type fix)


def prepare_dataframe(
        data: xr.Dataset,
        var: str, 
        var_dts: str,
        var_func: Optional[Callable[[float], bool]] = None,
        dts_func: Optional[Callable[[float], bool]] = None,
<<<<<<< HEAD
<<<<<<< HEAD
        scaler: Union[str, None] = 'StandardScaler'
=======
        scaler: str = 'StandardScaler'
>>>>>>> c6fc662 (Docstring and type fixes)
=======
        scaler: Union[str, None] = 'StandardScaler'
>>>>>>> 404d1af (Type fix)
    ) -> tuple[pd.DataFrame, list, np.ndarray, np.ndarray]:
    """Prepare data for clustering by filtering, extracting coordinates, and scaling.

    This function converts specified variables from an xarray Dataset to Pandas
    DataFrames, applies optional filtering functions, and scales the coordinates 
    for clustering. It also calculates importance weights based on the detection
    time series (dts) variable.

<<<<<<< HEAD
    >> Args:
        data:
            The input xarray Dataset containing the variable to be clustered and its detection time series.
        var:
            The name of the variable in the Dataset to be clustered.
        var_dts:
            The name of the detection time series variable in the Dataset.
        var_func:
            A function to filter the `var` values. Defaults to keeping all values.
        dts_func:
            A function to filter the `dts` values. Defaults to keeping all values.
        scaler:
            The scaler to use for normalizing coordinates ('StandardScaler', 'MinMaxScaler', or None). Defaults to 'StandardScaler'.

    >> Returns:
        tuple:
            A tuple containing:
=======
    Args:
        data: The input xarray Dataset containing the variable to be clustered and its detection time series.
        var: The name of the variable in the Dataset to be clustered.
        var_dts: The name of the detection time series variable in the Dataset.
        var_func: A function to filter the `var` values. Defaults to keeping all values.
        dts_func: A function to filter the `dts` values. Defaults to keeping all values.
        scaler: The scaler to use for normalizing coordinates ('StandardScaler', 'MinMaxScaler', or None). Defaults to 'StandardScaler'.

    Returns:
        tuple: A tuple containing:
>>>>>>> c6fc662 (Docstring and type fixes)
            - A Pandas DataFrame of the filtered data, including the original coordinates and the `dts` variable
            - A list of dimension names from the original xarray Dataset
            - A 1D NumPy array of the absolute values of the `dts` variable, used as sample weights
            - A 2D NumPy array of the scaled coordinates for clustering

    >> Raises:
        ValueError:
            If no data remains after filtering.
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

    # Scale if scaler is not None
    if scaler == 'StandardScaler':
        coords = StandardScaler().fit_transform(coords)
    elif scaler == 'MinMaxScaler':
        coords = MinMaxScaler().fit_transform(coords)
    elif scaler is not None:
        raise ValueError(f"Invalid scaler: {scaler}. Please choose 'StandardScaler', 'MinMaxScaler' or None.")

    # Compute importance weights as the absolute values of the dts variable
    importance_weights = np.abs(filtered_data_pandas[var_dts].to_numpy())

    return filtered_data_pandas, dims, importance_weights, coords