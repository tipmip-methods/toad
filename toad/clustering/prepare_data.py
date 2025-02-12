import numpy as np
import pandas as pd
<<<<<<< HEAD
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
        time_dim: str,
        space_dims: list[str],
        var_func: Optional[Callable[[float], bool]] = None,
        dts_func: Optional[Callable[[float], bool]] = None,
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        scaler: Union[str, None] = 'StandardScaler'
=======
        scaler: str = 'StandardScaler'
>>>>>>> c6fc662 (Docstring and type fixes)
=======
        scaler: Union[str, None] = 'StandardScaler'
>>>>>>> 404d1af (Type fix)
=======
        scaler: Union[str, None] = 'StandardScaler',
>>>>>>> 9a2a22a (Dim fixes for clustering)
    ) -> tuple[pd.DataFrame, list, np.ndarray, np.ndarray]:
    """Prepare data for clustering by filtering, extracting coordinates, and scaling.
=======
import xarray as xr
from typing import Optional, Union, Callable
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler


# TODO: this is should be removed from package, just used as a dev helper function 
def prepare_dataframe(data: xr.Dataset, 
                        var: str, 
                        shifts_label: Optional[str] = None, 
                        time_dim: str = "time", 
                        space_dims: list[str] = ["lon", "lat"],
                        var_filter_func: Callable = lambda _: True,
                        shifts_filter_func: Callable = lambda _: True,
                        scaler: Optional[Union[StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler]] = StandardScaler()
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Helper function for getting clustering input data."""
>>>>>>> 7aa71ab (Minor clustering refactoring)

    shifts_label = shifts_label if shifts_label else f'{var}_dts'
        
    df_data = {
        'var': data[var].to_dataframe().reset_index(),
        'dts': data[shifts_label].to_dataframe().reset_index()
    }

<<<<<<< HEAD
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
=======
    # Create combined mask
    mask = (
        np.vectorize(var_filter_func)(df_data['var'][var]) & 
        np.vectorize(shifts_filter_func)(df_data['dts'][shifts_label])
    )
>>>>>>> 7aa71ab (Minor clustering refactoring)
    
    # Apply mask
    filtered_df = df_data['dts'][mask]
    if filtered_df.empty:
        raise ValueError('No data left after filtering.')

    # Get coordinates and scale them if needed
    dims = np.array([time_dim] + space_dims)
    coords = filtered_df[dims].to_numpy()
    coords = scaler.fit_transform(coords) if scaler else coords

    # Compute importance weights as the absolute values of the dts variable
    weights = np.abs(filtered_df[shifts_label].to_numpy())

    return filtered_df, dims, weights, coords