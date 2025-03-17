import numpy as np
import pandas as pd
import xarray as xr
from typing import Optional, Union, Callable
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
)


# TODO: this is should be removed from package, just used as a dev helper function
def prepare_dataframe(
    data: xr.Dataset,
    var: str,
    shifts_label: Optional[str] = None,
    time_dim: str = "time",
    space_dims: list[str] = ["lon", "lat"],
    var_filter_func: Callable = lambda _: True,
    shifts_filter_func: Callable = lambda _: True,
    scaler: Optional[
        Union[StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler]
    ] = StandardScaler(),
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Helper function for getting clustering input data."""

    shifts_label = shifts_label if shifts_label else f"{var}_dts"

    df_data = {
        "var": data[var].to_dataframe().reset_index(),
        "dts": data[shifts_label].to_dataframe().reset_index(),
    }

    # Create combined mask
    mask = np.vectorize(var_filter_func)(df_data["var"][var]) & np.vectorize(
        shifts_filter_func
    )(df_data["dts"][shifts_label])

    # Apply mask
    filtered_df = df_data["dts"][mask]
    if filtered_df.empty:
        raise ValueError("No data left after filtering.")

    # Get coordinates and scale them if needed
    dims = np.array([time_dim] + space_dims)
    coords = filtered_df[dims].to_numpy()
    coords = scaler.fit_transform(coords) if scaler else coords

    # Compute importance weights as the absolute values of the dts variable
    weights = np.abs(filtered_df[shifts_label].to_numpy())

    return filtered_df, dims, weights, coords