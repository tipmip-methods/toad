import warnings
import functools
from typing import Union, Optional, Tuple
import xarray as xr
import numpy as np


def get_space_dims(xr_da: Union[xr.DataArray, xr.Dataset], tdim: Optional[str] = None) -> list[str]:
    """Get spatial dimensions from an xarray DataArray or Dataset.

    Args:
        xr_da: Input DataArray or Dataset to get dimensions from
        tdim: Optional name of temporal dimension. If provided, all other dims are considered spatial.
            If not provided, attempts to auto-detect spatial dims based on standard names.

    Returns:
        List of spatial dimension names as strings

    See Also:
        infer_dims: For full dimension inference including temporal dimension
    """
    return infer_dims(xr_da, tdim)[1]
        
        
def infer_dims(
    xr_da: Union[xr.DataArray, xr.Dataset], 
    tdim: Optional[str] = None
) -> Tuple[str, list[str]]:
    """
    Infers the temporal and spatial dimensions from an xarray DataArray or Dataset.

    Args:
        xr_da: The input DataArray or Dataset from which to infer dimensions.
        tdim: (Optional) The name of the temporal dimension. If provided, it will be used to 
            distinguish between temporal and spatial dimensions. If not provided, 
            the function will attempt to auto-detect the temporal dimension based 
            on standard spatial dimension names.

    Returns:
        - A tuple containing the time dimension as a string and a list of spatial dimensions as strings.

    Raises:
        ValueError: If the provided temporal dimension is not in the dimensions of the dataset.
        ValueError: If unable to infer temporal and spatial dimensions.

    Notes:
        - If `tdim` is provided, the function will use it to identify the temporal 
          dimension and consider all other dimensions as spatial.
        - If `tdim` is not provided, the function will attempt to auto-detect the 
          temporal dimension by looking for standard spatial dimension pairs such as 
          ('x', 'y'), ('lat', 'lon'), or ('latitude', 'longitude').
        
    Examples:
        >>> infer_dims(dataset)
        ('time', ['x', 'y'])
    """

    # Spatial dims are all non-temporal dims
    if tdim:
        sdims = list(xr_da.dims)
        if tdim not in xr_da.dims:
            raise ValueError(f"Provided temporal dim '{tdim}' is not in the dimensions of the dataset!")
        sdims.remove(tdim)
        sdims = [str(dim) for dim in sdims]
        sdims = sorted(sdims)
        return tdim, sdims
    else:
        # Auto-detect standard spatial dimension combinations
        for pair in [('x', 'y'), ('lat', 'lon'), ('latitude', 'longitude')]:
            if all(dim in xr_da.dims for dim in pair):
                sdims = list(pair)
                remaining_dims = [dim for dim in xr_da.dims if dim not in sdims]
                if len(remaining_dims) == 1:  # Ensure a single temporal dimension is left
                    tdim = str(remaining_dims[0])  # Explicitly convert to str
                    return tdim, sdims

        raise ValueError("Unable to infer temporal and spatial dimensions. Please provide `tdim` explicitly.")


def deprecated(message=None):
    """ Mark functions as deprecated with @deprecated decorator"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warn_message = message if message else f"{func.__name__} is deprecated and will be removed in a future version."
            warnings.warn(
                warn_message,
                category=DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator


def all_functions(obj) -> list[str]:
    return [x for x in dir(obj) if callable(getattr(obj, x)) and not x.startswith('__')]


def is_equal_to(x, value):
    """Check if x equals value, whether x is a scalar or sequence."""
    if np.isscalar(x):
        return x == value
    return np.array_equal(x, [value])


def contains_value(x, value):
    """Check if x contains value, whether x is a scalar or sequence."""
    if np.isscalar(x):
        return x == value
    return value in x