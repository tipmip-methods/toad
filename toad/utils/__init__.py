import warnings
import functools
from typing import Union
import xarray as xr
<<<<<<< HEAD
<<<<<<< HEAD
import numpy as np
<<<<<<< HEAD
import os
import requests
import zipfile
<<<<<<< HEAD
=======
import numpy as np
>>>>>>> 7d33054 ([Breaking changes] Refactored timeseries and Clustering + stats)
=======
>>>>>>> d35b270 (Merge TOADtorial repo with toad repo)
=======
>>>>>>> ffe41d0 (Added optional regridding for clustering)


def get_space_dims(xr_da: Union[xr.DataArray, xr.Dataset], tdim: str) -> list[str]:
    """Get spatial dimensions from an xarray DataArray or Dataset.

<<<<<<< HEAD
<<<<<<< HEAD
    >> Args:
        xr_da:
            Input DataArray or Dataset to get dimensions from
        tdim:
            Optional name of temporal dimension. If provided, all other dims are considered spatial. 
            If not provided, attempts to auto-detect spatial dims based on standard names.
=======
    Args:
        xr_da: Input DataArray or Dataset to get dimensions from
        tdim: Name of temporal dimension. All other dims are considered spatial.
>>>>>>> ffe41d0 (Added optional regridding for clustering)

    Returns:
        List of spatial dimension names as strings. If standard spatial dims
        (x/y, y/x) or (lon/lat, lat/lon) are found, returns only those.

<<<<<<< HEAD
    >> See Also:
        infer_dims:
            For full dimension inference including temporal dimension
=======
    Args:
        xr_da: Input DataArray or Dataset to get dimensions from
        tdim: Optional name of temporal dimension. If provided, all other dims are considered spatial.
            If not provided, attempts to auto-detect spatial dims based on standard names.

    Returns:
        List of spatial dimension names as strings

    See Also:
        infer_dims: For full dimension inference including temporal dimension
>>>>>>> 7d33054 ([Breaking changes] Refactored timeseries and Clustering + stats)
=======
    Raises:
        ValueError: If provided temporal dim is not in the dimensions
>>>>>>> ffe41d0 (Added optional regridding for clustering)
    """
    if tdim not in xr_da.dims:
        raise ValueError(f"Provided temporal dim '{tdim}' is not in the dimensions!")

    # Check for standard spatial dim pairs
    for pair in [("x", "y"), ("lon", "lat")]:
        if all(dim in xr_da.dims for dim in pair):
            return sorted(
                list(pair), key=lambda x: list(xr_da.dims).index(x)
            )  # keep original order from xr_da

    # Fallback: use all non-temporal dims
    sdims = list(xr_da.dims)
    sdims.remove(tdim)
    return [str(dim) for dim in sdims if "bnds" not in str(dim)]


def reorder_space_dims(space_dims: list[str]) -> list[str]:
    """Reorder space dimensions to ensure lat comes before lon if both present.

    Args:
        space_dims: List of spatial dimension names
<<<<<<< HEAD
        
<<<<<<< HEAD
        
<<<<<<< HEAD
=======


>>>>>>> c6fc662 (Docstring and type fixes)
=======
>>>>>>> 7d33054 ([Breaking changes] Refactored timeseries and Clustering + stats)
def infer_dims(
    xr_da: Union[xr.DataArray, xr.Dataset], 
    tdim: Optional[str] = None
) -> Tuple[str, list[str]]:
=======
=======

>>>>>>> 6ffac35 (Formatted codebase with Ruff)
    Returns:
        Reordered list with lat before lon if both present, otherwise original list
>>>>>>> ffe41d0 (Added optional regridding for clustering)
    """
    if all(dim in space_dims for dim in ["lat", "lon"]):
        return [dim for dim in ["lat", "lon"] if dim in space_dims] + [
            dim for dim in space_dims if dim not in ["lat", "lon"]
        ]
    return space_dims

<<<<<<< HEAD
<<<<<<< HEAD
    >> Args:
        xr_da:
            The input DataArray or Dataset from which to infer dimensions.
        >> tdim: (Optional)
            The name of the temporal dimension. If provided, it will be used to 
=======
    Args:
        xr_da: The input DataArray or Dataset from which to infer dimensions.
        tdim: (Optional) The name of the temporal dimension. If provided, it will be used to 
>>>>>>> c6fc662 (Docstring and type fixes)
            distinguish between temporal and spatial dimensions. If not provided, 
            the function will attempt to auto-detect the temporal dimension based 
            on standard spatial dimension names.

<<<<<<< HEAD
    >> Returns:
        - A tuple containing the time dimension as a string and a list of spatial dimensions as strings.

    >> Raises:
        ValueError:
            If the provided temporal dimension is not in the dimensions of the dataset.
        ValueError:
            If unable to infer temporal and spatial dimensions.

    >> Notes:

=======
    Returns:
        - A tuple containing the time dimension as a string and a list of spatial dimensions as strings.

    Raises:
        ValueError: If the provided temporal dimension is not in the dimensions of the dataset.
        ValueError: If unable to infer temporal and spatial dimensions.

    Notes:
>>>>>>> c6fc662 (Docstring and type fixes)
        - If `tdim` is provided, the function will use it to identify the temporal 
          dimension and consider all other dimensions as spatial.
        - If `tdim` is not provided, the function will attempt to auto-detect the 
          temporal dimension by looking for standard spatial dimension pairs such as 
          ('x', 'y'), ('lat', 'lon'), or ('latitude', 'longitude').
        
<<<<<<< HEAD
    >> Examples:
=======
    Examples:
>>>>>>> c6fc662 (Docstring and type fixes)
        >>> infer_dims(dataset)
        ('time', ['x', 'y'])
    """

    # Spatial dims are all non-temporal dims
    if tdim:
        sdims = list(xr_da.dims)
        if tdim not in xr_da.dims:
            raise ValueError(f"Provided temporal dim '{tdim}' is not in the dimensions of the dataset!")
        sdims.remove(tdim)

        # remove any dims that contains the bnds
        sdims = [dim for dim in sdims if 'bnds' not in dim]

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
=======
>>>>>>> ffe41d0 (Added optional regridding for clustering)

def deprecated(message=None):
    """Mark functions as deprecated with @deprecated decorator"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warn_message = (
                message
                if message
                else f"{func.__name__} is deprecated and will be removed in a future version."
            )
            warnings.warn(warn_message, category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def all_functions(obj) -> list[str]:
    return [x for x in dir(obj) if callable(getattr(obj, x)) and not x.startswith("__")]


def is_equal_to(x, value):
    """Check if x equals value, whether x is a scalar or sequence."""
    if np.isscalar(x):
        return x == value
    return np.array_equal(x, [value])


def contains_value(x, value):
    """Check if x contains value, whether x is a scalar or sequence."""
    if np.isscalar(x):
        return x == value
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> d35b270 (Merge TOADtorial repo with toad repo)
    return value in x


# Include this once we have a published release to fetch test data
# def download_test_data():
#     """Download test data sets

#     """
#     url = "https://github.com/tipmip-methods/toad/releases/download/[TAG_NAME]/test_data.zip"
#     extract_path = os.path.join(os.getcwd(), "test_data")  # Save to the current working directory
#     download_path = os.path.join(extract_path, "test_data.zip")

#     if not os.path.exists(extract_path):
#         print("Downloading test data...")
#         os.makedirs(extract_path, exist_ok=True)
#         response = requests.get(url, stream=True)
#         response.raise_for_status()

#         total_size = int(response.headers.get('content-length', 0))
#         downloaded_size = 0

#         with open(download_path, "wb") as f:
#             for chunk in response.iter_content(chunk_size=8192):
#                 if chunk:
#                     f.write(chunk)
#                     downloaded_size += len(chunk)
#                     # Print progress
#                     done = int(50 * downloaded_size / total_size)
#                     print(f"\r[{'=' * done}{' ' * (50 - done)}] {downloaded_size / total_size:.2%}", end='')

#         print("\nExtracting test data...")
#         with zipfile.ZipFile(download_path, "r") as zip_ref:
#             zip_ref.extractall(extract_path)

#         os.remove(download_path)

#         print(f"Test data extracted to: {extract_path}")
#     else:
#         print(f"test_data directory already exists at {extract_path}")
<<<<<<< HEAD

<<<<<<< HEAD
=======
    return value in x
>>>>>>> 7d33054 ([Breaking changes] Refactored timeseries and Clustering + stats)
=======
>>>>>>> d35b270 (Merge TOADtorial repo with toad repo)
=======
>>>>>>> 6ffac35 (Formatted codebase with Ruff)
