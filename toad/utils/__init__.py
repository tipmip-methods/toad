import warnings
import functools
from typing import Union
import xarray as xr
import numpy as np


def get_space_dims(xr_da: Union[xr.DataArray, xr.Dataset], tdim: str) -> list[str]:
    """Get spatial dimensions from an xarray DataArray or Dataset.

    Args:
        xr_da: Input DataArray or Dataset to get dimensions from
        tdim: Name of temporal dimension. All other dims are considered spatial.

    Returns:
        List of spatial dimension names as strings. If standard spatial dims
        (x/y, y/x) or (lon/lat, lat/lon) are found, returns only those.

    Raises:
        ValueError: If provided temporal dim is not in the dimensions
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

    Returns:
        Reordered list with lat before lon if both present, otherwise original list
    """
    if all(dim in space_dims for dim in ["lat", "lon"]):
        return [dim for dim in ["lat", "lon"] if dim in space_dims] + [
            dim for dim in space_dims if dim not in ["lat", "lon"]
        ]
    return space_dims


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
    return [x for x in dir(obj) if not x.startswith("__") and callable(getattr(obj, x))]


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
