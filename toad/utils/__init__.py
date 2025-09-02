import warnings
import functools
from typing import Union, Tuple, Optional
import xarray as xr
import numpy as np
import re

from .synthetic_data import create_global_dataset
from .toad_attrs import attrs

__all__ = [
    "create_global_dataset",
    "get_space_dims",
    "reorder_space_dims",
    "detect_latlon_names",
    "is_regular_grid",
    "deprecated",
    "all_functions",
    "is_equal_to",
    "contains_value",
    "get_unique_variable_name",
    "attrs",
]


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


def detect_latlon_names(data: xr.Dataset) -> Tuple[Optional[str], Optional[str]]:
    """Detect latitude and longitude coordinate names in a dataset.

    Searches for common latitude/longitude names in coordinates first,
    then falls back to data variables.

    Args:
        data: xarray Dataset to search

    Returns:
        Tuple of (lat_name, lon_name). Either can be None if not found.
    """
    lat_candidates = ["lat", "latitude"]
    lon_candidates = ["lon", "longitude"]

    # Try coordinates first
    lat_name = next((n for n in lat_candidates if n in data.coords), None)
    if lat_name is None:
        # Fall back to data variables
        lat_name = next((n for n in lat_candidates if n in data.variables), None)

    lon_name = next((n for n in lon_candidates if n in data.coords), None)
    if lon_name is None:
        # Fall back to data variables
        lon_name = next((n for n in lon_candidates if n in data.variables), None)

    return lat_name, lon_name


def is_regular_grid(data: xr.Dataset) -> bool:
    """Check if a dataset has a regular grid.

    Args:
        data: xarray Dataset to check

    Returns:
        True if the grid is regular (1D lat/lon), False otherwise
    """
    # Get lat/lon coordinate names
    lat_name, lon_name = detect_latlon_names(data)

    # Check if lat/lon coordinates are present
    if lat_name is None or lon_name is None:
        # No lat/lon coordinates → likely Cartesian/projected → regular
        return True

    # Check if both lat/lon coordinates are 1D (regular grid)
    try:
        lat = data[
            lat_name
        ]  # Use data[] instead of coords[] to handle both coords and data vars
        lon = data[lon_name]

        # Regular grid: both lat and lon must be 1D
        return lat.ndim == 1 and lon.ndim == 1

    except KeyError:
        # lat_name or lon_name not found in dataset
        return False


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


def get_unique_variable_name(base_name: str, existing_vars, logger=None) -> str:
    """Generate a unique variable name by appending sequential numbers if needed.

    Args:
        base_name: The desired variable name
        existing_vars: Container with existing variable names (Dataset, dict, list, set, etc.)
        logger: Optional logger for info messages

    Returns:
        Unique variable name (either base_name or base_name_N)

    Examples:
        >>> get_unique_variable_name("tas_cluster", ["tas_cluster"])
        "tas_cluster_1"
        >>> get_unique_variable_name("tas_cluster", ["tas_cluster", "tas_cluster_1"])
        "tas_cluster_2"
        >>> get_unique_variable_name("tas_cluster_5", ["tas_cluster_5", "tas_cluster_6"])
        "tas_cluster_7"
    """
    if base_name not in existing_vars:
        return base_name

    # Check if the name already has a number at the end
    match = re.search(r"_(\d+)$", base_name)
    if match:
        # Extract the base name and current number
        name_without_num = base_name[: match.start()]
        current_num = int(match.group(1))
        next_num = current_num + 1
    else:
        # No number at the end, start with _1
        name_without_num = base_name
        next_num = 1

    # Find the next available number
    while f"{name_without_num}_{next_num}" in existing_vars:
        next_num += 1

    new_name = f"{name_without_num}_{next_num}"

    if logger:
        logger.info(f"Variable {base_name} already exists. Using {new_name} instead.")

    return new_name


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
