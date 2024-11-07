def infer_dims(xr_da, tdim=None):
    """
    Infers the temporal and spatial dimensions from an xarray DataArray.

    Parameters:
    -----------
    xr_da : xarray.DataArray
        The input DataArray from which to infer dimensions.
    tdim : str, optional
        The name of the temporal dimension. If provided, it will be used to 
        distinguish between temporal and spatial dimensions. If not provided, 
        the function will attempt to auto-detect the temporal dimension based 
        on standard spatial dimension names.

    Returns:
    --------
    tuple
        A tuple containing the temporal dimension (str) and a list of spatial 
        dimensions (list of str).

    Raises:
    -------
    AssertionError
        If the provided temporal dimension is not in the dimensions of the dataset.

    Notes:
    ------
    - If `tdim` is provided, the function will use it to identify the temporal 
      dimension and consider all other dimensions as spatial.
    - If `tdim` is not provided, the function will attempt to auto-detect the 
      temporal dimension by looking for standard spatial dimension pairs such as 
      ('x', 'y'), ('lat', 'lon'), or ('latitude', 'longitude').
    """

    # spatial dims are all non-temporal dims
    if tdim:
        sdims = list(xr_da.dims)
        assert tdim in xr_da.dims, f"provided temporal dim '{tdim}' is not in the dimensions of the dataset!"
        sdims.remove(tdim)
        sdims = sorted(sdims)
        # print(f"inferring spatial dims {sdims} given temporal dim '{tdim}'")
        return (tdim, sdims)
    # check if one of the standard combinations in present and auto-infer
    else:
        for pair in [('x','y'),('lat','lon'),('latitude','longitude')]:
            if all(i in list(xr_da.dims) for i in pair):
                sdims = pair
                tdim = list(xr_da.dims)
                for sd in sdims:
                    tdim.remove(sd)

                # print(f"auto-detecting: spatial dims {sdims}, temporal dim '{tdim[0]}'")
                return (tdim[0], sdims)
            

import warnings
import functools

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