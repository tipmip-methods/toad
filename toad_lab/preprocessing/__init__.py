import xarray as xr


def preprocess(data, keep_only=None):
    """Preprocess the data.

    :param data:        xarray dataset
    :type data:         xr.Dataset
    :return:            ...
    :rtype:             ...
    """

    # Drop unnecessary variables
    if keep_only:
        data = data.drop_vars([v for v in data.data_vars if v not in keep_only])

    # TODO apply XMIP preprocessing

    return data


