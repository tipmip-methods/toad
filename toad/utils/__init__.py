def infer_dims(xr_da, tdim=None):

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