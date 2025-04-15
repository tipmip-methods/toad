import xarray as xr


class Preprocess:
    """
    Preprocessing methods for TOAD objects.

    Note: Docstrings here are short as this class is under heavy development
    """

    def __init__(self, toad):
        self.td = toad

    def preprocess(self, keep_only=None):
        """
        Preprocess the data. To be implemented.
        """

        raise NotImplementedError("Preprocessing is not yet implemented.")

        # Drop unnecessary variables
        if keep_only:
            self.data = self.data.drop_vars(
                [v for v in self.data.data_vars if v not in keep_only]
            )

        # apply XMIP preprocessing ...

        return self.data

    def dimension_to_variables(
        self,
        var: str,
        dim: str,
    ):
        """
        Convert a dimension in a dataset to separate variables.

        Args:
            var: Name of variable to process
            dim: Name of dimension to convert to variables

        Example:
            # Convert realization dimension to variables for 'thk'
            td.preprocess().dimension_to_variables(var='thk', dim='realization')
        """
        ds = self.td.data
        # Check if dimension exists
        if dim not in ds.dims and dim not in ds.coords:
            raise ValueError(
                f"Dimension '{dim}' not found in dataset. Available dimensions: {list(ds.dims.keys())}"
            )

        # Check if variable exists if specified
        if var not in ds.data_vars:
            raise ValueError(
                f"Variable '{var}' not found in dataset. Available variables: {list(ds.data_vars.keys())}"
            )

        new_ds = xr.Dataset()
        for val in ds[dim].values:
            data = ds[var].sel({dim: val}).drop_vars(dim)
            new_ds[f"{var}_{dim}_{val}"] = data

        # Copy remaining coordinates
        for coord in ds.coords:
            if coord != dim and coord not in new_ds.coords:
                new_ds[coord] = ds[coord]

        self.td.data = xr.merge([self.td.data, new_ds])
