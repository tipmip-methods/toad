import xarray as xr



class Preprocess:
    def __init__(self, toad):
        self.td = toad

    def preprocess(self, keep_only=None):
        """Preprocess the data.

        This method preprocesses the data by dropping unnecessary variables and applying XMIP preprocessing.

        Parameters
        ----------
        keep_only : list, optional
            List of variable names to keep. All other variables will be dropped.
            If None, all variables are kept.

        Returns
        -------
        xarray.Dataset
            The preprocessed dataset
        """

        raise NotImplementedError("Preprocessing is not yet implemented.")

        # Drop unnecessary variables
        if keep_only:
            self.data = self.data.drop_vars([v for v in self.data.data_vars if v not in keep_only])

        # TODO apply XMIP preprocessing

        return self.data


