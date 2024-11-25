import xarray as xr


class Preprocess:
    def __init__(self, toad):
        self.td = toad
        
    def check_for_coordinate_duplicates(self):
        """Check if data contains duplicate values for any coordinates."""
        dims = list(self.td.data.sizes.keys())
        duplicates = len(self.td.data.to_dataframe().groupby(dims).size().value_counts().values) > 1
        if duplicates:
            print("Data contains non-unqiue values for some coordinates. This may cause unexpected behavior in TOAD. Can be mitigated with xr.drop_duplicates(dim).")
        return duplicates
    

