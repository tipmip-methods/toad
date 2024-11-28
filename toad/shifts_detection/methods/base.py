from abc import ABC, abstractmethod
import xarray as xr

# Abstract class for shifts detection methods
class ShiftsMethod(ABC):
    @abstractmethod
    def fit_predict(self, dataarray: xr.DataArray, time_dim: str) -> xr.DataArray:
        """Apply the shifts detection method.
        
        Args:
            dataarray (xr.DataArray): Input data array to detect shifts in
            time_dim (str): Name of the temporal dimension
            
        Returns:
            - xr.DataArray: The detection time series
        """
        pass