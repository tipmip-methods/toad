from abc import ABC, abstractmethod
import xarray as xr

# Abstract class for shifts detection methods
class ShiftsMethod(ABC):
    @abstractmethod
    def apply(self, dataarray: xr.DataArray, time_dim: str) -> tuple[xr.DataArray, dict]:
        """Apply the shifts detection method.
        
        Args:
            dataarray (xr.DataArray): Input data array to detect shifts in
            time_dim (str): Name of the temporal dimension
            
        Returns:
            tuple: A tuple containing:
                - xr.DataArray: The detection time series
                - dict: Method details and parameters used
        """
        pass