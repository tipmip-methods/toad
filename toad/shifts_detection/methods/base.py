from abc import ABC, abstractmethod
import xarray as xr


# Abstract class for shifts detection methods
class ShiftsMethod(ABC):
    @abstractmethod
    def fit_predict(self, dataarray: xr.DataArray, time_dim: str) -> xr.DataArray:
        """Apply the shifts detection method.
<<<<<<< HEAD
        
<<<<<<< HEAD
=======

>>>>>>> 6ffac35 (Formatted codebase with Ruff)
        >> Args:
            dataarray : (xr.DataArray)
                Input data array to detect shifts in.
            time_dim : (str)
                Name of the temporal dimension.

        >> Returns:
            xr.DataArray:
                A detection time series with the same shape as the input,
                where each value indicates the presence or magnitude of a detected shift.
=======
        Args:
            dataarray (xr.DataArray): Input data array to detect shifts in.
            time_dim (str): Name of the temporal dimension.
            
        Returns:
            xr.DataArray: A detection time series with the same shape as the input, 
            where each value indicates the presence or magnitude of a detected shift.
>>>>>>> c6fc662 (Docstring and type fixes)
        """
        pass
