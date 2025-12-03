from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import xarray as xr

    from toad.core import TOAD


# Abstract class for shifts detection methods
class ShiftsMethod(ABC):
    @abstractmethod
    def fit_predict(
        self,
        values_1d: np.ndarray,
        times_1d: np.ndarray,
    ) -> np.ndarray:
        """Apply the shifts detection method.

        Args:
            values_1d: Input values.
            times_1d: Input times.

        Returns:
            np.ndarray: A detection time series with the same length as the input,
                where each value indicates the presence or magnitude of a detected shift.
        """
        pass

    def pre_validation(
        self,
        data_array: "xr.DataArray",
        td: "TOAD",
    ) -> None:
        """Optional validation method that runs once before applying the method to all grid cells.

        This method is called once in compute_shifts() before xr.apply_ufunc() processes
        all grid cells. Override this method in subclasses to implement method-specific
        validations (e.g., checking for regular temporal spacing, validating parameters,
        converting timescale parameters, etc.).

        Args:
            data_array: The masked data array that will be processed
            td: The TOAD object containing the dataset and metadata

        Raises:
            ValueError: If validation fails
        """
        # Default implementation does nothing
        # Subclasses can override this to add their own validations
        pass
