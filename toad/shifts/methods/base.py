from abc import ABC, abstractmethod
import numpy as np


# Abstract class for shifts detection methods
class ShiftsMethod(ABC):
    @abstractmethod
    def fit_predict(
        self,
        values_1d: np.ndarray,
        times_1d: np.ndarray,
    ) -> np.ndarray:
        """Apply the shifts detection method.

        >> Args:
            values_1d : (np.ndarray)
                Input values.
            times_1d : (np.ndarray)
                Input times.

        >> Returns:
            np.ndarray:
                A detection time series with the same length as the input,
                where each value indicates the presence or magnitude of a detected shift.
        """
        pass

    @classmethod
    @abstractmethod
    def on_timescale(self, time_scale: float):
        pass
