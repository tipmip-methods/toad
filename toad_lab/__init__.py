import logging
import numpy as np
import xarray as xr
from functools import wraps
from typing import Union, Any
import os

from . import shifts_detection
from . import clustering
from .utils import deprecated
from _version import __version__

class TOAD:
    def __init__(self, 
                 data: Union[xr.Dataset, str],
                 log_level = "WARNING"
        ):
        
        # load data from path if string
        if type(data) is str:
            print("hello")
            if not os.path.exists(data):
                raise ValueError(f"File {data} does not exist.")
            self.data = xr.open_dataset(data)
            self.data.attrs['title'] = os.path.basename(data).split('.')[0] # store path as title for saving toad file later
        elif type(data) is xr.Dataset or type(data) is xr.DataArray:
            self.data = data  # Original data
        
        # Initialize the logger for the TOAD object
        self.logger = logging.getLogger("TOAD")
        self.set_log_level(log_level) 


    def set_log_level(self, level):
        """Sets the logging level for the TOAD logger.
        
        Used like this:         
            logger.debug("This is a debug message.")
            logger.info("This is an info message.")
            logger.warning("This is a warning message.")
            logger.error("This is an error message.")
            logger.critical("This is a critical message.")

            In sub-modules get logger like this:
            logger = logging.getLogger("TOAD")
        """
        level = level.upper()
        if level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError("Invalid log level. Choose from 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'")

        self.logger.setLevel(getattr(logging, level))
        
        # Only add a handler if there are no handlers yet (to avoid duplicate messages)
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logging level set to {level}")
    

    # ================================================
    #               Module functions
    # ================================================

    @wraps(shifts_detection.compute_shifts)  # copies docstring // doesn't work in vs code though..
    def compute_shifts(self, *args, **kwargs):
        shifts = shifts_detection.compute_shifts(self.data, *args, **kwargs)
        self.data = xr.merge([self.data, shifts], combine_attrs="no_conflicts")
        return self
    
    
    @wraps(shifts_detection.compute_shifts)  # copies docstring
    def compute_clusters(self, *args, **kwargs):
        clusteres = clustering.compute_clusters(self.data, *args, **kwargs)
        self.data = xr.merge([self.data, clusteres], combine_attrs="no_conflicts")
        return self




# ================================================
#               Legacy entry points
# ================================================

@deprecated("detect is deprecated. Please use compute_shifts instead.")
def detect(
        data: Union[xr.Dataset, xr.DataArray],
        temporal_dim: str,
        method: str,
        var: str = None,
        keep_other_vars : bool = False, 
        **method_kwargs
    ) -> xr.Dataset :
    """Legacy function for the compute_shifts function."""
    
    data_array_dts = shifts_detection.compute_shifts(data=data, var=var, temporal_dim=temporal_dim, method=method, **method_kwargs)

    # If keep_other_vars is True, merge the original data with the detected shifts. Otherwise, return only the detected shifts.
    to_merge = data if keep_other_vars else data[var]
    return xr.merge([to_merge , data_array_dts], combine_attrs="no_conflicts")



@deprecated("cluster is deprecated. Please use compute_clusters instead.")
def cluster(
        data: xr.Dataset,
        var : str,
        method : str = "dbscan",
        method_kwargs = {}
    ) -> xr.Dataset:
    """ Legacy function for compute_clusters """

    clusters = clustering.compute_clusters(data=data, var=var, method=method, **method_kwargs)
    return xr.merge([data, clusters], combine_attrs="no_conflicts")
    

    