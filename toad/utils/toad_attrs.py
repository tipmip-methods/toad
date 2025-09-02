# this file contains the attributes that are saved to the dataset when computing shifts or clusters
from dataclasses import dataclass


@dataclass(frozen=True)
class Attrs:
    """Constants for xarray attribute names and values used throughout TOAD."""

    # Attribute names
    VARIABLE_TYPE: str = "variable_type"
    BASE_VARIABLE: str = "base_variable"
    SHIFTS_VARIABLE: str = "shifts_variable"
    CLUSTER_IDS: str = "cluster_ids"
    SHIFT_THRESHOLD: str = "shift_threshold"
    SHIFT_SIGN: str = "shift_sign"
    SCALER: str = "scaler"
    TIME_SCALE_FACTOR: str = "time_scale_factor"
    N_DATA_POINTS: str = "n_data_points"
    METHOD_NAME: str = "method_name"
    TOAD_VERSION: str = "toad_version"
    TIME_DIM: str = "time_dim"

    # Attribute values
    TYPE_SHIFT: str = "shift"
    TYPE_CLUSTER: str = "cluster"


attrs = Attrs()
