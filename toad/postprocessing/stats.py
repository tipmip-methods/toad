import numpy as np
from typing import Union
from toad.utils import infer_dims

class Stats:
    """
    Statistical analysis of clustering results.
    """

    def __init__(self, toad):
        self.td = toad