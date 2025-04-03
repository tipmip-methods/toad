"""
Source: https://gist.github.com/kadereub/9eae9cff356bb62cdbd672931e8e5ec4
"""

# Load relevant libraries
import numpy as np
from numba import jit, njit

# Goal is to implement a numba compatible polyfit (note does not include error handling)

# Define Functions Using Numba
# Idea here is to solve ax = b, using least squares, where a represents our coefficients e.g. x**2, x, constants
@njit
def _coeff_mat(x, deg):
    mat_ = np.zeros(shape=(x.shape[0],deg + 1))
    const = np.ones_like(x)
    mat_[:,0] = const
    mat_[:, 1] = x
    if deg > 1:
        for n in range(2, deg + 1):
            mat_[:, n] = x**n
    return mat_
@njit
def _fit_x(a, b):
    a = a.astype(np.float64)    # Ensure float64 consistency
    b = b.astype(np.float64)    # needed for numba's (faster) nopython-mode
    # linalg solves ax = b
    det_ = np.linalg.lstsq(a, b)[0]
    return det_
 
@njit
def polyfit(x, y, deg):
    a = _coeff_mat(x, deg)
    p = _fit_x(a, y)
    # Reverse order so p[0] is coefficient of highest order
    return p[::-1]