""" 
Shifts detection methods available in TOAD.

Currently implemented methods:
- ASDETECT: Implementation of the [Boulton+Lenton2019]_ algorithm for detecting abrupt shifts
"""

from toad.shifts_detection.methods.asdetect import ASDETECT

__all__ = ["ASDETECT"]
