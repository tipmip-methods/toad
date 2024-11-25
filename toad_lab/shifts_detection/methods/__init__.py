    
""" All shifts detection methods should be exposed here """

from toad_lab.shifts_detection.methods.asdetect import ASDETECT

__all__ = ["ASDETECT"]


# Default shifts detection method
default_shifts_method = ASDETECT(
    lmin=5,
    lmax=None
)
