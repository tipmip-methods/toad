import pytest
import numpy as np
import os
import sys

# change current dir to dir containing the running script file
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# append relative path to find library
sys.path.append('./../../../toad/tsanalysis/')
import asdetect as ad

"""
Things to test:
    - test if output is numpy.array
    - test simple example case
    - test truncation of division with
        ... even remainder
        ... uneven remainder
"""


def test_output_type():
    """
    Test if the type of the output object is numpy.array.
    """

    l_tot = 1       # range to be segmented
    l_seg = 1       # length of one segment
    out = ad.centered_segmentation(l_tot,l_seg)
    assert isinstance(out,(np.ndarray))


def test_simple_case():
    """
    Test simple case examples without truncation.

    **Example**

    >>> tsanalysis.asdetect.centered_segmentation(l_tot=10, l_seg=2)
    array([ 0, 2, 4, 6, 8, 10])
    # -> segmented list: [[0,1],[2,3],[4,5],[6,7],[8,9]]
    """

    l_tot = 10
    l_seg = 2
    out = ad.centered_segmentation(l_tot,l_seg)
    assert np.array_equal(out,np.array([0,2,4,6,8,10]))


def test_truncation_case():
    """
    Test case examples with of truncation of even and uneven remainder.

    **even Example**

    >>> tsanalysis.asdetect.centered_segmentation(l_tot=10, l_seg=4)
    array([ 1, 5, 9])
    # -> segmented list: [0,[1,2,3,4],[5,6,7,8],9];     0,9 is truncated
    """
    # even remainder truncation
    l_tot = 10
    l_seg = 4
    out = ad.centered_segmentation(l_tot,l_seg)
    assert np.array_equal(out,np.array([1,5,9]))

    """
    **uneven Example**

    >>> tsanalysis.asdetect.centered_segmentation(l_tot=10, l_seg=3)
    array([ 0, 3, 6, 9])
    # -> segmented list: [[0,1,2],[3,4,5],[6,7,8],9];   9 is truncated
    """
    # uneven remainder truncation
    l_tot = 10
    l_seg = 3
    out = ad.centered_segmentation(l_tot,l_seg)
    assert np.array_equal(out,np.array([0,3,6,9]))