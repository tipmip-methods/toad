"""asdetect method for shifts detection.

Contains the asdetect algorithm with associated helper functions. 

Crated: January 22, ? (Sina)
Refactored: Nov, 2024 (Jakob)
"""

import numpy as np
import xarray as xr
from scipy import stats

from toad.shifts_detection.methods.base import ShiftsMethod


class ASDETECT(ShiftsMethod):
    """
    Detect shifts algorithm applicable to 3D data arrays.
    TODO: write summary of the method (In notion already)

    Args:
        lmin (int, optional): The minimum segment length for detection. Defaults to 5.
        lmax (int, optional): The maximum segment length for detection. If not 
            specified, it defaults to one-third of the size of the temporal 
            dimension.
        
    """

    def __init__(self, lmin = 5, lmax = None):
        self.lmin = lmin
        self.lmax = lmax


    def apply(self, dataarray: xr.DataArray, temporal_dim: str):
        """Compute the detection time series for each grid cell in the 3D data array.

        Args:
            dataarray (xr.DataArray): A 3D xarray DataArray with shape (nt, nx, ny), where 
                `nt` is the length of the temporal dimension, and `nx` and `ny` are 
                the spatial dimensions.
            temporal_dim (str): The name of the temporal dimension in `data` used 
                for constructing the detection time series.

        Returns:
            tuple: A tuple containing:
                - xr.DataArray: A 3D xarray DataArray with the same shape as `data` 
                  (nt, nx, ny), representing the detection time series for each grid cell.
                  TODO: provide more info about the output, why are the numbers between 0 and 1? What does 1 mean?
                - dict: A dictionary summarizing the ASDETECT parameters used, suitable 
                  for storing as metadata or documentation.
        """
        shifts = xr.apply_ufunc(
            construct_detection_ts,
            dataarray,
            kwargs=dict(
                    times_1d=dataarray[temporal_dim].values,
                    lmin=self.lmin, lmax=self.lmax
                ),
            input_core_dims=[[temporal_dim]],
            output_core_dims=[[temporal_dim]],
            vectorize=True
        ).transpose(*dataarray.dims)

        method_params = {
            "lmin": self.lmin,
            "lmax": self.lmax
        }

        return shifts, method_params


# 1D time series analysis of abrupt shifts =====================================
def centered_segmentation(
            l_tot: int, l_seg: int, verbose : bool = False
        ) -> np.ndarray:
    """ Provide set of indices to divide a range into segments of equal length.

    The range of l_tot is divided into segments of equal length l_seg,
    with the remainder of the division being equally truncated at the
    beginning and end, with the end+1 for uneven division.

    :param l_tot:   Total length of the range to be segmented
    :type l_tot:    int
    :param l_seg:   Length of one segment.
    :type l_seg:    int
    :param verbose: If true, print segmentation indices.
    :type verbose:  bool
    :return:        List of indices of the segmentation; entry i are the first index of the ith segment.
    :rtype:         numpy.ndarray[int]

    **Example**

    >>> tsanalysis.asdetect.centered_segmentation(l_tot=103, l_seg=10)
    array([  1,  11,  21,  31,  41,  51,  61,  71,  81,  91, 101])

    """

    # number of segments
    n_seg = int(l_tot/l_seg)
    # uncovered points
    rest = l_tot-n_seg*l_seg
    # first index of first segment 
    idx0 = int(rest/2)
    # first index of each segment
    seg_idces = idx0+l_seg*np.arange(n_seg+1)

    # For checking correct indexing: Print the segmentation indices in the form
    # | a-b | where a/b = first/last index of the segment. (x) denotes the
    # number of ommitted indices at the beginning and the end of the time
    # series, respectively
    if verbose:
        # idx0 points are omitted in the beginning 
        if idx0 == 0:
            segments = " (0) |"
        elif idx0 == 1:
            segments = " (1) 0 |"
        else:
            segments = " ({}) 0-{} |".format(idx0, idx0-1)
        
        # nl segments of size l, starting at idx0
        for idx in seg_idces[:-1]:
            segments += " {}-{} |".format(idx, idx+l_seg-1)

        # (rest-idx0) points are omitted in the end
        if idx0 + n_seg*l_seg == n_seg:
            segments += " (0)"
        elif idx0 +n_seg*l_seg == n_seg-1:
            segments += " {} (1)".format(n_seg-1)
        else:
            segments += " {}-{} ({})".format(idx0+n_seg*l_seg,l_tot-1,rest-idx0)

        print("\nl_tot={}, l_seg={}: n_seg={}, rest={}, idx0={}\n".format(
            l_tot, l_seg, n_seg, rest, idx0
            )+"   "+segments)

    return seg_idces

def construct_detection_ts(
        values_1d : np.ndarray, times_1d: np.ndarray,
        lmin : int = 5, lmax : int = None) -> np.ndarray:
    """ Construct a detection time series (asdetect algorithm).

    Following [Boulton+Lenton2019]_, the time series (ts) is divided into
    segments of length l, for each of which the gradient is computed. Segments
    with gradients > 3 MAD of the gradients distribution are marked. Averaging
    over many segmentation choices (i.e. values of l) results in a detection
    time series that indicates the points of largest relative gradients.

    :param values_1d:   time series, shape (n,)
    :type values_1d:    1d-numpy.ndarray
    :param times_1d:    times, shape (n,), same length as values_1d
    :type times_1d:     1d-numpy.ndarray
    :param lmin:        smallest segment length, default = 5
    :type lmin:         int
    :param lmax:        largest segment length, default = n/3
    :type lmax:         int
    :return:            detection time series, shape (n,)
    :rtype:             1d-numpy.ndarray
    """

    n_tot = len(values_1d)

    detection_ts = np.zeros_like(values_1d)

    if np.isnan(values_1d).any():
        #print("you tried evaluating a ts with nan entries")
        return detection_ts

    # default to have at least three gradients (needed for grad distribution)
    if lmax is None:
        lmax = int(n_tot/3)

    # segmentation values [lmin, lmin+1, ..., lmax-1, lmax]
    segment_lengths = np.arange(lmin, lmax+1, 1)

    # construct a detection time series for each segmentation choice l
    for l in segment_lengths:

        # center the segments around the middle of the ts and drop the outer
        # points to get the segmented time series and the first index of each
        # segment, respectively
        seg_idces = centered_segmentation(n_tot, l)
        arr_segs = np.split(values_1d, seg_idces)[1:-1]
        t_segs = np.split(times_1d, seg_idces)[1:-1]

        # calculate gradient for each segment and median absolute deviation
        # of the resulting distribution
        gradients = [np.polyfit(tseg,aseg,1)[0] for (tseg,aseg) in zip(t_segs, arr_segs)]
        grad_MAD = stats.median_abs_deviation(gradients)
        grad_MEAN = np.median(gradients)

        # for each segment, check whether its gradient is larger than the 
        # threshold. if yes, update the detection time series accordingly.
        # i1/i2 are the first/last index of a segment
        for i, gradient in enumerate(gradients):
            i1,i2 = seg_idces[i], seg_idces[i+1]-1
            if gradient - grad_MEAN > 3*grad_MAD:
                detection_ts[i1:i2] += 1
            elif gradient - grad_MEAN < -3*grad_MAD:
                detection_ts[i1:i2] += -1
    
    # normalize the detection time series to one
    detection_ts /= len(segment_lengths)

    return detection_ts


# ==============================================================================
# Sina leftovers ===============================================================
# ==============================================================================


# Detection time series evaluation methods =====================================
# def re_evaluate_dts(
#     data_with_dts : xr.Dataset,
#     var: str,
#     dts_eval: str,
#     thresh: float = None,
#     tdist : int = None
# ):

#     # The following re-evaluation assumes that as_<var> is a detection time
#     # series, which is the case if this variable was generated with 
#     # method =='asdetect' and dts_eval == 'all'. 
#     assert data_with_dts.get(f'as_{var}') is not None, \
#             'No detection time series to re-evaluate: ' + \
#             f'No abrupt shift data for variable {var}!'
#     assert data_with_dts.attrs['as_detection_method'] == 'asdetect (all)', \
#             'No detection time series to re-evaluate: ' +\
#             f'as_{var} was not generated with asdetect (all)!'


# # def detect(**methodkwargs, redo_asd = True):
# #     pass
#     # check xarray 
#     # map_dts_to_xarray
#     # evaluate_dts
#     # return xr dataset with vars
#     # - var (len nt)
#     # - as_var  (len nt, nonzero where as, value=magnitude)
#     # - as_type_var (len nt, value=type)
#     # and attribute
#     # - types = dict{A:..., B:...}
#     # - git commit?s



# # evaluation of the detection time series =====================================
# # dts evaluations + dts viewer
# # 1 maxima
# # 2 threshold event bunching
# # 3 prob dist fit
# def get_dts_maxima(dts: xr.DataArray, sign='all'):
#     """dts: detection time series data array!"""
#     if sign=='pos':
#         maxt = dts.idmax(dim='time')
#     elif sign=='neg':
#         maxt = dts.idmin(dim='time')
#     else:
#         maxt = np.abs(dts).idxmax(dim="time")


#     dmax = xr.Dataset(
#         data_vars = dict(
#             # time of the maximum
#             maxtime = (["latitude", "longitude"], maxt.data),
#             # value of det_ts at that time
#             maxval = (["latitude", "longitude"], dts.sel(time=maxt).data)),
#         coords = dict(
#             longitude=dts.longitude,
#             latitude=dts.latitude)
#         )

#     # shift at first time stamp is due to a nan dts at that location. 
#     # set those to nan. also drop the now useless time label
#     dmax = dmax.where(dmax.maxtime>dts.time[0]).drop_vars("time")

#     return dmax


# demo zone ===================================================================
# if __name__=='__main__':
#     nt, nx, ny = 30,3,3
#     arr3d = np.arange(nt*nx*ny).reshape(nt,nx,ny).astype(float)
#     arr1d = arr3d[:,0,0]
#     times = np.arange(nt)
#     darr3d = xr.DataArray(
#         data=arr3d, dims=['t','x','y'],
#         coords = [ np.arange(nt), np.arange(nx), np.arange(ny) ]
#     )

#     res1d = construct_detection_ts(arr1d, times)
#     resxd = map_dts_to_xarray(darr3d, temporal_dim='t')    
