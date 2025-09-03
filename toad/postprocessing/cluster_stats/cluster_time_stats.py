import numpy as np
from toad.utils import all_functions
import inspect
from scipy.optimize import minimize_scalar
from scipy.signal import savgol_filter
import xarray as xr


class ClusterTimeStats:
    """Class containing functions for calculating time-related statistics for clusters, such as start time, peak time, etc."""

    def __init__(self, toad, var):
        """Initialize the ClusterTimeStats object.

        Args:
            toad (TOAD): TOAD object
            var (str): Base variable name (e.g. 'temperature', will look for 'temperature_cluster')
                or custom cluster variable name.
        """
        self.td = toad
        self.var = var

    def start(self, cluster_id) -> float:
        """Return the start time of the cluster"""
        return float(
            self.td.apply_cluster_mask(self.var, self.td.time_dim, cluster_id).min()
        )

    def start_timestep(self, cluster_id) -> float:
        """Return the start index of the cluster"""
        dens = self.td.get_cluster_spatial_density(self.var, cluster_id)
        idx_start = np.where(dens > 0)[0][0]
        return float(idx_start)

    def end(self, cluster_id) -> float:
        """Return the end time of the cluster"""
        return float(
            self.td.apply_cluster_mask(self.var, self.td.time_dim, cluster_id).max()
        )

    def end_timestep(self, cluster_id) -> int:
        """Return the end index of the cluster"""
        dens = self.td.get_cluster_spatial_density(self.var, cluster_id)
        idx_end = np.where(dens > 0)[0][-1]
        return int(idx_end)

    def duration(self, cluster_id) -> float:
        """Return duration of the cluster in time."""
        return float(float(self.end(cluster_id) - self.start(cluster_id)))

    def duration_timesteps(self, cluster_id) -> int:
        """Return duration of the cluster in timesteps."""
        return int(self.end_timestep(cluster_id) - self.start_timestep(cluster_id))

    def membership_peak(self, cluster_id) -> float:
        """Return the time of the largest cluster temporal density"""
        ctd = self.td.get_cluster_spatial_density(self.var, cluster_id)
        return float(ctd[self.td.time_dim][ctd.argmax()].values)

    def membership_peak_density(self, cluster_id) -> float:
        """Return the largest cluster temporal density"""
        ctd = self.td.get_cluster_spatial_density(self.var, cluster_id)
        return float(ctd.max().values)

    def steepest_gradient(self, cluster_id) -> float:
        """Return the time of the steepest gradient of the mean cluster timeseries inside the cluster time bounds"""
        ts = self.td.get_cluster_timeseries(
            self.var, cluster_id, aggregation="mean", keep_full_timeseries=False
        )

        # Check if all values are NaN before computing gradient
        if ts.isnull().all():
            import warnings

            warnings.warn(
                f"All-NaN timeseries found for cluster {cluster_id}. Returning first timestamp."
            )
            return float(ts.time.values[0])

        grad = ts.diff(self.td.time_dim)
        return float(grad.idxmin())

    def steepest_gradient_timestep(self, cluster_id) -> float:
        """Return the index of the steepest gradient of the mean cluster timeseries inside the cluster time bounds"""
        ts = self.td.get_cluster_timeseries(
            self.var, cluster_id, aggregation="mean", keep_full_timeseries=False
        )

        # Check if all values are NaN before computing gradient
        if ts.isnull().all():
            import warnings

            warnings.warn(
                f"All-NaN timeseries found for cluster {cluster_id}. Returning 0."
            )
            return 0.0

        grad = ts.diff(self.td.time_dim)
        return float(grad.argmin())

    def iqr(
        self, cluster_id, lower_quantile: float, upper_quantile: float
    ) -> tuple[float, float]:
        """Get start and end time of the specified interquantile range of the cluster temporal density.

        Args:
            cluster_id: ID of the cluster
            lower_quantile: Lower bound of the interquantile range (0-1)
            upper_quantile: Upper bound of the interquantile range (0-1)

        Returns:
            tuple[float, float]: Start time and end time of the interquantile range
        """
        ctd = self.td.get_cluster_spatial_density(self.var, cluster_id)
        cum_dist = ctd.cumsum()

        # Find both quantiles
        lower_time = ctd[self.td.time_dim].where(
            cum_dist >= lower_quantile * cum_dist[-1], drop=True
        )
        upper_time = ctd[self.td.time_dim].where(
            cum_dist >= upper_quantile * cum_dist[-1], drop=True
        )

        # Get first occurrence of each quantile
        lower = float(lower_time[0].values if lower_time.size > 0 else np.nan)
        upper = float(upper_time[0].values if upper_time.size > 0 else np.nan)

        return (lower, upper)

    def iqr_50(self, cluster_id) -> tuple[float, float]:
        """Get start and end time of the 50% interquantile range of the cluster temporal density"""
        return self.iqr(cluster_id, 0.25, 0.75)

    def iqr_68(self, cluster_id) -> tuple[float, float]:
        """Get start and end time of the 68% interquantile range of the cluster temporal density"""
        return self.iqr(cluster_id, 0.16, 0.84)

    def iqr_90(self, cluster_id) -> tuple[float, float]:
        """Get start and end time of the 90% interquantile range of the cluster temporal density"""
        return self.iqr(cluster_id, 0.05, 0.95)

    def mean(self, cluster_id) -> float:
        """Return mean time value of the cluster."""
        return float(
            self.td.apply_cluster_mask(self.var, self.td.time_dim, cluster_id).mean()
        )

    def median(self, cluster_id) -> float:
        """Return median time of the cluster"""
        return float(
            self.td.apply_cluster_mask(self.var, self.td.time_dim, cluster_id).median()
        )

    def std(self, cluster_id) -> float:
        """Return standard deviation of the time of the cluster"""
        return float(
            self.td.apply_cluster_mask(self.var, self.td.time_dim, cluster_id).std()
        )

    # TODO implement
    # - cluster_mean_membership_duration
    # - cluster_median_membership_duration
    # - cluster_std_membership_duration

    def all_stats(self, cluster_id) -> dict:
        """Return all cluster stats"""
        dict = {}
        for method_name in all_functions(self):
            if (
                not method_name.startswith("all_stats")
                and len(inspect.signature(getattr(self, method_name)).parameters) == 1
            ):
                dict[method_name] = getattr(self, method_name)(cluster_id)
        return dict

    # Global cluster stats ========================================================

    # TODO make a decision about these global stats, if we want them they need to be updated, possibly put in further subclass .global

    # @property
    # def global_mean_time(self) -> float:
    #     """Return mean of the times during which clusters are active"""
    #     return float(np.concatenate([self._cluster_active_time(cluster_id).values for cluster_id in self.cluster_ids if cluster_id != -1]).mean())

    # @property
    # def global_median_time(self) -> float:
    #     """Return median of the times during which clusters are active"""
    #     return float(np.median(np.concatenate([self._cluster_active_time(cluster_id).values for cluster_id in self.cluster_ids if cluster_id != -1])))

    # @property
    # def global_std_time(self) -> float:
    #     """Return standard deviation of the times during which clusters are active"""
    #     return float(np.concatenate([self._cluster_active_time(cluster_id).values for cluster_id in self.cluster_ids if cluster_id != -1]).std())

    # @property
    # def global_mean_start_time(self) -> float:
    #     """Return mean of the start time of all clusters"""
    #     start_times = [self.cluster_start_time(cluster_id) for cluster_id in self.cluster_ids if cluster_id != -1]
    #     return float(np.mean(start_times))

    # @property
    # def global_mean_end_time(self) -> float:
    #     """Return mean of the end time of all clusters"""
    #     end_times = [self.cluster_end_time(cluster_id) for cluster_id in self.cluster_ids if cluster_id != -1]
    #     return float(np.mean(end_times))

    # @property
    # def global_median_start_time(self) -> float:
    #     """Return median of the start time of all clusters"""
    #     start_times = [self.cluster_start_time(cluster_id) for cluster_id in self.cluster_ids if cluster_id != -1]
    #     return float(np.median(start_times))

    # @property
    # def global_median_end_time(self) -> float:
    #     """Return median of the end time of all clusters"""
    #     end_times = [self.cluster_end_time(cluster_id) for cluster_id in self.cluster_ids if cluster_id != -1]
    #     return float(np.median(end_times))

    # @property
    # def global_std_start_time(self) -> float:
    #     """Return standard deviation of the start time of all clusters"""
    #     start_times = [self.cluster_start_time(cluster_id) for cluster_id in self.cluster_ids if cluster_id != -1]
    #     return float(np.std(start_times))

    # @property
    # def global_std_end_time(self) -> float:
    #     """Return standard deviation of the end time of all clusters"""
    #     end_times = [self.cluster_end_time(cluster_id) for cluster_id in self.cluster_ids if cluster_id != -1]
    #     return float(np.std(end_times))

    # def all_global_stats(self) -> dict:
    #     """Return all global stats"""
    #     dict = {}
    #     for method_name in dir(self):
    #         if method_name.startswith('global_'):
    #             attr = getattr(self, method_name)
    #             dict[method_name] = attr() if callable(attr) else attr
    #     return dict

    def compute_transition_time(self, shifts=None, direction="absolute"):
        # get shifts
        shifts = self.td.get_shifts(self.var) if shifts is None else shifts

        if direction == "absolute":
            shifts = np.abs(shifts)
        elif direction == "negative":
            shifts = np.abs(shifts.where(shifts < 0))
        elif direction == "positive":
            shifts = shifts.where(shifts > 0)
        else:
            ValueError(
                f"{direction} is not a valid inpout for direction. Use absolute, negative or positive."
            )

        def compute_transition_time(shifts, t):
            m = np.isfinite(shifts) & np.isfinite(t)
            if m.sum() == 0:
                return np.nan
            return fit_gaussian_transition(t[m], shifts[m])["transition_time"]

        return xr.apply_ufunc(
            compute_transition_time,
            shifts,
            shifts[self.td.time_dim],
            input_core_dims=[[self.td.time_dim], [self.td.time_dim]],
            output_core_dims=[[]],
            vectorize=True,
            output_dtypes=[np.float64],
        )

    def compute_cluster_transition_time(self, cluster_ids, direction="absolute"):
        # get shifts and spatial cluster mask
        dts = self.td.get_shifts(self.var)
        cluster_mask = self.td.get_spatial_cluster_mask(
            self.var, cluster_id=cluster_ids
        )
        dts = dts.where(cluster_mask, drop=True)

        return self.compute_transition_time(dts, direction)


def fit_gaussian_transition(
    time: np.ndarray,
    values: np.ndarray,
    # preprocessing
    smooth_frac: float = 0.03,  # 0 disables smoothing; else Savitzky–Golay fraction of N
    # windowing
    baseline_frac: float = 0.05,  # define window by values >= baseline_frac * max
    margin_frac: float = 0.02,  # extend window on both sides (fraction of full series)
    # width (sigma) search
    min_sigma: float = 0.01,  # in normalized time [0,1]
    max_sigma: float = 0.20,
    k_sigma_bound: float = 2.0,  # center bounds depend on σ: [σ*k, 1-σ*k]
    # correlation target
    use_abs: bool = True,  # use |values| (ASDETECT is hump-like)
):
    """Fits a Gaussian transition function to time series data.

    This function attempts to fit a Gaussian curve to a transition in time series data.
    It includes preprocessing steps like smoothing and windowing to isolate the transition
    region before fitting.

    Args:
        time: Array of time values.
        values: Array of data values corresponding to time points.
        smooth_frac: Fraction of data length to use as window for Savitzky-Golay smoothing.
            Set to 0 to disable smoothing. Defaults to 0.03.
        baseline_frac: Fraction of maximum value to use as baseline threshold for defining
            the transition window. Defaults to 0.05.
        margin_frac: Fraction of full series length to extend window on both sides.
            Defaults to 0.02.
        min_sigma: Minimum allowed width (sigma) in normalized time units [0,1].
            Defaults to 0.01.
        max_sigma: Maximum allowed width (sigma) in normalized time units [0,1].
            Defaults to 0.20.
        k_sigma_bound: Factor determining how center position bounds depend on sigma.
            Center bounds are [σ*k, 1-σ*k]. Defaults to 2.0.
        use_abs: Whether to use absolute values for fitting. Set True for hump-like
            transitions. Defaults to True.

    Returns:
        dict: Dictionary containing fit results with the following keys:
            - transition_time: Gaussian peak position in original time units
            - center_norm: Normalized center position of Gaussian
            - sigma: Width of fitted Gaussian
            - amplitude: Height of fitted Gaussian
            - best_correlation: Weighted correlation coefficient of fit
            - window_indices: Tuple of (start, end) indices of fitting window
            - used_time: Time values used for fitting
            - used_values: Data values used for fitting
            - best_gaussian: Fitted Gaussian on original time grid (None if fit fails)
            - warnings: List of warning messages from fitting process

    Warnings:
        !! TODO: THIS IS WORKING VIBE CODE THAT NEEDS TO BE VALIDATED AND CHECKED FOR CORRECTNESS...  !!
    """

    warnings = []
    t = np.asarray(time)
    y = np.asarray(values)
    if use_abs:
        y = np.abs(y)

    # guard rails
    if len(t) != len(y) or len(t) < 5:
        return {
            "transition_time": None,
            "warnings": ["insufficient data"],
            "best_gaussian": None,
        }

    # optional light smoothing (preserves onset)
    if smooth_frac and smooth_frac > 0:
        win = max(5, int(round(smooth_frac * len(y))))
        if win % 2 == 0:
            win += 1
        win = min(win, len(y) - (1 - len(y) % 2))  # ensure valid odd size
        if win >= 5 and win % 2 == 1:
            y = savgol_filter(y, window_length=win, polyorder=2, mode="interp")

    # define analysis window from baseline crossings
    ymax = np.max(y)
    if ymax <= 0:
        return {
            "transition_time": None,
            "warnings": ["non-positive signal"],
            "best_gaussian": None,
        }

    baseline = baseline_frac * ymax
    above = y >= baseline
    if not np.any(above):
        return {
            "transition_time": None,
            "warnings": [f"no points ≥ baseline ({baseline:.3g})"],
            "best_gaussian": None,
        }

    i0 = np.argmax(above)  # first crossing
    i1 = len(y) - 1 - np.argmax(above[::-1])  # last crossing

    # add margins in absolute indices
    m = max(1, int(round(margin_frac * len(y))))
    i0 = max(0, i0 - m)  # type: ignore
    i1 = min(len(y) - 1, i1 + m)  # type: ignore

    t_win = t[i0 : i1 + 1]
    y_win = y[i0 : i1 + 1]
    if len(t_win) < 5:
        return {
            "transition_time": None,
            "warnings": ["window too small"],
            "best_gaussian": None,
        }

    # normalize time to [0,1] over window
    t0, t1 = t_win[0], t_win[-1]
    T = (t1 - t0) if (t1 > t0) else 1.0
    tn = (t_win - t0) / T

    # time-spacing weights (handle irregular grids)
    w_time = np.abs(np.gradient(tn))  # ~ dt normalized
    w_time /= np.mean(w_time)

    # initial sigma from half-width (if possible)
    try:
        half = 0.5 * np.max(y_win)
        mask_h = y_win >= half
        j0 = np.argmax(mask_h)
        j1 = len(y_win) - 1 - np.argmax(mask_h[::-1])
        fwhm_norm = (t_win[j1] - t_win[j0]) / T if j1 > j0 else None
        sigma0 = (
            (fwhm_norm / (2 * np.sqrt(2 * np.log(2))))
            if fwhm_norm and fwhm_norm > 0
            else 0.05
        )
    except Exception:
        sigma0 = 0.05
    sigma0 = float(np.clip(sigma0, min_sigma, max_sigma))

    # width set to try (tighter grid around sigma0)
    widths = np.unique(
        np.clip(sigma0 * np.array([0.5, 0.75, 1.0, 1.5, 2.0]), min_sigma, max_sigma)
    )

    def weighted_corr(v, g, w):
        # weighted Pearson correlation
        w = np.asarray(w)
        vw = np.average(v, weights=w)
        gw = np.average(g, weights=w)
        num = np.sum(w * (v - vw) * (g - gw))
        den = np.sqrt(np.sum(w * (v - vw) ** 2) * np.sum(w * (g - gw) ** 2))
        return 0.0 if den == 0 else num / den

    def best_for_sigma(sig):
        # constrain center away from edges proportional to sigma
        lo, hi = k_sigma_bound * sig, 1.0 - k_sigma_bound * sig
        if lo >= hi:
            return (-np.inf, 0.5, 0.0)  # invalid sigma for this window

        def objective(center):
            g = np.exp(-0.5 * ((tn - center) / sig) ** 2)
            # LS amplitude with time weights
            a = np.sum(w_time * y_win * g) / np.sum(w_time * g * g)
            a = max(0.0, a)
            corr = weighted_corr(y_win, a * g, w_time)
            return -corr  # maximize corr

        res = minimize_scalar(objective, bounds=(lo, hi), method="bounded")  # type: ignore
        return (-res.fun, float(res.x), float(sig))  # (corr, center, sigma)

    # search widths
    best_corr, best_center, best_sigma = -np.inf, 0.5, sigma0
    for sig in widths:
        corr, center, sig_used = best_for_sigma(sig)
        if corr > best_corr:
            best_corr, best_center, best_sigma = corr, center, sig_used

    if not np.isfinite(best_corr) or best_corr <= -1:
        warnings.append("optimization failed")
        return {"transition_time": None, "warnings": warnings, "best_gaussian": None}

    # final model on window
    g_win = np.exp(-0.5 * ((tn - best_center) / best_sigma) ** 2)
    amp = np.sum(w_time * y_win * g_win) / np.sum(w_time * g_win * g_win)
    amp = max(0.0, amp)

    # peak time (transition time) = Gaussian center
    transition_time = best_center * T + t0

    # build best Gaussian on full time grid for plotting
    tn_full = (t - t0) / T
    g_full = np.exp(-0.5 * ((tn_full - best_center) / best_sigma) ** 2)
    best_gaussian = amp * g_full

    return {
        "transition_time": float(transition_time),
        "center_norm": float(best_center),
        "sigma": float(best_sigma),
        "amplitude": float(amp),
        "best_correlation": float(best_corr),
        "window_indices": (int(i0), int(i1)),
        "used_time": t_win,
        "used_values": y_win,
        "best_gaussian": best_gaussian,
        "warnings": warnings,
    }
