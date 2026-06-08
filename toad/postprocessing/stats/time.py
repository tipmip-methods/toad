import inspect
import logging
from typing import Union

import cftime
import numpy as np
import pandas as pd
import xarray as xr

from toad.utils import (
    DEFAULT_SHIFT_THRESHOLD,
    _all_functions,
    convert_numeric_to_original_time,
)

logger = logging.getLogger("TOAD")


class TimeStats:
    """Class containing functions for calculating time-related statistics for clusters, such as start time, peak time, etc."""

    def __init__(self, toad, var):
        """Initialize the TimeStats object.

        Args:
            toad (TOAD): TOAD object
            var (str): Base variable name (e.g. 'temperature', will look for 'temperature_cluster')
                or custom cluster variable name.
        """
        self.td = toad
        self.var = var

    def _get_cluster_density(self, cluster_id) -> xr.DataArray:
        """Fraction of grid cells in cluster at each timestep (0-1)."""
        return self.td.get_cluster_mask(self.var, cluster_id).mean(
            dim=self.td.space_dims
        )

    def start(self, cluster_id) -> Union[float, cftime.datetime, np.datetime64]:
        """Return the start time of the cluster."""
        masked_numeric_time_values = self._get_cluster_numeric_times(cluster_id)

        # Calculate numeric min
        numeric_min = float(np.min(masked_numeric_time_values))

        # Convert back to original time format
        return convert_numeric_to_original_time(
            numeric_min, self.td.numeric_time_values, self.td.data[self.td.time_dim]
        )

    def start_timestep(self, cluster_id) -> float:
        """Return the start index of the cluster"""
        dens = self._get_cluster_density(cluster_id)
        idx_start = np.where(dens > 0)[0][0]
        return int(idx_start)

    def end(self, cluster_id) -> Union[float, cftime.datetime, np.datetime64]:
        """Return the end time of the cluster."""
        masked_numeric_time_values = self._get_cluster_numeric_times(cluster_id)

        # Calculate numeric max
        numeric_max = float(np.max(masked_numeric_time_values))

        # Convert back to original time format
        return convert_numeric_to_original_time(
            numeric_max, self.td.numeric_time_values, self.td.data[self.td.time_dim]
        )

    def end_timestep(self, cluster_id) -> int:
        """Return the end index of the cluster"""
        dens = self._get_cluster_density(cluster_id)
        idx_end = np.where(dens > 0)[0][-1]
        return int(idx_end)

    def duration(self, cluster_id) -> float:
        """Return duration of the cluster in time.

        Args:
            cluster_id: ID of the cluster to calculate duration for.

        Returns:
            float: Duration of the cluster. If the original dataset uses cftime format,
                the duration is returned in seconds.
        """
        numeric_times = self._get_cluster_numeric_times(cluster_id)
        return float(np.max(numeric_times) - np.min(numeric_times))

    def duration_timesteps(self, cluster_id) -> int:
        """Return duration of the cluster in timesteps."""
        return int(self.end_timestep(cluster_id) - self.start_timestep(cluster_id))

    def value_at_start(self, cluster_id, aggregation: str = "median") -> float:
        """Return aggregated cluster value at the start timestep."""
        ts = self.td.get_timeseries(
            self.var, cluster_id=cluster_id, aggregation=aggregation
        )
        start = self.start_timestep(cluster_id)
        return float(ts.isel({self.td.time_dim: start}))

    def value_at_end(self, cluster_id, aggregation: str = "median") -> float:
        """Return aggregated cluster value at the end timestep."""
        ts = self.td.get_timeseries(
            self.var, cluster_id=cluster_id, aggregation=aggregation
        )
        end = self.end_timestep(cluster_id)
        return float(ts.isel({self.td.time_dim: end}))

    def value_change(self, cluster_id, aggregation: str = "median") -> float:
        """Return signed aggregated value change across full span (end - start)."""
        return float(
            self.value_at_end(cluster_id, aggregation=aggregation)
            - self.value_at_start(cluster_id, aggregation=aggregation)
        )

    def value_at_iqr_90_start(self, cluster_id, aggregation: str = "median") -> float:
        """Return aggregated cluster value at the lower iqr_90 bound."""
        ts = self.td.get_timeseries(
            self.var, cluster_id=cluster_id, aggregation=aggregation
        )
        start_idx, _ = self._iqr_timestep_bounds(cluster_id, 0.05, 0.95)
        return float(ts.isel({self.td.time_dim: start_idx}))

    def value_at_iqr_90_end(self, cluster_id, aggregation: str = "median") -> float:
        """Return aggregated cluster value at the upper iqr_90 bound."""
        ts = self.td.get_timeseries(
            self.var, cluster_id=cluster_id, aggregation=aggregation
        )
        _, end_idx = self._iqr_timestep_bounds(cluster_id, 0.05, 0.95)
        return float(ts.isel({self.td.time_dim: end_idx}))

    def value_change_iqr_90(self, cluster_id, aggregation: str = "median") -> float:
        """Return signed aggregated value change across iqr_90 bounds (upper - lower)."""
        return float(
            self.value_at_iqr_90_end(cluster_id, aggregation=aggregation)
            - self.value_at_iqr_90_start(cluster_id, aggregation=aggregation)
        )

    def mean_shift_magnitude(self, cluster_id) -> float:
        """Alias for value_change(aggregation="mean")."""
        return self.value_change(cluster_id, aggregation="mean")

    def membership_peak(
        self, cluster_id
    ) -> Union[float, cftime.datetime, np.datetime64]:
        """Return the time of the largest cluster temporal density.

        If there's a plateau at the maximum value, returns the center of the plateau.
        """
        ctd = self._get_cluster_density(cluster_id)

        # Find the maximum value
        max_value = float(ctd.max())

        # Find all indices where the value equals the maximum (plateau detection)
        max_indices = np.where(ctd.values == max_value)[0]

        if len(max_indices) == 0:
            # Fallback to argmax if no exact matches (shouldn't happen)
            peak_idx = int(np.argmax(ctd.values))
        else:
            # Get the center of the plateau
            peak_idx = max_indices[len(max_indices) // 2]

        # Get the numeric time value at that index
        peak_numeric = float(self.td.numeric_time_values[peak_idx])

        # Convert back to original time format
        return self._return_time(peak_numeric)

    def membership_peak_density(self, cluster_id) -> float:
        """Return the largest cluster temporal density"""
        ctd = self._get_cluster_density(cluster_id)
        return float(ctd.max().values)

    def steepest_gradient(
        self, cluster_id
    ) -> Union[float, cftime.datetime, np.datetime64]:
        """Return the time of the steepest gradient (largest rate of change, up or down)
        of the median cluster timeseries."""
        cluster_var = str(self.td.get_clusters(self.var).name)
        base_var = str(self.td.get_base_var(self.var))

        ts = self.td.get_timeseries(
            base_var,
            cluster_id,
            cluster_var=cluster_var,
            aggregation="median",
            keep_full_timeseries=False,
        )

        # Handle undefined timeseries explicitly to avoid returning arbitrary timestamps.
        if ts.isnull().all():
            msg = (
                f"All-NaN timeseries found for cluster {cluster_id}. "
                "Steepest gradient is undefined."
            )
            logger.warning(msg)
            return np.nan

        grad = ts.diff(self.td.time_dim)

        # Use nanargmax on abs(grad) for steepest (up or down); grad's own coords
        # (diff[i] corresponds to t[i+1], not t[i]).
        if np.all(np.isnan(grad.values)):
            msg = (
                f"All-NaN gradient found for cluster {cluster_id}. "
                "Steepest gradient is undefined."
            )
            logger.warning(msg)
            return np.nan

        steepest_idx = int(np.nanargmax(np.abs(grad.values)))
        steepest_time_numeric = float(grad[self.td.time_dim].values[steepest_idx])

        # Convert back to original time format
        return self._return_time(steepest_time_numeric)

    def steepest_gradient_timestep(self, cluster_id) -> float:
        """Return the index of the steepest gradient (largest rate of change, up or down)
        of the median cluster timeseries inside the cluster time bounds."""

        cluster_var = str(self.td.get_clusters(self.var).name)
        base_var = str(self.td.get_base_var(self.var))

        ts = self.td.get_timeseries(
            base_var,
            cluster_id,
            cluster_var=cluster_var,
            aggregation="median",
            keep_full_timeseries=False,
        )

        # Handle undefined timeseries explicitly to avoid returning arbitrary indices.
        if ts.isnull().all():
            msg = (
                f"All-NaN timeseries found for cluster {cluster_id}. "
                "Steepest gradient timestep is undefined."
            )
            logger.warning(msg)
            return np.nan

        grad = ts.diff(self.td.time_dim)
        if np.all(np.isnan(grad.values)):
            msg = (
                f"All-NaN gradient found for cluster {cluster_id}. "
                "Steepest gradient timestep is undefined."
            )
            logger.warning(msg)
            return np.nan

        return float(np.nanargmax(np.abs(grad.values)))

    def iqr(
        self, cluster_id, lower_quantile: float, upper_quantile: float
    ) -> tuple[
        Union[float, cftime.datetime, np.datetime64],
        Union[float, cftime.datetime, np.datetime64],
    ]:
        """Get start and end time of the specified interquantile range of the cluster temporal density.

        Args:
            cluster_id: ID of the cluster
            lower_quantile: Lower bound of the interquantile range (0-1)
            upper_quantile: Upper bound of the interquantile range (0-1)

        Returns:
            tuple: Start time and end time of the interquantile range in original time format
        """
        lower_idx, upper_idx = self._iqr_timestep_bounds(
            cluster_id, lower_quantile, upper_quantile
        )

        # Get numeric time values at those indices
        lower_numeric = float(self.td.numeric_time_values[lower_idx])
        upper_numeric = float(self.td.numeric_time_values[upper_idx])

        # Convert back to original time format
        lower_original = self._return_time(lower_numeric)
        upper_original = self._return_time(upper_numeric)

        return (lower_original, upper_original)

    def _iqr_timestep_bounds(
        self, cluster_id, lower_quantile: float, upper_quantile: float
    ) -> tuple[int, int]:
        """Return lower and upper timestep indices for a cluster interquantile range."""
        ctd = self._get_cluster_density(cluster_id)
        cum_dist = ctd.cumsum()

        lower_idx = np.where(cum_dist >= lower_quantile * cum_dist[-1])[0]
        upper_idx = np.where(cum_dist >= upper_quantile * cum_dist[-1])[0]

        if len(lower_idx) == 0 or len(upper_idx) == 0:
            raise ValueError(
                f"Could not determine IQR bounds for cluster {cluster_id}. "
                "Check cluster density and quantile settings."
            )

        return int(lower_idx[0]), int(upper_idx[0])

    def iqr_50(
        self, cluster_id
    ) -> tuple[
        Union[float, cftime.datetime, np.datetime64],
        Union[float, cftime.datetime, np.datetime64],
    ]:
        """Get start and end time of the 50% interquantile range of the cluster temporal density"""
        return self.iqr(cluster_id, 0.25, 0.75)

    def iqr_68(
        self, cluster_id
    ) -> tuple[
        Union[float, cftime.datetime, np.datetime64],
        Union[float, cftime.datetime, np.datetime64],
    ]:
        """Get start and end time of the 68% interquantile range of the cluster temporal density"""
        return self.iqr(cluster_id, 0.16, 0.84)

    def iqr_90(
        self, cluster_id
    ) -> tuple[
        Union[float, cftime.datetime, np.datetime64],
        Union[float, cftime.datetime, np.datetime64],
    ]:
        """Get start and end time of the 90% interquantile range of the cluster temporal density"""
        return self.iqr(cluster_id, 0.05, 0.95)

    def _get_cluster_numeric_times(self, cluster_id):
        """Get numeric time values for timesteps where the cluster exists.

        Args:
            cluster_id: ID of the cluster to get times for.

        Returns:
            numpy.ndarray: Array of numeric time values where the cluster exists.
        """
        # Get cluster mask and apply to numeric times
        mask = self.td.get_cluster_mask(self.var, cluster_id)
        mask = mask.any(dim=self.td.space_dims)
        return self.td.numeric_time_values[mask]

    def mean(self, cluster_id) -> Union[float, cftime.datetime, np.datetime64]:
        """Return mean time value of the cluster."""
        numeric_times = self._get_cluster_numeric_times(cluster_id)
        return self._return_time(float(np.mean(numeric_times)))

    def median(self, cluster_id) -> Union[float, cftime.datetime, np.datetime64]:
        """Median model time while the cluster mask is active anywhere in space.

        This summarises the cluster's temporal footprint in the 3D cluster mask
        (equivalent to :meth:`median_activity_time`). It is **not** the median
        per-cell peak shift time; for that, use :meth:`pooled_median_transition_time`.
        """
        numeric_times = self._get_cluster_numeric_times(cluster_id)
        return self._return_time(float(np.median(numeric_times)))

    def median_activity_time(
        self, cluster_id
    ) -> Union[float, cftime.datetime, np.datetime64]:
        """Median model time while the cluster exists anywhere in space.

        See :meth:`median` for details.
        """
        return self.median(cluster_id)

    def _pooled_transition_time_values(
        self,
        cluster_id: int,
        shift_threshold: float = DEFAULT_SHIFT_THRESHOLD,
    ) -> np.ndarray:
        """Per-cell peak-shift times (finite values only) within one cluster."""
        transition_map = self.compute_transition_time(
            cluster_ids=[cluster_id],
            shift_threshold=shift_threshold,
        )
        values = transition_map.values.astype(float, copy=False).ravel()
        return values[np.isfinite(values)]

    def pooled_median_transition_time(
        self,
        cluster_id: int,
        shift_threshold: float = DEFAULT_SHIFT_THRESHOLD,
    ) -> Union[float, cftime.datetime, np.datetime64]:
        """Median of per-cell peak-shift times within the cluster.

        Each grid cell contributes one transition time: the model time of
        maximum ``|shift|`` above ``shift_threshold`` (same field as
        :meth:`compute_transition_time`). This pools all cells in the cluster,
        analogous to ``pooled_median_shift_time`` in
        :meth:`Aggregation.consensus_summary`.
        """
        values = self._pooled_transition_time_values(cluster_id, shift_threshold)
        if values.size == 0:
            return np.nan
        return self._return_time(float(np.median(values)))

    def pooled_std_transition_time(
        self,
        cluster_id: int,
        shift_threshold: float = DEFAULT_SHIFT_THRESHOLD,
    ) -> float:
        """Sample standard deviation of per-cell peak-shift times in the cluster."""
        values = self._pooled_transition_time_values(cluster_id, shift_threshold)
        if values.size == 0:
            return np.nan
        return float(np.std(values))

    def summary(
        self,
        cluster_ids: int | list[int] | range | None = None,
        shift_threshold: float = DEFAULT_SHIFT_THRESHOLD,
    ) -> pd.DataFrame:
        """Per-cluster table of activity-time vs pooled transition-time summaries.

        Returns one row per cluster with:

        * ``median_activity_time`` — median timestep while the cluster mask is
          active anywhere in space (:meth:`median_activity_time`).
        * ``pooled_median_transition_time`` / ``pooled_std_transition_time`` —
          median and sample std of per-cell peak-shift times (pooled over all
          cells), matching the spirit of ``pooled_*`` columns in
          :meth:`Aggregation.consensus_summary`.
        * ``n_transition_cells`` — number of cells with a finite transition time.
        * ``start`` / ``end`` — first and last timestep with any cluster member.
        """
        if cluster_ids is None:
            cluster_ids = list(self.td.get_cluster_ids(self.var))
        elif isinstance(cluster_ids, int):
            cluster_ids = [cluster_ids]
        else:
            cluster_ids = list(cluster_ids)

        rows: list[dict] = []
        for cluster_id in cluster_ids:
            values = self._pooled_transition_time_values(cluster_id, shift_threshold)
            rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "median_activity_time": self.median_activity_time(cluster_id),
                    "pooled_median_transition_time": self.pooled_median_transition_time(
                        cluster_id, shift_threshold
                    ),
                    "pooled_std_transition_time": self.pooled_std_transition_time(
                        cluster_id, shift_threshold
                    ),
                    "n_transition_cells": int(values.size),
                    "start": self.start(cluster_id),
                    "end": self.end(cluster_id),
                }
            )
        return pd.DataFrame(rows)

    def std(self, cluster_id) -> float:
        """Return standard deviation of the time of the cluster."""
        numeric_times = self._get_cluster_numeric_times(cluster_id)
        return float(np.std(numeric_times))

    def _return_time(
        self, value, convert_to_original_time: bool = True
    ) -> Union[float, cftime.datetime, np.datetime64]:
        """Return time value in original time format."""
        if convert_to_original_time:
            return convert_numeric_to_original_time(
                value, self.td.numeric_time_values, self.td.data[self.td.time_dim]
            )
        else:
            return value

    def all_stats(self, cluster_id) -> dict:
        """Return all cluster stats"""
        dict = {}
        for method_name in _all_functions(self):
            if method_name.startswith("all_stats") or method_name.startswith("_"):
                continue
            signature = inspect.signature(getattr(self, method_name))
            parameters = list(signature.parameters.values())
            if len(parameters) == 0:
                continue
            # Only include methods whose first argument is cluster_id and all other
            # arguments are optional (have defaults), so we can call with cluster_id only.
            has_cluster_id_first = parameters[0].name == "cluster_id"
            has_only_optional_rest = all(
                p.default is not inspect._empty for p in parameters[1:]
            )
            if has_cluster_id_first and has_only_optional_rest:
                dict[method_name] = getattr(self, method_name)(cluster_id)
        return dict

    def compute_transition_time(
        self,
        cluster_ids: int | list[int] | range | None = None,
        shift_threshold: float = DEFAULT_SHIFT_THRESHOLD,
    ) -> xr.DataArray:
        """Computes the transition time for each grid cell.

        This method identifies the time point of maximum rate of change (peak shift) for each
        spatial location in the data. It uses the absolute value of shifts to detect both
        positive and negative transitions.

        Args:
            cluster_ids: Optional integer or list of integers specifying which cluster IDs to analyze.
                If None, analyzes all clusters. If specified, only analyzes grid cells belonging
                to the given cluster(s).
            shift_threshold: Optional float specifying the minimum absolute shift value that should
                be considered a valid transition. Defaults to 0.5. Grid cells with maximum shift
                values below this threshold will be marked as having no transition (NaN).

        Returns:
            xarray DataArray containing the transition time for each grid cell. Grid cells
            with no detected transition will contain NaN values. The output has the same
            spatial dimensions as the input shifts data.

        Note:
            The transition time is determined by finding the time index where the absolute
            value of the shifts reaches its maximum for each grid cell. This corresponds to
            the point of most rapid change in the underlying data.

            For grid cells where the maximum absolute shift value is below shift_threshold,
            or where no clear transition is detected, NaN values will be returned.
        """
        from toad.clustering import _compute_dts_peak_sign_mask

        # If user has specified a cluster variable, we need to get the shifts variable from attrs
        shifts = self.td.get_shifts(self.var)

        # Filter by clusters if specified
        if cluster_ids is not None:
            mask = self.td.get_cluster_mask_spatial(self.var, cluster_ids)
            shifts = shifts.where(mask)
            start = self.td.stats(self.var).time.start(cluster_ids)
            end = self.td.stats(self.var).time.end(cluster_ids)
            shifts = shifts.where(shifts[self.td.time_dim] >= start, 0.0)
            shifts = shifts.where(shifts[self.td.time_dim] <= end, 0.0)

        # TODO could this be made faster by replacing with argmax(shifts)?
        max_dts_mask = _compute_dts_peak_sign_mask(
            shifts,
            self.td.time_dim,
            shift_selection="global",  # use global to largest shift
            shift_threshold=shift_threshold,
        )

        max_dts_mask = xr.apply_ufunc(np.abs, max_dts_mask)

        # Reductions use the named time dimension (not axis index), so (y, x, time) and similar orders work.
        time_dim = self.td.time_dim
        time_indices = max_dts_mask.argmax(dim=time_dim)
        has_peak = max_dts_mask.sum(dim=time_dim) > 0

        time_coords = self.td.numeric_time_values

        time_indices = xr.where(has_peak, time_indices, -1)

        idx_np = time_indices.values.astype(np.int64, copy=False)
        out_np = np.full(idx_np.shape, np.nan, dtype=float)
        valid = idx_np >= 0
        out_np[valid] = time_coords[idx_np[valid]]

        time_values = xr.DataArray(
            out_np,
            coords=time_indices.coords,
            dims=time_indices.dims,
            name="transition_time",
        )
        time_values.attrs["long_name"] = self.td.data[self.td.time_dim].name
        time_values.attrs["units"] = self.td.numeric_time_values_unit()
        time_values.attrs["description"] = (
            "Time point of maximum rate of change; pool over cells with "
            "TimeStats.pooled_median_transition_time"
        )
        return time_values
