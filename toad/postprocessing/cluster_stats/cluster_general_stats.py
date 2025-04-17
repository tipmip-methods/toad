from typing import Union, Literal, Optional
import numpy as np


class ClusterGeneralStats:
    """General cluster statistics, such as cluster score."""

    def __init__(self, toad, var):
        """
        >> Args:
            toad : (TOAD)
                TOAD object
            var:
                Base variable name (e.g. 'temperature', will look for 'temperature_cluster') or custom cluster variable name.
        """
        self.td = toad
        self.var = var
        # Initialize other necessary attributes

    def cluster_abruptness(
        self,
        cluster_id,
        return_score_fit=False,
        aggregation: Literal["mean", "sum", "std", "median", "percentile"] = "mean",
        percentile: Optional[float] = None,
        normalize: Optional[Literal["first", "max", "last"]] = None,
    ) -> Union[float, tuple[float, np.ndarray]]:
        """
        Evaluates how closely the spatially aggregated cluster time series resembles a perfect Heaviside function.
        A score of 1 indicates a perfect step function, while 0 indicates a linear trend.

        >> Args:
            cluster_id:
                id of the cluster to score.
            return_score_fit:
                If True, returns linear regression fit along with score.
            aggregation:
                How to aggregate spatial data:
                - "mean": Average across space
                - "median": Median across space
                - "sum": Sum across space
                - "std": Standard deviation across space
                - "percentile": Percentile across space (requires percentile arg)
            percentile:
                Percentile value between 0-1 when using percentile aggregation
            normalize:
                - "first": Normalize by the first non-zero, non-nan timestep
                - "max": Normalize by the maximum value
                - "last": Normalize by the last non-zero, non-nan timestep
                - "none": Do not normalize

        >> Returns:
            - score: Cluster score between 0-1.
            - If return_score_fit is True:
                - tuple: (score, linear fit)
        """

        # Does not work with raw
        assert aggregation != "raw", "raw is not supported for this method."

        # Get the variable values
        xvals = self.td.data[self.td.time_dim].values  # time values
        yvals = self.td.get_cluster_timeseries(
            self.var,
            cluster_id=cluster_id,
            aggregation=aggregation,
            percentile=percentile,
            normalize=normalize,
        ).values

        # Perform linear regression
        (a, b), res, _, _, _ = np.polyfit(xvals, yvals, 1, full=True)
        rss = res[0]  # Residual sum of squares

        # Compute theoretical maximum RSS for a perfect Heaviside function
        heaviside_rss = np.sum((yvals - np.median(yvals)) ** 2)

        # Standardized score
        standardized_score = float(rss / heaviside_rss if heaviside_rss > 0 else 0)

        # Linear fit (optional return)
        _score_fit = b + a * xvals

        if return_score_fit:
            return standardized_score, _score_fit
        else:
            return standardized_score
