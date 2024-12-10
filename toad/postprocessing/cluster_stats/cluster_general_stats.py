from typing import Union, Literal, Optional
import numpy as np


class ClusterGeneralStats:
    def __init__(self, toad, var):
        self.td = toad
        self.var = var
        # Initialize other necessary attributes

    def compute_cluster_score(
            self,
            cluster_id, 
            return_score_fit=False,
            aggregation: Literal["mean", "sum", "std", "median", "percentile"] = "mean",
            percentile: Optional[float] = None,
            normalize: Optional[Literal["first", "max", "last"]] = None
        ) -> Union[float, tuple[float, np.ndarray]]:
        """
        Calculates cluster score based on fit to Heaviside vs linear function.
        
        Score ranges from 0-1, where 1 indicates perfect Heaviside step function and 0 
        indicates perfect linear function. Higher scores mean more abrupt shifts.
        Score is calculated by fitting linear regression and evaluating residuals.

        Args:
            cluster_id: id of the cluster to score.
            return_score_fit: If True, returns linear regression fit along with score.
            aggregation: How to aggregate spatial data:
                - "mean": Average across space
                - "median": Median across space  
                - "sum": Sum across space
                - "std": Standard deviation across space
                - "percentile": Percentile across space (requires percentile arg)
            percentile: Percentile value between 0-1 when using percentile aggregation
            normalize: 
                - "first": Normalize by the first non-zero, non-nan timestep
                - "max": Normalize by the maximum value
                - "last": Normalize by the last non-zero, non-nan timestep
                - "none": Do not normalize

        Returns:
            - score: Cluster score between 0-1.
            - If return_score_fit is True:
                - tuple: (score, linear fit)
        """
        
        # Does not work with raw
        assert aggregation != "raw", f"raw is not supported for this method."

        # Get the variable values
        xvals = self.td.data[self.td.time_dim].values # time values
        # yvals = self.get_cluster_cell_aggregate(cluster_id, how=how)[var].values
        yvals = self.td.get_cluster_timeseries(self.var, cluster_id=cluster_id, aggregation=aggregation, percentile=percentile, normalize=normalize).values

        # Perform linear regression
        (a, b), res, _, _, _ = np.polyfit(xvals, yvals, 1, full=True)
        rss = res[0]  # Residual sum of squares

        # Compute theoretical maximum RSS for a perfect Heaviside function
        heaviside_rss = np.sum((yvals - np.median(yvals))**2)

        # Standardized score
        standardized_score = float(rss / heaviside_rss if heaviside_rss > 0 else 0)

        # Linear fit (optional return)
        _score_fit = b + a * xvals

        if return_score_fit:
            return standardized_score, _score_fit
        else:
            return standardized_score
        