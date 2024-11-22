import numpy as np

from toad_lab.utils import infer_dims


class Stats:
    def __init__(self, toad):
        self.td = toad

    def compute_cluster_score(
            self,
            var,
            cluster_id, 
            cluster_label = None,
            return_score_fit=False,
            how='mean'
        ):
        """
        Calculate the score of a cluster, ranging from 0 to 1, where:
        - A score of 1 corresponds to a perfect Heaviside step function.
        - A score of 0 corresponds to a perfect linear function.
        Clusters dominated by abrupt shifts yield higher scores.

        The score is derived by fitting a linear regression to the cluster's time series and evaluating the residual.

        Parameters:
        ----------
        var : str
            The variable to analyze.
        cluster_id : int or list
            The ID(s) of the cluster(s) to score.
        cluster_label : str, optional
            The cluster label to use. Defaults to `"{var}_cluster"` if not provided.
        return_score_fit : bool, optional
            If True, also return the linear regression fit. Defaults to False.
        how : str, optional
            Method for aggregating data across grid cells. Supported values:
            - 'mean': Mean value (default).
            - 'median': Median value.
            - 'aggr': Sum of values.
            - 'std': Standard deviation.
            - 'perc': Percentile value.

        Returns:
        -------
        float or tuple
            The cluster score, and optionally the linear regression fit if `return_score_fit` is True.
        """
        
        # Check if cluster_label is provided
        cluster_label = f"{var}_cluster" if cluster_label is None else cluster_label

        # Does not work with per_gridcell
        assert how != "per_gridcell", f"per_gridcell is not supported for this method."

        # Get the variable values
        tdim, _ = infer_dims(self.td.data)  
        xvals = self.td.data[tdim].values
        yvals = self.td.get_timeseries_in_cluster_aggregate(var, cluster_id, how=how)[var].values
        
        # Perform linear regression
        (a, b), res, _, _, _ = np.polyfit(xvals, yvals, 1, full=True)
        rss = res[0]  # Residual sum of squares

        # Compute theoretical maximum RSS for a perfect Heaviside function
        heaviside_rss = np.sum((yvals - np.median(yvals))**2)

        # Standardized score
        standardized_score = rss / heaviside_rss if heaviside_rss > 0 else 0

        # Linear fit (optional return)
        _score_fit = b + a * xvals

        if return_score_fit:
            return standardized_score, _score_fit
        else:
            return standardized_score