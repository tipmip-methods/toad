from typing import Literal, Optional, Tuple, overload, Union, Callable
import numpy as np
from scipy.cluster.hierarchy import linkage, inconsistent
from scipy.spatial.distance import squareform


class ClusterGeneralStats:
    """General cluster statistics, such as cluster score."""

    def __init__(self, toad, var:str):
        """Initialize ClusterGeneralStats.

        Args:
            toad: TOAD object
            var: Base variable name (e.g. 'temperature', will look for 'temperature_cluster') 
                or custom cluster variable name.
        """
        self.td = toad
        self.var = var


    @overload
    def score_heaviside(
        self,
        cluster_id: int,
        return_score_fit: Literal[False] = False,
        aggregation: Literal[
            "mean", "sum", "std", "median", "percentile", "max", "min"
        ] = "mean",
        percentile: Optional[float] = None,
        normalize: Optional[Literal["first", "max", "last"]] = None,
    ) -> float: ...

    @overload
    def score_heaviside(
        self,
        cluster_id: int,
        return_score_fit: Literal[True],
        aggregation: Literal[
            "mean", "sum", "std", "median", "percentile", "max", "min"
        ] = "mean",
        percentile: Optional[float] = None,
        normalize: Optional[Literal["first", "max", "last"]] = None,
    ) -> Tuple[float, np.ndarray]: ...

    def score_heaviside(
        self,
        cluster_id,
        return_score_fit=False,
        aggregation="mean",
        percentile=None,
        normalize=None,
    ):
        """Evaluates how closely the spatially aggregated cluster time series resembles a perfect Heaviside function.
        
        A score of 1 indicates a perfect step function, while 0 indicates a linear trend.

        Args:
            cluster_id: ID of the cluster to score.
            return_score_fit: If True, returns linear regression fit along with score.
            aggregation: How to aggregate spatial data. Options are:
                - "mean" - Average across space
                - "median" - Median across space 
                - "sum" - Sum across space
                - "std" - Standard deviation across space
                - "percentile" - Percentile across space (requires percentile arg)
                - "max" - Maximum across space
                - "min" - Minimum across space
            percentile: Percentile value between 0-1 when using percentile aggregation.
            normalize: How to normalize the data. Options are:
                - "first" - Normalize by first non-zero, non-nan timestep
                - "max" - Normalize by maximum value
                - "last" - Normalize by last non-zero, non-nan timestep
                - None - Do not normalize

        Returns:
            float: Cluster score between 0-1 if return_score_fit is False.
            tuple: (score, linear_fit) if return_score_fit is True, where score is a float between 0-1 and linear_fit is the fitted values.

        References:
            Kobe De Maeyer Master Thesis (2025)
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

    def score_consistency(
        self,
        cluster_id:int,
    ) -> float:
        """Measures how internally consistent a cluster is by analyzing the similarity between its time series.

        Uses hierarchical clustering to group similar time series and computes an inconsistency score.
        The final score is inverted so higher values indicate more consistency.

        The method works by:
        1. Computing pairwise R² correlations between all time series in the cluster
        2. Converting correlations to distances (1 - R²)
        3. Performing hierarchical clustering using Ward linkage
        4. Calculating inconsistency coefficients at the highest level
        5. Converting to a consistency score by taking the inverse

        Args:
            cluster_id: ID of the cluster to evaluate.

        Returns:
            float: Consistency score between 0-1, where:
                1.0: Perfect consistency (all time series are identical)
                ~0.5: Moderate consistency
                0.0: No consistency (single point or completely inconsistent)

        References:
            Kobe De Maeyer Master Thesis (2025)
        """
        # Get all time series in the cluster
        y_vals = self.td.get_cluster_timeseries(
            self.var,
            cluster_id=cluster_id,
        )

        # Convert to array
        time_series = np.array(y_vals)

        if len(time_series) <= 1:
            return 0.0  # Not enough data to assess consistency

        # Compute R² similarity matrix # prevent warning about division by zero variance
        with np.errstate(divide="ignore", invalid="ignore"):
            r_matrix = np.corrcoef(time_series)
        r_squared_matrix = np.nan_to_num(r_matrix**2)

        # Compute distance matrix
        distance_matrix = 1 - r_squared_matrix

        # Ensure symmetry and 0 diagonal
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)

        # Linkage for hierarchical clustering
        Z = linkage(squareform(distance_matrix), method="ward")
        d = len(time_series) - 1
        R = inconsistent(Z, d=d)

        # Inconsistency at top level
        inconsistency_value = R[-1, -1]

        # Convert to consistency score
        if inconsistency_value < 1e-10:  # to avoid numerical issues
            return 1.0
        else:
            return 1.0 / inconsistency_value

    def score_spatial_autocorrelation(
        self,
        cluster_id:int,
    ) -> float:
        """Computes average pairwise similarity (R²) between all time series in a cluster.

        This measures how spatially coherent the cluster behavior is.

        The score is calculated by:
        1. Getting all time series for cells in the cluster
        2. Computing pairwise R² correlations between all time series
        3. Taking the mean of the upper triangle of the correlation matrix

        Args:
            cluster_id: ID of the cluster to evaluate.

        Returns:
            float: Similarity score between 0-1, where:
                1.0: Perfect similarity (all time series identical)
                ~0.5: Moderate spatial coherence
                0.0: No similarity (completely uncorrelated)

        References:
            Kobe De Maeyer Master Thesis (2025)
        """
        # Get all time series in the cluster
        y_vals = self.td.get_cluster_timeseries(
            self.var,
            cluster_id=cluster_id,
        )

        # Convert to array
        time_series = np.array(y_vals)

        if len(time_series) <= 1:
            return 0.0  # Not enough data to assess similarity

        # Compute R² similarity matrix # prevent warning about division by zero variance
        with np.errstate(divide="ignore", invalid="ignore"):
            r_matrix = np.corrcoef(time_series)
        r_squared_matrix = np.nan_to_num(r_matrix**2)

        # Extract upper triangle (excluding diagonal)
        upper_triangle_indices = np.triu_indices_from(r_squared_matrix, k=1)
        avg_similarity = float(np.mean(r_squared_matrix[upper_triangle_indices]))

        return avg_similarity

    def score_nonlinearity(
        self,
        cluster_id:int,
        aggregation: Literal["mean", "sum", "std", "median", "percentile"] = "mean",
        percentile: Optional[float] = None,
        normalise_against_unclustered: bool = False,
    ) -> float:
        """Computes nonlinearity of a cluster's aggregated time series using RMSE from a linear fit.

        The score measures how much the time series deviates from a linear trend.

        When normalise_against_unclustered=True:
            - Score > 1: Cluster is more nonlinear than typical unclustered behavior
            - Score ≈ 1: Cluster has similar nonlinearity to unclustered data
            - Score < 1: Cluster is more linear than unclustered data
        When normalise_against_unclustered=False:
            - Returns raw RMSE (0 = perfectly linear, higher = more nonlinear)
            - Useful for comparing clusters to each other

        Args:
            cluster_id: Cluster ID to evaluate.
            aggregation: How to aggregate spatial data:
                - "mean": Average across space
                - "median": Median across space
                - "sum": Sum across space
                - "std": Standard deviation across space
                - "percentile": Percentile across space (requires percentile arg)
            percentile: Percentile value between 0–1 (only used if aggregation="percentile")
            normalize_against_unclustered: If True, normalize score by average RMSE of unclustered points.
                This helps identify clusters that stand out from background behavior.

        Returns:
            float: Nonlinearity score. Higher means more nonlinear behavior.
                Interpretation depends on normalize_against_unclustered parameter.

        References:
            Kobe De Maeyer Master Thesis (2025)
        """
        # Get aggregated cluster time series
        yvals = self.td.get_cluster_timeseries(
            self.var,
            cluster_id=cluster_id,
            aggregation=aggregation,
            percentile=percentile,
            normalize="max",
        ).values

        xvals = self.td.data[self.td.time_dim].values
        if len(xvals) != len(yvals):
            raise ValueError("Time and data dimensions do not match.")

        # Fit linear model with polyfit
        coeffs = np.polyfit(xvals, yvals, 1)
        predicted = np.polyval(coeffs, xvals)

        # RMSE for clustered series
        rmse_cluster = float(np.sqrt(np.mean((yvals - predicted) ** 2)))

        if not normalise_against_unclustered:
            return rmse_cluster

        # Get unclustered time series (raw, no aggregation)
        y_unclustered = self.td.get_cluster_timeseries(
            self.var,
            cluster_id=-1,
            aggregation="raw",
        ).values

        unclustered_rmses = []
        for ts in y_unclustered:
            ts = ts.astype(np.float32)
            if np.all(np.isnan(ts)) or np.nanmax(ts) - np.nanmin(ts) == 0:
                continue
            norm_ts = (ts - np.nanmin(ts)) / (
                np.nanmax(ts) - np.nanmin(ts)
            )  # Normalise each trajectory individually
            coeffs = np.polyfit(xvals, norm_ts, 1)
            pred = np.polyval(coeffs, xvals)
            rmse = np.sqrt(np.mean((norm_ts - pred) ** 2))
            unclustered_rmses.append(rmse)

        if len(unclustered_rmses) == 0:
            return rmse_cluster  # No valid unclustered points

        avg_rmse_unclustered = np.mean(unclustered_rmses)
        if avg_rmse_unclustered < 1e-10:
            avg_rmse_unclustered = 1e-10  # avoid divide-by-zero

        # Return normalized nonlinearity
        return float(rmse_cluster / avg_rmse_unclustered)

    def aggregate_cluster_scores(
        self,
        cluster_ids:list[int],
        score_method: str,
        aggregation: Union[str, Callable] = "mean",
        weights: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """Compute a score for multiple clusters and aggregate the results.

        Args:
            cluster_ids: List of cluster IDs
            score_method: Name of the scoring method (e.g., "score_nonlinearity")
            aggregation: "mean", "median", "weighted", or custom function
            weights: Weights for each cluster (if aggregation="weighted")
            **kwargs: Arguments passed to the scoring method

        Returns:
            float: Aggregated score across all clusters
        """
        method = getattr(self, score_method)
        scores = [method(cid, **kwargs) for cid in cluster_ids]

        if callable(aggregation):
            return aggregation(scores)
        elif aggregation == "mean":
            return float(np.mean(scores))
        elif aggregation == "median":
            return float(np.median(scores))
        elif aggregation == "weighted":
            return float(np.average(scores, weights=weights))
        else:
            return 0
