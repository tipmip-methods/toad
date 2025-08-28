import optuna
import time

import numpy as np
from toad.shifts import ASDETECT
from sklearn.cluster import HDBSCAN

__all__ = ["optimise", "combined_spatial_nonlinearity", "default_cluster_param_ranges"]


def combined_spatial_nonlinearity(td, var, weights=[1, 1]):
    """Compute a weighted combination of spatial autocorrelation and nonlinearity scores.

    Args:
        td: ToadDataset object containing the data
        cluster_ids: List of cluster IDs to evaluate
        var: Name of variable to analyze
        weights: List of two weights for spatial and nonlinearity scores. Defaults to [1,1]

    Returns:
        float: Weighted sum of spatial autocorrelation and nonlinearity scores
    """

    cluster_ids = td.get_cluster_ids(var)
    score1 = td.cluster_stats(var).general.aggregate_cluster_scores(
        cluster_ids=cluster_ids[:10],
        score_method="score_spatial_autocorrelation",
        aggregation="median",
    )

    score2 = td.cluster_stats(var).general.aggregate_cluster_scores(
        cluster_ids=cluster_ids[:10],
        score_method="score_nonlinearity",
        aggregation="median",
    )

    return float(weights[0] * score1 + weights[1] * score2)


default_cluster_param_ranges = dict(
    {
        "min_cluster_size": (10, 25),
        "shift_threshold": (0.6, 0.95),
        "time_scale_factor": (0.5, 1.5),
    }
)


def optimise(
    td,
    var,
    shifts_method=ASDETECT,
    cluster_method=HDBSCAN,
    shifts_param_ranges=dict({}),
    cluster_param_ranges=default_cluster_param_ranges,
    objective=combined_spatial_nonlinearity,
    n_trials=50,
    direction="maximize",
    log_level: int = optuna.logging.WARNING,
    show_progress_bar=True,
):
    """Apply clustering to a dataset's temporal shifts using a sklearn-compatible clustering algorithm.

    >> Args:
        var:
            Name of the variable to cluster.
        shifts_method:
            Class for shift detection. Defaults to ASDETECT.
        cluster_method:
            Class for clustering. Defaults to HDBSCAN.
        shifts_param_ranges:
            Dict of parameter ranges for shift detection. Defaults to empty dict.
        cluster_param_ranges:
            Dict of parameter ranges for clustering. Defaults to default_cluster_param_ranges.
        objective:
            Function or string specifying evaluation metric. Defaults to combined_spatial_nonlinearity.
            Can be one of:
            - callable: Custom objective function taking (td, cluster_ids, var) as arguments
            - "median_heaviside": Median heaviside score across clusters
            - "mean_heaviside": Mean heaviside score across clusters
            - "mean_consistency": Mean consistency score across clusters
            - "mean_spatial_autocorrelation": Mean spatial autocorrelation score
            - "mean_nonlinearity": Mean nonlinearity score across clusters
        n_trials:
            Number of optimization trials to run. Defaults to 50.
        direction:
            Whether to maximize or minimize objective. Defaults to "maximize".

    >> Returns:
        dict: Best parameters found during optimization

    >> Raises:
        ValueError: If objective is not valid
    """

    """
    Overview of the optimisation process:
    
    The optimization system is a sophisticated parameter tuning framework designed to find optimal 
    configurations for temporal shift detection and clustering in spatiotemporal data analysis. 
    The implementation uses Optuna as the underlying optimization engine and provides a flexible, 
    multi-objective approach to parameter selection.

    Core Architecture:
    - optimise() function: The main optimization routine that orchestrates the entire process
    - Cluster scoring system: A comprehensive set of metrics to evaluate cluster quality

    Key Features:

    1. Dual-Phase Parameter Optimization
       - Shift detection (e.g., ASDETECT parameters)
       - Clustering algorithms (e.g., HDBSCAN parameters)

    2. Smart Computation Management
       - Caching mechanism: Avoids recomputing shifts when shift parameters haven't changed
       - Memory-efficient storage: Only saves the best shift and cluster data
       - Automatic restoration: Restores the best configuration after optimization completes

    3. Flexible Objective Functions
       Built-in objectives:
       - "median_heaviside" / "mean_heaviside": Measures how well clusters resemble step functions
       - "mean_consistency": Evaluates internal coherence of cluster time series
       - "mean_spatial_autocorrelation": Assesses spatial coherence within clusters
       - "mean_nonlinearity": Measures deviation from linear trends
       
       Custom objectives:
       - combined_spatial_nonlinearity(): Default weighted combination of spatial autocorrelation 
         and nonlinearity scores
       - Support for user-defined objective functions

    4. Comprehensive Cluster Scoring Methods
       Heaviside Score (score_heaviside):
       - Evaluates how closely a cluster's aggregated time series resembles a perfect step function
       - Uses linear regression residuals compared to theoretical Heaviside function
       - Score of 1 = perfect step function, 0 = linear trend

       Consistency Score (score_consistency):
       - Measures internal coherence using hierarchical clustering of time series correlations
       - Converts R² correlations to distances, performs Ward linkage
       - Higher scores indicate more consistent temporal behavior within clusters

       Spatial Autocorrelation Score (score_spatial_autocorrelation):
       - Computes average pairwise R² correlations between all time series in a cluster
       - Measures spatial coherence and synchronization of cluster behavior

       Nonlinearity Score (score_nonlinearity):
       - Measures RMSE deviation from linear fit
       - Optional normalization against unclustered data to identify clusters that stand out
       - Supports various spatial aggregation methods (mean, median, percentile, etc.)

    5. Intelligent Parameter Handling
       - Automatic type detection: Distinguishes between integer and float parameters
       - Range-based sampling: Uses Optuna's suggest_float/suggest_int for parameter ranges
       - Fixed value support: Allows setting specific parameter values alongside ranges
       - Special parameter extraction: Handles time_scale_factor and shift_threshold separately

    6. Robust Error Handling
       - Empty cluster protection: Returns appropriate infinity values when no clusters are found
       - Numerical stability: Handles edge cases in scoring functions
       - Graceful degradation: Continues optimization even when individual trials fail

    """

    # Track the best results
    best_score = -np.inf if direction == "maximize" else np.inf
    best_shift_data = None
    best_cluster_data = None

    # Track previous shift parameters to avoid recomputation
    old_shift_params = None

    def get_params(trial, param_ranges):
        params = {}
        for k, v in param_ranges.items():
            if not isinstance(v, (list, tuple)):
                # If v is a single value, use it directly
                params[k] = v
            else:
                # If v is a range, suggest float or int based on type
                params[k] = (
                    trial.suggest_float(k, *v)
                    if isinstance(v[0], float)
                    else trial.suggest_int(k, *v)
                )
        return params

    def objective_fn(trial) -> float:
        nonlocal best_score, best_shift_data, best_cluster_data, old_shift_params

        shift_params = get_params(trial, shifts_param_ranges)
        cluster_params = get_params(trial, cluster_param_ranges)

        # Extract time_scale_factor and shift_threshold if present
        time_scale_factor = cluster_params.pop("time_scale_factor", 1)
        shift_threshold = cluster_params.pop("shift_threshold", 0.8)

        # Compute shifts if parameters have changed and shifts already present
        if shift_params != old_shift_params or f"{var}_dts" not in td.data:
            td.compute_shifts(var, method=shifts_method(**shift_params), overwrite=True)
            old_shift_params = shift_params.copy()

        # Compute clusters
        td.compute_clusters(
            var,
            method=cluster_method(**cluster_params),
            shift_threshold=shift_threshold,
            time_scale_factor=time_scale_factor,
            overwrite=True,
        )

        # If no clusters found, return -inf
        if len(td.get_cluster_ids(var, exclude_noise=True)) == 0:
            return -np.inf if direction == "maximize" else np.inf

        cluster_ids = td.get_cluster_ids(var)

        # Compute score
        if callable(objective):
            score = float(objective(td, var))  # type: ignore
        elif objective == "median_heaviside":
            score = float(
                np.median(
                    [
                        td.cluster_stats(var).general.score_heaviside(
                            cid, aggregation="median"
                        )
                        for cid in cluster_ids
                    ]
                )
            )
        elif objective == "mean_heaviside":
            score = float(
                np.mean(
                    [
                        td.cluster_stats(var).general.score_heaviside(
                            cid, aggregation="mean"
                        )
                        for cid in cluster_ids
                    ]
                )
            )
        elif objective == "mean_consistency":
            score = float(
                np.mean(
                    [
                        td.cluster_stats(var).general.score_consistency(cid)
                        for cid in cluster_ids
                    ]
                )
            )
        elif objective == "mean_spatial_autocorrelation":
            score = float(
                np.mean(
                    [
                        td.cluster_stats(var).general.score_spatial_autocorrelation(cid)
                        for cid in cluster_ids
                    ]
                )
            )
        elif objective == "mean_nonlinearity":
            score = float(
                np.mean(
                    [
                        td.cluster_stats(var).general.score_nonlinearity(
                            cid, aggregation="mean"
                        )
                        for cid in cluster_ids
                    ]
                )
            )
        else:
            raise ValueError("Invalid objective")

        # Save best results (only the relevant data)
        if (direction == "maximize" and score > best_score) or (
            direction == "minimize" and score < best_score
        ):
            best_score = score
            # Only save the shift and cluster data, not the entire dataset
            best_shift_data = td.data[f"{var}_dts"].copy()
            best_cluster_data = td.data[f"{var}_cluster"].copy()

        return score

    optuna.logging.set_verbosity(log_level)
    t0 = time.time()
    study = optuna.create_study(direction=direction)
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=show_progress_bar)
    t1 = time.time()

    # Restore the best results
    if best_shift_data is not None and best_cluster_data is not None:
        td.data[f"{var}_dts"] = best_shift_data
        td.data[f"{var}_cluster"] = best_cluster_data

    # print final score and best params
    print(f"Completed {n_trials} trials in {t1 - t0:.2f} seconds")
    print(f"Best trial: {study.best_trial.number} with score {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    print(f"Identified {len(td.get_cluster_ids(var))} clusters")

    return study.best_params
