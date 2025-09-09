import logging
import time
from typing import Union

import numpy as np
import optuna
import xarray as xr

logger = logging.getLogger("TOAD")


def combined_spatial_nonlinearity(td, var, weights=[1, 1]) -> float:
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


def _optimise_clusters(
    kwargs: dict,
) -> Union[xr.Dataset, xr.DataArray]:
    for key, value in kwargs.items():
        print(key, ":", value)

    # extract optimisation parameters
    var = kwargs.pop("var")
    method = kwargs.pop("method")
    cluster_param_ranges = kwargs.pop("cluster_param_ranges")
    objective = kwargs.pop("objective")
    direction = kwargs.pop("direction")
    log_level = kwargs.pop("log_level")
    show_progress_bar = kwargs.pop("show_progress_bar")
    n_trials = kwargs.pop("n_trials")
    td = kwargs.pop("td")

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
        cluster_params = get_params(trial, cluster_param_ranges)

        # Extract time_scale_factor and shift_threshold if present
        time_scale_factor = cluster_params.pop("time_scale_factor", 1)
        shift_threshold = cluster_params.pop("shift_threshold", 0.8)

        # Compute clusters
        td.compute_clusters(
            var,
            method=method(**cluster_params),
            shift_threshold=shift_threshold,
            time_scale_factor=time_scale_factor,
            **kwargs,
        )

        # If no clusters found, return -inf
        if len(td.get_cluster_ids(var, exclude_noise=True)) == 0:
            return -np.inf if direction == "maximize" else np.inf

        cluster_ids = td.get_cluster_ids(var)

        # Compute score
        if callable(objective):
            score = float(objective(td, var))  # type: ignore
        else:
            # fmt: off
            # Map objective names to their corresponding scoring functions
            score_funcs = {
                "median_heaviside":           lambda: np.median([td.cluster_stats(var).general.score_heaviside(cid, aggregation="median") for cid in cluster_ids]),
                "mean_heaviside":               lambda: np.mean([td.cluster_stats(var).general.score_heaviside(cid, aggregation="mean") for cid in cluster_ids]),
                "mean_consistency":             lambda: np.mean([td.cluster_stats(var).general.score_consistency(cid) for cid in cluster_ids]),
                "mean_spatial_autocorrelation": lambda: np.mean([td.cluster_stats(var).general.score_spatial_autocorrelation(cid) for cid in cluster_ids]),
                "mean_nonlinearity":            lambda: np.mean([td.cluster_stats(var).general.score_nonlinearity(cid, aggregation="mean") for cid in cluster_ids])
            }
            # fmt: on

            if objective not in score_funcs:
                raise ValueError("Invalid objective")

            score = float(score_funcs[objective]())

        return score

    optuna.logging.set_verbosity(log_level)
    toad_log_level = td.logger.level  # this gives an int
    td.set_log_level("WARNING")  # this fetches int of flag warning sets that
    t0 = time.time()
    study = optuna.create_study(direction=direction)
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=show_progress_bar)
    t1 = time.time()

    # restore toad log level
    td.logger.setLevel(toad_log_level)  # this sets int

    # print final score and best params
    print(f"Completed {n_trials} trials in {t1 - t0:.2f} seconds")
    print(f"Best trial: {study.best_trial.number} with score {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    print(f"Identified {len(td.get_cluster_ids(var))} clusters")
    print("Running best clustering again...")

    return td.compute_clusters(
        var,
        method=method(**study.best_params),
        shift_threshold=study.best_params["shift_threshold"],
        time_scale_factor=study.best_params["time_scale_factor"],
        **kwargs,
    )
