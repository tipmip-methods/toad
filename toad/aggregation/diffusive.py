"""diffusive aggregation module

Simple approach to aggregate various clusterings into a diffusive clustering.

October 24
"""
import xarray as xr
import numpy as np

def aggregate(data: xr.Dataset) -> xr.Dataset:
    """
    This function takes an xarray dataset with time, lat, and lon coordinates and multiple clusterings as variables.
    It calculates how many times each point is part of any cluster and normalizes 
    the result by the total number of clusterings.

    Parameters:
    -----------
    dataset : xr.Dataset
        The input dataset with time, lat, lon coordinates and discrete cluster numbers as values.
    
    Returns:
    --------
    xr.Dataset
        A dataset containing the normalized count of how often each point is part of a cluster and the original clusterings.
    """
    
    # Determine clustering variables
    cluster_vars = list(data.data_vars)

    # Initialize an empty array for counting the number of cluster occurrences
    shape = data[cluster_vars[0]].shape  # get the shape from any clustering var (time, lat, lon)
    cluster_count = xr.DataArray(np.zeros(shape), dims=data[cluster_vars[0]].dims, coords=data[cluster_vars[0]].coords)

    # Loop through each clustering variable
    for cluster_var in cluster_vars:

        # Set noise points (-1) to 0 and increment the cluster count for non-noise points
        cluster_count += xr.where(data[cluster_var] != -1, 1, 0)

    # Normalize by the total number of clusterings
    num_clusterings = len(cluster_vars)
    cluster_normalized = cluster_count / num_clusterings
    
    # Create a new dataset for the result and include the original clusterings
    diffusive_clustering = xr.Dataset({'Diffusive_clustering': cluster_normalized})
    
    # Add original clusterings to the result dataset
    for cluster_var in cluster_vars:
        diffusive_clustering[cluster_var] = data[cluster_var]
    
    return diffusive_clustering