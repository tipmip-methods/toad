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
        

    def get_timeseries_in_cluster_aggregate(self, var, cluster_id, cluster_label=None, how="mean"):
        """
        Get aggregated timeseries for a cluster using different statistical measures.
        
        Parameters:
        ----------
        var : str
            The variable to analyze (e.g., 'thk' for thickness)
        cluster_id : int
            The ID of the cluster to analyze. Use -1 for unclustered cells.
        cluster_label : str, optional
            Custom cluster label variable name. If None, uses "{var}_cluster"
        how : str, default="mean"
            Method for aggregating data across grid cells. Supported values:
            - 'mean': Mean value
            - 'median': Median value
            - 'aggr': Sum of values
            - 'std': Standard deviation
            - 'perc': Percentile value, e.g. how=('perc',0.9)
            
        Returns:
        -------
        xr.Dataset
            Dataset containing the aggregated timeseries for the specified cluster.
            The variable name in the dataset matches the input variable name.
            
        Notes:
        -----
        For cluster_id=-1, uses 'always_in_cluster' masking to properly handle 
        unclustered cells. For other cluster IDs, uses 'spatial' masking which 
        includes cells that were part of the cluster at any point in time.
        """
        from toad_lab.clustering import Clustering
        clusters = self.data[cluster_label] if cluster_label else self.get_clusters(var)
        return self.timeseries(
            self.data,
            clustering=Clustering(clusters),
            cluster_lbl=[cluster_id],
            masking='always_in_cluster' if cluster_id == -1 else 'spatial', # the spatial mask returns cells that at any point is in cluster_id, so for -1 you would get all cells. Therefore, we need another mask for unclustered cells (i.e. -1).
            how=how
        )
    

    def get_cluster_persistence_fraction(self, var, cluster_id):
        """
        Calculate the persistence fraction of cells in a given cluster over time.
        
        This function computes the fraction of grid cells that belong to a specified 
        cluster at each time step, relative to the total number of cells that have 
        ever belonged to that cluster. The "persistence fraction" represents how 
        many cells are part of the cluster at each moment, based on cells that 
        have been in the cluster at any point in time.

        Parameters:
        ----------
        var : str
            Name of the variable representing the cluster data. This variable will 
            be used to access the corresponding cluster field (e.g., "thk_cluster").
            
        cluster_id : int
            The identifier for the cluster of interest. If `cluster_id == -1`, the function 
            computes the fraction for unclustered cells using the 'always_in_cluster' mask. 
            Otherwise, it uses the 'spatial' mask to compute for the selected cluster.
        
        Returns:
        -------
        np.ndarray
            A 1D array of length equal to the number of time steps, where each element 
            represents the fraction of cells that are part of the cluster at that specific 
            time step. If there are no cells in the cluster, an array of zeros is returned.
        """
        from toad_lab.clustering import Clustering

        cluster_var = f"{var}_cluster"
        timeseries_data = self.timeseries(
            self.data,
            clustering=Clustering(self.data[cluster_var]),
            cluster_lbl=[cluster_id],
            masking='always_in_cluster' if cluster_id == -1 else 'spatial', # the spatial mask returns cells that at any point is in cluster_id, so for -1 you would get all cells. Therefore, we need another mask for unclustered cells (i.e. -1).
            how=('per_gridcell') # get time series for each grid cell
        )
            
        # Number of cells in the dataset
        num_cells = len(timeseries_data.cell_xy)
        
        # Preallocate cluster matrix
        cluster_matrix = np.full((num_cells, len(timeseries_data.time)), -1)
        
        for i in np.arange(num_cells):
            cluster_matrix[i, :] = timeseries_data.isel(cell_xy=i)[cluster_var].values

        # Calculate fraction of cells in the cluster at each time step
        num_cells_in_cluster = (cluster_matrix == cluster_id).sum(axis=0)
        
        # Avoid division by zero if no cells are in the cluster
        if num_cells == 0:
            return np.zeros(len(timeseries_data.time))

        return num_cells_in_cluster / num_cells