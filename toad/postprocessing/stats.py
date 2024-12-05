import numpy as np
from typing import Union
from toad.utils import infer_dims

class Stats:
    """
    Statistical analysis of clustering results.
    """

    def __init__(self, toad):
        self.td = toad

    def compute_cluster_score(
            self,
            var,
            cluster_id, 
            return_score_fit=False,
            time_dim="time",
            how='mean'
        ) -> Union[float, tuple[float, np.ndarray]]:
        """
        Calculates cluster score based on fit to Heaviside vs linear function.
        
        Score ranges from 0-1, where 1 indicates perfect Heaviside step function and 0 
        indicates perfect linear function. Higher scores mean more abrupt shifts.
        Score is calculated by fitting linear regression and evaluating residuals.

        Args:
            var: Name of the variable for which clusters have been computed or the name of the custom cluster variable.
            cluster_id: id of the cluster to score.
            return_score_fit: If True, returns linear regression fit along with score.
            how: Method for aggregating grid cells:
                - 'mean': Mean value (default)
                - 'median': Median value  
                - 'aggr': Sum of values
                - 'std': Standard deviation
                - 'perc': Percentile value, e.g. how=('perc', 0.9)

        Returns:
            - score: Cluster score between 0-1.
            - If return_score_fit is True:
                - tuple: (score, linear fit)
        """
        
        # Does not work with per_gridcell
        assert how != "per_gridcell", f"per_gridcell is not supported for this method."

        # Get the variable values
        xvals = self.td.data[time_dim].values # time values
        yvals = self.get_cluster_cell_aggregate(var, cluster_id, how=how)[var].values
        
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
        

    def get_cluster_cell_aggregate(self, var, cluster_id, how):
        """
        Aggregate data across all cells in the specified cluster using mean, median, sum, standard deviation, or percentile calculations.
        
        Args:
            var (str): Name of the variable for which clusters have been computed or the name of the custom cluster variable.
            cluster_id (int): The ID of the cluster to analyze. Use -1 for unclustered cells.
            how (str): Method for aggregating data across grid cells. Supported values:
                - 'mean': Mean value
                - 'median': Median value
                - 'aggr': Sum of values
                - 'std': Standard deviation
                - 'perc': Percentile value, e.g. how=('perc',0.9)

        Returns:
            xr.Dataset: Dataset containing the aggregated timeseries for the specified cluster.
                The variable name in the dataset matches the input variable name.

        Note:
            For cluster_id=-1, uses 'always_in_cluster' masking to properly handle
            unclustered cells. For other cluster IDs, uses 'spatial' masking which
            includes cells that were part of the cluster at any point in time.
        """
        from toad.core import Clustering
        clusters = self.td.get_clusters(var)
        return self.td.timeseries(
            self.td.data,
            clustering=Clustering(clusters),
            cluster_lbl=[cluster_id],
            masking='always_in_cluster' if cluster_id == -1 else 'spatial', # the spatial mask returns cells that at any point is in cluster_id, so for -1 you would get all cells. Therefore, we need another mask for unclustered cells (i.e. -1).
            how=how
        )
    

    def get_cluster_persistence_fraction(self, var, cluster_id):
        """
        Calculate the persistence fraction of cells in a given cluster over time.
        
        For each timestep, calculates what fraction of cells are currently in the cluster,
        compared to all cells that were ever part of that cluster. A value of 1.0 means
        all cells that were ever in the cluster are currently in it, while 0.0 means
        none of those cells are currently in the cluster.
        
        Args:
            var (str): Name of the variable representing the cluster data. This variable will
                be used to access the corresponding cluster field (e.g., "thk_cluster").
            cluster_id (int): The identifier for the cluster of interest. If `cluster_id == -1`, the function
                computes the fraction for unclustered cells using the 'always_in_cluster' mask.
                Otherwise, it uses the 'spatial' mask to compute for the selected cluster.

        Returns:
            np.ndarray: A 1D array of length equal to the number of time steps, where each element
                represents the fraction of cells that are part of the cluster at that specific
                time step. If there are no cells in the cluster, an array of zeros is returned.
        """
        from toad.core import Clustering

        cluster_var = f"{var}_cluster"
        timeseries_data = self.td.timeseries(
            self.td.data,
            clustering=Clustering(self.td.data[cluster_var]),
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