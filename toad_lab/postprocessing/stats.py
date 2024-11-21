import numpy as np

from ..utils import infer_dims

class Stats:
    def __init__(self, toad):
        self.td = toad

    def compute_cluster_score(
            self,
            var,
            cluster_id, 
            cluster_label = None,
            how='mean'
        ):
        """Compute the score of a cluster label.

        :param cluster_id:      Cluster label to compute the score for.
        :type cluster_id:       int, list
        :param how:             How to compute the score.
                                    * mean: mean value
                                    * median: median value
                                    * aggr: sum of values
                                    * std: standard deviation
                                    * perc: percentile value
                                    * per_gridcell: time series for each grid cell
        """
        from ..core import Clustering

        cluster_label = f"{var}_cluster" if cluster_label is None else cluster_label

        tdim, _ = infer_dims(self.td.data)  
        xvals = self.td.data.__getattr__(tdim).values
        yvals = self.td.timeseries(self.td.data, clustering=Clustering(self.td.data[cluster_label]), cluster_lbl=cluster_id, masking='spatial', how=how).values
        (a,b) , res, _, _, _ = np.polyfit(xvals, yvals, 1, full=True)
        
        _score = res[0] 
        _score_fit = b + a*xvals

        return _score, _score_fit