# TODO: write aggregation functions here: Combine clusters across different variables, datasets, etc

class Aggregation:
    """
    Aggregation methods for TOAD objects.
    
    Note: Docstrings here are short as this class is under heavy development
    """
    # TODO not sure this should work through the toad class because we may want to combine many different datasets

    def __init__(self, toad):
        self.td = toad

    def combine_clusters(self, var, cluster_ids, cluster_labels=None):
        """
        Combine clusters across different variables, datasets, etc. To be implemented.
        """

        raise NotImplementedError("Aggregation is not yet implemented.")
