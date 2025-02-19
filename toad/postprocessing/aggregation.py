# TODO: write aggregation functions here: Combine clusters across different variables, datasets, etc


class Aggregation:
    """
    Aggregation methods for TOAD objects.
<<<<<<< HEAD
    
<<<<<<< HEAD
<<<<<<< HEAD
=======

>>>>>>> 6ffac35 (Formatted codebase with Ruff)
    >> Note: TO BE IMPLEMENTED
    """

    # TODO not sure how much of this should work through the toad class directly as we will want to combine many different datasets
=======
    Note: Docstrings here are short as this class is under heavy development
    """
    # TODO not sure this should work through the toad class because we may want to combine many different datasets
>>>>>>> c6fc662 (Docstring and type fixes)
=======
    Note: TO BE IMPLEMENTED
    """
    # TODO not sure how much of this should work through the toad class directly as we will want to combine many different datasets
>>>>>>> ba8e9d6 (Clean up docstrings)

    def __init__(self, toad):
        self.td = toad

    def combine_clusters(self, var, cluster_ids, cluster_labels=None):
        """
<<<<<<< HEAD
<<<<<<< HEAD
        To be implemented: Combine clusters across different variables, datasets, etc.
=======
        Combine clusters across different variables, datasets, etc. To be implemented.
>>>>>>> c6fc662 (Docstring and type fixes)
=======
        To be implemented: Combine clusters across different variables, datasets, etc.
>>>>>>> ba8e9d6 (Clean up docstrings)
        """

        raise NotImplementedError("Aggregation is not yet implemented.")
