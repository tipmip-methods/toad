class Preprocess:
    """
    Preprocessing methods for TOAD objects.

    Note: Docstrings here are short as this class is under heavy development
    """

    def __init__(self, toad):
        self.td = toad

    def preprocess(self, keep_only=None):
        """
        Preprocess the data. To be implemented.
        """

        raise NotImplementedError("Preprocessing is not yet implemented.")

        # Drop unnecessary variables
        if keep_only:
            self.data = self.data.drop_vars(
                [v for v in self.data.data_vars if v not in keep_only]
            )

        # apply XMIP preprocessing ...

        return self.data
