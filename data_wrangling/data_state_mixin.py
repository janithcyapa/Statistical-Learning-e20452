class DataStateMixin:
    """
    Mixin to hold the internal state of the dataset.
    This manages the raw, numeric, categorical, and normalized dataframes.
    """
    def __init__(self):
        self.df = None
        self.original_df = None
        self.summary_df = None
        self.selected_df = None