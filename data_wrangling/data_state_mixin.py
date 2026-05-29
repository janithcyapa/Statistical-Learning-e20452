import pandas as pd

class DataStateMixin:
    """
    Mixin to hold the internal state of the dataset.
    This manages the raw, numeric, categorical, and normalized dataframes.
    """
    
    def __init__(self):
        self.df = None
        self.numeric_df = None
        self.categorical_df = None
        self.categorical_normalized_df = None
        self.normalized_data_df = None
        self.numeric_normalized_df = None
