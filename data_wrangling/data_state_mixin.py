import pandas as pd
import io
import os
import time
import json
try:
    from IPython.display import display
except ImportError:
    display = print

class DataStateMixin:
    """
    Mixin to hold the internal state of the dataset and handle I/O operations.
    """
    def __init__(self):
        self.df = None
        self.original_df = None
        self.summary_df = None
        self.selected_df = None

    def load_data(self, file_path=None, df=None):
        """
        Loads a CSV file or an existing DataFrame into the instance's state (`self.df`).
        It also handles missing values strings intelligently and attempts to automatically 
        coerce columns to their appropriate numeric types.
        
        Parameters
        ----------
        file_path : str, optional
            The local file path to a CSV dataset. If running in Google Colab and neither `df` 
            nor `file_path` is provided, a file upload prompt will be triggered automatically.
        df : pandas.DataFrame, optional
            A pre-loaded DataFrame to inject directly into the class state.
            
        Returns
        -------
        None
            The function does not return a value but updates `self.df` and `self.original_df`.
        """
        file_name = None
        
        if df is not None:
            self.df = df.copy()
            file_name = "Provided_DataFrame"
        # Local Environment
        elif file_path is not None:
            if not os.path.exists(file_path):
                print(f"Error: File '{file_path}' not found.")
                return
            file_name = os.path.basename(file_path)
            self.df = pd.read_csv(file_path, na_values=['?', 'n/a', 'N/A', 'NULL', 'null', ' '])
            
        # Google Colab Environment
        else:
            try:
                from google.colab import files
                uploaded = files.upload()
                if not uploaded:
                    print("No file uploaded.")
                    return
                file_name = list(uploaded.keys())[0]
                self.df = pd.read_csv(io.BytesIO(uploaded[file_name]),
                                    na_values=['?', 'n/a', 'N/A', 'NULL', 'null', ' '])
            except ImportError:
                print("Error: google.colab is not available. Please provide a file_path for local execution.")
                return

        self.df['count'] = 1

        for col in self.df.columns:
            # Attempt to convert the column to numeric, forcing errors to NaN
            numeric_col = pd.to_numeric(self.df[col], errors='coerce')
            if not numeric_col.isna().all():
                self.df[col] = numeric_col

        # Store original dataframe for resetting
        self.original_df = self.df.copy()

        print(f"\n✅ File '{file_name}' loaded successfully!")
        
        # Requires get_summary to be implemented in another mixin
        if hasattr(self, 'get_summary'):
            self.get_summary()

    def reset_df(self):
        """
        Resets the working DataFrame (`self.df`) back to its original state exactly as it was 
        when `load_data` was first called, abandoning any cleaning or transformations performed.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
            Updates `self.df` in-place and prints a summary if available.
        """
        if self.original_df is not None:
            self.df = self.original_df.copy()
            print("✅ DataFrame reset to original state.")
            if hasattr(self, 'get_summary'):
                self.get_summary()
        else:
            print("Error: No original data to reset to.")

    def export_data(self, dataset='working', file_name='export', export_summary=True):
        """
        Exports the specified dataset state to a CSV file appended with an epoch timestamp.
        Optionally exports a parallel JSON metadata file detailing column types, missing counts, 
        and statistical summaries. If running in Google Colab, triggers automatic browser downloads.
        
        Parameters
        ----------
        dataset : str, default 'working'
            The specific dataset state to export. Options are:
            - 'working' or 'df': The current modified dataset (`self.df`).
            - 'original': The initial pristine dataset (`self.original_df`).
            - 'selected': The dataset generated via extraction operations (`self.selected_df`).
        file_name : str, default 'export'
            The base prefix for the exported file names.
        export_summary : bool, default True
            If True, generates a paired `.json` file containing metadata and basic statistics.
            
        Returns
        -------
        None
            Generates files on the disk and triggers downloads.
        """
        target_df = None
        if dataset in ['df', 'working']:
            target_df = self.df
        elif dataset == 'original':
            target_df = self.original_df
        elif dataset == 'selected':
            target_df = self.selected_df
        else:
            return print("Invalid dataset. Choose 'working', 'original', or 'selected'.")
            
        if target_df is None:
            return print(f"Error: {dataset} dataset is empty or not initialized.")
            
        epoch_time = int(time.time())
        final_filename = f"{file_name}_{epoch_time}.csv"
        
        target_df.to_csv(final_filename, index=False)
        print(f"💾 Data exported to {final_filename}")
        
        if export_summary:
            summary_dict = {}
            for col in target_df.columns:
                col_data = target_df[col]
                col_summary = {
                    'Data Type': str(col_data.dtype),
                    'Missing Values': int(col_data.isnull().sum())
                }
                if pd.api.types.is_numeric_dtype(col_data):
                    col_summary['Mean'] = float(col_data.mean()) if not pd.isna(col_data.mean()) else None
                else:
                    col_summary['Unique Values'] = int(col_data.nunique())
                summary_dict[col] = col_summary
                
            summary_filename = f"{file_name}_{epoch_time}_summary.json"
            with open(summary_filename, 'w') as f:
                json.dump(summary_dict, f, indent=4)
            print(f"📄 Summary exported to {summary_filename}")
            
        try:
            from google.colab import files
            files.download(final_filename)
            if export_summary:
                files.download(summary_filename)
        except ImportError:
            pass