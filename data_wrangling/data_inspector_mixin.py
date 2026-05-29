import pandas as pd
import io
import os

class DataInspectorMixin:
    """
    Mixin for data ingestion, cleaning, and preprocessing.
    Contains methods for handling missing values, duplicates, outliers, and scaling.
    """
    def load_data(self, file_path=None):
        """
        Loads a CSV file into self.df. Supports both local paths and Google Colab upload.
        """
        file_name = None
        
        # Local Environment
        if file_path is not None:
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

            # If the conversion didn't turn the entire column into NaNs
            # (and it wasn't already all NaN), we apply the change.
            if not numeric_col.isna().all():
                self.df[col] = numeric_col

        print(f"\n✅ File '{file_name}' loaded successfully!")
        print(f"Number of columns: {self.df.shape[1]}")
        print(f"Number of records: {self.df.shape[0]}")
