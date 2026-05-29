import pandas as pd
import io
import os
try:
    from IPython.display import display
except ImportError:
    display = print

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
        
        # Automatically run get_summary after importing
        self.get_summary()

    def get_summary(self):
        """
        Generates and prints a summary DataFrame with column names, data types, 
        number of records, and number of missing values.
        """
        if self.df is None:
            print("Error: No data loaded.")
            return None
        
        summary_data = []
        for col in self.df.columns:
            summary_data.append({
                'Column Name': col,
                'Data Type': str(self.df[col].dtype),
                'Total Records': len(self.df[col]),
                'Missing Values': self.df[col].isnull().sum()
            })
        
        self.summary_df = pd.DataFrame(summary_data)
        print("--- Data Summary ---")
        display(self.summary_df)
        return self.summary_df

    def delete_rows(self, indices, summary=False):
        """
        Deletes rows based on a list of indices.
        """
        if self.df is None:
            return print("No data loaded.")
        
        existing_indices = [i for i in indices if i in self.df.index]
        self.df = self.df.drop(index=existing_indices).reset_index(drop=True)
        print(f"🗑️ Deleted {len(existing_indices)} rows. New count: {len(self.df)}")
        
        if summary:
            self.get_summary()

    def delete_columns(self, columns, summary=False):
        """
        Deletes columns based on a list of names.
        """
        if self.df is None:
            return print("No data loaded.")
        
        existing_cols = [c for c in columns if c in self.df.columns]
        self.df = self.df.drop(columns=existing_cols)
        print(f"🗑️ Deleted {len(existing_cols)} columns. Remaining count: {len(self.df.columns)}")
        
        if summary:
            self.get_summary()

    def get_missing_data(self, axis='rows'):
        """
        Shows and returns data missing even one value.
        axis: 'rows' or 'columns'
        """
        if self.df is None:
            return print("No data loaded.")
            
        if axis == 'rows':
            missing_mask = self.df.isnull().any(axis=1)
            missing_data = self.df[missing_mask]
            if missing_data.empty:
                print("✨ No missing data found in rows!")
            else:
                print(f"🔍 Found {len(missing_data)} rows with missing values.")
                display(missing_data.head())
            return missing_data
        elif axis == 'columns':
            missing_mask = self.df.isnull().any(axis=0)
            missing_data = self.df.loc[:, missing_mask]
            if missing_data.empty:
                print("✨ No missing data found in columns!")
            else:
                print(f"🔍 Found {missing_data.shape[1]} columns with missing values.")
                display(missing_data.head())
            return missing_data
        else:
            print("Invalid axis. Choose 'rows' or 'columns'.")
