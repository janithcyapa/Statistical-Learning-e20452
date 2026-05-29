import pandas as pd
import numpy as np
import io
import os
import time
import json
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

            if not numeric_col.isna().all():
                self.df[col] = numeric_col

        # Store original dataframe for resetting
        self.original_df = self.df.copy()

        print(f"\n✅ File '{file_name}' loaded successfully!")
        self.get_summary()

    def reset_df(self):
        """Resets self.df back to the original unmodified state."""
        if self.original_df is not None:
            self.df = self.original_df.copy()
            print("✅ DataFrame reset to original state.")
            self.get_summary()
        else:
            print("Error: No original data to reset to.")

    def get_summary(self, detailed=False):
        """
        Generates and prints a summary DataFrame. 
        If detailed=True, includes statistical metrics based on data types.
        """
        if self.df is None:
            print("Error: No data loaded.")
            return None
        
        summary_data = []
        for col in self.df.columns:
            col_data = self.df[col]
            summary_dict = {
                'Column Name': col,
                'Data Type': str(col_data.dtype),
                'Total Records': len(col_data),
                'Missing Values': col_data.isnull().sum()
            }
            
            if detailed:
                if pd.api.types.is_numeric_dtype(col_data):
                    summary_dict['Min'] = col_data.min()
                    summary_dict['Max'] = col_data.max()
                    summary_dict['Mean'] = col_data.mean()
                    summary_dict['Std Dev'] = col_data.std()
                    summary_dict['Unique Values'] = np.nan
                    summary_dict['Most Frequent'] = np.nan
                else:
                    summary_dict['Min'] = np.nan
                    summary_dict['Max'] = np.nan
                    summary_dict['Mean'] = np.nan
                    summary_dict['Std Dev'] = np.nan
                    summary_dict['Unique Values'] = col_data.nunique()
                    summary_dict['Most Frequent'] = col_data.mode()[0] if not col_data.mode().empty else np.nan
            
            summary_data.append(summary_dict)
        
        self.summary_df = pd.DataFrame(summary_data)
        print(f"--- Data Summary {'(Detailed)' if detailed else ''} ---")
        display(self.summary_df)
        return self.summary_df

    def delete_rows(self, indices, summary=False):
        """Deletes rows based on a list of indices."""
        if self.df is None:
            return print("No data loaded.")
        
        existing_indices = [i for i in indices if i in self.df.index]
        self.df = self.df.drop(index=existing_indices).reset_index(drop=True)
        print(f"🗑️ Deleted {len(existing_indices)} rows. New count: {len(self.df)}")
        
        if summary:
            self.get_summary()

    def delete_columns(self, columns, summary=False):
        """Deletes columns based on a list of names."""
        if self.df is None:
            return print("No data loaded.")
        
        existing_cols = [c for c in columns if c in self.df.columns]
        self.df = self.df.drop(columns=existing_cols)
        print(f"🗑️ Deleted {len(existing_cols)} columns. Remaining count: {len(self.df.columns)}")
        
        if summary:
            self.get_summary()

    def get_missing_data(self, axis='rows'):
        """Shows and returns data missing even one value."""
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

    def extract_data(self, columns=None, column_type=None, index_column=None, row_indices=None, row_ranges=None):
        """
        Extracts specific data rows/columns into self.selected_df.
        """
        if self.df is None:
            return print("No data loaded.")
            
        extracted = self.df.copy()
        
        # 1. Row selection
        selected_rows = []
        if row_indices:
            selected_rows.extend(row_indices)
        if row_ranges:
            for r in row_ranges:
                # Expecting tuples like (start, end)
                selected_rows.extend(list(range(r[0], r[1] + 1)))
                
        if selected_rows:
            selected_rows = list(set(selected_rows)) # Make unique
            extracted = extracted.loc[extracted.index.intersection(selected_rows)]
            
        # 2. Column selection
        selected_cols = []
        if columns:
            selected_cols = columns
        elif column_type == 'numerical':
            selected_cols = extracted.select_dtypes(include=[np.number]).columns.tolist()
        elif column_type == 'categorical':
            selected_cols = extracted.select_dtypes(exclude=[np.number]).columns.tolist()
        else:
            selected_cols = extracted.columns.tolist()
            
        # 3. Ensure index column is kept and forced to the front
        if index_column and index_column in self.df.columns:
            if index_column in selected_cols:
                selected_cols.remove(index_column)
            selected_cols.insert(0, index_column)
            
        selected_cols = [c for c in selected_cols if c in extracted.columns]
        extracted = extracted[selected_cols]
        
        self.selected_df = extracted
        print(f"✅ Data extracted into selected_df: {extracted.shape[0]} rows, {extracted.shape[1]} columns.")
        display(self.selected_df.head())
        return self.selected_df

    def export_data(self, dataset='working', file_name='export', export_summary=True):
        """
        Exports the specified dataset ('df', 'original', 'selected') to a CSV 
        with a timestamp, and optionally a JSON summary.
        """
        target_df = None
        if dataset == 'df':
            target_df = self.df
        elif dataset == 'original':
            target_df = self.original_df
        elif dataset == 'selected':
            target_df = self.selected_df
        else:
            return print("Invalid dataset. Choose 'df', 'original', or 'selected'.")
            
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
            pass # We are in a local environment
