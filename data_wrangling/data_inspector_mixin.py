import pandas as pd
import numpy as np
try:
    from IPython.display import display
except ImportError:
    display = print

class DataInspectorMixin:
    """
    Mixin for data cleaning, preprocessing, and modification.
    """
    def get_summary(self, detailed=False):
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

    def get_missing_data(self, axis='rows'):
        if self.df is None: return print("No data loaded.")
        if axis == 'rows':
            missing_mask = self.df.isnull().any(axis=1)
            missing_data = self.df[missing_mask]
            if missing_data.empty: print("✨ No missing data found in rows!")
            else:
                print(f"🔍 Found {len(missing_data)} rows with missing values.")
                display(missing_data.head())
            return missing_data
        elif axis == 'columns':
            missing_mask = self.df.isnull().any(axis=0)
            missing_data = self.df.loc[:, missing_mask]
            if missing_data.empty: print("✨ No missing data found in columns!")
            else:
                print(f"🔍 Found {missing_data.shape[1]} columns with missing values.")
                display(missing_data.head())
            return missing_data
        else:
            print("Invalid axis. Choose 'rows' or 'columns'.")

    def extract_data(self, columns=None, column_type=None, index_column=None, row_indices=None, row_ranges=None):
        if self.df is None: return print("No data loaded.")
        extracted = self.df.copy()
        
        selected_rows = []
        if row_indices: selected_rows.extend(row_indices)
        if row_ranges:
            for r in row_ranges: selected_rows.extend(list(range(r[0], r[1] + 1)))
        if selected_rows:
            selected_rows = list(set(selected_rows))
            extracted = extracted.loc[extracted.index.intersection(selected_rows)]
            
        selected_cols = []
        if columns: selected_cols = columns
        elif column_type == 'numerical': selected_cols = extracted.select_dtypes(include=[np.number]).columns.tolist()
        elif column_type == 'categorical': selected_cols = extracted.select_dtypes(exclude=[np.number]).columns.tolist()
        else: selected_cols = extracted.columns.tolist()
            
        if index_column and index_column in self.df.columns:
            if index_column in selected_cols: selected_cols.remove(index_column)
            selected_cols.insert(0, index_column)
            
        selected_cols = [c for c in selected_cols if c in extracted.columns]
        extracted = extracted[selected_cols]
        self.selected_df = extracted
        print(f"✅ Data extracted into selected_df: {extracted.shape[0]} rows, {extracted.shape[1]} columns.")
        display(self.selected_df.head())
        return self.selected_df

    def delete_rows(self, indices, summary=False):
        if self.df is None: return print("No data loaded.")
        existing_indices = [i for i in indices if i in self.df.index]
        self.df = self.df.drop(index=existing_indices).reset_index(drop=True)
        print(f"🗑️ Deleted {len(existing_indices)} rows. New count: {len(self.df)}")
        if summary: self.get_summary()

    def delete_columns(self, columns, summary=False):
        if self.df is None: return print("No data loaded.")
        existing_cols = [c for c in columns if c in self.df.columns]
        self.df = self.df.drop(columns=existing_cols)
        print(f"🗑️ Deleted {len(existing_cols)} columns. Remaining count: {len(self.df.columns)}")
        if summary: self.get_summary()

    # --- DATA MODIFIERS ---

    def modify_missing_values(self, columns=None, strategy='median', fill_value=None):
        """Imputes missing values in specified columns."""
        if self.df is None: return print("No data loaded.")
        target_cols = columns if columns else self.df.columns[self.df.isnull().any()].tolist()
        if isinstance(target_cols, str): target_cols = [target_cols]
        
        for col in target_cols:
            if col not in self.df.columns: continue
            
            is_numeric = pd.api.types.is_numeric_dtype(self.df[col])
            if strategy == 'mean' and is_numeric:
                self.df[col] = self.df[col].fillna(self.df[col].mean())
            elif strategy == 'median' and is_numeric:
                self.df[col] = self.df[col].fillna(self.df[col].median())
            elif strategy == 'mode':
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
            elif strategy == 'constant':
                self.df[col] = self.df[col].fillna(fill_value)
                
        print(f"🛠️ Missing values imputed using '{strategy}' strategy for: {target_cols}")

    def modify_duplicates(self):
        """Removes exact duplicate rows."""
        if self.df is None: return print("No data loaded.")
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        dropped = initial_count - len(self.df)
        print(f"✨ Removed {dropped} duplicate rows. New row count: {len(self.df)}")

    def modify_outliers(self, columns=None, find_and_delete=False):
        """Flags or deletes outliers using IQR logic."""
        if self.df is None: return print("No data loaded.")
        target_cols = columns if columns else self.df.select_dtypes(include=[np.number]).columns.tolist()
        if isinstance(target_cols, str): target_cols = [target_cols]
        all_outliers = set()

        for col in target_cols:
            if col not in self.df.columns or not pd.api.types.is_numeric_dtype(self.df[col]):
                continue
            Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.df[(self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))]
            all_outliers.update(outliers.index.tolist())
            if len(outliers) > 0:
                print(f"🚨 {col}: Found {len(outliers)} outliers.")

        if all_outliers:
            if find_and_delete:
                self.df = self.df.drop(index=list(all_outliers)).reset_index(drop=True)
                print(f"🗑️ Deleted {len(all_outliers)} outlier rows total.")
            else:
                print("Outliers identified but not deleted. Pass find_and_delete=True to remove them.")
        else:
            print("✅ No outliers found.")

    def modify_normalize_data(self, columns, method, replace=True):
        """
        Normalizes or encodes columns.
        Valid methods for numeric: 'minmax', 'standard', 'robust'
        Valid methods for categorical: 'uniform', 'ordinal', 'onehot', 'minmax_ordinal'
        """
        if self.df is None: return print("No data loaded.")
        if isinstance(columns, str): columns = [columns]
        
        numeric_methods = ['minmax', 'standard', 'robust']
        categorical_methods = ['uniform', 'ordinal', 'onehot', 'minmax_ordinal']
        method = method.lower().strip()
        
        from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OrdinalEncoder, OneHotEncoder
        
        for col in columns:
            if col not in self.df.columns:
                print(f"⚠️ Column '{col}' not found. Skipping.")
                continue
                
            is_numeric = pd.api.types.is_numeric_dtype(self.df[col])
            
            if is_numeric and method not in numeric_methods:
                raise ValueError(f"Method '{method}' is not suitable for numeric column '{col}'. Use one of {numeric_methods}.")
            if not is_numeric and method not in categorical_methods:
                raise ValueError(f"Method '{method}' is not suitable for categorical column '{col}'. Use one of {categorical_methods}.")
                
            col_data = self.df[[col]].copy()
            
            # Fill NaNs temporarily for fitting
            if col_data.isnull().any().any():
                if is_numeric:
                    col_data = col_data.fillna(col_data.median())
                else:
                    col_data = col_data.fillna("Missing")
            
            new_data = {}
            if method == 'minmax':
                new_data[col] = MinMaxScaler().fit_transform(col_data)[:, 0]
            elif method == 'standard':
                new_data[col] = StandardScaler().fit_transform(col_data)[:, 0]
            elif method == 'robust':
                new_data[col] = RobustScaler().fit_transform(col_data)[:, 0]
            elif method == 'uniform':
                codes = self.df[col].astype('category').cat.codes
                max_code = codes.max()
                new_data[col] = (codes / max_code) if max_code > 0 else 0.0
            elif method == 'ordinal':
                new_data[col] = OrdinalEncoder().fit_transform(col_data)[:, 0]
            elif method == 'onehot':
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded = encoder.fit_transform(col_data)
                for i, name in enumerate(encoder.get_feature_names_out([col])):
                    new_data[name] = encoded[:, i]
            elif method == 'minmax_ordinal':
                encoded = OrdinalEncoder().fit_transform(col_data)
                new_data[col] = MinMaxScaler().fit_transform(encoded)[:, 0]
                
            if replace:
                self.df = self.df.drop(columns=[col])
                for k, v in new_data.items():
                    self.df[k] = v
            else:
                for k, v in new_data.items():
                    new_name = f"{k}_{method}" if method != 'onehot' else k
                    self.df[new_name] = v

        print(f"✨ Successfully applied '{method}' to {columns}.")
