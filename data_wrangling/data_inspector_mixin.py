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

    def summary_plot(self, columns=None,
                     numeric_plots=None,
                     categorical_plots=None,
                     separate_plots=False, subplot_cols=3):
        """
        Creates a comprehensive multi-row Plotly visualization dashboard.

        Parameters
        ----------
        columns : list[str] or None
            Specific column names to visualize. If None, uses all columns in self.df.
        numeric_plots : list[str] or None
            Plot types for numerical columns. Supported: 'violin', 'scatter',
            'histogram', 'distplot'. Defaults to ['violin', 'scatter', 'distplot'].
        categorical_plots : list[str] or None
            Plot types for categorical columns. Supported: 'pie', 'histogram', 'bar'.
            Defaults to ['pie', 'histogram'].
        separate_plots : bool
            If True, each column gets its own dedicated row of subplots.
            If False (default), all numerical columns share one set of plots and
            all categorical columns share another.
        subplot_cols : int
            Number of subplot columns per row (default 3).
        """
        if self.df is None:
            print("⚠️ No data loaded.")
            return

        # --- Defaults ---
        if numeric_plots is None:
            numeric_plots = ['violin', 'scatter', 'distplot']
        if categorical_plots is None:
            categorical_plots = ['pie', 'histogram']

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.express as px
        import plotly.figure_factory as ff
        import math

        # --- Resolve & validate columns ---
        all_cols = self.df.columns.tolist()
        if columns is not None:
            invalid = [c for c in columns if c not in all_cols]
            if invalid:
                print(f"⚠️ Columns not found in data and skipped: {invalid}")
            cols = [c for c in columns if c in all_cols]
        else:
            cols = all_cols

        if not cols:
            print("⚠️ No valid columns to plot.")
            return

        colors = px.colors.qualitative.Vivid + px.colors.qualitative.Plotly

        # --- Helper: add a single plot trace to fig ---
        def _add_numeric_trace(fig, pt, col_name, row, col_idx, color):
            """Add a single numeric plot type to the figure."""
            data_series = self.df[col_name].dropna()
            if data_series.empty:
                return
            if pt == 'violin':
                fig.add_trace(
                    go.Violin(x=data_series, name=col_name, box_visible=True,
                              meanline_visible=True, points='all',
                              marker_color=color, showlegend=False),
                    row=row, col=col_idx)
            elif pt == 'scatter':
                fig.add_trace(
                    go.Scatter(x=data_series.index, y=data_series, mode='markers',
                               name=col_name, marker=dict(color=color, size=5,
                               opacity=0.7), showlegend=False),
                    row=row, col=col_idx)
            elif pt == 'histogram':
                fig.add_trace(
                    go.Histogram(x=data_series, name=col_name,
                                 marker_color=color, opacity=0.8,
                                 showlegend=False),
                    row=row, col=col_idx)
            elif pt == 'distplot':
                try:
                    if len(data_series) > 1:
                        fig_dist = ff.create_distplot(
                            [data_series.values], [col_name], colors=[color],
                            show_rug=False)
                        for trace in fig_dist['data']:
                            trace.xaxis = None
                            trace.yaxis = None
                            trace.showlegend = False
                            fig.add_trace(trace, row=row, col=col_idx)
                except Exception:
                    pass

        def _add_categorical_trace(fig, pt, col_name, row, col_idx, color):
            """Add a single categorical plot type to the figure."""
            val_counts = self.df[col_name].value_counts().reset_index()
            val_counts.columns = [col_name, 'count']
            if val_counts.empty:
                return
            if pt == 'pie':
                fig.add_trace(
                    go.Pie(labels=val_counts[col_name], values=val_counts['count'],
                           name=col_name, textinfo='percent+label', hole=0.3,
                           marker=dict(colors=px.colors.qualitative.Pastel)),
                    row=row, col=col_idx)
            elif pt in ('bar', 'histogram'):
                cat_colors = px.colors.qualitative.Set3[:len(val_counts)]
                fig.add_trace(
                    go.Bar(x=val_counts[col_name].astype(str),
                           y=val_counts['count'], name=col_name,
                           marker_color=cat_colors if len(cat_colors) == len(val_counts)
                           else color, showlegend=False),
                    row=row, col=col_idx)

        # =========================================================
        #  MODE 1: SEPARATE — each column gets its own row of plots
        # =========================================================
        if separate_plots:
            max_plot_cols = 0
            row_specs = []
            row_titles = []

            for col in cols:
                is_num = pd.api.types.is_numeric_dtype(self.df[col])
                plot_types = numeric_plots if is_num else categorical_plots
                max_plot_cols = max(max_plot_cols, len(plot_types))

                current_spec = []
                for pt in plot_types:
                    current_spec.append(
                        {"type": "domain"} if pt == 'pie' else {"type": "xy"})
                row_specs.append(current_spec)

                for pt in plot_types:
                    row_titles.append(f"{col} — {pt.capitalize()}")

            # Pad short rows with None so all rows have the same number of cols
            for spec_row in row_specs:
                while len(spec_row) < max_plot_cols:
                    spec_row.append(None)
                    row_titles.append("")

            if max_plot_cols == 0:
                print("⚠️ No plots to draw.")
                return

            fig = make_subplots(
                rows=len(cols), cols=max_plot_cols, specs=row_specs,
                subplot_titles=row_titles,
                vertical_spacing=max(0.02, 0.15 / len(cols)),
                horizontal_spacing=0.05)

            for i, col in enumerate(cols):
                row = i + 1
                is_num = pd.api.types.is_numeric_dtype(self.df[col])
                plot_types = numeric_plots if is_num else categorical_plots
                base_color = colors[i % len(colors)]

                for j, pt in enumerate(plot_types):
                    if is_num:
                        _add_numeric_trace(fig, pt, col, row, j + 1, base_color)
                    else:
                        _add_categorical_trace(fig, pt, col, row, j + 1, base_color)

            fig.update_layout(
                template="plotly_dark",
                height=max(400, 300 * len(cols)),
                showlegend=False,
                title_text="Summary Dashboard — Separate View",
                title_x=0.5,
                bargap=0.15,
                # paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter, sans-serif"))
            fig.show()

        # =========================================================
        #  MODE 2: COMBINED — all numericals together, categoricals together
        # =========================================================
        else:
            num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(self.df[c])]
            cat_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(self.df[c])]

            plot_cells = []
            if num_cols:
                for pt in numeric_plots:
                    plot_cells.append({
                        "type": "xy", "name": pt, "group": "num",
                        "title": f"Numerical — {pt.capitalize()}"})
            if cat_cols:
                for pt in categorical_plots:
                    spec_type = "domain" if pt == 'pie' else "xy"
                    plot_cells.append({
                        "type": spec_type, "name": pt, "group": "cat",
                        "title": f"Categorical — {pt.capitalize()}"})

            if not plot_cells:
                print("⚠️ No valid plots to draw.")
                return

            rows = math.ceil(len(plot_cells) / subplot_cols)
            row_specs = []
            row_titles = []

            for r in range(rows):
                spec_row = []
                for c in range(subplot_cols):
                    idx = r * subplot_cols + c
                    if idx < len(plot_cells):
                        spec_row.append({"type": plot_cells[idx]["type"]})
                        row_titles.append(plot_cells[idx]["title"])
                    else:
                        spec_row.append(None)
                        row_titles.append("")
                row_specs.append(spec_row)

            fig = make_subplots(
                rows=rows, cols=subplot_cols, specs=row_specs,
                subplot_titles=row_titles,
                vertical_spacing=0.10, horizontal_spacing=0.06)

            for idx, cell in enumerate(plot_cells):
                row = (idx // subplot_cols) + 1
                col = (idx % subplot_cols) + 1
                pt = cell["name"]

                if cell["group"] == "num":
                    if pt == 'distplot':
                        # Distplot merges all numeric columns into one subplot
                        try:
                            valid = [c for c in num_cols
                                     if len(self.df[c].dropna()) > 1]
                            if valid:
                                hist_data = [self.df[c].dropna().values
                                             for c in valid]
                                clrs = [colors[i % len(colors)]
                                        for i in range(len(valid))]
                                fig_dist = ff.create_distplot(
                                    hist_data, valid, colors=clrs,
                                    show_rug=False)
                                for trace in fig_dist['data']:
                                    trace.xaxis = None
                                    trace.yaxis = None
                                    fig.add_trace(trace, row=row, col=col)
                        except Exception:
                            pass
                    else:
                        for i, c in enumerate(num_cols):
                            _add_numeric_trace(
                                fig, pt, c, row, col,
                                colors[i % len(colors)])

                elif cell["group"] == "cat":
                    if pt == 'pie' and len(cat_cols) == 1:
                        # Single categorical column → simple pie
                        _add_categorical_trace(
                            fig, 'pie', cat_cols[0], row, col, colors[0])
                    elif pt == 'pie' and len(cat_cols) > 1:
                        # Multiple categoricals → sunburst
                        try:
                            cat_df = self.df[cat_cols].fillna("Unknown").astype(str)
                            fig_sun = px.sunburst(cat_df, path=cat_cols)
                            sun_trace = fig_sun.data[0]
                            sun_trace.domain = None
                            fig.add_trace(sun_trace, row=row, col=col)
                        except Exception:
                            pass
                    elif pt in ('histogram', 'bar'):
                        for i, c in enumerate(cat_cols):
                            _add_categorical_trace(
                                fig, pt, c, row, col,
                                colors[i % len(colors)])

            fig.update_layout(
                template="plotly_dark",
                height=max(450, 400 * rows),
                showlegend=True,
                title_text="Summary Dashboard — Combined View",
                title_x=0.5,
                bargap=0.15,
                barmode='group',
                # paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter, sans-serif"),
                legend=dict(orientation="h", yanchor="bottom",
                            y=-0.08, xanchor="center", x=0.5))
            fig.show()
