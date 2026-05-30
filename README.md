# ME526: Introduction to Statistical Learning - Data Wrangling

## Progress Checklist

### 1. 🏗️ Architecture & Project Setup
- [x] **PyProject Configuration**: Installable Python package.
- [x] **Modular Mixin Structure**: Segmented into `DataState`, `DataInspector`, and `DataPlotter`.
- [x] **Test Environment**: Generated edge-case test dataset and test notebook.

### 2. 💾 Data State & IO (`DataStateMixin`)
- [x] **Cross-Environment Loading**: Support Local files and Google Colab.
- [x] **Sanitization**: Auto-handle garbage strings (`?`, `N/A`, `NULL`, ` `).
- [x] **Type Coercion**: Convert valid strings into numeric types.
- [x] **State History**: `original_df` buffer for instant rollbacks (`reset_df`).
- [x] **Data Exporting**: Export original/working/selected data with JSON summaries.

### 3. 🧹 Data Inspection & Cleaning (`DataInspectorMixin`)
- [x] **Detailed Summaries**: Min, Max, Mean, Std Dev, and Unique counts based on type.
- [x] **Locate Missing Data**: Isolate rows/columns with NaNs.
- [x] **Advanced Extraction**: Slice by indices, ranges, data types, and enforce index columns.
- [x] **Targeted Deletion**: Manually remove rows or columns.
- [x] **Imputation**: Fix missing data (mean, median, mode, constant).
- [x] **Duplicate Management**: Find and remove identical rows.
- [x] **Outlier Management**: Flag and delete outliers using IQR logic.
- [x] **Unified Normalization**: Apply scaling and categorical encoding safely.

### 4. 📊 Interactive Visualizations (`DataPlotterMixin` or `DataInspectorMixin`)
- [x] **Numerical Subplots**: Plotly histograms/box plots for numeric features.
- [x] **Multi-Type Relationships**: Scatter plots and Box plots.
- [x] **Categorical Frequencies**: Colored bar charts for distributions.

### 5. 🧠 Statistical Insights & Associations
- [x] **Association Engine**: Pearson's r, Cramér's V, Point-Biserial, Eta calculations.
- [x] **Unified Heatmap**: Plotly heatmap mapping cross-type associations.

### 6. 🎨 Standalone Graphing
- [x] **PlottingMethods**: Isolated chart generation class.

---

## ✨ Extra Features Implemented
* **State Rollbacks**: Implemented `reset_df()` to revert data to its exact initial state without re-uploading.
* **Intelligent Data Extraction Engine**: `extract_data()` handles multi-range row slicing, datatype-based column filtering, and index column pinning.
* **Rich Data Exporting**: `export_data()` creates timestamped files and exports a parallel `.json` file containing dataset metadata.
* **Type-Safe Normalization Guards**: The `modify_normalize_data()` function throws intentional errors if you try to apply a numeric scaler to a categorical column (or vice-versa).
* **Automated Version Control**: Integrated `bumpver` for semantic version bumping.
* **Continuous Visual Testers**: Integrated `test_constant_mean` (MANOVA), `test_constant_covariance` (Box's M), and `test_row_independence` (Multivariate Ljung-Box) with real-time sequential visual reporting.

---

## 🚀 Advanced Features to Add (Inspired by `core.py`)
Based on imports and structures within the provided `core.py` (like `scipy.stats.multivariate_normal`), we can implement the following advanced features beyond the basic assignment parameters:
* **Multivariate Outlier Detection**: Using probability distributions (`multivariate_normal`) or Mahalanobis distance to find complex outliers, instead of just using basic 1D IQR logic.
* **Automated Outlier Capping (Winsorizing)**: Replacing outlier values with the 5th/95th percentile limits instead of aggressively deleting the entire row.
* **Advanced Imputation**: Iterative regression or KNN-based imputation for missing values instead of naive mean/median replacements.
* **One-Click HTML Reporting**: A function that generates and downloads a full HTML dashboard containing all Plotly charts and heatmaps at once.

---

## Installation & Setup
```bash
pip install -e .
```

### Versioning
```bash
bumpver update --patch
bumpver update --minor
```
