import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import pandas as pd
import numpy as np

class PlottingMethods:
    """
    Modular class to handle granular chart generation using Plotly Express & Graph Objects.
    These methods return the raw `go.Trace` objects if `as_trace=True` is passed (perfect for subplots),
    or a full `go.Figure` if `as_trace=False` (perfect for standalone viewing).
    """

    @staticmethod
    def _handle_return(fig_or_traces, layout_kwargs, as_trace):
        # Enforce Plotly Dark theme universally
        if "template" not in layout_kwargs:
            layout_kwargs["template"] = "plotly_dark"
            
        if isinstance(fig_or_traces, go.Figure):
            fig = fig_or_traces
            fig.update_layout(**layout_kwargs)
            if as_trace:
                return list(fig.data)
            return fig
        else:
            traces = fig_or_traces
            if not isinstance(traces, list) and not isinstance(traces, tuple):
                traces = [traces]
            if as_trace:
                return traces if len(traces) > 1 else traces[0]
            fig = go.Figure(data=traces)
            fig.update_layout(**layout_kwargs)
            return fig

    @classmethod
    def plot_bar(cls, data, x_col, y_col, color_col=None, title="Bar Chart", config=None, as_trace=False):
        config = config or {}
        # Automatically use x_col for coloring if color_col isn't specified to ensure variety
        c_col = color_col if color_col else (x_col if x_col in data.columns else None)
        fig = px.bar(data, x=x_col, y=y_col, color=c_col, **config)
        return cls._handle_return(fig, {"title": title}, as_trace)

    @classmethod
    def plot_pie(cls, data, names_col, values_col, color_col=None, title="Pie Chart", config=None, as_trace=False):
        config = config or {}
        c_col = color_col if color_col else names_col
        fig = px.pie(data, names=names_col, values=values_col, color=c_col, **config)
        return cls._handle_return(fig, {"title": title}, as_trace)

    @classmethod
    def plot_histogram(cls, data, x_col, color_col=None, title="Histogram", config=None, as_trace=False):
        config = config or {}
        fig = px.histogram(data, x=x_col, color=color_col, **config)
        return cls._handle_return(fig, {"title": title}, as_trace)

    @classmethod
    def plot_histogram_2d(cls, data, x_col, y_col, title="2D Histogram", config=None, as_trace=False):
        config = config or {}
        fig = px.density_heatmap(data, x=x_col, y=y_col, color_continuous_scale="Viridis", **config)
        return cls._handle_return(fig, {"title": title}, as_trace)

    @classmethod
    def plot_scatter_3d(cls, data, x_col, y_col, z_col, color_col=None, title="3D Scatter Plot", config=None, as_trace=False):
        config = config or {}
        c_col = color_col if color_col else z_col
        fig = px.scatter_3d(data, x=x_col, y=y_col, z=z_col, color=c_col, **config)
        return cls._handle_return(fig, {"title": title}, as_trace)

    @classmethod
    def plot_box_violin(cls, data, y_col, x_col=None, color_col=None, title="Violin & Box Plot", config=None, as_trace=False):
        config = config or {}
        c_col = color_col if color_col else x_col
        fig = px.violin(data, y=y_col, x=x_col, color=c_col, box=True, points="all", **config)
        return cls._handle_return(fig, {"title": title}, as_trace)

    @classmethod
    def plot_scatter(cls, data, x_col, y_col, color_col=None, title="Scatter Plot", config=None, as_trace=False):
        config = config or {}
        fig = px.scatter(data, x=x_col, y=y_col, color=color_col, **config)
        return cls._handle_return(fig, {"title": title}, as_trace)

    @classmethod
    def plot_bubble(cls, data, x_col, y_col, size_col, color_col=None, title="Bubble Chart", config=None, as_trace=False):
        config = config or {}
        c_col = color_col if color_col else size_col
        fig = px.scatter(data, x=x_col, y=y_col, size=size_col, color=c_col, **config)
        return cls._handle_return(fig, {"title": title}, as_trace)

    @classmethod
    def plot_heatmap(cls, z_matrix, x_labels, y_labels, title="Heatmap", config=None, as_trace=False):
        config = config or {}
        trace = go.Heatmap(z=z_matrix, x=x_labels, y=y_labels, colorscale='Viridis', **config)
        return cls._handle_return(trace, {"title": title}, as_trace)

    @classmethod
    def plot_distplot(cls, data, col_names, title="Distplot", config=None, as_trace=False):
        """
        Creates a distplot using plotly.figure_factory. 
        Requires col_names to be a list of numerical columns to plot.
        """
        config = config or {}
        hist_data = [data[col].dropna() for col in col_names]
        fig = ff.create_distplot(hist_data, col_names, **config)
        return cls._handle_return(fig, {"title": title}, as_trace)


class DataPlotterMixin(PlottingMethods):
    """
    Main DataInspector plotting integrations.
    Inherits PlottingMethods for atomic plot capabilities and provides
    complex analytical views.
    """
    
    def generate_univariate_subplots(self, numerical_cols=None):
        """
        Example of combining helper methods into a subplot visualization.
        Forces distinct colors per subplot automatically.
        """
        if self.df is None: return print("No data loaded.")
        cols = numerical_cols if numerical_cols else self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not cols: return print("No numerical columns to plot.")
        
        fig = make_subplots(rows=len(cols), cols=1, subplot_titles=[f"Distribution of {c}" for c in cols])
        
        colors = px.colors.qualitative.Plotly
        for i, col in enumerate(cols, 0):
            traces = self.plot_histogram(self.df, col, as_trace=True)
            if isinstance(traces, list):
                for t in traces:
                    t.marker.color = colors[i % len(colors)]
                    fig.add_trace(t, row=i+1, col=1)
            else:
                traces.marker.color = colors[i % len(colors)]
                fig.add_trace(traces, row=i+1, col=1)
            
        fig.update_layout(height=300 * len(cols), title_text="Univariate Numerical Analysis", template="plotly_dark")
        fig.show()
