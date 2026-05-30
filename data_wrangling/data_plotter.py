import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

class DataPlotter:
    """
    Standalone plotting engine for easy chart generation.
    Pass a dataframe `data` to any method to generate a Plotly chart.
    """

    @staticmethod
    def _data_validate(data, message_dict):
        """
        Universal data validator for all plotter methods.
        Accepts None, empty strings, DataFrames, lists of records, or JSON strings.
        Returns a dict with 'status' ('success'/'error'), and either 'data' (list of records)
        or an updated 'message_dict' with the error reason.
        """
        # 1. Handle empty/None input
        if data is None or (isinstance(data, str) and not data):
            message_dict.update({'message': 'No data'})
            return {'status': 'error', 'message_dict': message_dict}

        # 2. Check if data is a DataFrame
        if isinstance(data, pd.DataFrame):
            if data.empty:
                message_dict.update({'message': 'No data'})
                return {'status': 'error', 'message_dict': message_dict}
            return {'status': 'success', 'data': data.to_dict(orient='records')}

        # 3. Check if data is a list (records)
        if isinstance(data, list):
            if not data:
                message_dict.update({'message': 'No data'})
                return {'status': 'error', 'message_dict': message_dict}
            return {'status': 'success', 'data': data}

        # 4. Handle JSON string input
        try:
            parsed_data = json.loads(data)
            # Support both {'records': [...]} structure OR raw list of records
            records = parsed_data.get('records') if isinstance(parsed_data, dict) else parsed_data

            if not records:
                message_dict.update({'message': 'No data'})
                return {'status': 'error', 'message_dict': message_dict}

            return {'status': 'success', 'data': records}

        except (json.JSONDecodeError, TypeError):
            message_dict.update({'message': 'Invalid data format'})
            return {'status': 'error', 'message_dict': message_dict}

    @staticmethod
    def _handle_return(fig_or_traces, layout_kwargs, as_trace, as_html=False):
        if "template" not in layout_kwargs:
            layout_kwargs["template"] = "plotly_dark"
            
        if isinstance(fig_or_traces, go.Figure):
            fig = fig_or_traces
            fig.update_layout(**layout_kwargs)
            if as_trace:
                return list(fig.data)
            if as_html:
                return fig.to_html(full_html=False, include_plotlyjs='cdn')
            return fig
        else:
            traces = fig_or_traces
            if not isinstance(traces, list) and not isinstance(traces, tuple):
                traces = [traces]
            if as_trace:
                return traces if len(traces) > 1 else traces[0]
            fig = go.Figure(data=traces)
            fig.update_layout(**layout_kwargs)
            if as_html:
                return fig.to_html(full_html=False, include_plotlyjs='cdn')
            return fig

    @classmethod
    def bar(cls, data, x_col, y_col, color_col=None, title="Bar Chart", config=None, as_trace=False, as_html=False):
        config = config or {}
        c_col = color_col if color_col else (x_col if x_col in data.columns else None)
        fig = px.bar(data, x=x_col, y=y_col, color=c_col, **config)
        return cls._handle_return(fig, {"title": title}, as_trace, as_html)

    @classmethod
    def piechart(cls, data, names_col, values_col, color_col=None, title="Pie Chart", config=None, as_trace=False, as_html=False):
        config = config or {}
        c_col = color_col if color_col else names_col
        fig = px.pie(data, names=names_col, values=values_col, color=c_col, **config)
        return cls._handle_return(fig, {"title": title}, as_trace, as_html)

    @classmethod
    def histogram(cls, data, x_col, color_col=None, title="Histogram", config=None, as_trace=False, as_html=False):
        config = config or {}
        fig = px.histogram(data, x=x_col, color=color_col, **config)
        return cls._handle_return(fig, {"title": title}, as_trace, as_html)

    @classmethod
    def histogram_2d(cls, data, x_col, y_col, title="2D Histogram", config=None, as_trace=False, as_html=False):
        config = config or {}
        fig = px.density_heatmap(data, x=x_col, y=y_col, color_continuous_scale="Viridis", **config)
        return cls._handle_return(fig, {"title": title}, as_trace, as_html)

    @classmethod
    def scatter_3d(cls, data, x_col, y_col, z_col, color_col=None, title="3D Scatter Plot", config=None, as_trace=False, as_html=False):
        config = config or {}
        c_col = color_col if color_col else z_col
        fig = px.scatter_3d(data, x=x_col, y=y_col, z=z_col, color=c_col, **config)
        return cls._handle_return(fig, {"title": title}, as_trace, as_html)

    @classmethod
    def box_violin(cls, data, y_col, x_col=None, color_col=None, title="Violin & Box Plot", config=None, as_trace=False, as_html=False):
        config = config or {}
        c_col = color_col if color_col else x_col
        fig = px.violin(data, y=y_col, x=x_col, color=c_col, box=True, points="all", **config)
        return cls._handle_return(fig, {"title": title}, as_trace, as_html)

    @classmethod
    def scatter(cls, data, x_col, y_col, color_col=None, title="Scatter Plot", config=None, as_trace=False, as_html=False):
        config = config or {}
        fig = px.scatter(data, x=x_col, y=y_col, color=color_col, **config)
        return cls._handle_return(fig, {"title": title}, as_trace, as_html)

    @classmethod
    def bubble(cls, data, x_col, y_col, size_col, color_col=None, title="Bubble Chart", config=None, as_trace=False, as_html=False):
        config = config or {}
        c_col = color_col if color_col else size_col
        fig = px.scatter(data, x=x_col, y=y_col, size=size_col, color=c_col, **config)
        return cls._handle_return(fig, {"title": title}, as_trace, as_html)

    @classmethod
    def heatmap(cls, z_matrix, x_labels, y_labels, title="Heatmap", config=None, as_trace=False, as_html=False):
        config = config or {}
        trace = go.Heatmap(z=z_matrix, x=x_labels, y=y_labels, colorscale='Viridis', **config)
        return cls._handle_return(trace, {"title": title}, as_trace, as_html)

    @classmethod
    def distplot(cls, data, col_names, title="Distplot", config=None, as_trace=False, as_html=False):
        config = config or {}
        hist_data = [data[col].dropna() for col in col_names]
        fig = ff.create_distplot(hist_data, col_names, **config)
        return cls._handle_return(fig, {"title": title}, as_trace, as_html)

PlottingMethods = DataPlotter
