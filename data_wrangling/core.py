import logging
from .data_state_mixin import DataStateMixin
from .data_inspector_mixin import DataInspectorMixin
from .data_plotter_mixin import DataPlotterMixin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_version(version: str):
    """Logs the current version of the library."""
    logger.info(f"data_wrangling library version: {version}")
    print(f"data_wrangling library version: {version}")

class PlottingMethods:
    """Class to handle granular chart generation independently."""
    pass

class DataInspector(DataStateMixin, DataInspectorMixin, DataPlotterMixin):
    """
    Main DataInspector class that brings together state, cleaning, and plotting capabilities.
    Inherits from DataStateMixin, DataInspectorMixin, and DataPlotterMixin.
    """
    def __init__(self):
        super().__init__()
        self.df = None
