import logging
from .data_state_mixin import DataStateMixin
from .data_inspector_mixin import DataInspectorMixin
from .data_plotter import DataPlotter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_version(version: str):
    """Logs the current version of the library."""
    logger.info(f"data_wrangling library version: {version}")
    print(f"data_wrangling library version: {version}")

class DataInspector(DataStateMixin, DataInspectorMixin):
    """
    Main DataInspector class that brings together state and cleaning capabilities.
    Inherits from DataStateMixin and DataInspectorMixin.
    """
    def __init__(self):
        super().__init__()
