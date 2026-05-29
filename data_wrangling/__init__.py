__version__ = "0.1.0"

from .core import log_version, DataInspector, PlottingMethods

def get_version():
    """Returns the current version of the library."""
    log_version(__version__)
    return __version__
