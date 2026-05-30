__version__ = "0.4.2"

from .core import log_version, DataInspector, DataPlotter

def get_version():
    """Returns the current version of the library."""
    log_version(__version__)
    return __version__
