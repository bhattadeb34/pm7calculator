from .core import (
    calculate_pm7_properties_colab,
    calculate_pm7_batch_colab,
    calculate_pm7_dataframe_colab,
    display_properties_enhanced,
    ColabPM7Calculator
)

# Import utilities
from .utils import install_colab_dependencies, check_colab_environment

__version__ = "0.1.0"
__author__ = "bhattadeb34"

__all__ = [
    'calculate_pm7_properties_colab',
    'calculate_pm7_batch_colab',
    'calculate_pm7_dataframe_colab', 
    'display_properties_enhanced',
    'ColabPM7Calculator',
    'install_colab_dependencies',
    'check_colab_environment'
]


import sys

def _startup_message():
    """Show helpful startup message"""
    if check_colab_environment():
        print("PM7Calculator loaded for Google Colab")
        print("Run install_colab_dependencies() first if MOPAC not installed")
    else:
        print("PM7Calculator loaded")
        print("This package is optimized for Google Colab")

# Show message on import (but don't be annoying)
try:
    _startup_message()
except:
    pass  # Don't fail if something goes wrong with the message
