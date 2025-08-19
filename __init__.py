
"""
PM7Calculator: Comprehensive PM7 Quantum Chemistry Calculator

A powerful, user-friendly Python package for PM7 semi-empirical quantum chemistry 
calculations. Designed for researchers, educators, and students in computational 
chemistry, drug discovery, and materials science.

Author: bhattadeb34
Institution: The Pennsylvania State University
License: MIT
Repository: https://github.com/bhattadeb34/pm7calculator

Basic Usage:
    >>> from pm7calculator import PM7Calculator
    >>> calc = PM7Calculator()
    >>> props = calc.calculate("CCO")  # Ethanol
    >>> print(f"Heat of formation: {props['heat_of_formation']:.3f} kcal/mol")

Environment-specific usage:
    >>> from pm7calculator.environments import ColabCalculator
    >>> calc = ColabCalculator()  # Auto-installs dependencies in Colab
    >>> props = calc.calculate("CCO", cleanup=False)
    >>> calc.display_properties(props)

Batch processing:
    >>> smiles_list = ["CCO", "CC(=O)O", "CCN"]
    >>> results = calc.calculate_batch(smiles_list)
    >>> successful = [r for r in results if r['success']]
"""

# Core imports
from .core.calculator import PM7Calculator
from .core.parser import PM7Parser
from .core.utils import (
    smiles_to_3d,
    validate_smiles,
    format_properties,
    cleanup_temp_files,
    batch_calculate,
)
from .config.defaults import DEFAULT_CONFIG, PROPERTY_FORMATS

# Version and metadata
__version__ = "1.0.0"
__author__ = "bhattadeb34"
__email__ = "bhattadeb34@psu.edu"
__institution__ = "The Pennsylvania State University"
__license__ = "MIT"
__url__ = "https://github.com/bhattadeb34/pm7calculator"

# Main exports
__all__ = [
    "PM7Calculator",
    "PM7Parser",
    "smiles_to_3d",
    "validate_smiles", 
    "format_properties",
    "cleanup_temp_files",
    "batch_calculate",
    "DEFAULT_CONFIG",
    "PROPERTY_FORMATS",
    "__version__",
    "__author__",
    "__email__",
]

# Environment-specific imports with graceful fallback
try:
    from .environments.colab import ColabCalculator, calculate_pm7_properties_colab
    __all__.extend(["ColabCalculator", "calculate_pm7_properties_colab"])
except ImportError:
    pass

try:
    from .environments.local import LocalCalculator
    __all__.append("LocalCalculator")
except ImportError:
    pass

try:
    from .environments.cluster import ClusterCalculator  
    __all__.append("ClusterCalculator")
except ImportError:
    pass

# Optional visualization imports
try:
    from .visualization import PropertyPlotter, MoleculeVisualizer
    __all__.extend(["PropertyPlotter", "MoleculeVisualizer"])
except ImportError:
    pass

def get_version():
    """Get the current version of PM7Calculator."""
    return __version__

def get_info():
    """Get package information."""
    return {
        "name": "pm7calculator",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "institution": __institution__,
        "license": __license__,
        "url": __url__,
        "description": "Comprehensive PM7 quantum chemistry calculator for molecular property prediction",
    }

# Package-level configuration
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Welcome message for interactive usage
def _show_welcome():
    """Show welcome message in interactive environments."""
    try:
        import sys
        if hasattr(sys, 'ps1'):  # Interactive Python
            print(f"🧪 PM7Calculator v{__version__} loaded!")
            print(f"📚 Quick start: PM7Calculator().calculate('CCO')")
            print(f"📖 Documentation: {__url__}")
    except:
        pass

_show_welcome()
