"""
PM7Calculator: PM7 Quantum Chemistry Calculator for Google Colab

Author: bhattadeb34
Institution: The Pennsylvania State University
License: MIT
Repository: https://github.com/bhattadeb34/pm7calculator
"""

__version__ = "1.0.1"
__author__ = "bhattadeb34"
__email__ = "bhattadeb34@psu.edu"
__institution__ = "The Pennsylvania State University"
__license__ = "MIT"
__url__ = "https://github.com/bhattadeb34/pm7calculator"

# Import the working calculator
try:
    from .environments.colab import (
        ColabCalculator as PM7Calculator,
        calculate_pm7_properties_colab,
        calculate_pm7_batch_colab,
    )
    CALCULATOR_AVAILABLE = True
except ImportError:
    class PM7Calculator:
        def __init__(self, method: str = "PM7"):
            self.method = method

        def calculate_properties(self, smiles: str, cleanup: bool = True):
            return {
                "success": False,
                "error": "Dependencies not installed",
                "smiles": smiles,
            }

    def calculate_pm7_properties_colab(smiles: str, **kwargs):
        return {
            "success": False,
            "error": "Dependencies not installed",
            "smiles": smiles,
        }

    def calculate_pm7_batch_colab(smiles_list, **kwargs):
        return [
            {
                "success": False,
                "error": "Dependencies not installed",
                "smiles": s,
            }
            for s in smiles_list
        ]

    CALCULATOR_AVAILABLE = False

__all__ = [
    "PM7Calculator",
    "calculate_pm7_properties_colab",
    "calculate_pm7_batch_colab",
    "__version__",
    "__author__",
    "__email__",
]


def get_version() -> str:
    return __version__


def get_info() -> dict:
    return {
        "name": "pm7calculator",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "institution": __institution__,
        "license": __license__,
        "url": __url__,
        "description": "PM7 quantum chemistry calculator for Google Colab",
    }
