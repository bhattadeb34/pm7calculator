
"""
Environment-specific implementations for PM7Calculator.
"""

try:
    from .colab import ColabCalculator, calculate_pm7_properties_colab, calculate_pm7_batch_colab
    __all__ = ['ColabCalculator', 'calculate_pm7_properties_colab', 'calculate_pm7_batch_colab']
except ImportError:
    __all__ = []
