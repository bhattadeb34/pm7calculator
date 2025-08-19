
"""
Environment-specific implementations for PM7Calculator.

This module provides optimized calculator classes for different computing
environments including Google Colab, local machines, and computing clusters.
"""

# Import environment-specific calculators with graceful fallback
__all__ = []

try:
    from .colab import ColabCalculator, calculate_pm7_properties_colab, calculate_pm7_batch_colab
    __all__.extend(['ColabCalculator', 'calculate_pm7_properties_colab', 'calculate_pm7_batch_colab'])
except ImportError:
    pass

try:
    from .local import LocalCalculator
    __all__.append('LocalCalculator')
except ImportError:
    pass

try:
    from .cluster import ClusterCalculator
    __all__.append('ClusterCalculator')
except ImportError:
    pass
