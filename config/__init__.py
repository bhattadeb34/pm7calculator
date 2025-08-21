
"""
Configuration module for PM7Calculator.

Contains default settings, property formatting, and environment-specific
configurations for PM7 calculations.
"""

from .defaults import DEFAULT_CONFIG, PROPERTY_FORMATS


__all__ = [
    "DEFAULT_CONFIG",
    "PROPERTY_FORMATS", 
    "validate_config",
    "ConfigValidator",
]

