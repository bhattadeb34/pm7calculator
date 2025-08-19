
"""
Test suite for PM7Calculator package.

This module contains comprehensive tests for all PM7Calculator functionality
including core calculations, environment-specific features, and utilities.

Author: bhattadeb34
Institution: The Pennsylvania State University

Usage:
    Run all tests:
        pytest tests/
    
    Run specific test file:
        pytest tests/test_calculator.py
    
    Run with coverage:
        pytest --cov=pm7calculator tests/
"""

import pytest
import logging

# Configure test logging
logging.basicConfig(level=logging.WARNING)

# Test configuration
TEST_CONFIG = {
    'timeout': 60,  # seconds
    'test_molecules': [
        'CCO',        # Ethanol - simple alcohol
        'CC(=O)O',    # Acetic acid - simple carboxylic acid
        'c1ccccc1',   # Benzene - aromatic
        'CCN',        # Ethylamine - simple amine
    ],
    'skip_mopac_tests': False,  # Set to True if MOPAC not available
}

def get_test_config():
    """Get test configuration dictionary."""
    return TEST_CONFIG.copy()

# Test markers
pytest_plugins = []

# Custom fixtures available to all tests
@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration to all test functions."""
    return get_test_config()

@pytest.fixture(scope="session") 
def test_molecules():
    """Provide standard test molecules."""
    return TEST_CONFIG['test_molecules']

# Test utilities
def skip_if_no_mopac():
    """Skip test if MOPAC is not available."""
    try:
        import subprocess
        result = subprocess.run(['mopac'], capture_output=True, timeout=5)
        return False  # MOPAC is available
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return True   # MOPAC not available

def skip_if_no_rdkit():
    """Skip test if RDKit is not available.""" 
    try:
        import rdkit
        return False  # RDKit is available
    except ImportError:
        return True   # RDKit not available

# Test data
SAMPLE_SMILES = [
    "CCO",                                    # Ethanol
    "CC(=O)O",                               # Acetic acid
    "c1ccccc1",                              # Benzene
    "CCN",                                   # Ethylamine
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",        # Caffeine
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",       # Ibuprofen
]

INVALID_SMILES = [
    "",           # Empty string
    "XYZ123",     # Invalid characters
    "C[C@H](",    # Unmatched parenthesis
    "C=C=C=C",    # Invalid bonding
]

# Expected property ranges for validation
EXPECTED_RANGES = {
    'heat_of_formation': (-200, 200),     # kcal/mol
    'dipole_moment': (0, 10),             # Debye
    'homo_ev': (-15, -5),                 # eV
    'lumo_ev': (-5, 5),                   # eV
    'gap_ev': (0, 15),                    # eV
    'molecular_weight': (10, 1000),       # g/mol
    'computation_time': (0, 300),         # seconds
}

def validate_calculation_result(result):
    """
    Validate that a calculation result has expected structure and values.
    
    Args:
        result: Calculation result dictionary
        
    Returns:
        bool: True if result is valid
    """
    # Must have success flag
    if 'success' not in result:
        return False
    
    # Must have SMILES
    if 'smiles' not in result:
        return False
    
    if result['success']:
        # Successful calculations should have key properties
        required_props = ['heat_of_formation', 'num_atoms', 'method']
        
        for prop in required_props:
            if prop not in result:
                return False
        
        # Validate property ranges
        for prop, (min_val, max_val) in EXPECTED_RANGES.items():
            if prop in result:
                value = result[prop]
                if not isinstance(value, (int, float)):
                    continue
                if not (min_val <= value <= max_val):
                    return False
    
    else:
        # Failed calculations should have error message
        if 'error' not in result:
            return False
    
    return True

__all__ = [
    'TEST_CONFIG',
    'get_test_config', 
    'skip_if_no_mopac',
    'skip_if_no_rdkit',
    'SAMPLE_SMILES',
    'INVALID_SMILES',
    'EXPECTED_RANGES',
    'validate_calculation_result',
]
