
"""
Basic test suite for PM7Calculator.

Author: bhattadeb34
Institution: The Pennsylvania State University
"""

import pytest
import tempfile
import os
from pm7calculator import PM7Calculator
from pm7calculator.core.utils import validate_smiles, smiles_to_3d


class TestPM7Calculator:
    """Test cases for PM7Calculator core functionality."""
    
    @pytest.fixture
    def calculator(self):
        """Create a test calculator instance."""
        return PM7Calculator(temp_dir=tempfile.mkdtemp())
    
    def test_calculator_initialization(self, calculator):
        """Test calculator initialization."""
        assert calculator.method == "PM7"
        assert os.path.exists(calculator.temp_dir)
    
    def test_smiles_validation(self):
        """Test SMILES validation function."""
        # Valid SMILES
        assert validate_smiles("CCO") == True
        assert validate_smiles("c1ccccc1") == True
        assert validate_smiles("CC(=O)O") == True
        
        # Invalid SMILES
        assert validate_smiles("") == False
        assert validate_smiles("XYZ123") == False
    
    def test_3d_structure_generation(self):
        """Test 3D structure generation."""
        atoms, coords = smiles_to_3d("CCO")
        
        if atoms is not None:  # Only test if RDKit is available
            assert len(atoms) == 9  # C-C-O + 6 hydrogens
            assert coords.shape == (9, 3)
            assert "C" in atoms
            assert "O" in atoms
            assert "H" in atoms
    
    @pytest.mark.integration
    def test_simple_calculation(self, calculator):
        """Test a simple PM7 calculation."""
        # Skip if MOPAC not available
        if not calculator._mopac_available:
            pytest.skip("MOPAC not available")
        
        result = calculator.calculate("CCO")  # Ethanol
        
        # Should succeed or fail gracefully
        assert 'success' in result
        assert 'smiles' in result
        assert result['smiles'] == "CCO"
        
        if result['success']:
            assert 'heat_of_formation' in result
            assert isinstance(result['heat_of_formation'], (int, float))
    
    def test_batch_calculation(self, calculator):
        """Test batch calculation functionality."""
        smiles_list = ["CCO", "CC(=O)O"]
        results = calculator.calculate_batch(smiles_list)
        
        assert len(results) == 2
        assert all('success' in r for r in results)
        assert all('smiles' in r for r in results)
    
    def test_error_handling(self, calculator):
        """Test error handling for invalid inputs."""
        # Invalid SMILES should return failure
        result = calculator.calculate("INVALID_SMILES")
        assert result['success'] == False
        assert 'error' in result


if __name__ == "__main__":
    pytest.main([__file__])
