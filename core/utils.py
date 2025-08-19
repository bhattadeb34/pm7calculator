
"""
Utility functions for PM7Calculator package.

This module provides helper functions for molecular structure generation,
validation, property formatting, and file management.

Author: bhattadeb34
Institution: The Pennsylvania State University
"""

import os
import re
import tempfile
import logging
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Import with fallback handling
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')  # Suppress RDKit warnings
    RDKIT_AVAILABLE = True
except ImportError:
    logger.warning("RDKit not available - 3D structure generation will be limited")
    RDKIT_AVAILABLE = False


def validate_smiles(smiles: str) -> bool:
    """
    Validate SMILES string using RDKit.
    
    Args:
        smiles: SMILES string to validate
        
    Returns:
        True if valid SMILES, False otherwise
        
    Example:
        >>> validate_smiles("CCO")  # ethanol
        True
        >>> validate_smiles("C[C@H](O)C")  # valid chiral SMILES
        True
        >>> validate_smiles("XYZ123")  # invalid
        False
    """
    if not RDKIT_AVAILABLE:
        # Basic validation without RDKit
        if not isinstance(smiles, str) or len(smiles) == 0:
            return False
        # Check for obviously invalid characters (FIXED: removed syntax error)
        invalid_chars = ['$', '%', '^', '&', '*']  # Removed '@', '#' as they are valid in SMILES
        if any(char in smiles for char in invalid_chars):
            return False
        return True
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


def smiles_to_3d(
    smiles: str, 
    random_seed: int = 42,
    max_attempts: int = 5,
    optimize_ff: bool = True
) -> Tuple[Optional[List[str]], Optional[np.ndarray]]:
    """
    Convert SMILES string to 3D molecular coordinates.
    
    This function uses RDKit to generate 3D coordinates from SMILES,
    with force field optimization for better starting geometries.
    
    Args:
        smiles: SMILES string representation
        random_seed: Random seed for reproducible conformer generation
        max_attempts: Maximum attempts for conformer generation
        optimize_ff: Whether to optimize with force field before PM7
        
    Returns:
        Tuple of (atom_symbols, coordinates_array) or (None, None) if failed
        
    Example:
        >>> atoms, coords = smiles_to_3d("CCO")
        >>> print(f"Generated structure with {len(atoms)} atoms")
        >>> print(f"Coordinates shape: {coords.shape}")
    """
    if not RDKIT_AVAILABLE:
        logger.error("RDKit required for SMILES to 3D conversion")
        return None, None
    
    if not validate_smiles(smiles):
        logger.error(f"Invalid SMILES string: {smiles}")
        return None, None
    
    try:
        # Parse SMILES and add hydrogens
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        
        logger.debug(f"Processing molecule: {Chem.MolToSmiles(mol)}")
        
        # Configure 3D embedding parameters
        params = AllChem.ETKDGv3()
        params.randomSeed = random_seed
        params.maxAttempts = max_attempts
        params.useSmallRingTorsions = True
        params.useMacrocycleTorsions = True
        params.useRandomCoords = True
        
        # Generate 3D coordinates
        for attempt in range(max_attempts):
            params.randomSeed = random_seed + attempt
            embed_status = AllChem.EmbedMolecule(mol, params)
            
            if embed_status == 0:  # Success
                logger.debug(f"3D embedding successful on attempt {attempt + 1}")
                break
        else:
            logger.error(f"Failed to generate 3D coordinates after {max_attempts} attempts")
            return None, None
        
        # Force field optimization (optional but recommended)
        if optimize_ff:
            try:
                # Try MMFF94 first
                ff_result = AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
                if ff_result == 0:
                    logger.debug("MMFF94 optimization successful")
                else:
                    # Fallback to UFF
                    ff_result = AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
                    if ff_result == 0:
                        logger.debug("UFF optimization successful")
                    else:
                        logger.warning("Force field optimization failed, using raw coordinates")
            except Exception as e:
                logger.warning(f"Force field optimization failed: {e}")
        
        # Extract atomic symbols and coordinates
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        conf = mol.GetConformer()
        coordinates = np.array([
            [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
            for i in range(mol.GetNumAtoms())
        ])
        
        logger.debug(f"Generated 3D structure: {len(atoms)} atoms")
        return atoms, coordinates
        
    except Exception as e:
        logger.error(f"Error generating 3D structure for {smiles}: {e}")
        return None, None


def format_properties(
    properties: Dict[str, Any], 
    style: str = "standard",
    precision: int = 3
) -> str:
    """
    Format molecular properties for display.
    
    Args:
        properties: Dictionary of calculated properties
        style: Display style ('standard', 'colab', 'compact', 'detailed')
        precision: Number of decimal places for floating point values
        
    Returns:
        Formatted string representation of properties
        
    Example:
        >>> props = {'heat_of_formation': -57.859, 'dipole_moment': 2.057}
        >>> print(format_properties(props, style='colab'))
    """
    if not properties.get('success', False):
        return f"❌ Calculation failed: {properties.get('error', 'Unknown error')}"
    
    # Property formatting configuration
    property_config = {
        'heat_of_formation': {'unit': 'kcal/mol', 'emoji': '🔥', 'name': 'Heat of Formation'},
        'dipole_moment': {'unit': 'Debye', 'emoji': '⚡', 'name': 'Dipole Moment'},
        'homo_ev': {'unit': 'eV', 'emoji': '🔋', 'name': 'HOMO Energy'},
        'lumo_ev': {'unit': 'eV', 'emoji': '🔋', 'name': 'LUMO Energy'},
        'gap_ev': {'unit': 'eV', 'emoji': '⚡', 'name': 'HOMO-LUMO Gap'},
        'ionization_potential': {'unit': 'eV', 'emoji': '⚡', 'name': 'Ionization Potential'},
        'molecular_weight': {'unit': 'g/mol', 'emoji': '⚖️', 'name': 'Molecular Weight'},
        'cosmo_area': {'unit': 'Ų', 'emoji': '📐', 'name': 'COSMO Area'},
        'cosmo_volume': {'unit': 'ų', 'emoji': '📦', 'name': 'COSMO Volume'},
        'computation_time': {'unit': 'seconds', 'emoji': '⏱️', 'name': 'Computation Time'},
    }
    
    smiles = properties.get('smiles', 'Unknown')
    
    if style == 'compact':
        # Single line format
        key_props = ['heat_of_formation', 'dipole_moment', 'gap_ev']
        values = []
        for prop in key_props:
            if prop in properties:
                val = properties[prop]
                if isinstance(val, (int, float)):
                    values.append(f"{val:.{precision}f}")
                else:
                    values.append(str(val))
            else:
                values.append("N/A")
        return f"{smiles}: ΔHf={values[0]} kcal/mol, μ={values[1]} D, Gap={values[2]} eV"
    
    elif style == 'colab':
        # Enhanced format for Jupyter/Colab (FIXED: removed extra backslash)
        lines = [f"✅ PM7 Properties for {smiles}:"]
        lines.append("=" * 60)
        
        for prop_key, config in property_config.items():
            if prop_key in properties:
                value = properties[prop_key]
                if isinstance(value, (int, float)):
                    formatted_val = f"{value:.{precision}f}"
                    lines.append(f"{config['emoji']} {config['name']}: {formatted_val} {config['unit']}")
        
        # Add dipole components if available
        if all(k in properties for k in ['dipole_x', 'dipole_y', 'dipole_z']):
            dx, dy, dz = properties['dipole_x'], properties['dipole_y'], properties['dipole_z']
            lines.append(f"   Components: X={dx:.{precision}f}, Y={dy:.{precision}f}, Z={dz:.{precision}f}")
        
        # Add computational info
        if 'num_atoms' in properties:
            lines.append(f"🧮 Number of Atoms: {properties['num_atoms']}")
        
        return "\\n".join(lines)
    
    elif style == 'detailed':
        # Comprehensive format with all properties (FIXED: removed extra backslashes)
        lines = [f"PM7 Calculation Results for {smiles}"]
        lines.append("=" * 80)
        
        # Group properties by category
        categories = {
            'Thermodynamic Properties': ['heat_of_formation', 'total_energy_kcal_mol', 'total_energy_ev'],
            'Electronic Properties': ['homo_ev', 'lumo_ev', 'gap_ev', 'ionization_potential'],
            'Structural Properties': ['dipole_moment', 'molecular_weight', 'point_group'],
            'Surface Properties': ['cosmo_area', 'cosmo_volume'],
            'Computational Info': ['computation_time', 'scf_cycles', 'optimization_cycles']
        }
        
        for category, props in categories.items():
            category_props = [p for p in props if p in properties]
            if category_props:
                lines.append(f"\\n{category}:")
                lines.append("-" * len(category))
                for prop in category_props:
                    value = properties[prop]
                    config = property_config.get(prop, {'unit': '', 'name': prop.replace('_', ' ').title()})
                    if isinstance(value, (int, float)):
                        lines.append(f"  {config['name']}: {value:.{precision}f} {config['unit']}")
                    else:
                        lines.append(f"  {config['name']}: {value}")
        
        return "\\n".join(lines)
    
    else:  # standard
        # Simple format
        lines = [f"PM7 Properties for {smiles}:"]
        for prop_key, config in property_config.items():
            if prop_key in properties:
                value = properties[prop_key]
                if isinstance(value, (int, float)):
                    lines.append(f"  {config['name']}: {value:.{precision}f} {config['unit']}")
        
        return "\\n".join(lines)


def cleanup_temp_files(pattern: str = "mol_*", temp_dir: Optional[str] = None) -> List[str]:
    """
    Clean up temporary calculation files matching a pattern.
    
    Args:
        pattern: File pattern to match (e.g., "mol_*", "calc_*")
        temp_dir: Directory to clean (default: system temp directory)
        
    Returns:
        List of cleaned file paths
        
    Example:
        >>> cleaned = cleanup_temp_files("mol_*")
        >>> print(f"Cleaned {len(cleaned)} files")
    """
    import glob
    
    temp_dir = temp_dir or tempfile.gettempdir()
    extensions = ['.mop', '.out', '.arc', '.aux', '.log', '.end', '.mgf']
    
    cleaned_files = []
    
    for ext in extensions:
        pattern_path = os.path.join(temp_dir, f"{pattern}{ext}")
        matching_files = glob.glob(pattern_path)
        
        for file_path in matching_files:
            try:
                os.remove(file_path)
                cleaned_files.append(file_path)
                logger.debug(f"Removed: {os.path.basename(file_path)}")
            except Exception as e:
                logger.warning(f"Could not remove {file_path}: {e}")
    
    if cleaned_files:
        logger.info(f"🧹 Cleaned up {len(cleaned_files)} temporary files")
    else:
        logger.info("No temporary files found to clean")
    
    return cleaned_files


def batch_calculate(
    calculator, 
    smiles_list: List[str],
    **kwargs
) -> List[Dict]:
    """
    Convenience function for batch calculations.
    
    Args:
        calculator: PM7Calculator instance
        smiles_list: List of SMILES strings
        **kwargs: Additional arguments passed to calculate()
        
    Returns:
        List of calculation results
    """
    return calculator.calculate_batch(smiles_list, **kwargs)


def export_results(
    results: List[Dict], 
    output_file: str,
    format: str = "csv"
) -> bool:
    """
    Export calculation results to file.
    
    Args:
        results: List of calculation result dictionaries
        output_file: Output file path
        format: Export format ('csv', 'json', 'xlsx')
        
    Returns:
        True if export successful, False otherwise
        
    Example:
        >>> results = calc.calculate_batch(['CCO', 'CC(=O)O'])
        >>> export_results(results, 'results.csv', format='csv')
    """
    try:
        if format.lower() == 'csv':
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False)
            
        elif format.lower() == 'json':
            import json
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
                
        elif format.lower() == 'xlsx':
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_excel(output_file, index=False)
            
        else:
            logger.error(f"Unsupported export format: {format}")
            return False
        
        logger.info(f"Results exported to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting results: {e}")
        return False


def get_molecule_info(smiles: str) -> Dict[str, Any]:
    """
    Get basic molecular information from SMILES.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Dictionary with molecular formula, weight, etc.
    """
    if not RDKIT_AVAILABLE:
        return {'error': 'RDKit not available'}
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {'error': 'Invalid SMILES'}
        
        mol = Chem.AddHs(mol)
        
        info = {
            'smiles': smiles,
            'molecular_formula': Chem.rdMolDescriptors.CalcMolFormula(mol),
            'molecular_weight': Chem.rdMolDescriptors.CalcExactMolWt(mol),
            'num_atoms': mol.GetNumAtoms(),
            'num_heavy_atoms': mol.GetNumHeavyAtoms(),
            'num_bonds': mol.GetNumBonds(),
            'canonical_smiles': Chem.MolToSmiles(mol),
        }
        
        return info
        
    except Exception as e:
        return {'error': str(e)}
