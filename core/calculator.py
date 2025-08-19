
"""
Core PM7 Calculator class for quantum chemistry calculations.

Author: bhattadeb34
Institution: The Pennsylvania State University
"""

import os
import tempfile
import uuid
import subprocess
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from .parser import PM7Parser
from .utils import smiles_to_3d, validate_smiles
from ..config.defaults import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class PM7Calculator:
    """
    Core PM7 calculator for quantum chemistry calculations.
    
    This class provides comprehensive PM7 semi-empirical quantum chemistry
    calculations through MOPAC, with support for various molecular properties
    including thermodynamic, electronic, and structural characteristics.
    
    Calculated Properties:
        - Heat of formation (kcal/mol)
        - Dipole moment (Debye)
        - HOMO/LUMO energies (eV)
        - Ionization potential (eV)
        - COSMO surface area and volume
        - Molecular geometry and point group
        - Electronic properties
    
    Args:
        method: Semi-empirical method (default: "PM7")
        mopac_command: MOPAC executable command
        temp_dir: Directory for temporary calculation files
        keywords: Additional MOPAC calculation keywords
        parser_config: Custom parser configuration
    
    Example:
        >>> calc = PM7Calculator()
        >>> props = calc.calculate("CCO")  # Ethanol
        >>> print(f"ΔHf = {props['heat_of_formation']:.2f} kcal/mol")
        >>> print(f"μ = {props['dipole_moment']:.2f} Debye")
    """
    
    def __init__(
        self,
        method: str = "PM7",
        mopac_command: str = "mopac", 
        temp_dir: Optional[str] = None,
        keywords: Optional[str] = None,
        parser_config: Optional[Dict] = None,
        timeout: int = 300,
    ):
        self.method = method
        self.mopac_command = mopac_command
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.keywords = keywords or DEFAULT_CONFIG["mopac_keywords"]
        self.timeout = timeout
        self.parser = PM7Parser(config=parser_config)
        
        # Ensure temp directory exists
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate MOPAC availability
        self._mopac_available = self._check_mopac()
        
        logger.info(f"PM7Calculator initialized with method={method}")
    
    def _check_mopac(self) -> bool:
        """Check if MOPAC is available and responsive."""
        try:
            result = subprocess.run(
                [self.mopac_command], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            logger.info("✅ MOPAC is available and responsive")
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.warning(f"❌ MOPAC not found or not responding: {e}")
            return False
    
    def calculate(
        self, 
        smiles: str, 
        cleanup: bool = True,
        charge: int = 0,
        multiplicity: int = 1,
        custom_keywords: Optional[str] = None,
        label: Optional[str] = None,
    ) -> Dict:
        """
        Calculate PM7 properties for a molecule from SMILES.
        
        Args:
            smiles: SMILES string representation of the molecule
            cleanup: Whether to remove temporary calculation files
            charge: Molecular charge (default: 0 for neutral)
            multiplicity: Spin multiplicity (1=singlet, 2=doublet, etc.)
            custom_keywords: Additional MOPAC keywords for this calculation
            label: Custom label for this calculation (auto-generated if None)
            
        Returns:
            Dictionary containing:
                - success: Boolean indicating calculation success
                - smiles: Input SMILES string
                - Properties: heat_of_formation, dipole_moment, homo_ev, lumo_ev, etc.
                - Metadata: num_atoms, computation_time, temp_files, etc.
                - Error info: error message if calculation failed
        
        Example:
            >>> calc = PM7Calculator()
            >>> result = calc.calculate("c1ccccc1", charge=1, multiplicity=2)
            >>> if result['success']:
            ...     print(f"Benzene cation HOMO: {result['homo_ev']:.2f} eV")
        """
        # Input validation
        if not validate_smiles(smiles):
            return {
                'success': False, 
                'error': 'Invalid SMILES string', 
                'smiles': smiles
            }
        
        if not self._mopac_available:
            return {
                'success': False,
                'error': 'MOPAC not available',
                'smiles': smiles
            }
        
        # Generate unique label for this calculation
        calc_label = label or f"mol_{uuid.uuid4().hex[:8]}"
        
        try:
            logger.info(f"🧬 Processing SMILES: {smiles}")
            
            # Step 1: Generate 3D molecular structure
            atoms, coords = smiles_to_3d(smiles)
            if atoms is None or coords is None:
                return {
                    'success': False, 
                    'error': 'Failed to generate 3D molecular structure', 
                    'smiles': smiles
                }
            
            logger.info(f"   ✅ Generated 3D structure: {len(atoms)} atoms")
            
            # Step 2: Write MOPAC input file
            input_file = self._write_mopac_input(
                atoms, coords, calc_label, charge, multiplicity, custom_keywords
            )
            logger.info(f"   📝 Created MOPAC input: {os.path.basename(input_file)}")
            
            # Step 3: Execute MOPAC calculation
            success = self._run_mopac_calculation(input_file)
            if not success:
                return {
                    'success': False, 
                    'error': 'MOPAC calculation failed or timed out', 
                    'smiles': smiles,
                    'label': calc_label
                }
            
            logger.info(f"   ⚡ MOPAC calculation completed successfully")
            
            # Step 4: Parse calculation results
            output_file = os.path.join(self.temp_dir, f"{calc_label}.out")
            properties = self.parser.parse(output_file)
            
            if not properties:
                return {
                    'success': False, 
                    'error': 'Failed to parse calculation results', 
                    'smiles': smiles,
                    'label': calc_label
                }
            
            # Step 5: Compile final results
            result = self._compile_results(
                properties, smiles, atoms, calc_label, charge, multiplicity
            )
            
            # Step 6: Handle temporary files
            temp_files = self._get_temp_files(calc_label)
            result['temp_files'] = temp_files
            result['files_kept'] = not cleanup
            
            if cleanup:
                cleaned_files = self._cleanup_files(calc_label)
                result['cleaned_files'] = cleaned_files
                logger.info(f"   🗑️  Cleaned up {len(cleaned_files)} temporary files")
            else:
                logger.info(f"   📁 Keeping {len(temp_files)} temporary files")
            
            logger.info(f"   🎯 Calculation completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in PM7 calculation: {e}")
            return {
                'success': False, 
                'error': f"Calculation error: {str(e)}", 
                'smiles': smiles,
                'label': calc_label
            }
        
        finally:
            # Safety cleanup if requested
            if cleanup:
                try:
                    self._cleanup_files(calc_label)
                except Exception as e:
                    logger.warning(f"Cleanup warning: {e}")
    
    def calculate_batch(
        self, 
        smiles_list: List[str], 
        cleanup: bool = True,
        max_molecules: Optional[int] = None,
        progress_callback: Optional[callable] = None,
        **kwargs
    ) -> List[Dict]:
        """
        Calculate PM7 properties for multiple molecules efficiently.
        
        Args:
            smiles_list: List of SMILES strings
            cleanup: Whether to clean up temporary files after each calculation
            max_molecules: Maximum number of molecules to process
            progress_callback: Function called with (current, total) for progress tracking
            **kwargs: Additional arguments passed to calculate()
            
        Returns:
            List of result dictionaries from individual calculations
            
        Example:
            >>> smiles = ["CCO", "CC(=O)O", "CCN", "c1ccccc1"]
            >>> results = calc.calculate_batch(smiles)
            >>> successful = [r for r in results if r['success']]
            >>> print(f"Processed {len(successful)}/{len(smiles)} molecules")
        """
        if max_molecules:
            smiles_list = smiles_list[:max_molecules]
        
        results = []
        total = len(smiles_list)
        
        logger.info(f"🚀 Starting batch calculation: {total} molecules")
        
        for i, smiles in enumerate(smiles_list, 1):
            logger.info(f"📊 Processing molecule {i}/{total}: {smiles}")
            
            # Calculate properties for this molecule
            result = self.calculate(smiles, cleanup=cleanup, **kwargs)
            results.append(result)
            
            # Progress callback
            if progress_callback:
                progress_callback(i, total)
            
            # Log progress
            if result['success']:
                hof = result.get('heat_of_formation', 'N/A')
                dipole = result.get('dipole_moment', 'N/A')
                logger.info(f"   ✅ Success - ΔHf: {hof}, μ: {dipole}")
            else:
                logger.warning(f"   ❌ Failed: {result['error']}")
        
        # Summary statistics
        successful = sum(1 for r in results if r['success'])
        failed = total - successful
        
        logger.info(f"📈 Batch complete: {successful} successful, {failed} failed")
        
        return results
    
    def calculate_dataframe(self, df, smiles_column: str = 'smiles', **kwargs):
        """
        Calculate PM7 properties for molecules in a pandas DataFrame.
        
        Args:
            df: pandas DataFrame containing SMILES
            smiles_column: Name of column containing SMILES strings
            **kwargs: Additional arguments passed to calculate_batch()
            
        Returns:
            pandas DataFrame with added PM7 property columns
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for DataFrame processing")
        
        # Extract SMILES list
        smiles_list = df[smiles_column].tolist()
        
        logger.info(f"📊 Processing DataFrame: {len(df)} rows")
        
        # Calculate properties
        results = self.calculate_batch(smiles_list, **kwargs)
        
        # Convert to DataFrame and merge
        props_df = pd.DataFrame(results)
        result_df = pd.concat([df.reset_index(drop=True), props_df], axis=1)
        
        return result_df
    
    def _write_mopac_input(
        self, 
        atoms: List[str], 
        coordinates: List[List[float]], 
        label: str,
        charge: int = 0,
        multiplicity: int = 1,
        custom_keywords: Optional[str] = None
    ) -> str:
        """Write MOPAC input file with specified parameters."""
        input_file = os.path.join(self.temp_dir, f"{label}.mop")
        
        # Build keyword line
        keywords = self.keywords
        
        if custom_keywords:
            keywords = f"{keywords} {custom_keywords}"
        
        if charge != 0:
            keywords = f"{keywords} CHARGE={charge}"
            
        if multiplicity != 1:
            # UHF for open shell, MS for spin multiplicity
            keywords = f"{keywords} UHF MS={multiplicity-1}"
        
        # Write input file
        with open(input_file, 'w') as f:
            f.write(f"{self.method} {keywords}\\n")
            f.write(f"PM7 calculation for {label}\\n")
            f.write("\\n")
            
            # Atomic coordinates (1 = optimize this coordinate)
            for atom, coord in zip(atoms, coordinates):
                f.write(f"{atom:2s} {coord[0]:12.6f} 1 {coord[1]:12.6f} 1 {coord[2]:12.6f} 1\\n")
        
        return input_file
    
    def _run_mopac_calculation(self, input_file: str) -> bool:
        """Execute MOPAC calculation with timeout and error handling."""
        try:
            result = subprocess.run(
                [self.mopac_command, input_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode != 0:
                logger.error(f"MOPAC failed with return code {result.returncode}")
                if result.stderr:
                    logger.error(f"STDERR: {result.stderr}")
                return False
            
            return True
            
        except subprocess.TimeoutExpired:
            logger.error(f"MOPAC calculation timed out after {self.timeout} seconds")
            return False
        except Exception as e:
            logger.error(f"Error executing MOPAC: {e}")
            return False
    
    def _compile_results(
        self, 
        properties: Dict, 
        smiles: str, 
        atoms: List[str], 
        label: str,
        charge: int,
        multiplicity: int
    ) -> Dict:
        """Compile final results dictionary with metadata."""
        result = {
            'success': True,
            'smiles': smiles,
            'label': label,
            'num_atoms': len(atoms),
            'charge': charge,
            'multiplicity': multiplicity,
            'method': self.method,
        }
        
        # Add all calculated properties
        result.update(properties)
        
        return result
    
    def _get_temp_files(self, label: str) -> List[str]:
        """Get list of all temporary files for a calculation."""
        extensions = ['.mop', '.out', '.arc', '.aux', '.log', '.end', '.mgf']
        temp_files = []
        
        for ext in extensions:
            temp_file = os.path.join(self.temp_dir, f"{label}{ext}")
            if os.path.exists(temp_file):
                temp_files.append(temp_file)
        
        return temp_files
    
    def _cleanup_files(self, label: str) -> List[str]:
        """Remove temporary calculation files."""
        temp_files = self._get_temp_files(label)
        cleaned_files = []
        
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
                cleaned_files.append(temp_file)
            except Exception as e:
                logger.warning(f"Could not remove {temp_file}: {e}")
        
        return cleaned_files
    
    def get_version_info(self) -> Dict:
        """Get version information for PM7Calculator and dependencies."""
        info = {
            'pm7calculator': '1.0.0',
            'python': os.sys.version,
            'mopac_available': self._mopac_available,
            'temp_dir': self.temp_dir,
        }
        
        try:
            import rdkit
            info['rdkit'] = rdkit.__version__
        except ImportError:
            info['rdkit'] = 'Not available'
        
        try:
            import ase
            info['ase'] = ase.__version__
        except ImportError:
            info['ase'] = 'Not available'
            
        return info
