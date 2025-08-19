
"""
Google Colab-optimized PM7Calculator implementation.

This module provides specialized functionality for Google Colab environments,
including automatic dependency installation and Jupyter-friendly output.

Author: bhattadeb34
Institution: The Pennsylvania State University
"""

import subprocess
import sys
import os
import logging
from typing import Optional, Dict, List

from ..core.calculator import PM7Calculator
from ..core.utils import format_properties
from ..config.defaults import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class ColabCalculator(PM7Calculator):
    """
    PM7Calculator optimized for Google Colab environment.
    
    This class provides Colab-specific features including:
    - Automatic dependency installation (MOPAC, RDKit, etc.)
    - Optimized temporary file handling
    - Enhanced progress display for Jupyter notebooks
    - Interactive property visualization
    
    Args:
        auto_install: Whether to automatically install dependencies
        method: Semi-empirical method (default: "PM7")
        show_progress: Whether to show calculation progress
        **kwargs: Additional arguments passed to PM7Calculator
    
    Example:
        >>> # In Google Colab
        >>> from pm7calculator.environments import ColabCalculator
        >>> calc = ColabCalculator()  # Auto-installs dependencies
        >>> props = calc.calculate("CCO", cleanup=False)
        >>> calc.display_properties(props)
    """
    
    def __init__(
        self,
        auto_install: bool = True,
        method: str = "PM7",
        show_progress: bool = True,
        **kwargs
    ):
        # Set Colab-specific defaults
        colab_config = DEFAULT_CONFIG["environments"]["colab"]
        
        kwargs.setdefault("temp_dir", colab_config["temp_dir"])
        kwargs.setdefault("mopac_command", "mopac")
        
        self.show_progress = show_progress
        
        # Install dependencies if requested
        if auto_install:
            self.install_dependencies()
        
        # Initialize parent calculator
        super().__init__(method=method, **kwargs)
        
        logger.info("🚀 ColabCalculator initialized for Google Colab")
    
    @staticmethod
    def install_dependencies():
        """
        Install required packages in Google Colab environment.
        
        This method handles the installation of MOPAC, RDKit, and other
        dependencies required for PM7 calculations in Colab.
        """
        print("🔧 Installing PM7Calculator dependencies in Google Colab...")
        print("This may take a few minutes on first run...")
        
        try:
            # Install condacolab for better package management
            print("📦 Installing condacolab...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-q", "condacolab"
            ], check=True)
            
            # Import and install conda
            print("🐍 Setting up conda environment...")
            import condacolab
            condacolab.install()
            
            # Install packages via conda-forge
            packages = DEFAULT_CONFIG["environments"]["colab"]["conda_packages"]
            for package in packages:
                print(f"📚 Installing {package}...")
                subprocess.run([
                    "conda", "install", "-c", "conda-forge", package, "-y"
                ], check=True)
            
            print("✅ Installation complete! PM7Calculator is ready to use.")
            print("🔄 You may need to restart the runtime after first installation.")
            
        except Exception as e:
            print(f"❌ Installation failed: {e}")
            print("💡 Try running this cell again or restart the runtime.")
            raise
    
    def calculate(self, smiles: str, cleanup: bool = True, **kwargs) -> Dict:
        """
        Calculate PM7 properties with Colab-specific enhancements.
        
        Args:
            smiles: SMILES string
            cleanup: Whether to remove temporary files
            **kwargs: Additional arguments passed to parent calculate()
            
        Returns:
            Dictionary containing calculation results and metadata
        """
        if self.show_progress:
            print(f"🧬 Processing in Colab: {smiles}")
        
        # Run calculation using parent method
        result = super().calculate(smiles, cleanup=cleanup, **kwargs)
        
        # Enhanced progress reporting for Colab
        if self.show_progress:
            if result.get('success'):
                print("   🎯 Calculation completed successfully")
                # Show key properties inline
                if 'heat_of_formation' in result:
                    print(f"   🔥 ΔHf: {result['heat_of_formation']:.3f} kcal/mol")
                if 'dipole_moment' in result:
                    print(f"   ⚡ μ: {result['dipole_moment']:.3f} Debye")
            else:
                print(f"   ❌ Calculation failed: {result.get('error')}")
        
        return result
    
    def calculate_batch(self, smiles_list: List[str], cleanup: bool = True, **kwargs) -> List[Dict]:
        """
        Batch calculation with Colab progress tracking.
        
        Args:
            smiles_list: List of SMILES strings
            cleanup: Whether to clean up temporary files
            **kwargs: Additional arguments
            
        Returns:
            List of calculation results
        """
        total = len(smiles_list)
        cleanup_msg = "with cleanup" if cleanup else "keeping files"
        
        if self.show_progress:
            print(f"🚀 Processing {total} molecules in Colab {cleanup_msg}...")
            
            # Try to use tqdm for progress bar if available
            try:
                from tqdm import tqdm
                progress_bar = tqdm(total=total, desc="Calculating")
                
                def progress_callback(current, total):
                    progress_bar.update(1)
                
                results = super().calculate_batch(
                    smiles_list, cleanup=cleanup, 
                    progress_callback=progress_callback, **kwargs
                )
                progress_bar.close()
                
            except ImportError:
                # Fallback to simple progress reporting
                results = super().calculate_batch(smiles_list, cleanup=cleanup, **kwargs)
        else:
            results = super().calculate_batch(smiles_list, cleanup=cleanup, **kwargs)
        
        # Summary for Colab
        successful = sum(1 for r in results if r.get('success', False))
        if self.show_progress:
            print(f"📊 Batch Summary: {successful}/{total} successful calculations")
            if not cleanup:
                total_files = sum(len(r.get('temp_files', [])) for r in results if r.get('success'))
                print(f"📁 Total files kept: {total_files}")
        
        return results
    
    def display_properties(self, props: Dict, style: str = "colab"):
        """
        Display properties with Colab-friendly formatting.
        
        Args:
            props: Properties dictionary from calculation
            style: Display style ("colab", "detailed", "compact")
        """
        formatted_output = format_properties(props, style=style)
        print(formatted_output)
        
        # Additional Colab-specific display features
        if props.get('success') and props.get('temp_files'):
            print(f"\n📁 Temporary files:")
            for file_path in props['temp_files']:
                file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                print(f"   - {os.path.basename(file_path)} ({file_size:,} bytes)")
    
    def create_summary_dataframe(self, results: List[Dict]):
        """
        Create a pandas DataFrame summary for Colab display.
        
        Args:
            results: List of calculation results
            
        Returns:
            pandas DataFrame with key properties
        """
        try:
            import pandas as pd
        except ImportError:
            print("❌ pandas not available for DataFrame creation")
            return None
        
        # Extract key properties for summary
        summary_data = []
        for result in results:
            if result.get('success'):
                row = {
                    'SMILES': result.get('smiles', ''),
                    'Heat_of_Formation': result.get('heat_of_formation'),
                    'Dipole_Moment': result.get('dipole_moment'),
                    'HOMO_eV': result.get('homo_ev'),
                    'LUMO_eV': result.get('lumo_ev'),
                    'Gap_eV': result.get('gap_ev'),
                    'Mol_Weight': result.get('molecular_weight'),
                    'Comp_Time': result.get('computation_time'),
                }
                summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        print(f"📊 Summary DataFrame created with {len(df)} successful calculations")
        return df


# Convenience functions for backward compatibility
def calculate_pm7_properties_colab(smiles: str, method: str = "PM7", cleanup: bool = True):
    """
    One-line function for Colab PM7 calculations.
    
    Args:
        smiles: SMILES string
        method: Semi-empirical method
        cleanup: Whether to remove temporary files
        
    Returns:
        Properties dictionary
        
    Example:
        >>> props = calculate_pm7_properties_colab("CCO")
        >>> print(f"Heat of formation: {props['heat_of_formation']:.3f} kcal/mol")
    """
    calc = ColabCalculator(method=method)
    return calc.calculate(smiles, cleanup=cleanup)


def calculate_pm7_batch_colab(
    smiles_list: List[str], 
    method: str = "PM7", 
    cleanup: bool = True, 
    max_molecules: Optional[int] = None,
    **kwargs
):
    """
    Batch calculation function for Colab.
    
    Args:
        smiles_list: List of SMILES strings
        method: Semi-empirical method
        cleanup: Whether to remove temporary files
        max_molecules: Maximum number to process
        **kwargs: Additional arguments
        
    Returns:
        List of calculation results
        
    Example:
        >>> results = calculate_pm7_batch_colab(['CCO', 'CC(=O)O'], cleanup=False)
        >>> successful = [r for r in results if r['success']]
    """
    calc = ColabCalculator(method=method)
    
    if max_molecules:
        smiles_list = smiles_list[:max_molecules]
    
    return calc.calculate_batch(smiles_list, cleanup=cleanup, **kwargs)


def calculate_pm7_dataframe_colab(df, smiles_column: str = 'smiles', method: str = "PM7", cleanup: bool = True):
    """
    Calculate PM7 properties for SMILES in a pandas DataFrame (Colab version).
    
    Args:
        df: pandas DataFrame
        smiles_column: Name of column containing SMILES
        method: Semi-empirical method
        cleanup: Whether to remove temporary files
        
    Returns:
        pandas DataFrame with added PM7 properties
        
    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'smiles': ['CCO', 'CC(=O)O']})
        >>> result_df = calculate_pm7_dataframe_colab(df, cleanup=False)
    """
    calc = ColabCalculator(method=method)
    return calc.calculate_dataframe(df, smiles_column=smiles_column, cleanup=cleanup)
