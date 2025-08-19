
"""
Local machine PM7Calculator implementation.

Optimized for local development environments with enhanced file management
and dependency checking.

Author: bhattadeb34
Institution: The Pennsylvania State University
"""

import os
import shutil
import multiprocessing
from pathlib import Path
from typing import Optional, Dict, List
import logging

from ..core.calculator import PM7Calculator
from ..config.defaults import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class LocalCalculator(PM7Calculator):
    """
    PM7Calculator optimized for local machine environments.
    
    Features include:
    - Automatic dependency checking
    - Enhanced file organization
    - Optional parallel processing
    - Detailed logging and debugging
    
    Args:
        check_dependencies: Whether to verify all dependencies on init
        parallel: Whether to enable parallel batch processing
        max_workers: Maximum number of parallel workers
        **kwargs: Additional arguments passed to PM7Calculator
        
    Example:
        >>> from pm7calculator.environments import LocalCalculator
        >>> calc = LocalCalculator(parallel=True, max_workers=4)
        >>> results = calc.calculate_batch(smiles_list, cleanup=False)
    """
    
    def __init__(
        self,
        check_dependencies: bool = True,
        parallel: bool = False,
        max_workers: Optional[int] = None,
        **kwargs
    ):
        self.parallel = parallel
        self.max_workers = max_workers or multiprocessing.cpu_count()
        
        # Set local-specific defaults
        local_config = DEFAULT_CONFIG["environments"]["local"]
        kwargs.setdefault("temp_dir", local_config["temp_dir"])
        
        # Initialize parent calculator
        super().__init__(**kwargs)
        
        # Check dependencies if requested
        if check_dependencies:
            self.check_all_dependencies()
        
        logger.info(f"LocalCalculator initialized (parallel={parallel}, workers={self.max_workers})")
    
    def check_all_dependencies(self):
        """Check availability of all required dependencies."""
        print("🔍 Checking PM7Calculator dependencies...")
        
        dependencies = {
            'MOPAC': self._mopac_available,
            'RDKit': self._check_rdkit(),
            'NumPy': self._check_numpy(),
            'Pandas': self._check_pandas(),
        }
        
        missing = []
        for dep, available in dependencies.items():
            if available:
                print(f"   ✅ {dep}: Available")
            else:
                print(f"   ❌ {dep}: Missing")
                missing.append(dep)
        
        if missing:
            print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
            print("💡 Install with: pip install pm7calculator[all]")
        else:
            print("✅ All dependencies satisfied!")
    
    def _check_rdkit(self) -> bool:
        """Check RDKit availability."""
        try:
            import rdkit
            return True
        except ImportError:
            return False
    
    def _check_numpy(self) -> bool:
        """Check NumPy availability."""
        try:
            import numpy
            return True
        except ImportError:
            return False
    
    def _check_pandas(self) -> bool:
        """Check Pandas availability."""
        try:
            import pandas
            return True
        except ImportError:
            return False
    
    def calculate_batch(
        self, 
        smiles_list: List[str], 
        cleanup: bool = True,
        **kwargs
    ) -> List[Dict]:
        """
        Batch calculation with optional parallel processing.
        
        Args:
            smiles_list: List of SMILES strings
            cleanup: Whether to clean up temporary files
            **kwargs: Additional arguments
            
        Returns:
            List of calculation results
        """
        if self.parallel and len(smiles_list) > 1:
            return self._calculate_batch_parallel(smiles_list, cleanup, **kwargs)
        else:
            return super().calculate_batch(smiles_list, cleanup=cleanup, **kwargs)
    
    def _calculate_batch_parallel(
        self, 
        smiles_list: List[str], 
        cleanup: bool = True,
        **kwargs
    ) -> List[Dict]:
        """Parallel batch processing for local machines."""
        import multiprocessing as mp
        from functools import partial
        
        print(f"🚀 Starting parallel batch calculation with {self.max_workers} workers...")
        
        # Create partial function with fixed arguments
        calc_func = partial(self._single_calculation_worker, cleanup=cleanup, **kwargs)
        
        # Use multiprocessing pool
        with mp.Pool(processes=self.max_workers) as pool:
            results = pool.map(calc_func, smiles_list)
        
        successful = sum(1 for r in results if r.get('success', False))
        print(f"📊 Parallel batch complete: {successful}/{len(results)} successful")
        
        return results
    
    def _single_calculation_worker(self, smiles: str, cleanup: bool = True, **kwargs) -> Dict:
        """Worker function for parallel calculations."""
        try:
            # Create a new calculator instance for this worker
            worker_calc = PM7Calculator(
                method=self.method,
                mopac_command=self.mopac_command,
                temp_dir=self.temp_dir,
                keywords=self.keywords,
                timeout=self.timeout
            )
            return worker_calc.calculate(smiles, cleanup=cleanup, **kwargs)
        except Exception as e:
            return {'success': False, 'error': str(e), 'smiles': smiles}
    
    def organize_results(self, results: List[Dict], output_dir: str = "pm7_results"):
        """
        Organize calculation results into a structured directory.
        
        Args:
            results: List of calculation results
            output_dir: Output directory for organized results
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create subdirectories
        (output_path / "successful").mkdir(exist_ok=True)
        (output_path / "failed").mkdir(exist_ok=True)
        (output_path / "temp_files").mkdir(exist_ok=True)
        
        successful_count = 0
        failed_count = 0
        
        for i, result in enumerate(results):
            if result.get('success', False):
                # Copy successful calculation files
                if 'temp_files' in result:
                    calc_dir = output_path / "successful" / f"calc_{i:04d}_{result.get('label', 'unknown')}"
                    calc_dir.mkdir(exist_ok=True)
                    
                    for temp_file in result['temp_files']:
                        if os.path.exists(temp_file):
                            shutil.copy2(temp_file, calc_dir)
                    
                    # Save result metadata
                    import json
                    with open(calc_dir / "result.json", 'w') as f:
                        json.dump(result, f, indent=2, default=str)
                
                successful_count += 1
            else:
                # Log failed calculations
                with open(output_path / "failed" / f"failed_{failed_count:04d}.txt", 'w') as f:
                    f.write(f"SMILES: {result.get('smiles', 'Unknown')}\n")
                    f.write(f"Error: {result.get('error', 'Unknown error')}\n")
                
                failed_count += 1
        
        # Create summary
        summary = {
            'total_calculations': len(results),
            'successful': successful_count,
            'failed': failed_count,
            'success_rate': successful_count / len(results) * 100
        }
        
        with open(output_path / "summary.json", 'w') as f:
            import json
            json.dump(summary, f, indent=2)
        
        print(f"📁 Results organized in {output_dir}/")
        print(f"📊 Summary: {successful_count} successful, {failed_count} failed")
