
"""
Google Colab PM7Calculator implementation with working PM7 calculations.

Author: bhattadeb34
Institution: The Pennsylvania State University
"""

import subprocess
import sys
import os
import tempfile
import uuid
import re
import logging
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


class ColabCalculator:
    """PM7Calculator optimized for Google Colab environment."""
    
    def __init__(self, method="PM7", auto_install=True):
        self.method = method
        self.temp_dir = "/tmp"
        self.keywords = "PRECISE GNORM=0.001 SCFCRT=1.D-8"
        
        if auto_install:
            print("To install dependencies, run: ColabCalculator.install_dependencies()")
    
    @staticmethod
    def install_dependencies():
        """Install required packages in Google Colab."""
        print("Installing PM7Calculator dependencies in Google Colab...")
        
        try:
            # Install condacolab
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "condacolab"], check=True)
            
            # Import and install conda
            import condacolab
            condacolab.install()
            
            # Install packages
            packages = ["mopac", "rdkit", "ase", "pandas", "numpy<2.0", "matplotlib", "seaborn"]
            for package in packages:
                print(f"Installing {package}...")
                subprocess.run(["conda", "install", "-c", "conda-forge", package, "-y"], check=True)
            
            print("Installation complete! Please restart the runtime.")
            
        except Exception as e:
            print(f"Installation failed: {e}")
            raise
    
    def _check_mopac(self):
        """Check if MOPAC is available."""
        try:
            subprocess.run(["mopac"], capture_output=True, text=True)
            return True
        except FileNotFoundError:
            print("MOPAC not found. Run ColabCalculator.install_dependencies() first.")
            return False
    
    def smiles_to_3d(self, smiles):
        """Convert SMILES to 3D coordinates using RDKit."""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            import numpy as np
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None, None
            
            mol = Chem.AddHs(mol)
            
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            params.useSmallRingTorsions = True
            
            status = AllChem.EmbedMolecule(mol, params)
            if status != 0:
                return None, None
            
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
            except:
                try:
                    AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
                except:
                    pass
            
            atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
            conf = mol.GetConformer()
            coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
            
            return atoms, coords
            
        except ImportError:
            print("RDKit not available. Install dependencies first.")
            return None, None
        except Exception as e:
            print(f"Failed to generate 3D structure for {smiles}: {e}")
            return None, None
    
    def write_mopac_input(self, atoms, coordinates, label):
        """Write MOPAC input file."""
        input_file = os.path.join(self.temp_dir, f"{label}.mop")
        
        with open(input_file, 'w') as f:
            f.write(f"{self.method} {self.keywords} CHARGE=0\\n")
            f.write(f"PM7 calculation for {label}\\n")
            f.write("\\n")
            
            for atom, coord in zip(atoms, coordinates):
                f.write(f"{atom:2s} {coord[0]:12.6f} 1 {coord[1]:12.6f} 1 {coord[2]:12.6f} 1\\n")
        
        return input_file
    
    def run_mopac_calculation(self, input_file):
        """Run MOPAC calculation."""
        try:
            result = subprocess.run(
                ["mopac", input_file],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                print(f"MOPAC failed with return code {result.returncode}")
                return False
            
            return True
            
        except subprocess.TimeoutExpired:
            print("MOPAC calculation timed out")
            return False
        except Exception as e:
            print(f"Error running MOPAC: {e}")
            return False
    
    def parse_mopac_output(self, output_file):
        """Parse MOPAC output file for properties."""
        properties = {}
        
        try:
            with open(output_file, 'r') as f:
                content = f.read()
            
            # Parse heat of formation
            hof_pattern = r"FINAL\\s+HEAT\\s+OF\\s+FORMATION\\s*=\\s*([-+]?\\d+\\.\\d+)\\s*KCAL/MOL"
            hof_match = re.search(hof_pattern, content, re.IGNORECASE)
            if hof_match:
                properties['heat_of_formation'] = float(hof_match.group(1))
            
            # Parse dipole moment
            dipole_pattern = r"SUM\\s+([-+]?\\d+\\.\\d+)\\s+([-+]?\\d+\\.\\d+)\\s+([-+]?\\d+\\.\\d+)\\s+([-+]?\\d+\\.\\d+)"
            dipole_match = re.search(dipole_pattern, content)
            if dipole_match:
                properties['dipole_moment'] = float(dipole_match.group(4))
                properties['dipole_x'] = float(dipole_match.group(1))
                properties['dipole_y'] = float(dipole_match.group(2))
                properties['dipole_z'] = float(dipole_match.group(3))
            
            # Parse HOMO/LUMO
            homo_lumo_pattern = r"HOMO\\s+LUMO\\s+ENERGIES\\s*\\(EV\\)\\s*=\\s*([-+]?\\d+\\.\\d+)\\s+([-+]?\\d+\\.\\d+)"
            homo_lumo_match = re.search(homo_lumo_pattern, content)
            if homo_lumo_match:
                properties['homo_ev'] = float(homo_lumo_match.group(1))
                properties['lumo_ev'] = float(homo_lumo_match.group(2))
                properties['gap_ev'] = properties['lumo_ev'] - properties['homo_ev']
            
            # Parse ionization potential
            ip_pattern = r"IONIZATION\\s+POTENTIAL\\s*=\\s*([-+]?\\d+\\.\\d+)\\s*EV"
            ip_match = re.search(ip_pattern, content, re.IGNORECASE)
            if ip_match:
                properties['ionization_potential'] = float(ip_match.group(1))
            
            # Parse molecular weight
            mw_pattern = r"MOLECULAR\\s+WEIGHT\\s*=\\s*([-+]?\\d+\\.\\d+)"
            mw_match = re.search(mw_pattern, content, re.IGNORECASE)
            if mw_match:
                properties['molecular_weight'] = float(mw_match.group(1))
            
            # Parse computation time
            comp_time_pattern = r"COMPUTATION\\s+TIME\\s*=\\s*([\\d.]+)\\s*SECONDS"
            comp_time_match = re.search(comp_time_pattern, content, re.IGNORECASE)
            if comp_time_match:
                properties['computation_time'] = float(comp_time_match.group(1))
            
        except Exception as e:
            print(f"Error parsing MOPAC output: {e}")
        
        return properties
    
    def get_temp_files(self, label):
        """Get list of temporary files."""
        extensions = ['.mop', '.out', '.arc', '.aux', '.log', '.end']
        temp_files = []
        for ext in extensions:
            temp_file = os.path.join(self.temp_dir, f"{label}{ext}")
            if os.path.exists(temp_file):
                temp_files.append(temp_file)
        return temp_files
    
    def cleanup_files(self, label):
        """Clean up temporary files."""
        temp_files = self.get_temp_files(label)
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass
        return temp_files
    
    def calculate_properties(self, smiles, cleanup=True):
        """Calculate PM7 properties for a SMILES string."""
        if not self._check_mopac():
            return {'success': False, 'error': 'MOPAC not available', 'smiles': smiles}
        
        label = f"mol_{uuid.uuid4().hex[:8]}"
        
        try:
            print(f"Processing: {smiles}")
            
            # Generate 3D structure
            atoms, coords = self.smiles_to_3d(smiles)
            if atoms is None:
                return {'success': False, 'error': 'Failed to generate 3D structure', 'smiles': smiles}
            
            print(f"Generated 3D structure ({len(atoms)} atoms)")
            
            # Write MOPAC input
            input_file = self.write_mopac_input(atoms, coords, label)
            
            # Run calculation
            success = self.run_mopac_calculation(input_file)
            if not success:
                return {'success': False, 'error': 'MOPAC calculation failed', 'smiles': smiles}
            
            # Parse output
            output_file = os.path.join(self.temp_dir, f"{label}.out")
            properties = self.parse_mopac_output(output_file)
            
            if not properties:
                return {'success': False, 'error': 'Failed to parse properties', 'smiles': smiles}
            
            # Add metadata
            properties['success'] = True
            properties['smiles'] = smiles
            properties['num_atoms'] = len(atoms)
            properties['label'] = label
            
            # Handle files
            temp_files = self.get_temp_files(label)
            properties['temp_files'] = temp_files
            properties['files_kept'] = not cleanup
            
            if cleanup:
                self.cleanup_files(label)
            else:
                print(f"Keeping {len(temp_files)} temporary files")
            
            print("Properties calculated successfully")
            return properties
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'smiles': smiles}
        
        finally:
            if cleanup:
                self.cleanup_files(label)
    
    def calculate_batch(self, smiles_list, cleanup=True, max_molecules=None):
        """Calculate properties for multiple SMILES."""
        if max_molecules:
            smiles_list = smiles_list[:max_molecules]
        
        results = []
        total = len(smiles_list)
        
        print(f"Processing {total} molecules...")
        
        for i, smiles in enumerate(smiles_list):
            print(f"Molecule {i+1}/{total}")
            result = self.calculate_properties(smiles, cleanup=cleanup)
            results.append(result)
        
        successful = sum(1 for r in results if r['success'])
        print(f"Summary: {successful}/{len(results)} successful calculations")
        
        return results
    
    def display_properties(self, props):
        """Display properties with formatting."""
        if not props.get('success', False):
            print(f"Calculation failed: {props.get('error', 'Unknown error')}")
            return
        
        print(f"PM7 Properties for {props.get('smiles', 'Unknown')}:")
        print("=" * 50)
        
        if 'heat_of_formation' in props:
            print(f"Heat of Formation: {props['heat_of_formation']:.3f} kcal/mol")
        
        if 'dipole_moment' in props:
            print(f"Dipole Moment: {props['dipole_moment']:.3f} Debye")
        
        if 'homo_ev' in props and 'lumo_ev' in props:
            print(f"HOMO Energy: {props['homo_ev']:.3f} eV")
            print(f"LUMO Energy: {props['lumo_ev']:.3f} eV")
            print(f"HOMO-LUMO Gap: {props['gap_ev']:.3f} eV")
        
        if 'molecular_weight' in props:
            print(f"Molecular Weight: {props['molecular_weight']:.2f} g/mol")
        
        if 'computation_time' in props:
            print(f"Computation Time: {props['computation_time']:.3f} seconds")
        
        print(f"Number of Atoms: {props.get('num_atoms', 'N/A')}")


# Convenience functions
def calculate_pm7_properties_colab(smiles, method="PM7", cleanup=True):
    """One-line function for PM7 calculations."""
    calc = ColabCalculator(method=method, auto_install=False)
    return calc.calculate_properties(smiles, cleanup=cleanup)


def calculate_pm7_batch_colab(smiles_list, method="PM7", cleanup=True, max_molecules=None):
    """Batch calculation function."""
    calc = ColabCalculator(method=method, auto_install=False)
    return calc.calculate_batch(smiles_list, cleanup=cleanup, max_molecules=max_molecules)
