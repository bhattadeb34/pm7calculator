
"""
MOPAC output file parser for extracting molecular properties.

This module provides comprehensive parsing of MOPAC PM7 calculation results,
extracting thermodynamic, electronic, and structural properties.

Author: bhattadeb34
Institution: The Pennsylvania State University
"""

import re
import logging
from typing import Dict, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class PM7Parser:
    """
    Parser for MOPAC PM7 output files.
    
    Extracts comprehensive molecular properties from MOPAC output including:
    - Thermodynamic properties (heat of formation, total energy)
    - Electronic properties (HOMO/LUMO, ionization potential)
    - Structural properties (dipole moment, point group)
    - Surface properties (COSMO area/volume)
    - Computational metadata (timing, convergence)
    
    Args:
        config: Optional parser configuration dictionary
        
    Example:
        >>> parser = PM7Parser()
        >>> props = parser.parse("calculation.out")
        >>> print(f"Heat of formation: {props['heat_of_formation']} kcal/mol")
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._setup_patterns()
        logger.debug("PM7Parser initialized")
    
    def _setup_patterns(self):
        """Define regex patterns for property extraction."""
        self.patterns = {
            # Thermodynamic properties
            'heat_of_formation': r"FINAL\\s+HEAT\\s+OF\\s+FORMATION\\s*=\\s*([-+]?\\d+\\.\\d+)\\s*KCAL/MOL",
            
            # Electronic properties  
            'homo_lumo': r"HOMO\\s+LUMO\\s+ENERGIES\\s*\\(EV\\)\\s*=\\s*([-+]?\\d+\\.\\d+)\\s+([-+]?\\d+\\.\\d+)",
            'ionization_potential': r"IONIZATION\\s+POTENTIAL\\s*=\\s*([-+]?\\d+\\.\\d+)\\s*EV",
            
            # Dipole moment (from SUM line in dipole section)
            'dipole_moment': r"SUM\\s+([-+]?\\d+\\.\\d+)\\s+([-+]?\\d+\\.\\d+)\\s+([-+]?\\d+\\.\\d+)\\s+([-+]?\\d+\\.\\d+)",
            
            # Surface properties
            'cosmo_area': r"COSMO\\s+AREA\\s*=\\s*([-+]?\\d+\\.\\d+)\\s*SQUARE\\s+ANGSTROMS",
            'cosmo_volume': r"COSMO\\s+VOLUME\\s*=\\s*([-+]?\\d+\\.\\d+)\\s*CUBIC\\s+ANGSTROMS",
            
            # Molecular properties
            'molecular_weight': r"MOLECULAR\\s+WEIGHT\\s*=\\s*([-+]?\\d+\\.\\d+)",
            'point_group': r"POINT\\s+GROUP:\\s*([A-Za-z0-9]+)",
            'filled_levels': r"NO\\.\\s+OF\\s+FILLED\\s+LEVELS\\s*=\\s*(\\d+)",
            
            # Computational information
            'computation_time': r"COMPUTATION\\s+TIME\\s*=\\s*([\\d.]+)\\s*SECONDS",
            'scf_cycles': r"SCF\\s+CALCULATIONS\\s*=\\s*(\\d+)",
            
            # Geometry optimization
            'gradient_norm': r"GRADIENT\\s+NORM\\s*=\\s*([\\d.]+)",
            'optimization_cycles': r"CYCLE:\\s*(\\d+)",
        }
    
    def parse(self, output_file: str) -> Dict[str, Any]:
        """
        Parse MOPAC output file and extract all available properties.
        
        Args:
            output_file: Path to MOPAC .out file
            
        Returns:
            Dictionary containing extracted properties with descriptive keys
            
        Example:
            >>> parser = PM7Parser()
            >>> props = parser.parse("ethanol.out")
            >>> print(f"Dipole: {props.get('dipole_moment', 'N/A')} Debye")
        """
        if not Path(output_file).exists():
            logger.error(f"Output file not found: {output_file}")
            return {}
        
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"🔍 Parsing MOPAC output: {Path(output_file).name}")
            
            properties = {}
            
            # Extract all properties using patterns
            properties.update(self._extract_thermodynamic_properties(content))
            properties.update(self._extract_electronic_properties(content))
            properties.update(self._extract_structural_properties(content))
            properties.update(self._extract_surface_properties(content))
            properties.update(self._extract_molecular_properties(content))
            properties.update(self._extract_computational_info(content))
            
            # Calculate derived properties
            properties.update(self._calculate_derived_properties(properties))
            
            # Count successful extractions
            extracted_count = len([v for v in properties.values() if v is not None])
            logger.info(f"   📊 Successfully extracted {extracted_count} properties")
            
            return properties
            
        except Exception as e:
            logger.error(f"❌ Error parsing output file {output_file}: {e}")
            return {}
    
    def _extract_thermodynamic_properties(self, content: str) -> Dict:
        """Extract thermodynamic properties."""
        props = {}
        
        # Heat of formation
        match = re.search(self.patterns['heat_of_formation'], content, re.IGNORECASE)
        if match:
            props['heat_of_formation'] = float(match.group(1))
            logger.debug(f"   ✅ Heat of Formation: {props['heat_of_formation']:.3f} kcal/mol")
        else:
            logger.debug("   ❌ Heat of Formation: Not found")
            
        return props
    
    def _extract_electronic_properties(self, content: str) -> Dict:
        """Extract electronic properties."""
        props = {}
        
        # HOMO/LUMO energies
        match = re.search(self.patterns['homo_lumo'], content, re.IGNORECASE)
        if match:
            props['homo_ev'] = float(match.group(1))
            props['lumo_ev'] = float(match.group(2))
            props['gap_ev'] = props['lumo_ev'] - props['homo_ev']
            logger.debug(f"   ✅ HOMO: {props['homo_ev']:.3f} eV")
            logger.debug(f"   ✅ LUMO: {props['lumo_ev']:.3f} eV")
            logger.debug(f"   ✅ Gap: {props['gap_ev']:.3f} eV")
        else:
            logger.debug("   ❌ HOMO/LUMO energies: Not found")
        
        # Ionization potential
        match = re.search(self.patterns['ionization_potential'], content, re.IGNORECASE)
        if match:
            props['ionization_potential'] = float(match.group(1))
            logger.debug(f"   ✅ Ionization Potential: {props['ionization_potential']:.3f} eV")
        else:
            logger.debug("   ❌ Ionization Potential: Not found")
            
        return props
    
    def _extract_structural_properties(self, content: str) -> Dict:
        """Extract structural properties."""
        props = {}
        
        # Dipole moment (from SUM line)
        match = re.search(self.patterns['dipole_moment'], content)
        if match:
            props['dipole_moment'] = float(match.group(4))  # Total magnitude
            props['dipole_x'] = float(match.group(1))
            props['dipole_y'] = float(match.group(2))
            props['dipole_z'] = float(match.group(3))
            logger.debug(f"   ✅ Dipole Moment: {props['dipole_moment']:.3f} Debye")
        else:
            logger.debug("   ❌ Dipole Moment: Not found")
            
        return props
    
    def _extract_surface_properties(self, content: str) -> Dict:
        """Extract surface and solvation properties."""
        props = {}
        
        # COSMO area
        match = re.search(self.patterns['cosmo_area'], content, re.IGNORECASE)
        if match:
            props['cosmo_area'] = float(match.group(1))
            logger.debug(f"   ✅ COSMO Area: {props['cosmo_area']:.2f} Ų")
        
        # COSMO volume
        match = re.search(self.patterns['cosmo_volume'], content, re.IGNORECASE)
        if match:
            props['cosmo_volume'] = float(match.group(1))
            logger.debug(f"   ✅ COSMO Volume: {props['cosmo_volume']:.2f} ų")
            
        return props
    
    def _extract_molecular_properties(self, content: str) -> Dict:
        """Extract general molecular properties."""
        props = {}
        
        # Molecular weight
        match = re.search(self.patterns['molecular_weight'], content, re.IGNORECASE)
        if match:
            props['molecular_weight'] = float(match.group(1))
            logger.debug(f"   ✅ Molecular Weight: {props['molecular_weight']:.2f} g/mol")
        
        # Point group
        match = re.search(self.patterns['point_group'], content, re.IGNORECASE)
        if match:
            props['point_group'] = match.group(1)
            logger.debug(f"   ✅ Point Group: {props['point_group']}")
        
        # Number of filled levels
        match = re.search(self.patterns['filled_levels'], content, re.IGNORECASE)
        if match:
            props['filled_levels'] = int(match.group(1))
            logger.debug(f"   ✅ Filled Levels: {props['filled_levels']}")
            
        return props
    
    def _extract_computational_info(self, content: str) -> Dict:
        """Extract computational metadata."""
        props = {}
        
        # Computation time
        match = re.search(self.patterns['computation_time'], content, re.IGNORECASE)
        if match:
            props['computation_time'] = float(match.group(1))
            logger.debug(f"   ✅ Computation Time: {props['computation_time']:.3f} seconds")
        
        # SCF cycles
        match = re.search(self.patterns['scf_cycles'], content, re.IGNORECASE)
        if match:
            props['scf_cycles'] = int(match.group(1))
            logger.debug(f"   ✅ SCF Cycles: {props['scf_cycles']}")
        
        # Final gradient norm
        matches = list(re.finditer(self.patterns['gradient_norm'], content, re.IGNORECASE))
        if matches:
            # Get the last (final) gradient norm
            props['final_gradient_norm'] = float(matches[-1].group(1))
            logger.debug(f"   ✅ Final Gradient Norm: {props['final_gradient_norm']:.6f}")
        
        # Optimization cycles (count CYCLE: occurrences)
        matches = list(re.finditer(self.patterns['optimization_cycles'], content))
        if matches:
            props['optimization_cycles'] = len(matches)
            logger.debug(f"   ✅ Optimization Cycles: {props['optimization_cycles']}")
            
        return props
    
    def _calculate_derived_properties(self, properties: Dict) -> Dict:
        """Calculate derived properties from extracted values."""
        derived = {}
        
        # Convert heat of formation to other energy units
        if 'heat_of_formation' in properties:
            hof_kcal = properties['heat_of_formation']
            derived['total_energy_kcal_mol'] = hof_kcal
            derived['total_energy_ev'] = hof_kcal * 0.043363  # kcal/mol to eV
            derived['total_energy_kj_mol'] = hof_kcal * 4.184  # kcal/mol to kJ/mol
        
        # Calculate dipole moment magnitude from components (redundant check)
        if all(k in properties for k in ['dipole_x', 'dipole_y', 'dipole_z']):
            import math
            dx, dy, dz = properties['dipole_x'], properties['dipole_y'], properties['dipole_z']
            calculated_magnitude = math.sqrt(dx**2 + dy**2 + dz**2)
            derived['dipole_magnitude_calc'] = calculated_magnitude
        
        # Electronic properties
        if 'homo_ev' in properties and 'lumo_ev' in properties:
            # Electron affinity approximation (negative of LUMO)
            derived['electron_affinity_approx'] = -properties['lumo_ev']
            
            # Chemical hardness (approximation)
            derived['chemical_hardness'] = (properties['ionization_potential'] - derived['electron_affinity_approx']) / 2 if 'ionization_potential' in properties else None
            
            # Chemical softness
            if derived['chemical_hardness'] and derived['chemical_hardness'] != 0:
                derived['chemical_softness'] = 1 / derived['chemical_hardness']
        
        # Surface area to volume ratio
        if 'cosmo_area' in properties and 'cosmo_volume' in properties:
            if properties['cosmo_volume'] > 0:
                derived['surface_to_volume_ratio'] = properties['cosmo_area'] / properties['cosmo_volume']
        
        return derived
    
    def get_available_properties(self) -> List[str]:
        """Return list of properties that can be extracted."""
        return [
            'heat_of_formation', 'total_energy_kcal_mol', 'total_energy_ev', 
            'dipole_moment', 'dipole_x', 'dipole_y', 'dipole_z',
            'homo_ev', 'lumo_ev', 'gap_ev', 'ionization_potential',
            'cosmo_area', 'cosmo_volume', 'molecular_weight', 'point_group',
            'filled_levels', 'computation_time', 'scf_cycles', 'optimization_cycles'
        ]

