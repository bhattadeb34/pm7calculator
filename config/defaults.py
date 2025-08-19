
"""
Default configuration settings for PM7Calculator.

Author: bhattadeb34
Institution: The Pennsylvania State University
"""

DEFAULT_CONFIG = {
    # MOPAC calculation settings
    "mopac_keywords": "PRECISE GNORM=0.001 SCFCRT=1.D-8",
    "mopac_timeout": 300,  # seconds
    "mopac_command": "mopac",
    
    # Parser settings
    "parser": {
        "extract_all_properties": True,
        "include_derived_properties": True,
        "property_precision": 6,
    },
    
    # 3D structure generation settings
    "structure_generation": {
        "random_seed": 42,
        "max_attempts": 5,
        "use_small_ring_torsions": True,
        "force_field_optimization": True,
        "max_ff_iterations": 1000,
        "embedding_method": "ETKDG",
    },
    
    # File management
    "file_management": {
        "cleanup_by_default": True,
        "keep_successful_outputs": False,
        "temp_dir_prefix": "pm7calc_",
    },
    
    # Environment-specific settings
    "environments": {
        "colab": {
            "temp_dir": "/tmp",
            "auto_install": True,
            "conda_packages": ["mopac", "rdkit", "ase"],
            "pip_packages": ["condacolab"],
            "display_progress": True,
        },
        "local": {
            "temp_dir": None,  # Use system default
            "check_dependencies": True,
            "parallel_calculations": False,
        },
        "cluster": {
            "job_template": "slurm",
            "default_queue": "normal",
            "default_walltime": "01:00:00",
            "max_concurrent_jobs": 50,
        }
    },
    
    # Logging settings
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file_logging": False,
        "log_file": "pm7calculator.log",
    }
}

# Property display formatting
PROPERTY_FORMATS = {
    "heat_of_formation": {
        "unit": "kcal/mol", 
        "precision": 3, 
        "emoji": "🔥",
        "name": "Heat of Formation",
        "description": "Enthalpy change for formation from elements"
    },
    "dipole_moment": {
        "unit": "Debye", 
        "precision": 3, 
        "emoji": "⚡",
        "name": "Dipole Moment",
        "description": "Measure of molecular polarity"
    },
    "homo_ev": {
        "unit": "eV", 
        "precision": 3, 
        "emoji": "🔋",
        "name": "HOMO Energy",
        "description": "Highest Occupied Molecular Orbital energy"
    },
    "lumo_ev": {
        "unit": "eV", 
        "precision": 3, 
        "emoji": "🔋",
        "name": "LUMO Energy", 
        "description": "Lowest Unoccupied Molecular Orbital energy"
    },
    "gap_ev": {
        "unit": "eV", 
        "precision": 3, 
        "emoji": "⚡",
        "name": "HOMO-LUMO Gap",
        "description": "Electronic band gap"
    },
    "ionization_potential": {
        "unit": "eV", 
        "precision": 3, 
        "emoji": "⚡",
        "name": "Ionization Potential",
        "description": "Energy required to remove an electron"
    },
    "molecular_weight": {
        "unit": "g/mol", 
        "precision": 2, 
        "emoji": "⚖️",
        "name": "Molecular Weight",
        "description": "Molar mass of the molecule"
    },
    "cosmo_area": {
        "unit": "Ų", 
        "precision": 2, 
        "emoji": "📐",
        "name": "COSMO Area",
        "description": "Solvent-accessible surface area"
    },
    "cosmo_volume": {
        "unit": "ų", 
        "precision": 2, 
        "emoji": "📦",
        "name": "COSMO Volume",
        "description": "Molecular volume in solvent"
    },
    "computation_time": {
        "unit": "seconds", 
        "precision": 3, 
        "emoji": "⏱️",
        "name": "Computation Time",
        "description": "CPU time for calculation"
    },
}

# Method-specific settings
METHOD_CONFIGS = {
    "PM7": {
        "description": "Latest parametrized semi-empirical method",
        "elements_supported": ["H", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"],
        "typical_accuracy": "±5 kcal/mol for heat of formation",
        "speed": "Very fast",
    },
    "PM6": {
        "description": "Previous generation semi-empirical method", 
        "elements_supported": ["H", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br"],
        "typical_accuracy": "±8 kcal/mol for heat of formation",
        "speed": "Very fast",
    },
    "AM1": {
        "description": "Austin Model 1 semi-empirical method",
        "elements_supported": ["H", "C", "N", "O", "F", "Si", "P", "S", "Cl"],
        "typical_accuracy": "±10 kcal/mol for heat of formation", 
        "speed": "Fast",
    }
}
