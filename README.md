# PM7Calculator 

[![PyPI Version](https://img.shields.io/pypi/v/pm7calculator.svg)](https://pypi.org/project/pm7calculator/)
[![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Issues](https://img.shields.io/github/issues/bhattadeb34/pm7calculator.svg)](https://github.com/bhattadeb34/pm7calculator/issues)
[![Downloads](https://pepy.tech/badge/pm7calculator)](https://pepy.tech/project/pm7calculator)

**A wrapper around MOPAC and a user-friendly Python package for PM7 semi-empirical quantum chemistry calculations, designed for researchers, educators, and students in computational chemistry, drug discovery, and materials science.**

---

## Features

###  Molecular Properties Calculations
Calculate essential molecular properties including:
- **Thermodynamic Properties**: Heat of formation, total energy  
- **Electronic Properties**: HOMO/LUMO energies, ionization potential, electron affinity  
- **Structural Properties**: Dipole moment, molecular geometry, point group  
- **Surface Properties**: COSMO area and volume for solvation studies  

###  Multi-Environment Support
- ** Google Colab**: Auto-installation and optimized workflows  
- ** Local Machines**: Full-featured calculations with file management  
- ** Computing Clusters**: Scalable batch processing capabilities  

###  Advanced Capabilities
- **Batch Processing**: Efficiently process thousands of molecules  
- **Smart File Management**: Automatic cleanup with debugging options  
- **Flexible Input**: SMILES strings, SDF files, or coordinate files  
- **Customizable Calculations**: Charge states, spin multiplicities, custom keywords  

---

##  Demo (Example GIF)

![Usage Demo](https://raw.githubusercontent.com/bhattadeb34/pm7calculator/main/docs/demo.gif)  
*Example workflow of calculating properties in Google Colab*  

---

## Quick Start

### Installation

```bash
# From PyPI (recommended)
pip install pm7calculator

# From GitHub (latest version)
pip install git+https://github.com/bhattadeb34/pm7calculator.git

# With Google Colab support
pip install "pm7calculator[colab]"

# With visualization tools
pip install "pm7calculator[visualization]"

# Complete installation
pip install "pm7calculator[all]"
````

### Basic Usage

```python
from pm7calculator import PM7Calculator

# Initialize calculator
calc = PM7Calculator()

# Calculate properties for caffeine
props = calc.calculate("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")

# Display results
print(f"Heat of Formation: {props['heat_of_formation']:.2f} kcal/mol")
print(f"Dipole Moment: {props['dipole_moment']:.2f} Debye")
print(f"HOMO Energy: {props['homo_ev']:.2f} eV")
print(f"LUMO Energy: {props['lumo_ev']:.2f} eV")
```

### Google Colab Usage

```python
# In Google Colab - automatic dependency installation
from pm7calculator.environments import ColabCalculator

# Auto-install MOPAC and dependencies
calc = ColabCalculator()

# Calculate properties with file retention for inspection
props = calc.calculate("CCO", cleanup=False)

# Enhanced display for Jupyter environments
calc.display_properties(props)
```

### Batch Processing

```python
# Process multiple molecules efficiently
drug_molecules = [
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"    # Caffeine
]

results = calc.calculate_batch(drug_molecules)

# Create pandas DataFrame for analysis
import pandas as pd
df = pd.DataFrame([r for r in results if r['success']])
print(df[['smiles', 'heat_of_formation', 'dipole_moment', 'gap_ev']])
```

---

##  Requirements

* **Python**: 3.7 or higher
* **MOPAC**: Quantum chemistry software (auto-installed in Colab)
* **RDKit**: Molecular structure handling
* **NumPy / Pandas**: Data manipulation

---

##  Technical Details

* **Method**: PM7 semi-empirical quantum mechanics
* **3D Structure**: Generated using RDKit with MMFF/UFF force fields
* **Calculations**: Single-point energy and properties
* **Convergence**: High precision settings (GNORM=0.001, SCFCRT=1.D-8)
* **Default**: Neutral molecules in gas phase with COSMO solvation

---

##  License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

##  Citation

```bibtex
@software{pm7calculator2025,
  title={PM7Calculator: A Comprehensive Python Package for PM7 Quantum Chemistry Calculations},
  author={bhattadeb34},
  institution={The Pennsylvania State University},
  year={2025},
  version={1.0.0},
  url={https://github.com/bhattadeb34/pm7calculator}
}
```

---

 **Star this repository if you find it useful!**

