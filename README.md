# PM7Calculator

A simple Python package for PM7 semi-empirical quantum chemistry calculations, optimized for Google Colab.

## Installation

Simply install from GitHub:

```bash
pip install git+https://github.com/bhattadeb34/pm7calculator.git
```

### For Google Colab Users

After installation, run this once per Colab session to install MOPAC and dependencies:

```python
from pm7calculator import install_colab_dependencies
install_colab_dependencies()
```

### For Local Users

You need to install MOPAC separately before using this package. The easiest way is with conda:

```bash
conda install -c conda-forge mopac rdkit pandas numpy
```

This package is optimized for Google Colab, but works anywhere MOPAC is available.

## Quick Start

```python
# Import the main function
from pm7calculator import calculate_pm7_properties_colab

# Calculate properties for ethanol
props = calculate_pm7_properties_colab("CCO")

# Display results
print(f"Heat of Formation: {props['heat_of_formation']:.2f} kcal/mol")
print(f"Dipole Moment: {props['dipole_moment']:.2f} Debye")
print(f"HOMO-LUMO Gap: {props['gap_ev']:.2f} eV")
```

## Properties Calculated

- **Thermodynamic**: Heat of formation, total energy
- **Electronic**: HOMO/LUMO energies, ionization potential, HOMO-LUMO gap
- **Structural**: Dipole moment, molecular geometry, point group
- **Surface**: COSMO area and volume
- **General**: Molecular weight, computation time

## Usage Examples

### Single Molecule

```python
from pm7calculator import calculate_pm7_properties_colab, display_properties_enhanced

# Calculate caffeine properties
props = calculate_pm7_properties_colab("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")

# Enhanced display with formatting
display_properties_enhanced(props)
```

### Multiple Molecules

```python
from pm7calculator import calculate_pm7_batch_colab

# Process a list of SMILES
smiles_list = ["CCO", "CC(=O)O", "CCN"]  # ethanol, acetic acid, ethylamine
results = calculate_pm7_batch_colab(smiles_list)

# Print results
for result in results:
    if result['success']:
        print(f"{result['smiles']}: {result['heat_of_formation']:.2f} kcal/mol")
```

### DataFrame Processing

```python
from pm7calculator import calculate_pm7_dataframe_colab
import pandas as pd

# Create DataFrame with SMILES
df = pd.DataFrame({'smiles': ["CCO", "CC(=O)O", "CCN"]})

# Add PM7 properties as new columns
df_with_props = calculate_pm7_dataframe_colab(df)

# Display selected columns
print(df_with_props[['smiles', 'heat_of_formation', 'dipole_moment']])
```

## Available Functions

- `calculate_pm7_properties_colab(smiles)` - Single molecule calculation
- `calculate_pm7_batch_colab(smiles_list)` - Multiple molecules
- `calculate_pm7_dataframe_colab(df)` - Process pandas DataFrame
- `display_properties_enhanced(props)` - Pretty print results
- `install_colab_dependencies()` - Install MOPAC and dependencies (Colab only)

## Output Format

Each calculation returns a dictionary containing:

```python
{
    'success': True,
    'smiles': 'CCO',
    'heat_of_formation': -57.859,
    'dipole_moment': 2.057,
    'homo_ev': -10.641,
    'lumo_ev': 2.965,
    'gap_ev': 13.606,
    'ionization_potential': 10.641,
    'molecular_weight': 46.07,
    'point_group': 'C1',
    'num_atoms': 9,
    'computation_time': 0.207,
    # ... additional properties
}
```

## Example Output

```
MOPAC is available
Processing: CCO
Generated 3D structure (9 atoms)
MOPAC calculation completed
Parsing MOPAC output: /tmp/mol_83d0fa62.out
Heat of Formation: -57.859 kcal/mol
Dipole Moment: 2.057 Debye
HOMO: -10.641 eV
LUMO: 2.965 eV
Successfully parsed 17 properties
Properties calculated successfully
```

## Requirements

- Python 3.7+
- Google Colab (recommended) or local MOPAC installation
- Dependencies auto-installed in Colab: MOPAC, RDKit, pandas, numpy

## Why This Package

- **Simple**: Works exactly like the original Colab code
- **No configuration**: Just install and use
- **Reliable**: Uses the proven MOPAC quantum chemistry package
- **Fast**: Optimized for batch processing
- **Colab-ready**: Auto-installs dependencies in Google Colab

## Complete Google Colab Example

```python
# 1. Install the package
!pip install git+https://github.com/bhattadeb34/pm7calculator.git

# 2. Install dependencies (once per session)
from pm7calculator import install_colab_dependencies
install_colab_dependencies()

# 3. Calculate properties
from pm7calculator import calculate_pm7_properties_colab, display_properties_enhanced

# Single molecule
props = calculate_pm7_properties_colab("CCO")
display_properties_enhanced(props)

# Multiple molecules
drug_molecules = [
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",   # Caffeine
    "CC(=O)OC1=CC=CC=C1C(=O)O"        # Aspirin
]

results = calculate_pm7_batch_colab(drug_molecules)

# Create DataFrame for analysis
import pandas as pd
df = pd.DataFrame([r for r in results if r['success']])
print(df[['smiles', 'heat_of_formation', 'dipole_moment', 'gap_ev']])
```

## License

MIT License

## Issues

Report issues at: https://github.com/bhattadeb34/pm7calculator/issues