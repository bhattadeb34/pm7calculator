
"""
Basic usage examples for PM7Calculator.

Author: bhattadeb34
Institution: The Pennsylvania State University
"""

def run_basic_examples():
    """Run basic PM7Calculator examples."""
    print("🧪 PM7Calculator Basic Usage Examples")
    print("=" * 50)
    
    from pm7calculator import PM7Calculator
    
    # Example 1: Simple calculation
    print("\n1️⃣ Simple Calculation:")
    calc = PM7Calculator()
    props = calc.calculate("CCO")  # Ethanol
    
    if props['success']:
        print(f"   Heat of Formation: {props['heat_of_formation']:.3f} kcal/mol")
        print(f"   Dipole Moment: {props['dipole_moment']:.3f} Debye")
    
    # Example 2: Different charge states
    print("\n2️⃣ Charged Species:")
    benzene_cation = calc.calculate("c1ccccc1", charge=1, multiplicity=2)
    if benzene_cation['success']:
        print(f"   Benzene cation HOMO: {benzene_cation['homo_ev']:.3f} eV")
    
    # Example 3: Keep files for inspection
    print("\n3️⃣ File Management:")
    props_debug = calc.calculate("CCN", cleanup=False)
    if props_debug['success']:
        print(f"   Files kept: {len(props_debug['temp_files'])}")
        for f in props_debug['temp_files']:
            print(f"      - {f}")

if __name__ == "__main__":
    run_basic_examples()
