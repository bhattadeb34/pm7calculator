
"""
Batch processing examples for PM7Calculator.

Author: bhattadeb34
Institution: The Pennsylvania State University
"""

def run_batch_examples():
    """Run batch processing examples."""
    print("🚀 PM7Calculator Batch Processing Examples")
    print("=" * 50)
    
    from pm7calculator import PM7Calculator
    
    # Example drug molecules
    drug_molecules = [
        ("Ethanol", "CCO"),
        ("Acetic Acid", "CC(=O)O"),
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
        ("Ibuprofen", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"),
    ]
    
    print(f"\n🧬 Processing {len(drug_molecules)} drug molecules...")
    
    calc = PM7Calculator()
    smiles_list = [smiles for name, smiles in drug_molecules]
    results = calc.calculate_batch(smiles_list)
    
    # Display results
    print("\n📊 Results Summary:")
    print("-" * 80)
    print(f"{'Name':<15} {'SMILES':<25} {'ΔHf (kcal/mol)':<15} {'μ (Debye)':<12} {'Gap (eV)':<10}")
    print("-" * 80)
    
    for i, (name, smiles) in enumerate(drug_molecules):
        result = results[i]
        if result['success']:
            hof = result.get('heat_of_formation', 'N/A')
            dipole = result.get('dipole_moment', 'N/A')
            gap = result.get('gap_ev', 'N/A')
            print(f"{name:<15} {smiles:<25} {hof:<15.3f} {dipole:<12.3f} {gap:<10.3f}")
        else:
            print(f"{name:<15} {smiles:<25} {'FAILED':<15} {'FAILED':<12} {'FAILED':<10}")
    
    # Statistics
    successful = [r for r in results if r['success']]
    print(f"\n✅ Success rate: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")
    
    if successful:
        avg_hof = sum(r['heat_of_formation'] for r in successful) / len(successful)
        print(f"📊 Average heat of formation: {avg_hof:.3f} kcal/mol")

if __name__ == "__main__":
    run_batch_examples()
