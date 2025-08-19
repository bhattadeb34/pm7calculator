
"""
Command-line interface for PM7Calculator.

Author: bhattadeb34
Institution: The Pennsylvania State University
"""

import argparse
import sys
import json
from pathlib import Path
from pm7calculator import PM7Calculator


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PM7Calculator: Quantum chemistry calculations from the command line",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pm7calc "CCO"                          # Calculate ethanol properties
  pm7calc "CCO" --output results.json   # Save to JSON file
  pm7calc --batch molecules.txt         # Process file with SMILES list
  pm7calc "c1ccccc1" --charge 1 --mult 2 # Benzene cation calculation
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('smiles', nargs='?', help='SMILES string to calculate')
    input_group.add_argument('--batch', '-b', help='File containing SMILES strings (one per line)')
    
    # Calculation options
    parser.add_argument('--method', '-m', default='PM7', help='Calculation method (default: PM7)')
    parser.add_argument('--charge', '-c', type=int, default=0, help='Molecular charge (default: 0)')
    parser.add_argument('--multiplicity', '--mult', type=int, default=1, help='Spin multiplicity (default: 1)')
    parser.add_argument('--keywords', '-k', help='Additional MOPAC keywords')
    
    # Output options
    parser.add_argument('--output', '-o', help='Output file (JSON format)')
    parser.add_argument('--format', '-f', choices=['json', 'csv'], default='json', help='Output format')
    parser.add_argument('--cleanup', action='store_true', help='Clean up temporary files (default: keep)')
    
    # Other options
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--version', action='version', version='PM7Calculator 1.0.0')
    
    args = parser.parse_args()
    
    # Initialize calculator
    calc = PM7Calculator(method=args.method)
    
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
    
    try:
        if args.smiles:
            # Single molecule calculation
            result = calc.calculate(
                args.smiles,
                cleanup=args.cleanup,
                charge=args.charge,
                multiplicity=args.multiplicity,
                custom_keywords=args.keywords
            )
            results = [result]
        
        elif args.batch:
            # Batch calculation
            with open(args.batch, 'r') as f:
                smiles_list = [line.strip() for line in f if line.strip()]
            
            results = calc.calculate_batch(
                smiles_list,
                cleanup=args.cleanup,
                charge=args.charge,
                multiplicity=args.multiplicity,
                custom_keywords=args.keywords
            )
        
        # Output results
        if args.output:
            output_path = Path(args.output)
            
            if args.format == 'json':
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
            
            elif args.format == 'csv':
                import pandas as pd
                df = pd.DataFrame(results)
                df.to_csv(output_path, index=False)
            
            print(f"Results saved to {output_path}")
        
        else:
            # Print to stdout
            for result in results:
                if result['success']:
                    print(f"SMILES: {result['smiles']}")
                    print(f"Heat of Formation: {result.get('heat_of_formation', 'N/A')} kcal/mol")
                    print(f"Dipole Moment: {result.get('dipole_moment', 'N/A')} Debye")
                    if 'homo_ev' in result and 'lumo_ev' in result:
                        print(f"HOMO: {result['homo_ev']:.3f} eV, LUMO: {result['lumo_ev']:.3f} eV")
                    print()
                else:
                    print(f"FAILED: {result['smiles']} - {result.get('error', 'Unknown error')}")
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        if len(results) > 1:
            print(f"Summary: {successful}/{len(results)} calculations successful")
        
        return 0 if successful > 0 else 1
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
