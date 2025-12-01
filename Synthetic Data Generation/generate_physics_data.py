#!/usr/bin/env python3
"""
Simple Physics Data Generator
============================
Generate CSV file with actual physics data from all models.

Usage:
    python3 generate_physics_data.py [--cycles N] [--output filename.csv]
"""

import sys
import argparse
sys.path.insert(0, 'src')

def main():
    parser = argparse.ArgumentParser(description='Generate physics data CSV')
    parser.add_argument('--cycles', type=int, default=100, help='Number of cycles per model (default: 100)')
    parser.add_argument('--output', type=str, default='physics_data.csv', help='Output CSV filename (default: physics_data.csv)')
    args = parser.parse_args()
    
    print(f"🔄 Generating {args.cycles} cycles per model...")
    print(f"📁 Output file: {args.output}")
    
    # Import and run the fixed integration pipeline
    from integrated_validation_pipeline import IntegratedPhysicsValidator
    import pandas as pd
    
    # Initialize and run
    validator = IntegratedPhysicsValidator()
    results = validator.validate_all_physics_models(n_test_cycles=args.cycles)
    
    # Extract ACTUAL comprehensive data from each model separately
    all_rows = []
    
    for model_name, model_results in results['model_results'].items():
        model_data = model_results.get('generated_data', [])
        
        for i, cycle_data in enumerate(model_data):
            # Create model-specific row with ALL available data
            row = {
                'cycle_id': i,
                'model_name': model_name,
                'model_type': cycle_data.get('model_type', model_name),
                'validation_score': model_results.get('model_score', 1.0),
                'model_status': model_results.get('model_status', 'PASS'),
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            # Extract ALL data from this cycle (not just single values)
            def flatten_data(data, prefix=''):
                """Recursively flatten nested data structures"""
                for key, value in data.items():
                    if key in ['cycle_id', 'model_type']:
                        continue
                    
                    full_key = f"{prefix}{key}" if prefix else key
                    
                    if isinstance(value, dict):
                        flatten_data(value, f"{full_key}_")
                    elif isinstance(value, (list, tuple)) and len(value) > 0:
                        if isinstance(value[0], (int, float)):
                            row[f"{full_key}_mean"] = sum(value) / len(value)
                            row[f"{full_key}_min"] = min(value)
                            row[f"{full_key}_max"] = max(value)
                            row[f"{full_key}_count"] = len(value)
                        else:
                            row[f"{full_key}_count"] = len(value)
                            row[f"{full_key}_type"] = str(type(value[0]).__name__)
                    elif isinstance(value, (int, float, str, bool)):
                        row[full_key] = value
                    elif hasattr(value, '__dict__'):
                        # Handle dataclass/object instances
                        flatten_data(value.__dict__, f"{full_key}_")
            
            flatten_data(cycle_data)
            all_rows.append(row)
    
    # Save CSV
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(args.output, index=False)
        print(f"✅ Generated {len(df)} rows from {df['model_name'].nunique()} models")
        print(f"📊 Saved to: {args.output}")
    else:
        print("❌ No data generated")

if __name__ == "__main__":
    main()