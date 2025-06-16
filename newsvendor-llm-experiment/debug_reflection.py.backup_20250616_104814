#!/usr/bin/env python3
"""Debug reflection patterns"""

import pandas as pd

# Load data
data = pd.read_csv("./full_results/processed/complete_20250615_171248.csv")
print(f"Loaded {len(data)} rows")

# Check reflection column
print(f"\nColumns: {list(data.columns)}")

if 'reflection_pattern' in data.columns:
    print(f"\nReflection pattern counts:")
    print(data['reflection_pattern'].value_counts())
    print(f"\nUnique values: {data['reflection_pattern'].unique()}")
else:
    print("❌ No 'reflection_pattern' column found!")
    print("Available columns containing 'reflect':")
    reflect_cols = [col for col in data.columns if 'reflect' in col.lower()]
    print(reflect_cols)