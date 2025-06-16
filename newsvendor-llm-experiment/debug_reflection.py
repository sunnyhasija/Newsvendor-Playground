#!/usr/bin/env python3
"""Debug reflection patterns with auto-detection"""

import pandas as pd
import sys
from pathlib import Path

# Auto-detection imports
sys.path.append(str(Path(__file__).parent / "src" / "utils"))

try:
    from file_finder import auto_find_and_load_data
except ImportError:
    print("⚠️  Auto-detection module not found, using fallback...")
    def auto_find_and_load_data():
        try:
            return pd.read_csv("./full_results/processed/complete_20250615_171248.csv"), "fallback"
        except:
            return None, None

def main():
    print("🔍 Debug Reflection Patterns")
    print("="*50)
    
    data, data_path = auto_find_and_load_data()
    
    if data is None:
        print("❌ No valid data file found!")
        return
    
    print(f"📊 Loaded {len(data)} rows from: {data_path}")
    print(f"\nColumns: {list(data.columns)}")

    if 'reflection_pattern' in data.columns:
        print(f"\n📋 Reflection pattern counts:")
        print(data['reflection_pattern'].value_counts())
    else:
        print("❌ No 'reflection_pattern' column found!")

if __name__ == '__main__':
    main()