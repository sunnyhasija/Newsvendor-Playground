#!/usr/bin/env python3
"""
src/utils/file_finder.py - Auto-detection module for latest data files
Centralized file detection logic for all analysis scripts
"""

import glob
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple
import pandas as pd

class DataFileFinder:
    """Centralized auto-detection of latest experiment data files"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.search_patterns = [
            "./full_results/processed/complete_*.csv",
            "./full_results/processed/complete_*.csv.gz", 
            "./complete_*.csv",
            "./temp_results.csv",
            "./data/complete_*.csv",
            "./results/complete_*.csv",
            "./output/complete_*.csv"
        ]
    
    def find_latest_data_file(self, custom_patterns: Optional[List[str]] = None) -> Optional[str]:
        """Find the most recent data file from experiment results"""
        
        patterns = custom_patterns or self.search_patterns
        latest_file = None
        latest_time = 0
        all_files = []
        
        if self.verbose:
            print("🔍 Searching for data files...")
        
        for pattern in patterns:
            files = glob.glob(pattern)
            for file in files:
                file_path = Path(file)
                if file_path.exists():
                    mtime = file_path.stat().st_mtime
                    all_files.append((file, mtime))
                    
                    if mtime > latest_time:
                        latest_time = mtime
                        latest_file = file_path
                        
                    if self.verbose:
                        print(f"  Found: {file} (modified: {datetime.fromtimestamp(mtime)})")
        
        if latest_file:
            if self.verbose:
                print(f"✅ Using latest file: {latest_file}")
            return str(latest_file)
        else:
            if self.verbose:
                print("❌ No data files found!")
            return None
    
    def find_all_data_files(self, custom_patterns: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """Find all data files sorted by modification time (newest first)"""
        
        patterns = custom_patterns or self.search_patterns
        all_files = []
        
        for pattern in patterns:
            files = glob.glob(pattern)
            for file in files:
                file_path = Path(file)
                if file_path.exists():
                    mtime = file_path.stat().st_mtime
                    all_files.append((str(file_path), mtime))
        
        # Sort by modification time (newest first)
        all_files.sort(key=lambda x: x[1], reverse=True)
        return all_files
    
    def validate_data_file(self, file_path: str) -> Tuple[bool, str]:
        """Validate that a data file has the expected structure"""
        
        try:
            # Handle compressed files
            if file_path.endswith('.gz'):
                data = pd.read_csv(file_path, compression='gzip')
            else:
                data = pd.read_csv(file_path)
            
            # Check if empty
            if len(data) == 0:
                return False, "Data file is empty"
            
            # Check required columns
            required_columns = [
                'negotiation_id', 'buyer_model', 'supplier_model', 
                'reflection_pattern', 'completed', 'agreed_price',
                'total_rounds', 'total_tokens'
            ]
            
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                return False, f"Missing required columns: {missing_columns}"
            
            # Check for successful negotiations with valid prices
            successful = data[data['completed'] == True]
            valid_prices = successful['agreed_price'].dropna()
            
            if len(valid_prices) == 0:
                return False, "No successful negotiations with valid prices found"
            
            return True, f"Valid data file with {len(data)} negotiations, {len(valid_prices)} with valid prices"
            
        except Exception as e:
            return False, f"Error reading file: {e}"
    
    def find_best_data_file(self, custom_patterns: Optional[List[str]] = None) -> Optional[str]:
        """Find the best (latest + valid) data file"""
        
        all_files = self.find_all_data_files(custom_patterns)
        
        if self.verbose:
            print(f"\n🔍 Validating {len(all_files)} candidate files...")
        
        for file_path, mtime in all_files:
            is_valid, message = self.validate_data_file(file_path)
            
            if self.verbose:
                status = "✅" if is_valid else "❌"
                print(f"  {status} {file_path}: {message}")
            
            if is_valid:
                if self.verbose:
                    print(f"🎯 Selected: {file_path}")
                return file_path
        
        if self.verbose:
            print("❌ No valid data files found!")
        return None

def load_data_smart(file_path: str, verbose: bool = True) -> Optional[pd.DataFrame]:
    """Smart data loading that handles both regular and compressed files"""
    
    if verbose:
        print(f"📊 Loading data from: {file_path}")
    
    try:
        # Check if it's a compressed file
        if file_path.endswith('.gz'):
            if verbose:
                print("  📦 Detected compressed file, using gzip decompression...")
            data = pd.read_csv(file_path, compression='gzip')
        else:
            if verbose:
                print("  📄 Loading regular CSV file...")
            data = pd.read_csv(file_path)
        
        if verbose:
            print(f"✅ Successfully loaded {len(data):,} rows with {len(data.columns)} columns")
        return data
        
    except Exception as e:
        if verbose:
            print(f"❌ Error loading file: {e}")
        return None

def auto_find_and_load_data(custom_patterns: Optional[List[str]] = None, 
                           verbose: bool = True) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Complete auto-detection and loading workflow"""
    
    finder = DataFileFinder(verbose=verbose)
    
    # Find the best data file
    data_path = finder.find_best_data_file(custom_patterns)
    
    if not data_path:
        return None, None
    
    # Load the data
    data = load_data_smart(data_path, verbose=verbose)
    
    return data, data_path

# Convenience functions for backward compatibility
def find_latest_data_file(verbose: bool = True) -> Optional[str]:
    """Legacy function - find latest data file"""
    finder = DataFileFinder(verbose=verbose)
    return finder.find_latest_data_file()

def load_and_validate_data(file_path: Optional[str] = None, verbose: bool = True) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Load and validate data with auto-detection fallback"""
    
    if file_path:
        # Use provided file path
        data = load_data_smart(file_path, verbose=verbose)
        if data is not None:
            return data, file_path
    
    # Auto-detect if no file provided or loading failed
    if verbose:
        print("🔄 Auto-detecting data file...")
    
    return auto_find_and_load_data(verbose=verbose)

if __name__ == '__main__':
    # Test the auto-detection
    print("🧪 Testing auto-detection...")
    
    finder = DataFileFinder()
    
    # Find all files
    all_files = finder.find_all_data_files()
    print(f"\nFound {len(all_files)} data files:")
    for file_path, mtime in all_files:
        print(f"  {file_path} (modified: {datetime.fromtimestamp(mtime)})")
    
    # Find best file
    best_file = finder.find_best_data_file()
    
    if best_file:
        # Load and show summary
        data, _ = auto_find_and_load_data()
        if data is not None:
            successful = data[data['completed'] == True]
            valid_prices = successful['agreed_price'].dropna()
            
            print(f"\n📊 Data Summary:")
            print(f"   Total negotiations: {len(data):,}")
            print(f"   Success rate: {data['completed'].mean()*100:.1f}%")
            print(f"   Valid prices: {len(valid_prices):,}")
            print(f"   Average price: ${valid_prices.mean():.2f}")
    else:
        print("\n❌ No valid data files found for testing.")