#!/usr/bin/env python3
"""
Debug script to check what auto-detection is finding
"""

import glob
from pathlib import Path
from datetime import datetime
import pandas as pd

def test_auto_detection():
    """Test what the auto-detection is finding"""
    
    print("🔍 DEBUGGING AUTO-DETECTION")
    print("="*50)
    
    # Test the exact patterns used in file_finder.py
    search_patterns = [
        "./full_results/processed/complete_*.csv",
        "./full_results/processed/complete_*.csv.gz", 
        "./complete_*.csv",
        "./temp_results.csv",
        "./data/complete_*.csv",
        "./results/complete_*.csv",
        "./output/complete_*.csv"
    ]
    
    all_files = []
    
    print("🔍 Testing each search pattern:")
    for pattern in search_patterns:
        files = glob.glob(pattern)
        print(f"   Pattern: {pattern}")
        if files:
            for file in files:
                file_path = Path(file)
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"     ✅ Found: {file} ({size_mb:.1f}MB, {mtime})")
                all_files.append((file, file_path.stat().st_mtime))
        else:
            print(f"     ❌ No files found")
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total files found: {len(all_files)}")
    
    if all_files:
        # Sort by modification time (newest first)
        all_files.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n📅 Files by modification time (newest first):")
        for i, (file_path, mtime) in enumerate(all_files, 1):
            mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            print(f"   {i}. {file_path} ({mtime_str})")
        
        # Test loading the newest file
        newest_file = all_files[0][0]
        print(f"\n🧪 TESTING LOAD OF NEWEST FILE:")
        print(f"   File: {newest_file}")
        
        try:
            if newest_file.endswith('.gz'):
                print("   📦 Loading compressed file...")
                data = pd.read_csv(newest_file, compression='gzip')
            else:
                print("   📄 Loading regular CSV...")
                data = pd.read_csv(newest_file)
            
            print(f"   ✅ SUCCESS: Loaded {len(data):,} rows, {len(data.columns)} columns")
            
            # Check key columns
            key_columns = ['negotiation_id', 'buyer_model', 'supplier_model', 'completed', 'agreed_price']
            missing_cols = [col for col in key_columns if col not in data.columns]
            
            if missing_cols:
                print(f"   ⚠️  Missing columns: {missing_cols}")
            else:
                print(f"   ✅ All key columns present")
            
            # Basic stats
            if 'completed' in data.columns:
                success_rate = data['completed'].mean() * 100
                print(f"   📈 Success rate: {success_rate:.1f}%")
            
            if 'agreed_price' in data.columns:
                successful = data[data['completed'] == True] if 'completed' in data.columns else data
                valid_prices = successful['agreed_price'].dropna()
                if len(valid_prices) > 0:
                    print(f"   💰 Valid prices: {len(valid_prices):,} (avg: ${valid_prices.mean():.2f})")
            
            return newest_file, True
            
        except Exception as e:
            print(f"   ❌ FAILED to load: {e}")
            return newest_file, False
    
    else:
        print("❌ No files found with any pattern!")
        return None, False

def check_specific_files():
    """Check the specific files you mentioned"""
    
    print(f"\n🎯 CHECKING YOUR SPECIFIC FILES:")
    print("-" * 40)
    
    specific_files = [
        "./full_results/processed/complete_20250616_043831.csv.gz",
        "./full_results/processed/complete_20250615_171248.csv",
    ]
    
    for file_path in specific_files:
        path = Path(file_path)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            print(f"✅ {file_path}")
            print(f"   Size: {size_mb:.1f} MB")
            print(f"   Modified: {mtime}")
            
            # Try to load
            try:
                if file_path.endswith('.gz'):
                    data = pd.read_csv(file_path, compression='gzip')
                else:
                    data = pd.read_csv(file_path)
                print(f"   Data: {len(data):,} rows, {len(data.columns)} columns")
            except Exception as e:
                print(f"   ❌ Load error: {e}")
        else:
            print(f"❌ {file_path} - NOT FOUND")

def test_file_finder_module():
    """Test the actual file_finder module if available"""
    
    print(f"\n🔧 TESTING FILE_FINDER MODULE:")
    print("-" * 40)
    
    try:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent / "src" / "utils"))
        
        from file_finder import DataFileFinder, auto_find_and_load_data
        
        print("✅ file_finder module imported successfully")
        
        # Test the finder
        finder = DataFileFinder(verbose=True)
        latest_file = finder.find_latest_data_file()
        
        if latest_file:
            print(f"🎯 file_finder found: {latest_file}")
            
            # Test validation
            is_valid, message = finder.validate_data_file(latest_file)
            print(f"   Validation: {message}")
            
            # Test auto_find_and_load_data
            data, data_path = auto_find_and_load_data(verbose=True)
            if data is not None:
                print(f"   ✅ auto_find_and_load_data: {len(data)} rows from {data_path}")
            else:
                print(f"   ❌ auto_find_and_load_data failed")
                
        else:
            print("❌ file_finder found no files")
            
    except ImportError as e:
        print(f"❌ Could not import file_finder: {e}")
    except Exception as e:
        print(f"❌ Error testing file_finder: {e}")

def main():
    """Run all debugging tests"""
    
    # Test 1: Basic auto-detection patterns
    newest_file, load_success = test_auto_detection()
    
    # Test 2: Check your specific files
    check_specific_files()
    
    # Test 3: Test the file_finder module
    test_file_finder_module()
    
    # Summary
    print(f"\n🎯 RECOMMENDATION:")
    print("="*30)
    
    if newest_file and load_success:
        print(f"✅ Use this file: {newest_file}")
        print("   The auto-detection should work correctly")
    else:
        print("❌ Auto-detection issues found")
        print("   Recommended actions:")
        print("   1. Check if file_finder.py exists in src/utils/")
        print("   2. Verify the compressed file can be read")
        print("   3. Check file permissions")
        
        # Suggest manual path
        latest_compressed = "./full_results/processed/complete_20250616_043831.csv.gz"
        if Path(latest_compressed).exists():
            print(f"   4. Try manual path: {latest_compressed}")

if __name__ == '__main__':
    main()