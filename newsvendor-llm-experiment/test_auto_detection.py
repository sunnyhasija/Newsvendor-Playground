#!/usr/bin/env python3
"""
Test Auto-Detection Functionality
Run this script to verify auto-detection works across all your analysis scripts
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

def test_file_finder_module():
    """Test the core file_finder module"""
    print("🧪 Testing file_finder module...")
    
    try:
        # Test import
        sys.path.append(str(Path("src/utils")))
        from file_finder import DataFileFinder, auto_find_and_load_data
        
        print("✅ Import successful")
        
        # Test file detection
        finder = DataFileFinder(verbose=False)
        latest_file = finder.find_latest_data_file()
        
        if latest_file:
            print(f"✅ Auto-detection found: {latest_file}")
            
            # Test validation
            is_valid, message = finder.validate_data_file(latest_file)
            print(f"✅ Validation: {message}")
            
            # Test loading
            data, data_path = auto_find_and_load_data(verbose=False)
            if data is not None:
                print(f"✅ Data loading: {len(data)} rows from {data_path}")
                return True
            else:
                print("❌ Data loading failed")
                return False
        else:
            print("❌ No data files found")
            return False
            
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_script_auto_detection(script_path: str, script_name: str):
    """Test auto-detection in a specific script"""
    print(f"\n🔍 Testing {script_name}...")
    
    if not Path(script_path).exists():
        print(f"⚠️  Script not found: {script_path}")
        return False
    
    try:
        # Run the script with a timeout
        result = subprocess.run(
            [sys.executable, script_path, "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print(f"✅ {script_name} runs successfully")
            return True
        else:
            print(f"⚠️  {script_name} returned code {result.returncode}")
            if result.stderr:
                print(f"   Error: {result.stderr[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⚠️  {script_name} timed out")
        return False
    except FileNotFoundError:
        print(f"❌ Python not found or script missing")
        return False
    except Exception as e:
        print(f"❌ {script_name} test failed: {e}")
        return False

def test_data_file_detection():
    """Test that we can find valid data files"""
    print("\n📂 Testing data file detection...")
    
    # Common patterns for newsvendor data
    patterns = [
        "./full_results/processed/complete_*.csv",
        "./complete_*.csv", 
        "./temp_results.csv",
        "./data/complete_*.csv"
    ]
    
    import glob
    found_files = []
    
    for pattern in patterns:
        files = glob.glob(pattern)
        found_files.extend(files)
    
    if found_files:
        print(f"✅ Found {len(found_files)} potential data files:")
        for file in found_files[:5]:  # Show first 5
            file_path = Path(file)
            size_mb = file_path.stat().st_size / (1024 * 1024)
            mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            print(f"   📄 {file} ({size_mb:.1f}MB, modified: {mod_time})")
        
        if len(found_files) > 5:
            print(f"   ... and {len(found_files) - 5} more")
        
        return True
    else:
        print("❌ No data files found in common locations")
        print("💡 Try creating sample data first:")
        print("   python src/utils/debug_data.py --create-sample")
        return False

def run_integration_test():
    """Run a complete integration test"""
    print("\n🔗 Integration Test - Complete Analysis Flow")
    print("-" * 50)
    
    try:
        # Import the auto-detection module
        sys.path.append(str(Path("src/utils")))
        from file_finder import auto_find_and_load_data
        
        # Test the complete flow
        print("1. Auto-detecting data file...")
        data, data_path = auto_find_and_load_data()
        
        if data is None:
            print("❌ No valid data found")
            return False
        
        print(f"✅ Loaded {len(data)} negotiations from {data_path}")
        
        # Basic analysis
        print("2. Testing basic analysis...")
        successful = data[data['completed'] == True]
        valid_prices = successful['agreed_price'].dropna()
        
        print(f"✅ Analysis results:")
        print(f"   - Success rate: {data['completed'].mean()*100:.1f}%")
        print(f"   - Valid prices: {len(valid_prices)}")
        print(f"   - Average price: ${valid_prices.mean():.2f}")
        
        # Test reflection patterns
        if 'reflection_pattern' in data.columns:
            print("3. Testing reflection pattern analysis...")
            patterns = data['reflection_pattern'].value_counts()
            print(f"✅ Reflection patterns: {len(patterns)} unique")
            
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all auto-detection tests"""
    
    print("🚀 AUTO-DETECTION TEST SUITE")
    print("="*60)
    print(f"Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test results
    results = []
    
    # 1. Test data file detection
    results.append(("Data File Detection", test_data_file_detection()))
    
    # 2. Test core module
    results.append(("File Finder Module", test_file_finder_module()))
    
    # 3. Test integration
    results.append(("Integration Test", run_integration_test()))
    
    # 4. Test individual scripts (if they exist)
    scripts_to_test = [
        ("debug_reflection.py", "Debug Reflection"),
        ("src/analysis/conversation_analyzer.py", "Conversation Analyzer"),
        ("src/utils/debug_data.py", "Debug Data"),
    ]
    
    for script_path, script_name in scripts_to_test:
        if Path(script_path).exists():
            results.append((script_name, test_script_auto_detection(script_path, script_name)))
    
    # Summary
    print(f"\n📊 TEST SUMMARY")
    print("="*30)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Auto-detection is working correctly.")
        print("\n💡 Ready to use:")
        print("   python debug_reflection.py")
        print("   python final_comprehensive_analysis.py")
        print("   python src/analysis/conversation_analyzer.py")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        print("\n🔧 Troubleshooting:")
        print("   1. Ensure file_finder.py exists in src/utils/")
        print("   2. Run the bulk update script: python bulk_update_analysis.py")
        print("   3. Create sample data: python src/utils/debug_data.py --create-sample")

if __name__ == '__main__':
    main()