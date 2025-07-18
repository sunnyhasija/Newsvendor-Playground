#!/usr/bin/env python3
"""
Debug runner to check data availability and run analysis
"""

import json
from pathlib import Path
import sys

def check_data_files():
    """Check for available data files."""
    print("🔍 Checking for data files...")
    
    # Common locations to check
    locations_to_check = [
        "./experiment_results",
        "./results", 
        "./data",
        "./",
        "../experiment_results",
        "../results",
        "../data"
    ]
    
    found_files = []
    
    for location in locations_to_check:
        location_path = Path(location)
        if location_path.exists():
            print(f"✅ Directory exists: {location}")
            
            # Check for JSON files
            json_files = list(location_path.glob("*.json"))
            result_files = list(location_path.glob("*result*.json"))
            complete_files = list(location_path.glob("complete_results_*.json"))
            
            if json_files:
                print(f"   📄 JSON files found: {len(json_files)}")
                for file in json_files[:5]:  # Show first 5
                    print(f"      - {file.name}")
                if len(json_files) > 5:
                    print(f"      ... and {len(json_files) - 5} more")
                
                found_files.extend(json_files)
            
            if result_files:
                print(f"   🎯 Result files found: {len(result_files)}")
                for file in result_files:
                    print(f"      - {file.name}")
            
            if complete_files:
                print(f"   ✨ Complete result files found: {len(complete_files)}")
                for file in complete_files:
                    print(f"      - {file.name}")
        else:
            print(f"❌ Directory not found: {location}")
    
    return found_files

def check_file_format(file_path):
    """Check if a file has the expected format."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"\n📊 Analyzing file: {file_path.name}")
        print(f"   File size: {file_path.stat().st_size / 1024:.1f} KB")
        
        # Check structure
        if isinstance(data, dict):
            print(f"   Structure: Dictionary with keys: {list(data.keys())}")
            
            if 'results' in data:
                results = data['results']
                print(f"   Results: {len(results)} items")
                if len(results) > 0:
                    sample = results[0]
                    print(f"   Sample keys: {list(sample.keys()) if isinstance(sample, dict) else 'Not a dict'}")
                    return True
            else:
                print(f"   No 'results' key found")
                
        elif isinstance(data, list):
            print(f"   Structure: List with {len(data)} items")
            if len(data) > 0:
                sample = data[0]
                print(f"   Sample keys: {list(sample.keys()) if isinstance(sample, dict) else 'Not a dict'}")
                return True
        
        return False
        
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
        return False

def main():
    print("🐛 LLM Negotiation Analysis - Debug Mode")
    print("=" * 50)
    
    # Check current directory
    current_dir = Path(".")
    print(f"📁 Current directory: {current_dir.absolute()}")
    
    # List contents
    print(f"\n📋 Directory contents:")
    for item in sorted(current_dir.iterdir()):
        if item.is_dir():
            print(f"   📁 {item.name}/")
        else:
            print(f"   📄 {item.name}")
    
    # Check for data files
    found_files = check_data_files()
    
    if not found_files:
        print("\n❌ No data files found!")
        print("\n💡 Expected file locations:")
        print("   - ./experiment_results/complete_results_*.json")
        print("   - ./results/*.json")
        print("   - Any JSON file with negotiation results")
        
        print("\n🔧 To create test data, you could:")
        print("   1. Run your negotiation experiment first")
        print("   2. Or provide a path to existing results")
        print("   3. Or create a sample results file")
        return
    
    print(f"\n✅ Found {len(found_files)} potential data files")
    
    # Check the most promising files
    valid_files = []
    for file_path in found_files:
        if check_file_format(file_path):
            valid_files.append(file_path)
    
    if not valid_files:
        print("\n❌ No valid data files found!")
        return
    
    print(f"\n🎯 Found {len(valid_files)} valid data files")
    
    # Choose the best file (most recent complete_results file)
    complete_results = [f for f in valid_files if 'complete_results' in f.name]
    if complete_results:
        best_file = max(complete_results, key=lambda f: f.stat().st_mtime)
    else:
        best_file = max(valid_files, key=lambda f: f.stat().st_mtime)
    
    print(f"\n🚀 Will use: {best_file}")
    
    # Now try to run the analysis
    try:
        # Import the analyzer
        sys.path.append('.')
        from analysis.consolidated_analysis_runner import UnifiedLLMNegotiationAnalyzer
        
        print(f"\n🔬 Running analysis...")
        analyzer = UnifiedLLMNegotiationAnalyzer()
        
        # Manually set the results file
        analyzer.results_file = best_file
        
        results = analyzer.run_complete_analysis()
        
        if results and 'analysis_directory' in results:
            print(f"\n🎉 Analysis completed successfully!")
            print(f"📁 Results saved to: {results['analysis_directory']}")
        else:
            print(f"\n❌ Analysis failed - no results returned")
            
    except ImportError as e:
        print(f"\n❌ Could not import analyzer: {e}")
        print("   Make sure the consolidated_analysis_runner.py file is in the analysis/ directory")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()