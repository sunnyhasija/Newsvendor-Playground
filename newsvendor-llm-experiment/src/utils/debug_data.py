#!/usr/bin/env python3
"""
src/utils/debug_data.py - Debug data issues for Poetry workflow
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import click

# Auto-detection imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src" / "utils"))
sys.path.append(str(Path(__file__).parent.parent / "utils"))

try:
    from file_finder import auto_find_and_load_data, DataFileFinder, load_data_smart
    AUTO_DETECTION_AVAILABLE = True
except ImportError:
    print("⚠️  Auto-detection module not found, using fallback...")
    AUTO_DETECTION_AVAILABLE = False
    
    def auto_find_and_load_data():
        # Fallback - try common paths
        common_paths = [
            "./full_results/processed/complete_20250615_171248.csv",
            "./temp_results.csv",
            "./complete_*.csv"
        ]
        for path in common_paths:
            try:
                import pandas as pd
                return pd.read_csv(path), path
            except:
                continue
        return None, None




@click.command()
@click.option('--data', default='temp_results.csv', help='Data file to debug')
@click.option('--create-sample', is_flag=True, help='Create sample data for testing')
def debug_data(data: str, create_sample: bool):
    """Debug data file issues"""
    
    print(f"🔍 Debugging data file: {data}")
    print("=" * 60)
    
    data_path = Path(data)
    
    # Check if file exists
    if not data_path.exists():
        print(f"❌ File does not exist: {data}")
        if create_sample:
            create_sample_data(data)
        return
    
    try:
        # Load the data
        df = pd.read_csv(data_path)
        print(f"✅ File loaded successfully")
        print(f"📊 Shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Check if empty
        if len(df) == 0:
            print("❌ Data file is empty!")
            if create_sample:
                create_sample_data(data)
            return
        
        # Show first few rows
        print("\n🔎 First 5 rows:")
        print(df.head())
        
        # Check required columns
        required_columns = [
            'negotiation_id', 'buyer_model', 'supplier_model', 
            'reflection_pattern', 'completed', 'agreed_price',
            'total_rounds', 'total_tokens'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return
        else:
            print("✅ All required columns present")
        
        # Check data summary
        print(f"\n📈 Data Summary:")
        print(f"• Total rows: {len(df)}")
        
        # Check completed negotiations
        if 'completed' in df.columns:
            completed_count = df['completed'].sum()
            print(f"• Completed negotiations: {completed_count}")
            print(f"• Success rate: {df['completed'].mean()*100:.1f}%")
            
            if completed_count == 0:
                print("❌ No successful negotiations found!")
                if create_sample:
                    create_sample_data(data)
                return
        
        # Check prices
        if 'agreed_price' in df.columns:
            successful_df = df[df['completed'] == True]
            valid_prices = successful_df['agreed_price'].dropna()
            
            print(f"• Successful negotiations with valid prices: {len(valid_prices)}")
            
            if len(valid_prices) == 0:
                print("❌ No valid prices found in successful negotiations!")
                if create_sample:
                    create_sample_data(data)
                return
            else:
                print(f"• Price range: ${valid_prices.min():.2f} - ${valid_prices.max():.2f}")
                print(f"• Average price: ${valid_prices.mean():.2f}")
        
        # Check for null values
        print("\n🔍 Null values per column:")
        null_counts = df.isnull().sum()
        for col, count in null_counts.items():
            if count > 0:
                print(f"  {col}: {count} nulls ({count/len(df)*100:.1f}%)")
        
        print("\n✅ Data file looks good for analysis!")
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        if create_sample:
            create_sample_data(data)

def create_sample_data(output_path: str = "temp_results_sample.csv"):
    """Create sample data for testing"""
    print(f"\n🔧 Creating sample data: {output_path}")
    
    # Sample data matching expected structure
    np.random.seed(42)  # For reproducible results
    sample_data = []
    
    models = ["tinyllama:latest", "qwen2:1.5b", "gemma2:2b", "phi3:mini"]
    patterns = ["00", "01", "10", "11"]
    
    for i in range(50):  # Create 50 sample negotiations
        buyer_model = np.random.choice(models)
        supplier_model = np.random.choice(models)
        pattern = np.random.choice(patterns)
        completed = np.random.choice([True, False], p=[0.85, 0.15])  # 85% success rate
        
        if completed:
            # Create realistic price negotiations around optimal $65
            if pattern == "00":  # No reflection - less optimal
                agreed_price = int(np.random.normal(67, 8))
            elif pattern == "11":  # Both reflection - more optimal
                agreed_price = int(np.random.normal(65, 5))
            else:  # Partial reflection
                agreed_price = int(np.random.normal(66, 6))
            
            # Ensure reasonable range
            agreed_price = max(35, min(agreed_price, 95))
            
            total_rounds = np.random.randint(3, 8)
            total_tokens = np.random.randint(200, 800)
        else:
            agreed_price = None
            total_rounds = np.random.randint(8, 10)  # Failed negotiations take longer
            total_tokens = np.random.randint(400, 1000)
        
        sample_data.append({
            'negotiation_id': f'neg_{i:03d}',
            'buyer_model': buyer_model,
            'supplier_model': supplier_model,
            'reflection_pattern': pattern,
            'completed': completed,
            'agreed_price': agreed_price,
            'total_rounds': total_rounds,
            'total_tokens': total_tokens,
            'buyer_profit': (100 - agreed_price) * 40 if agreed_price else None,
            'supplier_profit': (agreed_price - 30) * 40 if agreed_price else None,
            'distance_from_optimal': abs(agreed_price - 65) if agreed_price else None,
            'termination_type': 'acceptance' if completed else 'timeout'
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(sample_data)
    df.to_csv(output_path, index=False)
    
    successful_prices = df[df['completed'] == True]['agreed_price'].dropna()
    
    print(f"✅ Sample data created with {len(df)} rows")
    print(f"📊 Success rate: {df['completed'].mean()*100:.1f}%")
    print(f"💰 Price range: ${successful_prices.min():.2f} - ${successful_prices.max():.2f}")
    print(f"🎯 Average price: ${successful_prices.mean():.2f} (optimal: $65)")
    
    # Show reflection pattern breakdown
    print("\n📋 Reflection patterns:")
    pattern_names = {'00': 'No Reflection', '01': 'Buyer Only', '10': 'Supplier Only', '11': 'Both'}
    for pattern in ['00', '01', '10', '11']:
        pattern_data = df[df['reflection_pattern'] == pattern]
        success_rate = pattern_data['completed'].mean()
        count = len(pattern_data)
        print(f"  {pattern} ({pattern_names[pattern]}): {count} negotiations, {success_rate*100:.1f}% success")

if __name__ == '__main__':
    debug_data()