#!/usr/bin/env python3
"""
Debug script to identify and fix analysis issues
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def debug_data_file(file_path):
    """Debug the input data file"""
    print(f"🔍 Debugging data file: {file_path}")
    print("=" * 60)
    
    # Check if file exists
    if not Path(file_path).exists():
        print(f"❌ File does not exist: {file_path}")
        return False
    
    try:
        # Load the data
        data = pd.read_csv(file_path)
        print(f"✅ File loaded successfully")
        print(f"📊 Shape: {data.shape}")
        print(f"📋 Columns: {list(data.columns)}")
        
        # Check if empty
        if len(data) == 0:
            print("❌ Data file is empty!")
            return False
        
        # Show first few rows
        print("\n🔎 First 5 rows:")
        print(data.head())
        
        # Check required columns
        required_columns = [
            'negotiation_id', 'buyer_model', 'supplier_model', 
            'reflection_pattern', 'completed', 'agreed_price',
            'total_rounds', 'total_tokens'
        ]
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
        else:
            print("✅ All required columns present")
        
        # Check data types and values
        print("\n📈 Data Summary:")
        print(f"• Total rows: {len(data)}")
        print(f"• Completed negotiations: {data['completed'].sum() if 'completed' in data.columns else 'N/A'}")
        print(f"• Success rate: {data['completed'].mean()*100:.1f}%" if 'completed' in data.columns else 'N/A')
        
        if 'agreed_price' in data.columns:
            successful_prices = data[data['completed'] == True]['agreed_price'].dropna()
            print(f"• Successful negotiations with prices: {len(successful_prices)}")
            if len(successful_prices) > 0:
                print(f"• Price range: ${successful_prices.min():.2f} - ${successful_prices.max():.2f}")
            else:
                print("❌ No successful negotiations with valid prices!")
        
        # Check for null values
        print("\n🔍 Null values:")
        null_counts = data.isnull().sum()
        for col, count in null_counts.items():
            if count > 0:
                print(f"  {col}: {count} nulls")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return False

def create_sample_data(output_path="temp_results_sample.csv"):
    """Create sample data for testing"""
    print(f"\n🔧 Creating sample data: {output_path}")
    
    # Sample data matching expected structure
    sample_data = []
    
    models = ["tinyllama:latest", "qwen2:1.5b", "gemma2:2b", "phi3:mini"]
    patterns = ["00", "01", "10", "11"]
    
    for i in range(20):  # Create 20 sample negotiations
        buyer_model = np.random.choice(models)
        supplier_model = np.random.choice(models)
        pattern = np.random.choice(patterns)
        completed = np.random.choice([True, False], p=[0.8, 0.2])  # 80% success rate
        
        if completed:
            agreed_price = np.random.randint(45, 85)  # Reasonable price range
            total_rounds = np.random.randint(3, 8)
            total_tokens = np.random.randint(200, 800)
        else:
            agreed_price = None
            total_rounds = np.random.randint(8, 10)
            total_tokens = np.random.randint(300, 600)
        
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
            'distance_from_optimal': abs(agreed_price - 65) if agreed_price else None
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(sample_data)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Sample data created with {len(df)} rows")
    print(f"📊 Success rate: {df['completed'].mean()*100:.1f}%")
    print(f"💰 Price range: ${df['agreed_price'].min():.2f} - ${df['agreed_price'].max():.2f}")
    
    return output_path

def fix_analysis_runner():
    """Create a fixed version of the analysis runner"""
    
    fixed_code = '''
"""
Fixed Complete Analysis Runner
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class FixedCompleteAnalysisRunner:
    """Fixed analysis runner with better error handling."""
    
    def __init__(self, 
                 data_path: str = "temp_results.csv",
                 output_dir: str = "./analysis_output",
                 optimal_price: float = 65.0):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.optimal_price = optimal_price
        
        # Create subdirectories
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        
    def load_and_validate_data(self) -> pd.DataFrame:
        """Load and validate data with better error handling."""
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        data = pd.read_csv(self.data_path)
        
        if len(data) == 0:
            raise ValueError("Data file is empty")
        
        # Validate required columns
        required_columns = [
            'negotiation_id', 'buyer_model', 'supplier_model', 
            'reflection_pattern', 'completed', 'agreed_price',
            'total_rounds', 'total_tokens'
        ]
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert data types safely
        data['completed'] = data['completed'].astype(bool)
        data['agreed_price'] = pd.to_numeric(data['agreed_price'], errors='coerce')
        data['total_rounds'] = pd.to_numeric(data['total_rounds'], errors='coerce')
        data['total_tokens'] = pd.to_numeric(data['total_tokens'], errors='coerce')
        
        logger.info(f"Loaded {len(data)} negotiations")
        return data
    
    def run_safe_analysis(self) -> Dict[str, Any]:
        """Run analysis with comprehensive error handling."""
        
        try:
            # Load data
            data = self.load_and_validate_data()
            
            # Filter successful negotiations
            successful = data[data['completed'] == True].copy()
            
            if len(successful) == 0:
                return {
                    "error": "No successful negotiations found",
                    "total_negotiations": len(data),
                    "success_rate": 0.0
                }
            
            # Get valid prices (not null)
            valid_prices = successful['agreed_price'].dropna()
            
            if len(valid_prices) == 0:
                return {
                    "error": "No valid prices found in successful negotiations",
                    "total_negotiations": len(data),
                    "successful_negotiations": len(successful),
                    "success_rate": len(successful) / len(data)
                }
            
            # Basic metrics
            analysis = {
                "experiment_overview": {
                    "total_negotiations": len(data),
                    "successful_negotiations": len(successful),
                    "success_rate": len(successful) / len(data),
                    "negotiations_with_valid_prices": len(valid_prices)
                },
                "price_analysis": {
                    "mean_price": float(valid_prices.mean()),
                    "median_price": float(valid_prices.median()),
                    "std_price": float(valid_prices.std()),
                    "min_price": float(valid_prices.min()),
                    "max_price": float(valid_prices.max()),
                    "distance_from_optimal": float(abs(valid_prices - self.optimal_price).mean())
                },
                "efficiency_analysis": {
                    "mean_rounds": float(successful['total_rounds'].mean()),
                    "mean_tokens": float(successful['total_tokens'].mean()),
                    "tokens_per_round": float(successful['total_tokens'].sum() / successful['total_rounds'].sum())
                }
            }
            
            # Reflection pattern analysis
            reflection_analysis = []
            for pattern in ['00', '01', '10', '11']:
                pattern_data = successful[successful['reflection_pattern'] == pattern]
                pattern_prices = pattern_data['agreed_price'].dropna()
                
                if len(pattern_prices) > 0:
                    reflection_analysis.append({
                        "pattern": pattern,
                        "count": len(pattern_data),
                        "success_rate": len(pattern_data) / len(data[data['reflection_pattern'] == pattern]),
                        "avg_price": float(pattern_prices.mean()),
                        "avg_rounds": float(pattern_data['total_rounds'].mean()),
                        "distance_from_optimal": float(abs(pattern_prices - self.optimal_price).mean())
                    })
            
            analysis["reflection_analysis"] = reflection_analysis
            
            # Model analysis
            model_analysis = []
            all_models = set(data['buyer_model'].unique()) | set(data['supplier_model'].unique())
            
            for model in all_models:
                # As buyer
                buyer_data = successful[successful['buyer_model'] == model]
                buyer_prices = buyer_data['agreed_price'].dropna()
                
                # As supplier  
                supplier_data = successful[successful['supplier_model'] == model]
                supplier_prices = supplier_data['agreed_price'].dropna()
                
                model_analysis.append({
                    "model_name": model,
                    "as_buyer": {
                        "count": len(buyer_data),
                        "avg_price": float(buyer_prices.mean()) if len(buyer_prices) > 0 else None,
                        "avg_tokens": float(buyer_data['total_tokens'].mean()) if len(buyer_data) > 0 else None
                    },
                    "as_supplier": {
                        "count": len(supplier_data),
                        "avg_price": float(supplier_prices.mean()) if len(supplier_prices) > 0 else None,
                        "avg_tokens": float(supplier_data['total_tokens'].mean()) if len(supplier_data) > 0 else None
                    }
                })
            
            analysis["model_analysis"] = model_analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {"error": str(e)}
    
    def save_analysis(self, analysis: Dict[str, Any]) -> str:
        """Save analysis results."""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / "metrics" / f"analysis_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        return str(output_file)
    
    def generate_report(self, analysis: Dict[str, Any]) -> str:
        """Generate a readable report."""
        
        if "error" in analysis:
            return f"""
# Analysis Report - ERROR

**Error:** {analysis['error']}

**Data Status:**
- Total negotiations: {analysis.get('total_negotiations', 'Unknown')}
- Success rate: {analysis.get('success_rate', 0)*100:.1f}%

**Recommendation:** Check your data file and ensure it contains completed negotiations with valid prices.
"""
        
        overview = analysis.get('experiment_overview', {})
        price_analysis = analysis.get('price_analysis', {})
        efficiency = analysis.get('efficiency_analysis', {})
        
        report = f"""
# Newsvendor LLM Negotiation Analysis Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Experiment Overview
- **Total Negotiations:** {overview.get('total_negotiations', 0):,}
- **Successful Negotiations:** {overview.get('successful_negotiations', 0):,}
- **Success Rate:** {overview.get('success_rate', 0)*100:.1f}%
- **Valid Prices:** {overview.get('negotiations_with_valid_prices', 0):,}

## Price Analysis
- **Average Price:** ${price_analysis.get('mean_price', 0):.2f}
- **Median Price:** ${price_analysis.get('median_price', 0):.2f}
- **Price Range:** ${price_analysis.get('min_price', 0):.2f} - ${price_analysis.get('max_price', 0):.2f}
- **Distance from Optimal (${self.optimal_price}):** ${price_analysis.get('distance_from_optimal', 0):.2f}

## Efficiency Analysis
- **Average Rounds:** {efficiency.get('mean_rounds', 0):.1f}
- **Average Tokens:** {efficiency.get('mean_tokens', 0):.0f}
- **Tokens per Round:** {efficiency.get('tokens_per_round', 0):.1f}

## Reflection Pattern Analysis
"""
        
        for pattern_data in analysis.get('reflection_analysis', []):
            pattern_name = {
                '00': 'No Reflection',
                '01': 'Buyer Reflection',
                '10': 'Supplier Reflection', 
                '11': 'Both Reflection'
            }.get(pattern_data['pattern'], pattern_data['pattern'])
            
            report += f"""
### {pattern_name} ({pattern_data['pattern']})
- Count: {pattern_data['count']}
- Success Rate: {pattern_data['success_rate']*100:.1f}%
- Average Price: ${pattern_data['avg_price']:.2f}
- Average Rounds: {pattern_data['avg_rounds']:.1f}
- Distance from Optimal: ${pattern_data['distance_from_optimal']:.2f}
"""
        
        return report

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='temp_results.csv')
    parser.add_argument('--output', default='./analysis_results')
    parser.add_argument('--optimal-price', type=float, default=65.0)
    
    args = parser.parse_args()
    
    runner = FixedCompleteAnalysisRunner(args.data, args.output, args.optimal_price)
    
    print("🔍 Running fixed analysis...")
    analysis = runner.run_safe_analysis()
    
    # Save results
    analysis_file = runner.save_analysis(analysis)
    print(f"📊 Analysis saved to: {analysis_file}")
    
    # Generate report
    report = runner.generate_report(analysis)
    report_file = runner.output_dir / "reports" / "analysis_report.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"📋 Report saved to: {report_file}")
    print("\\n" + "="*50)
    print(report)

if __name__ == '__main__':
    main()
'''
    
    with open('fixed_analysis_runner.py', 'w') as f:
        f.write(fixed_code)
    
    print("✅ Created fixed_analysis_runner.py")
    return 'fixed_analysis_runner.py'

if __name__ == '__main__':
    import sys
    
    print("🔧 Newsvendor Analysis Debugger")
    print("=" * 50)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = "temp_results.csv"
    
    # Debug the data file
    data_ok = debug_data_file(data_file)
    
    if not data_ok:
        print("\\n🔧 Creating sample data for testing...")
        sample_file = create_sample_data()
        print(f"✅ Sample data created: {sample_file}")
        print("\\n🧪 Testing with sample data:")
        debug_data_file(sample_file)
    
    # Create fixed analysis runner
    print("\\n🔧 Creating fixed analysis runner...")
    fixed_runner = fix_analysis_runner()
    
    print(f"""
🎯 Next Steps:

1. If your data file is empty/missing, use the sample data:
   python fixed_analysis_runner.py --data temp_results_sample.csv

2. If your data file has issues, fix them based on the debug output above

3. Run the fixed analysis:
   python fixed_analysis_runner.py --data {data_file}

4. The fixed version has better error handling and will show you exactly what's wrong
""")