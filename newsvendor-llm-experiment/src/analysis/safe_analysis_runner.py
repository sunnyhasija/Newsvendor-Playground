#!/usr/bin/env python3
"""
Quick fix for your analysis pipeline
Save this as src/analysis/safe_analysis_runner.py
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
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




logger = logging.getLogger(__name__)

class SafeAnalysisRunner:
    """Safe analysis runner that handles your data structure"""
    
    def __init__(self, data_path: str, output_dir: str = "./analysis_output", optimal_price: float = 65.0):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.optimal_price = optimal_price
        
        # Create subdirectories
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
    
    def load_and_clean_data(self) -> pd.DataFrame:
        """Load and clean your data"""
        
        print("📊 Loading data...")
        data = pd.read_csv(self.data_path)
        
        print(f"✅ Loaded {len(data)} negotiations")
        print(f"📈 Completion rate: {data['completed'].mean()*100:.1f}%")
        
        # Clean price data
        data['agreed_price'] = pd.to_numeric(data['agreed_price'], errors='coerce')
        data['total_rounds'] = pd.to_numeric(data['total_rounds'], errors='coerce')
        data['total_tokens'] = pd.to_numeric(data['total_tokens'], errors='coerce')
        
        return data
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run comprehensive analysis on your data"""
        
        data = self.load_and_clean_data()
        
        # Basic overview
        successful = data[data['completed'] == True].copy()
        valid_prices = successful['agreed_price'].dropna()
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "data_summary": {
                "total_negotiations": len(data),
                "successful_negotiations": len(successful), 
                "success_rate": len(successful) / len(data) * 100,
                "negotiations_with_prices": len(valid_prices),
                "price_completion_rate": len(valid_prices) / len(successful) * 100 if len(successful) > 0 else 0
            }
        }
        
        if len(valid_prices) == 0:
            analysis["error"] = "No valid prices found"
            return analysis
        
        # Price analysis
        analysis["price_analysis"] = {
            "count": len(valid_prices),
            "mean": float(valid_prices.mean()),
            "median": float(valid_prices.median()),
            "std": float(valid_prices.std()),
            "min": float(valid_prices.min()),
            "max": float(valid_prices.max()),
            "q25": float(valid_prices.quantile(0.25)),
            "q75": float(valid_prices.quantile(0.75)),
            "distance_from_optimal": {
                "mean": float(abs(valid_prices - self.optimal_price).mean()),
                "median": float(abs(valid_prices - self.optimal_price).median()),
                "within_5_dollars": int(sum(abs(valid_prices - self.optimal_price) <= 5)),
                "within_10_dollars": int(sum(abs(valid_prices - self.optimal_price) <= 10))
            }
        }
        
        # Efficiency analysis
        analysis["efficiency_analysis"] = {
            "mean_rounds": float(successful['total_rounds'].mean()),
            "median_rounds": float(successful['total_rounds'].median()),
            "mean_tokens": float(successful['total_tokens'].mean()),
            "median_tokens": float(successful['total_tokens'].median()),
            "tokens_per_round": float(successful['total_tokens'].sum() / successful['total_rounds'].sum())
        }
        
        # Reflection pattern analysis
        analysis["reflection_patterns"] = self._analyze_reflection_patterns(successful)
        
        # Model analysis
        analysis["model_analysis"] = self._analyze_models(successful)
        
        # Model pairing analysis
        analysis["model_pairings"] = self._analyze_model_pairings(successful)
        
        return analysis
    
    def _analyze_reflection_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze reflection patterns safely"""
        
        pattern_names = {
            '00': 'No Reflection',
            '01': 'Buyer Reflection Only', 
            '10': 'Supplier Reflection Only',
            '11': 'Both Reflection'
        }
        
        results = {}
        
        for pattern in ['00', '01', '10', '11']:
            pattern_data = data[data['reflection_pattern'] == pattern]
            pattern_prices = pattern_data['agreed_price'].dropna()
            
            if len(pattern_data) > 0:
                results[pattern] = {
                    "name": pattern_names.get(pattern, pattern),
                    "count": len(pattern_data),
                    "success_rate": len(pattern_data) / len(data[data['reflection_pattern'] == pattern]) * 100,
                    "price_stats": {
                        "count": len(pattern_prices),
                        "mean": float(pattern_prices.mean()) if len(pattern_prices) > 0 else None,
                        "std": float(pattern_prices.std()) if len(pattern_prices) > 1 else None,
                        "distance_from_optimal": float(abs(pattern_prices - self.optimal_price).mean()) if len(pattern_prices) > 0 else None
                    },
                    "efficiency": {
                        "mean_rounds": float(pattern_data['total_rounds'].mean()),
                        "mean_tokens": float(pattern_data['total_tokens'].mean())
                    }
                }
        
        # Find best patterns (safely)
        valid_patterns = [p for p in results.values() if p['price_stats']['mean'] is not None]
        
        if len(valid_patterns) > 0:
            best_price = min(valid_patterns, key=lambda x: abs(x['price_stats']['mean'] - self.optimal_price))
            most_efficient = min(valid_patterns, key=lambda x: x['efficiency']['mean_rounds'])
            
            results["summary"] = {
                "closest_to_optimal": best_price["name"],
                "most_efficient": most_efficient["name"],
                "total_patterns_analyzed": len(valid_patterns)
            }
        
        return results
    
    def _analyze_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze model performance safely"""
        
        results = {}
        all_models = set(data['buyer_model'].unique()) | set(data['supplier_model'].unique())
        
        for model in all_models:
            # As buyer
            buyer_data = data[data['buyer_model'] == model]
            buyer_prices = buyer_data['agreed_price'].dropna()
            
            # As supplier
            supplier_data = data[data['supplier_model'] == model]
            supplier_prices = supplier_data['agreed_price'].dropna()
            
            results[model] = {
                "as_buyer": {
                    "count": len(buyer_data),
                    "avg_price": float(buyer_prices.mean()) if len(buyer_prices) > 0 else None,
                    "avg_tokens": float(buyer_data['total_tokens'].mean()) if len(buyer_data) > 0 else None,
                    "avg_rounds": float(buyer_data['total_rounds'].mean()) if len(buyer_data) > 0 else None
                },
                "as_supplier": {
                    "count": len(supplier_data),
                    "avg_price": float(supplier_prices.mean()) if len(supplier_prices) > 0 else None,
                    "avg_tokens": float(supplier_data['total_tokens'].mean()) if len(supplier_data) > 0 else None,
                    "avg_rounds": float(supplier_data['total_rounds'].mean()) if len(supplier_data) > 0 else None
                }
            }
            
            # Overall performance score
            all_model_data = data[(data['buyer_model'] == model) | (data['supplier_model'] == model)]
            all_prices = all_model_data['agreed_price'].dropna()
            
            if len(all_prices) > 0:
                results[model]["overall"] = {
                    "total_negotiations": len(all_model_data),
                    "avg_price": float(all_prices.mean()),
                    "distance_from_optimal": float(abs(all_prices - self.optimal_price).mean()),
                    "efficiency_score": float(all_model_data['total_tokens'].mean())
                }
        
        return results
    
    def _analyze_model_pairings(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze model pairing effects"""
        
        # Homogeneous vs heterogeneous pairings
        data['pairing_type'] = data.apply(
            lambda row: 'homogeneous' if row['buyer_model'] == row['supplier_model'] else 'heterogeneous',
            axis=1
        )
        
        results = {}
        
        for pairing_type in ['homogeneous', 'heterogeneous']:
            type_data = data[data['pairing_type'] == pairing_type]
            type_prices = type_data['agreed_price'].dropna()
            
            if len(type_data) > 0:
                results[pairing_type] = {
                    "count": len(type_data),
                    "price_stats": {
                        "mean": float(type_prices.mean()) if len(type_prices) > 0 else None,
                        "std": float(type_prices.std()) if len(type_prices) > 1 else None,
                        "distance_from_optimal": float(abs(type_prices - self.optimal_price).mean()) if len(type_prices) > 0 else None
                    },
                    "efficiency": {
                        "mean_rounds": float(type_data['total_rounds'].mean()),
                        "mean_tokens": float(type_data['total_tokens'].mean())
                    }
                }
        
        return results
    
    def generate_report(self, analysis: Dict[str, Any]) -> str:
        """Generate readable report"""
        
        if "error" in analysis:
            return f"❌ Analysis Error: {analysis['error']}"
        
        data_summary = analysis["data_summary"]
        price_analysis = analysis["price_analysis"]
        efficiency = analysis["efficiency_analysis"]
        reflection = analysis["reflection_patterns"]
        
        report = f"""
# Newsvendor LLM Negotiation Analysis Report

**Generated:** {analysis['timestamp']}

## 📊 Data Summary
- **Total Negotiations:** {data_summary['total_negotiations']:,}
- **Successful Negotiations:** {data_summary['successful_negotiations']:,}
- **Success Rate:** {data_summary['success_rate']:.1f}%
- **Negotiations with Valid Prices:** {data_summary['negotiations_with_prices']:,}
- **Price Completion Rate:** {data_summary['price_completion_rate']:.1f}%

## 💰 Price Analysis
- **Average Price:** ${price_analysis['mean']:.2f}
- **Median Price:** ${price_analysis['median']:.2f}
- **Price Range:** ${price_analysis['min']:.2f} - ${price_analysis['max']:.2f}
- **Standard Deviation:** ${price_analysis['std']:.2f}

### Distance from Optimal (${self.optimal_price})
- **Mean Distance:** ${price_analysis['distance_from_optimal']['mean']:.2f}
- **Within $5:** {price_analysis['distance_from_optimal']['within_5_dollars']} negotiations ({price_analysis['distance_from_optimal']['within_5_dollars']/price_analysis['count']*100:.1f}%)
- **Within $10:** {price_analysis['distance_from_optimal']['within_10_dollars']} negotiations ({price_analysis['distance_from_optimal']['within_10_dollars']/price_analysis['count']*100:.1f}%)

## ⚡ Efficiency Analysis
- **Average Rounds:** {efficiency['mean_rounds']:.1f}
- **Average Tokens:** {efficiency['mean_tokens']:.0f}
- **Tokens per Round:** {efficiency['tokens_per_round']:.1f}

## 🤔 Reflection Pattern Analysis
"""
        
        if "summary" in reflection:
            report += f"""
### Summary
- **Closest to Optimal:** {reflection['summary']['closest_to_optimal']}
- **Most Efficient:** {reflection['summary']['most_efficient']}
"""
        
        for pattern_id, pattern_data in reflection.items():
            if pattern_id != "summary":
                price_stats = pattern_data['price_stats']
                efficiency_stats = pattern_data['efficiency']
                
                report += f"""
### {pattern_data['name']} ({pattern_id})
- **Count:** {pattern_data['count']} negotiations
- **Average Price:** ${price_stats['mean']:.2f if price_stats['mean'] else 'N/A'}
- **Distance from Optimal:** ${price_stats['distance_from_optimal']:.2f if price_stats['distance_from_optimal'] else 'N/A'}
- **Efficiency:** {efficiency_stats['mean_rounds']:.1f} rounds, {efficiency_stats['mean_tokens']:.0f} tokens
"""
        
        return report
    
    def save_results(self, analysis: Dict[str, Any]) -> str:
        """Save analysis results"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON
        json_file = self.output_dir / "metrics" / f"analysis_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Save report
        report = self.generate_report(analysis)
        report_file = self.output_dir / "reports" / f"report_{timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        return str(report_file)

@click.command()
@click.option('--data', default='temp_results.csv', help='Data file to analyze')
@click.option('--output', default='./analysis_output', help='Output directory')
@click.option('--optimal-price', type=float, default=65.0, help='Optimal price for analysis')
def main(data, output, optimal_price):
    """Run safe analysis on your newsvendor data"""
    
    print("🚀 Starting Safe Analysis Runner...")
    
    runner = SafeAnalysisRunner(data, output, optimal_price)
    
    try:
        analysis = runner.run_comprehensive_analysis()
        report_file = runner.save_results(analysis)
        
        print(f"✅ Analysis complete!")
        print(f"📁 Results saved to: {runner.output_dir}")
        print(f"📋 Report: {report_file}")
        
        # Print summary
        if "error" not in analysis:
            summary = analysis["data_summary"]
            price_summary = analysis["price_analysis"]
            print(f"\n📊 Quick Summary:")
            print(f"   • {summary['total_negotiations']} negotiations, {summary['success_rate']:.1f}% success")
            print(f"   • Average price: ${price_summary['mean']:.2f} (optimal: ${optimal_price})")
            print(f"   • Distance from optimal: ${price_summary['distance_from_optimal']['mean']:.2f}")
        
        print("\n" + runner.generate_report(analysis))
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        raise

if __name__ == '__main__':
    main()