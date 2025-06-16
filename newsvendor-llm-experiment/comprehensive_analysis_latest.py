#!/usr/bin/env python3
"""
Auto-Detecting Comprehensive Analysis for Latest Newsvendor Data
Automatically finds and analyzes your latest experiment results
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def find_latest_data_file():
    """Find the most recent data file from your experiment"""
    
    # Look for CSV files in the processed directory
    csv_patterns = [
        "./full_results/processed/complete_*.csv",
        "./full_results/processed/complete_*.csv.gz",
        "./complete_*.csv",
        "./temp_results.csv"
    ]
    
    latest_file = None
    latest_time = 0
    
    print("🔍 Searching for data files...")
    
    for pattern in csv_patterns:
        files = glob.glob(pattern)
        for file in files:
            file_path = Path(file)
            if file_path.exists():
                mtime = file_path.stat().st_mtime
                if mtime > latest_time:
                    latest_time = mtime
                    latest_file = file_path
                    
                print(f"  Found: {file} (modified: {datetime.fromtimestamp(mtime)})")
    
    if latest_file:
        print(f"✅ Using latest file: {latest_file}")
        return str(latest_file)
    else:
        print("❌ No data files found!")
        return None

def load_data_smart(file_path):
    """Smart data loading that handles both regular and compressed files"""
    
    print(f"📊 Loading data from: {file_path}")
    
    try:
        # Check if it's a compressed file
        if file_path.endswith('.gz'):
            print("  📦 Detected compressed file, using gzip decompression...")
            data = pd.read_csv(file_path, compression='gzip')
        else:
            print("  📄 Loading regular CSV file...")
            data = pd.read_csv(file_path)
        
        print(f"✅ Successfully loaded {len(data):,} rows with {len(data.columns)} columns")
        return data
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None

def run_comprehensive_analysis():
    """Complete analysis of your latest newsvendor data"""
    
    print("🎯 COMPREHENSIVE NEWSVENDOR LLM ANALYSIS")
    print("="*60)
    
    # Find and load the latest data
    file_path = find_latest_data_file()
    if not file_path:
        return
    
    data = load_data_smart(file_path)
    if data is None:
        return
    
    # Basic data overview
    print(f"\n📋 DATASET OVERVIEW")
    print(f"   Total negotiations: {len(data):,}")
    print(f"   Columns: {list(data.columns)}")
    
    # Success rate analysis
    if 'completed' in data.columns:
        success_rate = data['completed'].mean() * 100
        successful_count = data['completed'].sum()
        print(f"   Success rate: {success_rate:.1f}% ({successful_count:,} successful)")
    
    # Filter to successful negotiations with valid prices
    successful = data[data['completed'] == True].copy()
    valid_prices = successful['agreed_price'].dropna()
    
    if len(valid_prices) == 0:
        print("❌ No successful negotiations with valid prices found!")
        return
    
    print(f"   Valid prices: {len(valid_prices):,} negotiations")
    print(f"   Price range: ${valid_prices.min():.2f} - ${valid_prices.max():.2f}")
    print(f"   Average price: ${valid_prices.mean():.2f} (optimal: $65)")
    
    # === HYPOTHESIS TESTING ===
    print(f"\n🧪 HYPOTHESIS TESTING")
    print("="*50)
    
    # H1: Reflection Benefits Analysis
    analyze_reflection_benefits(successful)
    
    # H3: Role Asymmetry Analysis  
    analyze_role_asymmetry(valid_prices)
    
    # === MODEL PERFORMANCE ANALYSIS ===
    print(f"\n🤖 MODEL PERFORMANCE ANALYSIS")
    print("="*50)
    
    analyze_model_performance(successful)
    
    # === EFFICIENCY ANALYSIS ===
    print(f"\n⚡ EFFICIENCY ANALYSIS")
    print("="*50)
    
    analyze_efficiency(successful)
    
    # === EXTREME OUTCOMES ===
    print(f"\n⚠️ EXTREME OUTCOMES ANALYSIS")
    print("="*50)
    
    analyze_extreme_outcomes(valid_prices, successful)
    
    # === VISUALIZATIONS ===
    print(f"\n📊 CREATING VISUALIZATIONS...")
    create_comprehensive_visualizations(successful, valid_prices)
    
    # === SAVE RESULTS ===
    save_analysis_results(successful, valid_prices, file_path)
    
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print("="*30)

def analyze_reflection_benefits(data):
    """Analyze H1: Reflection Benefits"""
    
    print("1️⃣ H1: REFLECTION BENEFITS ANALYSIS")
    
    # Normalize reflection patterns
    if 'reflection_pattern' not in data.columns:
        print("   ❌ No reflection_pattern column found")
        return
    
    # Convert to string and normalize
    data['reflection_pattern'] = data['reflection_pattern'].astype(str).str.zfill(2)
    
    pattern_names = {
        '00': 'No Reflection',
        '01': 'Buyer Reflection',
        '10': 'Supplier Reflection', 
        '11': 'Both Reflection'
    }
    
    print("   📊 Price Analysis by Reflection Pattern:")
    reflection_results = {}
    
    for pattern in ['00', '01', '10', '11']:
        pattern_data = data[data['reflection_pattern'] == pattern]
        pattern_prices = pattern_data['agreed_price'].dropna()
        
        if len(pattern_prices) > 0:
            avg_price = pattern_prices.mean()
            distance = abs(pattern_prices - 65).mean()
            efficiency = pattern_data['total_rounds'].mean()
            
            reflection_results[pattern] = {
                'count': len(pattern_data),
                'avg_price': avg_price,
                'distance_from_optimal': distance,
                'avg_rounds': efficiency
            }
            
            print(f"     {pattern_names[pattern]:<20} "
                  f"${avg_price:>6.2f} (n={len(pattern_data):>3}, "
                  f"dist=${distance:>5.2f}, rounds={efficiency:>4.1f})")
    
    # Statistical significance test
    if len(reflection_results) >= 2:
        print("\n   🔬 Statistical Significance Tests:")
        
        # Compare no reflection vs both reflection
        if '00' in reflection_results and '11' in reflection_results:
            no_refl = data[data['reflection_pattern'] == '00']['agreed_price'].dropna()
            both_refl = data[data['reflection_pattern'] == '11']['agreed_price'].dropna()
            
            if len(no_refl) > 0 and len(both_refl) > 0:
                t_stat, p_val = stats.ttest_ind(no_refl, both_refl)
                effect_size = (both_refl.mean() - no_refl.mean()) / np.sqrt((no_refl.var() + both_refl.var()) / 2)
                
                print(f"     No Reflection vs Both Reflection:")
                print(f"       Price difference: ${both_refl.mean() - no_refl.mean():+.2f}")
                print(f"       T-statistic: {t_stat:.3f}, P-value: {p_val:.4f}")
                print(f"       Effect size (Cohen's d): {effect_size:.3f}")
                print(f"       Result: {'✅ SIGNIFICANT' if p_val < 0.05 else '❌ NOT SIGNIFICANT'}")
        
        # ANOVA for all patterns
        groups = [data[data['reflection_pattern'] == p]['agreed_price'].dropna() 
                 for p in reflection_results.keys()]
        groups = [g for g in groups if len(g) > 0]
        
        if len(groups) > 2:
            f_stat, p_val = stats.f_oneway(*groups)
            print(f"\n     Overall ANOVA (all patterns):")
            print(f"       F-statistic: {f_stat:.3f}, P-value: {p_val:.4f}")
            print(f"       Result: {'✅ SIGNIFICANT DIFFERENCES' if p_val < 0.05 else '❌ NO SIGNIFICANT DIFFERENCES'}")

def analyze_role_asymmetry(prices):
    """Analyze H3: Role Asymmetry"""
    
    print("\n3️⃣ H3: ROLE ASYMMETRY ANALYSIS")
    
    # Calculate buyer advantage (lower prices = better for buyers)
    buyer_advantage = 65 - prices.mean()  # Positive = buyer advantage
    
    print(f"   Average price: ${prices.mean():.2f}")
    print(f"   Optimal price: $65.00")
    print(f"   Buyer advantage: ${buyer_advantage:+.2f}")
    
    # One-sample t-test against optimal price
    t_stat, p_val = stats.ttest_1samp(prices, 65)
    
    print(f"\n   🔬 Statistical Test vs Optimal Price ($65):")
    print(f"     T-statistic: {t_stat:.3f}")
    print(f"     P-value: {p_val:.6f}")
    
    if p_val < 0.001:
        if prices.mean() < 65:
            print(f"     Result: ✅ STRONG BUYER BIAS CONFIRMED")
        else:
            print(f"     Result: ✅ STRONG SUPPLIER BIAS CONFIRMED")
    elif p_val < 0.05:
        print(f"     Result: ⚠️ MODERATE BIAS DETECTED")
    else:
        print(f"     Result: ❌ NO SIGNIFICANT BIAS")
    
    # Price distribution analysis
    very_low = sum(prices <= 50)
    low = sum((prices > 50) & (prices <= 60))
    optimal = sum((prices > 60) & (prices <= 70))
    high = sum((prices > 70) & (prices <= 80))
    very_high = sum(prices > 80)
    
    total = len(prices)
    print(f"\n   📊 Price Distribution:")
    print(f"     Very Low (≤$50):    {very_low:>4} ({very_low/total*100:>5.1f}%)")
    print(f"     Low ($50-60):       {low:>4} ({low/total*100:>5.1f}%)")
    print(f"     Optimal ($60-70):   {optimal:>4} ({optimal/total*100:>5.1f}%)")
    print(f"     High ($70-80):      {high:>4} ({high/total*100:>5.1f}%)")
    print(f"     Very High (>$80):   {very_high:>4} ({very_high/total*100:>5.1f}%)")

def analyze_model_performance(data):
    """Analyze model performance"""
    
    print("📊 Model Performance Rankings:")
    
    models = sorted(data['buyer_model'].unique())
    
    print("\n   As BUYER (lower prices = better performance):")
    buyer_performance = []
    
    for model in models:
        model_data = data[data['buyer_model'] == model]
        prices = model_data['agreed_price'].dropna()
        
        if len(prices) > 0:
            buyer_performance.append({
                'model': model,
                'count': len(model_data),
                'avg_price': prices.mean(),
                'distance_from_optimal': abs(prices - 65).mean()
            })
    
    # Sort by performance (closer to optimal = better)
    buyer_performance.sort(key=lambda x: x['distance_from_optimal'])
    
    for i, perf in enumerate(buyer_performance[:5], 1):
        print(f"     {i}. {perf['model']:<20} ${perf['avg_price']:>6.2f} "
              f"(n={perf['count']:>3}, dist=${perf['distance_from_optimal']:>5.2f})")
    
    print("\n   As SUPPLIER (higher prices = better performance):")
    supplier_performance = []
    
    for model in models:
        model_data = data[data['supplier_model'] == model]
        prices = model_data['agreed_price'].dropna()
        
        if len(prices) > 0:
            supplier_performance.append({
                'model': model,
                'count': len(model_data),
                'avg_price': prices.mean(),
                'distance_from_optimal': abs(prices - 65).mean()
            })
    
    # Sort by performance (closer to optimal = better for overall analysis)
    supplier_performance.sort(key=lambda x: x['distance_from_optimal'])
    
    for i, perf in enumerate(supplier_performance[:5], 1):
        print(f"     {i}. {perf['model']:<20} ${perf['avg_price']:>6.2f} "
              f"(n={perf['count']:>3}, dist=${perf['distance_from_optimal']:>5.2f})")

def analyze_efficiency(data):
    """Analyze negotiation efficiency"""
    
    print("📈 Efficiency Metrics:")
    
    print(f"   Average rounds: {data['total_rounds'].mean():.1f}")
    print(f"   Average tokens: {data['total_tokens'].mean():.0f}")
    print(f"   Tokens per round: {data['total_tokens'].sum() / data['total_rounds'].sum():.1f}")
    
    # Efficiency by reflection pattern
    if 'reflection_pattern' in data.columns:
        print(f"\n   📊 Efficiency by Reflection Pattern:")
        
        data['reflection_pattern'] = data['reflection_pattern'].astype(str).str.zfill(2)
        
        for pattern in ['00', '01', '10', '11']:
            pattern_data = data[data['reflection_pattern'] == pattern]
            if len(pattern_data) > 0:
                avg_rounds = pattern_data['total_rounds'].mean()
                avg_tokens = pattern_data['total_tokens'].mean()
                print(f"     {pattern}: {avg_rounds:>4.1f} rounds, {avg_tokens:>4.0f} tokens")

def analyze_extreme_outcomes(prices, data):
    """Analyze extreme price outcomes"""
    
    very_low = prices[prices <= 40]
    very_high = prices[prices >= 90]
    
    print(f"📊 Extreme Outcomes:")
    print(f"   Very low prices (≤$40): {len(very_low)} negotiations ({len(very_low)/len(prices)*100:.1f}%)")
    print(f"   Very high prices (≥$90): {len(very_high)} negotiations ({len(very_high)/len(prices)*100:.1f}%)")
    
    if len(very_low) > 0:
        print(f"   Lowest price: ${very_low.min():.2f}")
    
    if len(very_high) > 0:
        print(f"   Highest price: ${very_high.max():.2f}")

def create_comprehensive_visualizations(data, prices):
    """Create key visualizations"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Newsvendor LLM Negotiation Analysis - Key Results', fontsize=16, fontweight='bold')
    
    # 1. Price distribution by reflection pattern
    ax1 = axes[0, 0]
    if 'reflection_pattern' in data.columns:
        data['reflection_pattern'] = data['reflection_pattern'].astype(str).str.zfill(2)
        sns.boxplot(data=data, x='reflection_pattern', y='agreed_price', ax=ax1)
        ax1.axhline(y=65, color='red', linestyle='--', alpha=0.7, label='Optimal ($65)')
        ax1.set_title('Price by Reflection Pattern')
        ax1.set_xlabel('Reflection Pattern')
        ax1.set_ylabel('Agreed Price ($)')
        ax1.legend()
    
    # 2. Price distribution histogram
    ax2 = axes[0, 1]
    prices.hist(bins=30, ax=ax2, alpha=0.7, color='lightblue')
    ax2.axvline(x=65, color='red', linestyle='--', alpha=0.7, label='Optimal ($65)')
    ax2.axvline(x=prices.mean(), color='orange', linestyle='-', alpha=0.7, label=f'Mean (${prices.mean():.2f})')
    ax2.set_title('Price Distribution')
    ax2.set_xlabel('Agreed Price ($)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    
    # 3. Model performance as buyer
    ax3 = axes[0, 2]
    buyer_perf = data.groupby('buyer_model')['agreed_price'].mean().sort_values()
    buyer_perf.plot(kind='bar', ax=ax3, color='lightcoral')
    ax3.axhline(y=65, color='red', linestyle='--', alpha=0.7)
    ax3.set_title('Average Price by Buyer Model')
    ax3.set_xlabel('Buyer Model')
    ax3.set_ylabel('Average Price ($)')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Efficiency analysis
    ax4 = axes[1, 0]
    data['total_rounds'].hist(bins=range(1, 12), ax=ax4, alpha=0.7, color='lightgreen')
    ax4.axvline(x=data['total_rounds'].mean(), color='red', linestyle='--', alpha=0.7, 
               label=f'Mean: {data["total_rounds"].mean():.1f}')
    ax4.set_title('Distribution of Negotiation Length')
    ax4.set_xlabel('Total Rounds')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    
    # 5. Token efficiency
    ax5 = axes[1, 1]
    ax5.scatter(data['total_rounds'], data['total_tokens'], alpha=0.6)
    ax5.set_title('Tokens vs Rounds')
    ax5.set_xlabel('Total Rounds')
    ax5.set_ylabel('Total Tokens')
    
    # Add trend line
    z = np.polyfit(data['total_rounds'], data['total_tokens'], 1)
    p = np.poly1d(z)
    ax5.plot(data['total_rounds'], p(data['total_rounds']), "r--", alpha=0.8)
    
    # 6. Model pairing analysis
    ax6 = axes[1, 2]
    data['pairing_type'] = data.apply(
        lambda row: 'Homogeneous' if row['buyer_model'] == row['supplier_model'] else 'Heterogeneous',
        axis=1
    )
    sns.boxplot(data=data, x='pairing_type', y='agreed_price', ax=ax6)
    ax6.axhline(y=65, color='red', linestyle='--', alpha=0.7)
    ax6.set_title('Homogeneous vs Heterogeneous Pairings')
    ax6.set_xlabel('Pairing Type')
    ax6.set_ylabel('Agreed Price ($)')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = f"comprehensive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   💾 Visualizations saved to: {output_file}")
    plt.show()

def save_analysis_results(data, prices, original_file):
    """Save analysis results"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create summary results
    results = {
        'analysis_timestamp': timestamp,
        'source_file': str(original_file),
        'dataset_overview': {
            'total_negotiations': len(data),
            'valid_prices': len(prices),
            'success_rate': len(data) / len(data) * 100,  # All data here is successful
            'average_price': float(prices.mean()),
            'median_price': float(prices.median()),
            'price_std': float(prices.std()),
            'min_price': float(prices.min()),
            'max_price': float(prices.max()),
            'buyer_advantage': float(65 - prices.mean()),
            'distance_from_optimal': float(abs(prices - 65).mean())
        },
        'efficiency_metrics': {
            'average_rounds': float(data['total_rounds'].mean()),
            'average_tokens': float(data['total_tokens'].mean()),
            'tokens_per_round': float(data['total_tokens'].sum() / data['total_rounds'].sum())
        }
    }
    
    # Add reflection analysis if available
    if 'reflection_pattern' in data.columns:
        data['reflection_pattern'] = data['reflection_pattern'].astype(str).str.zfill(2)
        reflection_analysis = {}
        
        pattern_names = {
            '00': 'No Reflection',
            '01': 'Buyer Reflection',
            '10': 'Supplier Reflection', 
            '11': 'Both Reflection'
        }
        
        for pattern in ['00', '01', '10', '11']:
            pattern_data = data[data['reflection_pattern'] == pattern]
            if len(pattern_data) > 0:
                pattern_prices = pattern_data['agreed_price'].dropna()
                reflection_analysis[pattern] = {
                    'name': pattern_names[pattern],
                    'count': len(pattern_data),
                    'avg_price': float(pattern_prices.mean()),
                    'distance_from_optimal': float(abs(pattern_prices - 65).mean()),
                    'avg_rounds': float(pattern_data['total_rounds'].mean())
                }
        
        results['reflection_analysis'] = reflection_analysis
    
    # Save results
    output_file = f"analysis_results_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"   💾 Analysis results saved to: {output_file}")

if __name__ == '__main__':
    run_comprehensive_analysis()