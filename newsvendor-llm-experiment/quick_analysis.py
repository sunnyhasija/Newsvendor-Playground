#!/usr/bin/env python3
"""
Quick analysis scripts for your newsvendor results
Save as quick_analysis.py
"""

import pandas as pd
import json
import glob
from scipy import stats
import numpy as np

def analyze_reflection_patterns():
    """Analyze reflection pattern effects"""
    print("🤔 REFLECTION PATTERN ANALYSIS")
    print("=" * 50)
    
    # Load analysis results
    json_files = glob.glob('analysis_results/metrics/analysis_*.json')
    if json_files:
        with open(json_files[0], 'r') as f:
            data = json.load(f)
        
        print("Average Prices by Reflection Pattern:")
        patterns = []
        for pattern, stats in data['reflection_patterns'].items():
            if pattern != 'summary' and stats['price_stats']['mean']:
                price = stats['price_stats']['mean']
                count = stats['price_stats']['count']
                distance = stats['price_stats']['distance_from_optimal']
                rounds = stats['efficiency']['mean_rounds']
                tokens = stats['efficiency']['mean_tokens']
                
                patterns.append({
                    'pattern': pattern,
                    'name': stats['name'],
                    'price': price,
                    'count': count,
                    'distance': distance,
                    'rounds': rounds,
                    'tokens': tokens
                })
                
                print(f"  {stats['name']:<25} ${price:>6.2f} (n={count:>3}, dist=${distance:>5.2f})")
        
        if len(patterns) > 1:
            best_price = min(patterns, key=lambda x: abs(x['price'] - 65))
            closest_optimal = min(patterns, key=lambda x: x['distance'])
            most_efficient = min(patterns, key=lambda x: x['rounds'])
            
            print(f"\n📊 Key Findings:")
            print(f"  Best Price (closest to $65): {best_price['name']} (${best_price['price']:.2f})")
            print(f"  Closest to Optimal: {closest_optimal['name']} (${closest_optimal['distance']:.2f} away)")
            print(f"  Most Efficient: {most_efficient['name']} ({most_efficient['rounds']:.1f} rounds)")
        
        return patterns
    else:
        print("❌ No analysis results found")
        return []

def analyze_models():
    """Analyze model performance"""
    print("\n🤖 MODEL PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    data = pd.read_csv('temp_results.csv')
    
    print("As Buyer (wanting lower prices):")
    buyer_perf = data.groupby('buyer_model')['agreed_price'].agg(['mean', 'count', 'std']).round(2)
    buyer_perf = buyer_perf.sort_values('mean')  # Lower is better for buyers
    for model, row in buyer_perf.iterrows():
        print(f"  {model:<20} ${row['mean']:>6.2f} ±{row['std']:>5.2f} (n={row['count']:>3})")
    
    print(f"\nAs Supplier (wanting higher prices):")
    supplier_perf = data.groupby('supplier_model')['agreed_price'].agg(['mean', 'count', 'std']).round(2)
    supplier_perf = supplier_perf.sort_values('mean', ascending=False)  # Higher is better for suppliers
    for model, row in supplier_perf.iterrows():
        print(f"  {model:<20} ${row['mean']:>6.2f} ±{row['std']:>5.2f} (n={row['count']:>3})")
    
    return buyer_perf, supplier_perf

def test_reflection_significance():
    """Test statistical significance of reflection effects"""
    print("\n📈 STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 50)
    
    data = pd.read_csv('temp_results.csv')
    
    # Compare all reflection patterns
    patterns = ['00', '01', '10', '11']
    pattern_names = {
        '00': 'No Reflection',
        '01': 'Buyer Only', 
        '10': 'Supplier Only',
        '11': 'Both Reflection'
    }
    
    pattern_data = {}
    for pattern in patterns:
        prices = data[data['reflection_pattern'] == pattern]['agreed_price'].dropna()
        if len(prices) > 0:
            pattern_data[pattern] = prices
    
    print("Reflection Pattern Comparison:")
    for pattern, prices in pattern_data.items():
        print(f"  {pattern_names[pattern]:<15} Mean: ${prices.mean():>6.2f}, N: {len(prices):>3}")
    
    # Test specific comparisons
    if '00' in pattern_data and '11' in pattern_data:
        no_refl = pattern_data['00']
        both_refl = pattern_data['11']
        
        t_stat, p_val = stats.ttest_ind(no_refl, both_refl)
        effect_size = (both_refl.mean() - no_refl.mean()) / np.sqrt((no_refl.var() + both_refl.var()) / 2)
        
        print(f"\n🔬 No Reflection vs Both Reflection:")
        print(f"  Mean difference: ${both_refl.mean() - no_refl.mean():+.2f}")
        print(f"  T-statistic: {t_stat:.3f}")
        print(f"  P-value: {p_val:.4f}")
        print(f"  Effect size (Cohen's d): {effect_size:.3f}")
        print(f"  Significant: {'✅ Yes' if p_val < 0.05 else '❌ No'}")
    
    # ANOVA for all patterns
    if len(pattern_data) > 2:
        all_prices = list(pattern_data.values())
        f_stat, p_val = stats.f_oneway(*all_prices)
        print(f"\n🔬 ANOVA (all patterns):")
        print(f"  F-statistic: {f_stat:.3f}")
        print(f"  P-value: {p_val:.4f}")
        print(f"  Significant: {'✅ Yes' if p_val < 0.05 else '❌ No'}")

def analyze_extreme_outcomes():
    """Analyze extreme price outcomes"""
    print("\n⚠️  EXTREME OUTCOMES ANALYSIS")
    print("=" * 50)
    
    data = pd.read_csv('temp_results.csv')
    prices = data['agreed_price'].dropna()
    
    # Define extreme outcomes
    very_low = prices[prices <= 10]
    very_high = prices[prices >= 150]
    optimal_range = prices[(prices >= 60) & (prices <= 70)]
    
    print(f"Price Distribution:")
    print(f"  Very Low (≤$10): {len(very_low)} ({len(very_low)/len(prices)*100:.1f}%)")
    print(f"  Optimal Range ($60-70): {len(optimal_range)} ({len(optimal_range)/len(prices)*100:.1f}%)")
    print(f"  Very High (≥$150): {len(very_high)} ({len(very_high)/len(prices)*100:.1f}%)")
    
    if len(very_low) > 0:
        print(f"\n🔍 Very Low Prices (≤$10):")
        low_analysis = data[data['agreed_price'] <= 10]
        print(f"  Common buyer models: {low_analysis['buyer_model'].value_counts().head(3).to_dict()}")
        print(f"  Common reflection patterns: {low_analysis['reflection_pattern'].value_counts().head(3).to_dict()}")
    
    if len(very_high) > 0:
        print(f"\n🔍 Very High Prices (≥$150):")
        high_analysis = data[data['agreed_price'] >= 150]
        print(f"  Common supplier models: {high_analysis['supplier_model'].value_counts().head(3).to_dict()}")
        print(f"  Common reflection patterns: {high_analysis['reflection_pattern'].value_counts().head(3).to_dict()}")

def analyze_model_pairings():
    """Analyze homogeneous vs heterogeneous model pairings"""
    print("\n🤝 MODEL PAIRING ANALYSIS")
    print("=" * 50)
    
    data = pd.read_csv('temp_results.csv')
    
    # Create pairing type
    data['pairing_type'] = data.apply(
        lambda row: 'Homogeneous' if row['buyer_model'] == row['supplier_model'] else 'Heterogeneous',
        axis=1
    )
    
    pairing_stats = data.groupby('pairing_type')['agreed_price'].agg(['mean', 'count', 'std']).round(2)
    
    print("Pairing Performance:")
    for pairing_type, row in pairing_stats.iterrows():
        print(f"  {pairing_type:<15} ${row['mean']:>6.2f} ±{row['std']:>5.2f} (n={row['count']:>3})")
    
    # Test significance
    homo_prices = data[data['pairing_type'] == 'Homogeneous']['agreed_price'].dropna()
    hetero_prices = data[data['pairing_type'] == 'Heterogeneous']['agreed_price'].dropna()
    
    if len(homo_prices) > 0 and len(hetero_prices) > 0:
        t_stat, p_val = stats.ttest_ind(homo_prices, hetero_prices)
        print(f"\n🔬 Homogeneous vs Heterogeneous:")
        print(f"  Mean difference: ${hetero_prices.mean() - homo_prices.mean():+.2f}")
        print(f"  P-value: {p_val:.4f}")
        print(f"  Significant: {'✅ Yes' if p_val < 0.05 else '❌ No'}")

def main():
    """Run all analyses"""
    print("🎯 NEWSVENDOR LLM EXPERIMENT - DETAILED ANALYSIS")
    print("=" * 60)
    
    try:
        patterns = analyze_reflection_patterns()
        buyer_perf, supplier_perf = analyze_models()
        test_reflection_significance()
        analyze_extreme_outcomes() 
        analyze_model_pairings()
        
        print(f"\n🎉 SUMMARY INSIGHTS")
        print("=" * 30)
        print(f"✅ Analysis complete! Key findings:")
        print(f"   • Systematic buyer advantage (avg ${pd.read_csv('temp_results.csv')['agreed_price'].mean():.2f} vs optimal $65)")
        print(f"   • {len(patterns)} reflection patterns analyzed")
        print(f"   • High efficiency: {pd.read_csv('temp_results.csv')['total_rounds'].mean():.1f} rounds average")
        print(f"   • Check individual sections above for detailed insights")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure temp_results.csv and analysis results exist")

if __name__ == '__main__':
    main()