#!/usr/bin/env python3
"""
Comprehensive Statistical Analysis for Newsvendor LLM Experiment
Run this on your complete_20250615_171248.csv file
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import itertools

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




def load_and_prepare_data(file_path="./full_results/processed/complete_20250615_171248.csv"):
    """Load and prepare data for analysis"""
    print("📊 Loading data...")
    data = pd.read_csv(file_path)
    
    # Filter successful negotiations with valid prices
    successful = data[data['completed'] == True].copy()
    valid_data = successful[successful['agreed_price'].notna()].copy()
    
    print(f"✅ Loaded {len(data)} total negotiations")
    print(f"✅ {len(valid_data)} successful negotiations with valid prices")
    
    return data, valid_data

def hypothesis_testing_suite(data):
    """Comprehensive hypothesis testing"""
    print("\n🧪 HYPOTHESIS TESTING SUITE")
    print("="*60)
    
    results = {}
    
    # H1: Reflection Benefits Analysis
    print("\n1️⃣ H1: REFLECTION BENEFITS")
    print("-"*40)
    
    # Convert reflection patterns to string format for consistency
    data['reflection_pattern_str'] = data['reflection_pattern'].astype(str).str.zfill(2)
    
    reflection_groups = {}
    for pattern in ['00', '01', '10', '11']:
        prices = data[data['reflection_pattern_str'] == pattern]['agreed_price'].dropna()
        if len(prices) > 0:
            reflection_groups[pattern] = prices
            print(f"   Pattern {pattern}: n={len(prices)}, mean=${prices.mean():.2f}, std=${prices.std():.2f}")
        else:
            print(f"   Pattern {pattern}: n=0 (no data)")
    
    # ANOVA for reflection patterns
    if len(reflection_groups) > 2:
        groups_list = list(reflection_groups.values())
        f_stat, p_val = stats.f_oneway(*groups_list)
        
        # Effect size (eta-squared)
        grand_mean = np.mean(np.concatenate(groups_list))
        ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in groups_list)
        ss_total = sum(sum((x - grand_mean)**2 for x in group) for group in groups_list)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        print(f"\n   🔬 ANOVA Results:")
        print(f"      F-statistic: {f_stat:.3f}")
        print(f"      P-value: {p_val:.6f}")
        print(f"      Effect size (η²): {eta_squared:.4f}")
        print(f"      Significant: {'✅ YES' if p_val < 0.05 else '❌ NO'}")
        
        results['reflection_anova'] = {
            'f_stat': f_stat, 'p_val': p_val, 'eta_squared': eta_squared,
            'significant': p_val < 0.05
        }
        
        # Post-hoc pairwise comparisons
        print(f"\n   📋 Pairwise Comparisons (Welch's t-test):")
        pairwise_results = {}
        
        pattern_names = {'00': 'None', '01': 'Buyer', '10': 'Supplier', '11': 'Both'}
        
        for p1, p2 in itertools.combinations(reflection_groups.keys(), 2):
            if p1 in reflection_groups and p2 in reflection_groups:
                group1, group2 = reflection_groups[p1], reflection_groups[p2]
                
                # Welch's t-test (unequal variances)
                t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
                
                # Cohen's d effect size
                pooled_std = np.sqrt((group1.var() + group2.var()) / 2)
                cohens_d = (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0
                
                print(f"      {pattern_names[p1]} vs {pattern_names[p2]}: "
                      f"t={t_stat:.3f}, p={p_val:.4f}, d={cohens_d:.3f}")
                
                pairwise_results[f"{p1}_vs_{p2}"] = {
                    't_stat': t_stat, 'p_val': p_val, 'cohens_d': cohens_d
                }
        
        results['reflection_pairwise'] = pairwise_results
    
    # H2: Model Size Effects
    print(f"\n2️⃣ H2: MODEL SIZE EFFECTS")
    print("-"*40)
    
    # Define model tiers
    model_tiers = {
        'tinyllama:latest': 'Ultra', 'qwen2:1.5b': 'Ultra',
        'gemma2:2b': 'Compact', 'phi3:mini': 'Compact', 'llama3.2:latest': 'Compact',
        'mistral:instruct': 'Mid', 'qwen:7b': 'Mid',
        'qwen3:latest': 'Large'
    }
    
    # Add tier information
    data['buyer_tier'] = data['buyer_model'].map(model_tiers)
    data['supplier_tier'] = data['supplier_model'].map(model_tiers)
    
    # Test buyer tier effects
    tier_groups = {}
    for tier in ['Ultra', 'Compact', 'Mid', 'Large']:
        prices = data[data['buyer_tier'] == tier]['agreed_price'].dropna()
        if len(prices) > 5:  # Minimum sample size
            tier_groups[tier] = prices
            print(f"   {tier} as Buyer: n={len(prices)}, mean=${prices.mean():.2f}")
    
    if len(tier_groups) > 2:
        f_stat, p_val = stats.f_oneway(*tier_groups.values())
        print(f"\n   🔬 Buyer Tier ANOVA: F={f_stat:.3f}, p={p_val:.4f}")
        results['buyer_tier_anova'] = {'f_stat': f_stat, 'p_val': p_val}
    
    # H3: Role Asymmetry (Buyer vs Supplier Advantage)
    print(f"\n3️⃣ H3: ROLE ASYMMETRY")
    print("-"*40)
    
    optimal_price = 65
    buyer_advantage = optimal_price - data['agreed_price'].dropna()
    
    # One-sample t-test against zero (no bias)
    t_stat, p_val = stats.ttest_1samp(buyer_advantage, 0)
    
    print(f"   Buyer Advantage: ${buyer_advantage.mean():.2f} (positive = buyer wins)")
    print(f"   One-sample t-test vs 0: t={t_stat:.3f}, p={p_val:.6f}")
    print(f"   Effect size: {buyer_advantage.mean() / buyer_advantage.std():.3f}")
    print(f"   Strong buyer bias: {'✅ CONFIRMED' if p_val < 0.001 and buyer_advantage.mean() > 5 else '❌ NOT CONFIRMED'}")
    
    results['role_asymmetry'] = {
        'buyer_advantage_mean': buyer_advantage.mean(),
        't_stat': t_stat, 'p_val': p_val,
        'strong_bias': p_val < 0.001 and buyer_advantage.mean() > 5
    }
    
    return results

def efficiency_analysis(data):
    """Analyze negotiation efficiency"""
    print("\n⚡ EFFICIENCY ANALYSIS")
    print("="*40)
    
    # Efficiency by reflection pattern
    print("\n📊 Efficiency by Reflection Pattern:")
    data['reflection_pattern_str'] = data['reflection_pattern'].astype(str).str.zfill(2)
    for pattern in ['00', '01', '10', '11']:
        pattern_data = data[data['reflection_pattern_str'] == pattern]
        if len(pattern_data) > 0:
            print(f"   Pattern {pattern}: "
                  f"Rounds={pattern_data['total_rounds'].mean():.1f}, "
                  f"Tokens={pattern_data['total_tokens'].mean():.0f}")
    
    # Correlation analysis
    print(f"\n📈 Correlation Analysis:")
    correlations = {}
    
    # Price vs Efficiency correlations
    price_data = data['agreed_price'].dropna()
    rounds_data = data.loc[price_data.index, 'total_rounds']
    tokens_data = data.loc[price_data.index, 'total_tokens']
    
    corr_price_rounds, p_val = stats.pearsonr(price_data, rounds_data)
    corr_price_tokens, p_val2 = stats.pearsonr(price_data, tokens_data)
    
    print(f"   Price vs Rounds: r={corr_price_rounds:.3f}, p={p_val:.4f}")
    print(f"   Price vs Tokens: r={corr_price_tokens:.3f}, p={p_val2:.4f}")
    
    correlations['price_rounds'] = {'r': corr_price_rounds, 'p': p_val}
    correlations['price_tokens'] = {'r': corr_price_tokens, 'p': p_val2}
    
    return correlations

def model_performance_deep_dive(data):
    """Detailed model performance analysis"""
    print("\n🤖 MODEL PERFORMANCE DEEP DIVE")
    print("="*50)
    
    models = sorted(data['buyer_model'].unique())
    
    # Performance matrix
    performance_matrix = []
    
    for model in models:
        # As buyer performance
        buyer_data = data[data['buyer_model'] == model]
        buyer_prices = buyer_data['agreed_price'].dropna()
        
        # As supplier performance  
        supplier_data = data[data['supplier_model'] == model]
        supplier_prices = supplier_data['agreed_price'].dropna()
        
        if len(buyer_prices) > 0 and len(supplier_prices) > 0:
            # T-test comparing same model as buyer vs supplier
            t_stat, p_val = stats.ttest_ind(buyer_prices, supplier_prices, equal_var=False)
            
            performance_matrix.append({
                'model': model,
                'buyer_mean': buyer_prices.mean(),
                'supplier_mean': supplier_prices.mean(),
                'role_difference': supplier_prices.mean() - buyer_prices.mean(),
                't_stat': t_stat,
                'p_val': p_val,
                'significant_role_effect': p_val < 0.05
            })
    
    # Display results
    print("\n📋 Model Role Performance (Supplier Mean - Buyer Mean):")
    for result in sorted(performance_matrix, key=lambda x: x['role_difference'], reverse=True):
        print(f"   {result['model']:<20} "
              f"Δ=${result['role_difference']:>+6.2f} "
              f"(t={result['t_stat']:>6.3f}, p={result['p_val']:>6.4f}) "
              f"{'✅' if result['significant_role_effect'] else '❌'}")
    
    return performance_matrix

def advanced_statistical_tests(data):
    """Advanced statistical analyses"""
    print("\n🔬 ADVANCED STATISTICAL TESTS")
    print("="*50)
    
    results = {}
    
    # 1. Multiple Regression Analysis
    print("\n1️⃣ Multiple Regression: Price ~ Reflection + Model Tiers")
    
    # Prepare data for regression
    reg_data = data.dropna(subset=['agreed_price']).copy()
    
    # Create dummy variables
    reg_data['reflection_pattern_str'] = reg_data['reflection_pattern'].astype(str).str.zfill(2)
    reg_data['reflection_01'] = (reg_data['reflection_pattern_str'] == '01').astype(int)
    reg_data['reflection_10'] = (reg_data['reflection_pattern_str'] == '10').astype(int) 
    reg_data['reflection_11'] = (reg_data['reflection_pattern_str'] == '11').astype(int)
    
    # Model tier dummies (buyer)
    reg_data['buyer_compact'] = (reg_data['buyer_tier'] == 'Compact').astype(int)
    reg_data['buyer_mid'] = (reg_data['buyer_tier'] == 'Mid').astype(int)
    reg_data['buyer_large'] = (reg_data['buyer_tier'] == 'Large').astype(int)
    
    try:
        # Fit regression model
        model_formula = '''agreed_price ~ reflection_01 + reflection_10 + reflection_11 + 
                          buyer_compact + buyer_mid + buyer_large + total_rounds'''
        
        regression_model = ols(model_formula, data=reg_data).fit()
        
        print(f"   R²: {regression_model.rsquared:.4f}")
        print(f"   Adjusted R²: {regression_model.rsquared_adj:.4f}")
        print(f"   F-statistic: {regression_model.fvalue:.3f}")
        print(f"   p-value: {regression_model.f_pvalue:.6f}")
        
        # Significant coefficients
        print(f"\n   📊 Significant Coefficients (p < 0.05):")
        for param, p_val in regression_model.pvalues.items():
            if p_val < 0.05:
                coef = regression_model.params[param]
                print(f"      {param}: β={coef:.3f}, p={p_val:.4f}")
        
        results['regression'] = {
            'r_squared': regression_model.rsquared,
            'f_stat': regression_model.fvalue,
            'p_val': regression_model.f_pvalue
        }
        
    except Exception as e:
        print(f"   ❌ Regression failed: {e}")
    
    # 2. Non-parametric Tests
    print(f"\n2️⃣ Non-parametric Tests")
    
    # Kruskal-Wallis test for reflection patterns
    data['reflection_pattern_str'] = data['reflection_pattern'].astype(str).str.zfill(2)
    reflection_groups = [data[data['reflection_pattern_str'] == p]['agreed_price'].dropna() 
                        for p in ['00', '01', '10', '11']]
    reflection_groups = [g for g in reflection_groups if len(g) > 0]
    
    if len(reflection_groups) > 2:
        h_stat, p_val = stats.kruskal(*reflection_groups)
        print(f"   Kruskal-Wallis (reflection): H={h_stat:.3f}, p={p_val:.4f}")
        results['kruskal_wallis'] = {'h_stat': h_stat, 'p_val': p_val}
    
    # 3. Effect Size Analysis
    print(f"\n3️⃣ Effect Size Summary")
    
    # Calculate effect sizes for key comparisons
    data['reflection_pattern_str'] = data['reflection_pattern'].astype(str).str.zfill(2)
    no_refl = data[data['reflection_pattern_str'] == '00']['agreed_price'].dropna()
    both_refl = data[data['reflection_pattern_str'] == '11']['agreed_price'].dropna()
    
    if len(no_refl) > 0 and len(both_refl) > 0:
        # Cohen's d
        pooled_std = np.sqrt((no_refl.var() + both_refl.var()) / 2)
        cohens_d = (both_refl.mean() - no_refl.mean()) / pooled_std
        
        # Interpret effect size
        if abs(cohens_d) < 0.2:
            interpretation = "negligible"
        elif abs(cohens_d) < 0.5:
            interpretation = "small"
        elif abs(cohens_d) < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        print(f"   No Reflection vs Both Reflection:")
        print(f"      Cohen's d: {cohens_d:.3f} ({interpretation} effect)")
        print(f"      Mean difference: ${both_refl.mean() - no_refl.mean():.2f}")
        
        results['effect_size'] = {'cohens_d': cohens_d, 'interpretation': interpretation}
    
    return results

def create_statistical_visualizations(data):
    """Create comprehensive statistical visualizations"""
    print("\n📊 Creating Statistical Visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Statistical Analysis Results', fontsize=16, fontweight='bold')
    
    # 1. Box plot: Price by reflection pattern
    ax1 = axes[0, 0]
    
    # Convert reflection patterns to string format
    data['reflection_pattern_str'] = data['reflection_pattern'].astype(str).str.zfill(2)
    
    reflection_data = []
    labels = []
    for pattern in ['00', '01', '10', '11']:
        prices = data[data['reflection_pattern_str'] == pattern]['agreed_price'].dropna()
        if len(prices) > 0:
            reflection_data.append(prices)
            labels.append(f"{pattern}\n(n={len(prices)})")
    
    if reflection_data:  # Only plot if we have data
        bp = ax1.boxplot(reflection_data)
        ax1.set_xticklabels(labels)
    ax1.axhline(y=65, color='red', linestyle='--', alpha=0.7, label='Optimal ($65)')
    ax1.set_title('Price Distribution by Reflection Pattern')
    ax1.set_ylabel('Agreed Price ($)')
    ax1.legend()
    
    # 2. Scatter: Price vs Rounds with regression line
    ax2 = axes[0, 1]
    price_data = data['agreed_price'].dropna()
    rounds_data = data.loc[price_data.index, 'total_rounds']
    
    ax2.scatter(rounds_data, price_data, alpha=0.6)
    
    # Add regression line
    z = np.polyfit(rounds_data, price_data, 1)
    p = np.poly1d(z)
    ax2.plot(rounds_data, p(rounds_data), "r--", alpha=0.8)
    
    # Calculate and display correlation
    corr, p_val = stats.pearsonr(price_data, rounds_data)
    ax2.set_title(f'Price vs Rounds (r={corr:.3f}, p={p_val:.3f})')
    ax2.set_xlabel('Total Rounds')
    ax2.set_ylabel('Agreed Price ($)')
    
    # 3. Model performance heatmap
    ax3 = axes[0, 2]
    
    # Create model performance matrix
    models = sorted(data['buyer_model'].unique())
    perf_matrix = np.zeros((len(models), len(models)))
    
    for i, buyer in enumerate(models):
        for j, supplier in enumerate(models):
            pair_data = data[(data['buyer_model'] == buyer) & 
                           (data['supplier_model'] == supplier)]
            prices = pair_data['agreed_price'].dropna()
            if len(prices) > 0:
                perf_matrix[i, j] = prices.mean()
            else:
                perf_matrix[i, j] = np.nan
    
    im = ax3.imshow(perf_matrix, cmap='RdYlBu_r', aspect='auto')
    ax3.set_title('Model Pairing Performance Matrix')
    ax3.set_xlabel('Supplier Model')
    ax3.set_ylabel('Buyer Model')
    
    # Set ticks and labels
    ax3.set_xticks(range(len(models)))
    ax3.set_yticks(range(len(models))) 
    ax3.set_xticklabels([m.split(':')[0] for m in models], rotation=45, ha='right')
    ax3.set_yticklabels([m.split(':')[0] for m in models])
    
    plt.colorbar(im, ax=ax3, label='Avg Price ($)')
    
    # 4. Efficiency distribution
    ax4 = axes[1, 0]
    ax4.hist(data['total_rounds'], bins=range(1, 12), alpha=0.7, color='skyblue')
    ax4.set_title('Distribution of Negotiation Length')
    ax4.set_xlabel('Total Rounds')
    ax4.set_ylabel('Frequency')
    ax4.axvline(x=data['total_rounds'].mean(), color='red', linestyle='--', 
               label=f'Mean: {data["total_rounds"].mean():.1f}')
    ax4.legend()
    
    # 5. Q-Q plot for normality check
    ax5 = axes[1, 1]
    stats.probplot(data['agreed_price'].dropna(), dist="norm", plot=ax5)
    ax5.set_title('Q-Q Plot: Price Normality Check')
    
    # 6. Effect sizes visualization
    ax6 = axes[1, 2]
    
    # Calculate effect sizes for reflection comparisons
    data['reflection_pattern_str'] = data['reflection_pattern'].astype(str).str.zfill(2)
    baseline = data[data['reflection_pattern_str'] == '00']['agreed_price'].dropna()
    effect_sizes = []
    patterns = []
    
    for pattern in ['01', '10', '11']:
        group = data[data['reflection_pattern_str'] == pattern]['agreed_price'].dropna()
        if len(group) > 0 and len(baseline) > 0:
            pooled_std = np.sqrt((baseline.var() + group.var()) / 2)
            cohens_d = (group.mean() - baseline.mean()) / pooled_std
            effect_sizes.append(cohens_d)
            patterns.append(pattern)
    
    bars = ax6.bar(patterns, effect_sizes, color=['orange', 'green', 'red'], alpha=0.7)
    ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax6.set_title('Effect Sizes vs No Reflection')
    ax6.set_xlabel('Reflection Pattern')
    ax6.set_ylabel("Cohen's d")
    
    # Add effect size interpretation lines
    ax6.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect')
    ax6.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Medium effect')
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig('./statistical_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualizations saved to ./statistical_analysis_results.png")

def main():
    """Run complete statistical analysis"""
    print("🚀 COMPREHENSIVE STATISTICAL ANALYSIS")
    print("="*60)
    
    # Load data
    try:
        data, valid_data = load_and_prepare_data()
    except FileNotFoundError:
        print("❌ Data file not found. Please check the file path.")
        return
    
    # Run all analyses
    hypothesis_results = hypothesis_testing_suite(valid_data)
    efficiency_results = efficiency_analysis(valid_data)
    model_results = model_performance_deep_dive(valid_data)
    advanced_results = advanced_statistical_tests(valid_data)
    
    # Create visualizations
    create_statistical_visualizations(valid_data)
    
    # Summary report
    print(f"\n📋 STATISTICAL ANALYSIS SUMMARY")
    print("="*50)
    print(f"✅ Hypothesis testing completed")
    print(f"✅ Efficiency analysis completed")
    print(f"✅ Model performance analysis completed")
    print(f"✅ Advanced statistical tests completed")
    print(f"✅ Visualizations created")
    
    # Save results
    import json
    all_results = {
        'hypothesis_tests': hypothesis_results,
        'efficiency_analysis': efficiency_results,
        'model_performance': model_results,
        'advanced_tests': advanced_results
    }
    
    with open('./comprehensive_statistical_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to ./comprehensive_statistical_results.json")

if __name__ == '__main__':
    main()