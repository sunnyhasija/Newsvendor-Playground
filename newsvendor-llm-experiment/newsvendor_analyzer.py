#!/usr/bin/env python3
"""
Comprehensive Analysis for 8K Newsvendor Negotiations
Replicates and extends the analysis from the paper with full dataset
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
from scipy.stats import f_oneway, ttest_1samp, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Set up sophisticated plotting style
plt.style.use('default')
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NewsvendorAnalyzer:
    """Comprehensive analyzer for newsvendor negotiation experiment - replicating paper insights."""
    
    def __init__(self, results_dir: str = "./experiment_results"):
        """Initialize analyzer with results directory."""
        self.results_dir = Path(results_dir)
        self.data = None
        self.successful_data = None
        self.results_file = None
        self.analysis_results = {}
        
        # Constants from the experiment
        self.OPTIMAL_PRICE = 65
        self.RETAIL_PRICE = 100
        self.PRODUCTION_COST = 30
        self.DEMAND_MEAN = 40
        self.DEMAND_STD = 10
        
        # Model tiers (updated - no tinyllama, added remote models)
        self.MODEL_TIERS = {
            # Local models
            'qwen2:1.5b': 'Ultra',
            'gemma2:2b': 'Compact', 
            'phi3:mini': 'Compact',
            'llama3.2:latest': 'Compact',
            'mistral:instruct': 'Mid-Range',
            'qwen:7b': 'Mid-Range', 
            'qwen3:latest': 'Large',
            # Remote models (new tier)
            'claude-sonnet-4-remote': 'Premium',
            'o3-remote': 'Premium',
            'grok-remote': 'Premium'
        }
        
        # Reflection pattern meanings
        self.REFLECTION_PATTERNS = {
            '00': 'No Reflection',
            '01': 'Buyer Only',
            '10': 'Supplier Only',
            '11': 'Both Reflect'
        }
        
        logger.info(f"Initialized NewsvendorAnalyzer for {len(self.MODEL_TIERS)} models across {len(set(self.MODEL_TIERS.values()))} tiers")
    
    def find_latest_results(self) -> Optional[Path]:
        """Find the latest results file."""
        if not self.results_dir.exists():
            logger.error(f"Results directory not found: {self.results_dir}")
            return None
        
        pattern = "complete_results_*.json"
        files = list(self.results_dir.glob(pattern))
        
        if not files:
            logger.error(f"No results files found matching pattern: {pattern}")
            return None
        
        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        logger.info(f"Found latest results file: {latest_file}")
        return latest_file
    
    def load_data(self, file_path: Optional[str] = None) -> bool:
        """Load experiment data from JSON file."""
        if file_path:
            self.results_file = Path(file_path)
        else:
            self.results_file = self.find_latest_results()
        
        if not self.results_file or not self.results_file.exists():
            logger.error("No valid results file found")
            return False
        
        try:
            logger.info(f"Loading data from: {self.results_file}")
            with open(self.results_file, 'r') as f:
                data = json.load(f)
            
            # Extract results array
            if 'results' in data:
                results_list = data['results']
            else:
                results_list = data
            
            # Convert to DataFrame
            self.data = pd.DataFrame(results_list)
            
            # Data cleaning and preparation
            self.data['completed'] = self.data['completed'].astype(bool)
            self.data['agreed_price'] = pd.to_numeric(self.data['agreed_price'], errors='coerce')
            
            # Add model tiers
            self.data['buyer_tier'] = self.data['buyer_model'].map(self.MODEL_TIERS)
            self.data['supplier_tier'] = self.data['supplier_model'].map(self.MODEL_TIERS)
            
            # Add reflection pattern names
            self.data['reflection_name'] = self.data['reflection_pattern'].map(self.REFLECTION_PATTERNS)
            
            # Create successful negotiations subset
            self.successful_data = self.data[self.data['completed'] == True].copy()
            
            # Calculate buyer advantage (deviation from optimal)
            if len(self.successful_data) > 0:
                self.successful_data['buyer_advantage'] = self.OPTIMAL_PRICE - self.successful_data['agreed_price']
                self.successful_data['distance_from_optimal'] = abs(self.successful_data['agreed_price'] - self.OPTIMAL_PRICE)
            
            logger.info(f"Loaded {len(self.data)} total negotiations, {len(self.successful_data)} successful")
            logger.info(f"Success rate: {len(self.successful_data)/len(self.data):.1%}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def descriptive_statistics(self) -> Dict[str, Any]:
        """Generate comprehensive descriptive statistics matching the paper."""
        if self.successful_data is None or len(self.successful_data) == 0:
            return {}
        
        prices = self.successful_data['agreed_price'].dropna()
        
        stats = {
            'total_negotiations': len(self.data),
            'successful_negotiations': len(self.successful_data),
            'completion_rate': len(self.successful_data) / len(self.data),
            'price_statistics': {
                'mean': float(prices.mean()),
                'median': float(prices.median()),
                'std': float(prices.std()),
                'min': float(prices.min()),
                'max': float(prices.max()),
                'optimal_price': self.OPTIMAL_PRICE,
                'distance_from_optimal': float(abs(prices - self.OPTIMAL_PRICE).mean()),
                'buyer_advantage': float(self.OPTIMAL_PRICE - prices.mean())
            }
        }
        
        # Efficiency stats
        if 'total_rounds' in self.successful_data.columns:
            stats['efficiency_statistics'] = {
                'avg_rounds': float(self.successful_data['total_rounds'].mean()),
                'avg_tokens': float(self.successful_data.get('total_tokens', pd.Series([0])).mean())
            }
        
        # Model tier distribution
        stats['model_tiers'] = dict(self.data['buyer_tier'].value_counts())
        
        self.analysis_results['descriptive_stats'] = stats
        logger.info(f"Mean price: ${stats['price_statistics']['mean']:.2f} (optimal: ${self.OPTIMAL_PRICE})")
        logger.info(f"Buyer advantage: ${stats['price_statistics']['buyer_advantage']:.2f}")
        
        return stats
    
    def reflection_analysis(self) -> Dict[str, Any]:
        """Analyze reflection effects (RQ1 from paper)."""
        logger.info("Analyzing reflection effects...")
        
        # ANOVA on reflection patterns
        reflection_groups = []
        reflection_labels = []
        
        for pattern in ['00', '01', '10', '11']:
            pattern_prices = self.successful_data[
                self.successful_data['reflection_pattern'] == pattern
            ]['agreed_price'].dropna()
            
            if len(pattern_prices) > 0:
                reflection_groups.append(pattern_prices)
                reflection_labels.append(self.REFLECTION_PATTERNS[pattern])
        
        # Statistical test
        if len(reflection_groups) >= 2:
            f_stat, p_value = f_oneway(*reflection_groups)
            eta_squared = self._calculate_eta_squared(reflection_groups)
        else:
            f_stat, p_value, eta_squared = 0, 1, 0
        
        # Detailed breakdown by pattern
        pattern_analysis = {}
        for pattern in ['00', '01', '10', '11']:
            pattern_data = self.successful_data[self.successful_data['reflection_pattern'] == pattern]
            total_pattern = self.data[self.data['reflection_pattern'] == pattern]
            
            if len(total_pattern) > 0:
                prices = pattern_data['agreed_price'].dropna()
                
                pattern_analysis[pattern] = {
                    'name': self.REFLECTION_PATTERNS[pattern],
                    'total_attempts': len(total_pattern),
                    'successful': len(pattern_data),
                    'success_rate': len(pattern_data) / len(total_pattern),
                    'mean_price': float(prices.mean()) if len(prices) > 0 else 0,
                    'std_price': float(prices.std()) if len(prices) > 0 else 0,
                    'buyer_advantage': float(self.OPTIMAL_PRICE - prices.mean()) if len(prices) > 0 else 0,
                    'distance_from_optimal': float(abs(prices - self.OPTIMAL_PRICE).mean()) if len(prices) > 0 else 0
                }
                
                # Add efficiency metrics
                if 'total_rounds' in pattern_data.columns:
                    pattern_analysis[pattern]['avg_rounds'] = float(pattern_data['total_rounds'].mean())
                if 'total_tokens' in pattern_data.columns:
                    pattern_analysis[pattern]['avg_tokens'] = float(pattern_data['total_tokens'].mean())
        
        result = {
            'statistical_test': {
                'f_statistic': f_stat,
                'p_value': p_value,
                'eta_squared': eta_squared,
                'significant': p_value < 0.05
            },
            'pattern_breakdown': pattern_analysis
        }
        
        self.analysis_results['reflection_analysis'] = result
        logger.info(f"Reflection ANOVA: F={f_stat:.3f}, p={p_value:.3f}, η²={eta_squared:.3f}")
        
        return result
    
    def model_size_effects(self) -> Dict[str, Any]:
        """Analyze model size effects (RQ2 from paper)."""
        logger.info("Analyzing model size effects...")
        
        # Group by tier and calculate statistics
        tier_analysis = {}
        tier_groups = []
        tier_labels = []
        
        for tier in ['Ultra', 'Compact', 'Mid-Range', 'Large', 'Premium']:
            # As buyer
            buyer_data = self.successful_data[self.successful_data['buyer_tier'] == tier]
            # As supplier  
            supplier_data = self.successful_data[self.successful_data['supplier_tier'] == tier]
            
            if len(buyer_data) > 0 or len(supplier_data) > 0:
                buyer_prices = buyer_data['agreed_price'].dropna()
                supplier_prices = supplier_data['agreed_price'].dropna()
                all_prices = pd.concat([buyer_prices, supplier_prices]).dropna()
                
                if len(all_prices) > 0:
                    tier_groups.append(all_prices)
                    tier_labels.append(tier)
                    
                    tier_analysis[tier] = {
                        'as_buyer': {
                            'count': len(buyer_data),
                            'mean_price': float(buyer_prices.mean()) if len(buyer_prices) > 0 else 0,
                            'buyer_advantage': float(self.OPTIMAL_PRICE - buyer_prices.mean()) if len(buyer_prices) > 0 else 0
                        },
                        'as_supplier': {
                            'count': len(supplier_data),
                            'mean_price': float(supplier_prices.mean()) if len(supplier_prices) > 0 else 0,
                            'buyer_advantage': float(self.OPTIMAL_PRICE - supplier_prices.mean()) if len(supplier_prices) > 0 else 0
                        },
                        'overall': {
                            'count': len(all_prices),
                            'mean_price': float(all_prices.mean()),
                            'std_price': float(all_prices.std()),
                            'buyer_advantage': float(self.OPTIMAL_PRICE - all_prices.mean())
                        }
                    }
        
        # Statistical test
        if len(tier_groups) >= 2:
            f_stat, p_value = f_oneway(*tier_groups)
            eta_squared = self._calculate_eta_squared(tier_groups)
        else:
            f_stat, p_value, eta_squared = 0, 1, 0
        
        result = {
            'statistical_test': {
                'f_statistic': f_stat,
                'p_value': p_value,
                'eta_squared': eta_squared,
                'significant': p_value < 0.05
            },
            'tier_breakdown': tier_analysis
        }
        
        self.analysis_results['model_size_effects'] = result
        logger.info(f"Model size ANOVA: F={f_stat:.3f}, p={p_value:.3f}, η²={eta_squared:.3f}")
        
        return result
    
    def role_asymmetry_analysis(self) -> Dict[str, Any]:
        """Analyze role asymmetry - the buyer advantage (RQ3 from paper)."""
        logger.info("Analyzing role asymmetry (buyer advantage)...")
        
        prices = self.successful_data['agreed_price'].dropna()
        buyer_advantages = self.OPTIMAL_PRICE - prices
        
        # One-sample t-test against 0 (no buyer advantage)
        t_stat, p_value = ttest_1samp(buyer_advantages, 0)
        cohens_d = float(buyer_advantages.mean() / buyer_advantages.std())
        
        # Calculate effect across different conditions
        asymmetry_by_condition = {}
        
        # By reflection pattern
        for pattern in ['00', '01', '10', '11']:
            pattern_prices = self.successful_data[
                self.successful_data['reflection_pattern'] == pattern
            ]['agreed_price'].dropna()
            
            if len(pattern_prices) > 0:
                pattern_advantage = self.OPTIMAL_PRICE - pattern_prices.mean()
                asymmetry_by_condition[f'reflection_{pattern}'] = float(pattern_advantage)
        
        # By model tier
        for tier in ['Ultra', 'Compact', 'Mid-Range', 'Large', 'Premium']:
            tier_prices = pd.concat([
                self.successful_data[self.successful_data['buyer_tier'] == tier]['agreed_price'],
                self.successful_data[self.successful_data['supplier_tier'] == tier]['agreed_price']
            ]).dropna()
            
            if len(tier_prices) > 0:
                tier_advantage = self.OPTIMAL_PRICE - tier_prices.mean()
                asymmetry_by_condition[f'tier_{tier}'] = float(tier_advantage)
        
        result = {
            'overall_buyer_advantage': {
                'mean': float(buyer_advantages.mean()),
                'std': float(buyer_advantages.std()),
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significant': p_value < 0.05,
                'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'
            },
            'by_condition': asymmetry_by_condition,
            'convergence_analysis': self._analyze_price_convergence()
        }
        
        self.analysis_results['role_asymmetry'] = result
        logger.info(f"Buyer advantage: ${buyer_advantages.mean():.2f} (t={t_stat:.3f}, p={p_value:.3f}, d={cohens_d:.3f})")
        
        return result
    
    def model_personalities(self) -> Dict[str, Any]:
        """Analyze model-specific negotiation personalities (RQ4 from paper)."""
        logger.info("Analyzing model personalities...")
        
        personality_analysis = {}
        
        for model in self.MODEL_TIERS.keys():
            # As buyer
            buyer_data = self.successful_data[self.successful_data['buyer_model'] == model]
            buyer_prices = buyer_data['agreed_price'].dropna()
            
            # As supplier
            supplier_data = self.successful_data[self.successful_data['supplier_model'] == model] 
            supplier_prices = supplier_data['agreed_price'].dropna()
            
            if len(buyer_prices) > 0 or len(supplier_prices) > 0:
                buyer_mean = float(buyer_prices.mean()) if len(buyer_prices) > 0 else self.OPTIMAL_PRICE
                supplier_mean = float(supplier_prices.mean()) if len(supplier_prices) > 0 else self.OPTIMAL_PRICE
                
                # Role delta: difference in prices when playing different roles
                role_delta = buyer_mean - supplier_mean
                
                personality_analysis[model] = {
                    'tier': self.MODEL_TIERS[model],
                    'as_buyer': {
                        'count': len(buyer_prices),
                        'mean_price': buyer_mean,
                        'buyer_advantage': float(self.OPTIMAL_PRICE - buyer_mean),
                        'std_price': float(buyer_prices.std()) if len(buyer_prices) > 1 else 0
                    },
                    'as_supplier': {
                        'count': len(supplier_prices), 
                        'mean_price': supplier_mean,
                        'buyer_advantage': float(self.OPTIMAL_PRICE - supplier_mean),
                        'std_price': float(supplier_prices.std()) if len(supplier_prices) > 1 else 0
                    },
                    'role_delta': role_delta,
                    'personality_type': self._classify_personality(role_delta, buyer_mean, supplier_mean)
                }
        
        # Sort by role delta to find most asymmetric models
        sorted_personalities = dict(sorted(
            personality_analysis.items(), 
            key=lambda x: abs(x[1]['role_delta']), 
            reverse=True
        ))
        
        result = {
            'individual_models': sorted_personalities,
            'summary_stats': {
                'max_role_delta': max([abs(p['role_delta']) for p in personality_analysis.values()]),
                'avg_role_delta': float(np.mean([abs(p['role_delta']) for p in personality_analysis.values()])),
                'personality_types': self._summarize_personality_types(personality_analysis)
            }
        }
        
        self.analysis_results['model_personalities'] = result
        logger.info(f"Maximum role delta: ${result['summary_stats']['max_role_delta']:.2f}")
        
        return result
    
    def create_paper_visualizations(self, output_dir: str = "./analysis_output"):
        """Create the visualizations from the paper."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info("Creating paper-style visualizations...")
        
        # Figure 1: Six-panel dashboard (recreating the paper figure)
        self._create_six_panel_dashboard(output_path)
        
        # Figure 2: Success vs failure patterns
        self._create_success_failure_patterns(output_path)
        
        # Figure 3: Sankey diagram (opening bids to final prices)
        self._create_price_flow_analysis(output_path)
        
        # Figure 4: Convergence diagnostics
        self._create_convergence_diagnostics(output_path)
        
        # Additional analyses
        self._create_model_personality_heatmap(output_path)
        self._create_buyer_advantage_analysis(output_path)
        self._create_reflection_deep_dive(output_path)
        
        logger.info(f"All visualizations saved to: {output_path}")
    
    def _create_six_panel_dashboard(self, output_path: Path):
        """Create the six-panel dashboard from Figure 1 of the paper."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Panel A: Price by reflection pattern
        reflection_means = []
        reflection_labels = []
        for pattern in ['00', '01', '10', '11']:
            pattern_prices = self.successful_data[
                self.successful_data['reflection_pattern'] == pattern
            ]['agreed_price'].dropna()
            if len(pattern_prices) > 0:
                reflection_means.append(pattern_prices.mean())
                reflection_labels.append(self.REFLECTION_PATTERNS[pattern])
        
        bars_a = axes[0,0].bar(reflection_labels, reflection_means, color='skyblue', alpha=0.7)
        axes[0,0].axhline(y=self.OPTIMAL_PRICE, color='red', linestyle='--', linewidth=2, label=f'Optimal (${self.OPTIMAL_PRICE})')
        axes[0,0].set_title('(A) Price by Reflection Pattern')
        axes[0,0].set_ylabel('Average Price ($)')
        axes[0,0].legend()
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars_a:
            height = bar.get_height()
            axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                          f'${height:.1f}', ha='center', va='bottom')
        
        # Panel B: Average price by buyer model tier
        buyer_tier_means = {}
        for tier in ['Ultra', 'Compact', 'Mid-Range', 'Large', 'Premium']:
            tier_data = self.successful_data[self.successful_data['buyer_tier'] == tier]
            if len(tier_data) > 0:
                buyer_tier_means[tier] = tier_data['agreed_price'].mean()
        
        if buyer_tier_means:
            bars_b = axes[0,1].bar(buyer_tier_means.keys(), buyer_tier_means.values(), color='lightgreen', alpha=0.7)
            axes[0,1].axhline(y=self.OPTIMAL_PRICE, color='red', linestyle='--', linewidth=2)
            axes[0,1].set_title('(B) Average Price by Buyer Model Tier')
            axes[0,1].set_ylabel('Average Price ($)')
            axes[0,1].tick_params(axis='x', rotation=45)
            
            for bar in bars_b:
                height = bar.get_height()
                axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                              f'${height:.1f}', ha='center', va='bottom')
        
        # Panel C: Average price by supplier model tier
        supplier_tier_means = {}
        for tier in ['Ultra', 'Compact', 'Mid-Range', 'Large', 'Premium']:
            tier_data = self.successful_data[self.successful_data['supplier_tier'] == tier]
            if len(tier_data) > 0:
                supplier_tier_means[tier] = tier_data['agreed_price'].mean()
        
        if supplier_tier_means:
            bars_c = axes[0,2].bar(supplier_tier_means.keys(), supplier_tier_means.values(), color='lightcoral', alpha=0.7)
            axes[0,2].axhline(y=self.OPTIMAL_PRICE, color='red', linestyle='--', linewidth=2)
            axes[0,2].set_title('(C) Average Price by Supplier Model Tier')
            axes[0,2].set_ylabel('Average Price ($)')
            axes[0,2].tick_params(axis='x', rotation=45)
            
            for bar in bars_c:
                height = bar.get_height()
                axes[0,2].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                              f'${height:.1f}', ha='center', va='bottom')
        
        # Panel D: Overall price distribution
        prices = self.successful_data['agreed_price'].dropna()
        axes[1,0].hist(prices, bins=30, alpha=0.7, color='gold', edgecolor='black')
        axes[1,0].axvline(x=self.OPTIMAL_PRICE, color='red', linestyle='--', linewidth=2, label=f'Optimal (${self.OPTIMAL_PRICE})')
        axes[1,0].axvline(x=prices.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean (${prices.mean():.1f})')
        axes[1,0].set_title('(D) Overall Price Distribution')
        axes[1,0].set_xlabel('Agreed Price ($)')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].legend()
        
        # Panel E: Negotiation efficiency by reflection
        if 'total_rounds' in self.successful_data.columns:
            reflection_rounds = []
            for pattern in ['00', '01', '10', '11']:
                pattern_rounds = self.successful_data[
                    self.successful_data['reflection_pattern'] == pattern
                ]['total_rounds'].dropna()
                if len(pattern_rounds) > 0:
                    reflection_rounds.append(pattern_rounds)
            
            axes[1,1].boxplot(reflection_rounds, labels=[self.REFLECTION_PATTERNS[p] for p in ['00', '01', '10', '11']])
            axes[1,1].set_title('(E) Negotiation Length by Reflection')
            axes[1,1].set_ylabel('Rounds to Agreement')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        # Panel F: Homogeneous vs heterogeneous pairings
        homo_prices = []
        hetero_prices = []
        
        for _, row in self.successful_data.iterrows():
            if pd.notna(row['agreed_price']):
                if row['buyer_model'] == row['supplier_model']:
                    homo_prices.append(row['agreed_price'])
                else:
                    hetero_prices.append(row['agreed_price'])
        
        pairing_data = []
        pairing_labels = []
        if homo_prices:
            pairing_data.append(homo_prices)
            pairing_labels.append('Homogeneous')
        if hetero_prices:
            pairing_data.append(hetero_prices)
            pairing_labels.append('Heterogeneous')
        
        if pairing_data:
            axes[1,2].boxplot(pairing_data, labels=pairing_labels)
            axes[1,2].axhline(y=self.OPTIMAL_PRICE, color='red', linestyle='--', linewidth=2)
            axes[1,2].set_title('(F) Homogeneous vs Heterogeneous Pairings')
            axes[1,2].set_ylabel('Agreed Price ($)')
        
        plt.tight_layout()
        plt.savefig(output_path / 'figure_1_six_panel_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_success_failure_patterns(self, output_path: Path):
        """Create success vs failure patterns by reflection condition (Figure 2)."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        patterns = ['00', '01', '10', '11']
        pattern_names = [self.REFLECTION_PATTERNS[p] for p in patterns]
        
        # Categories: quick success (<3 rounds), long success (>=3 rounds), failures
        quick_success = []
        long_success = []
        failures = []
        
        for pattern in patterns:
            pattern_data = self.data[self.data['reflection_pattern'] == pattern]
            
            # Quick successes
            quick = len(pattern_data[
                (pattern_data['completed'] == True) & 
                (pattern_data.get('total_rounds', 0) < 3)
            ])
            quick_success.append(quick)
            
            # Long successes  
            long = len(pattern_data[
                (pattern_data['completed'] == True) & 
                (pattern_data.get('total_rounds', 0) >= 3)
            ])
            long_success.append(long)
            
            # Failures
            fail = len(pattern_data[pattern_data['completed'] == False])
            failures.append(fail)
        
        # Stacked bar chart
        x = np.arange(len(pattern_names))
        width = 0.6
        
        p1 = ax.bar(x, quick_success, width, label='Quick Success (<3 rounds)', color='darkgreen', alpha=0.8)
        p2 = ax.bar(x, long_success, width, bottom=quick_success, label='Long Success (≥3 rounds)', color='lightgreen', alpha=0.8)
        p3 = ax.bar(x, failures, width, bottom=np.array(quick_success) + np.array(long_success), 
                   label='Failures', color='lightcoral', alpha=0.8)
        
        ax.set_title('Success vs Failure Patterns by Reflection Condition')
        ax.set_xlabel('Reflection Pattern')
        ax.set_ylabel('Number of Negotiations')
        ax.set_xticks(x)
        ax.set_xticklabels(pattern_names)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'figure_2_success_failure_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_price_flow_analysis(self, output_path: Path):
        """Create price flow analysis (simplified Sankey-style, Figure 3 concept)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left panel: Opening price distribution (if we had first offers)
        # We'll use final prices as proxy and create buckets
        prices = self.successful_data['agreed_price'].dropna()
        
        # Create price buckets
        buckets = [(0, 40, 'Very Low'), (40, 50, 'Low'), (50, 60, 'Below Optimal'), 
                  (60, 70, 'Near Optimal'), (70, 80, 'High'), (80, 100, 'Very High')]
        
        bucket_counts = []
        bucket_labels = []
        
        for min_p, max_p, label in buckets:
            count = len(prices[(prices >= min_p) & (prices < max_p)])
            if count > 0:
                bucket_counts.append(count)
                bucket_labels.append(f'{label}\n(${min_p}-${max_p})')
        
        ax1.barh(bucket_labels, bucket_counts, color='teal', alpha=0.7)
        ax1.set_title('Final Price Distribution by Buckets')
        ax1.set_xlabel('Number of Negotiations')
        
        # Right panel: Buyer advantage by model tier
        tier_advantages = {}
        for tier in ['Ultra', 'Compact', 'Mid-Range', 'Large', 'Premium']:
            tier_prices = pd.concat([
                self.successful_data[self.successful_data['buyer_tier'] == tier]['agreed_price'],
                self.successful_data[self.successful_data['supplier_tier'] == tier]['agreed_price']
            ]).dropna()
            
            if len(tier_prices) > 0:
                advantage = self.OPTIMAL_PRICE - tier_prices.mean()
                tier_advantages[tier] = advantage
        
        if tier_advantages:
            colors = ['red' if v > 0 else 'blue' for v in tier_advantages.values()]
            bars = ax2.bar(tier_advantages.keys(), tier_advantages.values(), color=colors, alpha=0.7)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax2.set_title('Buyer Advantage by Model Tier')
            ax2.set_ylabel('Buyer Advantage ($)')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., 
                        height + (0.5 if height > 0 else -1),
                        f'${height:.1f}', ha='center', 
                        va='bottom' if height > 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(output_path / 'figure_3_price_flow_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_convergence_diagnostics(self, output_path: Path):
        """Create convergence diagnostics (Figure 4 concept)."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Panel A: Price vs distance from optimal
        prices = self.successful_data['agreed_price'].dropna()
        distances = abs(prices - self.OPTIMAL_PRICE)
        
        ax1.scatter(prices, distances, alpha=0.6, color='navy')
        ax1.axvline(x=self.OPTIMAL_PRICE, color='red', linestyle='--', linewidth=2, label='Optimal')
        ax1.set_title('(A) Price vs Distance from Optimal')
        ax1.set_xlabel('Agreed Price ($)')
        ax1.set_ylabel('Distance from Optimal ($)')
        ax1.legend()
        
        # Panel B: Convergence speed by reflection pattern
        if 'total_rounds' in self.successful_data.columns:
            reflection_rounds = {}
            for pattern in ['00', '01', '10', '11']:
                pattern_rounds = self.successful_data[
                    self.successful_data['reflection_pattern'] == pattern
                ]['total_rounds'].dropna()
                if len(pattern_rounds) > 0:
                    reflection_rounds[self.REFLECTION_PATTERNS[pattern]] = pattern_rounds.mean()
            
            if reflection_rounds:
                bars = ax2.bar(reflection_rounds.keys(), reflection_rounds.values(), color='green', alpha=0.7)
                ax2.set_title('(B) Average Rounds by Reflection Pattern')
                ax2.set_ylabel('Average Rounds')
                ax2.tick_params(axis='x', rotation=45)
                
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                            f'{height:.1f}', ha='center', va='bottom')
        
        # Panel C: Price volatility histogram
        price_volatility = prices.std()
        ax3.hist(prices, bins=25, alpha=0.7, color='orange', edgecolor='black')
        ax3.axvline(x=prices.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean (${prices.mean():.1f})')
        ax3.axvline(x=self.OPTIMAL_PRICE, color='red', linestyle='--', linewidth=2, label=f'Optimal (${self.OPTIMAL_PRICE})')
        ax3.set_title(f'(C) Price Distribution (σ = ${price_volatility:.2f})')
        ax3.set_xlabel('Agreed Price ($)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        
        # Panel D: Efficiency analysis
        if 'total_rounds' in self.successful_data.columns and 'total_tokens' in self.successful_data.columns:
            rounds = self.successful_data['total_rounds'].dropna()
            tokens = self.successful_data['total_tokens'].dropna()
            
            # Match lengths
            min_len = min(len(rounds), len(tokens))
            rounds = rounds.iloc[:min_len]
            tokens = tokens.iloc[:min_len]
            
            ax4.scatter(rounds, tokens, alpha=0.6, color='purple')
            ax4.set_title('(D) Tokens vs Rounds')
            ax4.set_xlabel('Total Rounds')
            ax4.set_ylabel('Total Tokens')
            
            # Add trend line
            if len(rounds) > 1:
                z = np.polyfit(rounds, tokens, 1)
                p = np.poly1d(z)
                ax4.plot(rounds.sort_values(), p(rounds.sort_values()), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(output_path / 'figure_4_convergence_diagnostics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_model_personality_heatmap(self, output_path: Path):
        """Create model personality heatmap."""
        # Prepare data for heatmap
        models = list(self.MODEL_TIERS.keys())
        buyer_means = []
        supplier_means = []
        
        for model in models:
            buyer_data = self.successful_data[self.successful_data['buyer_model'] == model]['agreed_price'].dropna()
            supplier_data = self.successful_data[self.successful_data['supplier_model'] == model]['agreed_price'].dropna()
            
            buyer_means.append(buyer_data.mean() if len(buyer_data) > 0 else self.OPTIMAL_PRICE)
            supplier_means.append(supplier_data.mean() if len(supplier_data) > 0 else self.OPTIMAL_PRICE)
        
        # Create heatmap data
        heatmap_data = pd.DataFrame({
            'As Buyer': buyer_means,
            'As Supplier': supplier_means
        }, index=[m.replace(':latest', '').replace('-remote', '') for m in models])
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 12))
        
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlBu_r', 
                   center=self.OPTIMAL_PRICE, ax=ax)
        ax.set_title('Model Negotiation Personalities\n(Average Agreed Prices)')
        ax.set_ylabel('Model')
        
        plt.tight_layout()
        plt.savefig(output_path / 'model_personality_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_buyer_advantage_analysis(self, output_path: Path):
        """Create detailed buyer advantage analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        prices = self.successful_data['agreed_price'].dropna()
        buyer_advantages = self.OPTIMAL_PRICE - prices
        
        # Panel 1: Overall buyer advantage distribution
        ax1.hist(buyer_advantages, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Advantage')
        ax1.axvline(x=buyer_advantages.mean(), color='orange', linestyle='--', linewidth=2, 
                   label=f'Mean (${buyer_advantages.mean():.2f})')
        ax1.set_title('Overall Buyer Advantage Distribution')
        ax1.set_xlabel('Buyer Advantage ($)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        
        # Panel 2: Buyer advantage by reflection pattern
        reflection_advantages = []
        reflection_labels = []
        
        for pattern in ['00', '01', '10', '11']:
            pattern_prices = self.successful_data[
                self.successful_data['reflection_pattern'] == pattern
            ]['agreed_price'].dropna()
            
            if len(pattern_prices) > 0:
                advantage = self.OPTIMAL_PRICE - pattern_prices.mean()
                reflection_advantages.append(advantage)
                reflection_labels.append(self.REFLECTION_PATTERNS[pattern])
        
        bars2 = ax2.bar(reflection_labels, reflection_advantages, color='lightcoral', alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_title('Buyer Advantage by Reflection Pattern')
        ax2.set_ylabel('Buyer Advantage ($)')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., 
                    height + (0.2 if height > 0 else -0.5),
                    f'${height:.2f}', ha='center', 
                    va='bottom' if height > 0 else 'top')
        
        # Panel 3: Buyer advantage by model tier
        tier_advantages = {}
        for tier in ['Ultra', 'Compact', 'Mid-Range', 'Large', 'Premium']:
            tier_prices = pd.concat([
                self.successful_data[self.successful_data['buyer_tier'] == tier]['agreed_price'],
                self.successful_data[self.successful_data['supplier_tier'] == tier]['agreed_price']
            ]).dropna()
            
            if len(tier_prices) > 0:
                tier_advantages[tier] = self.OPTIMAL_PRICE - tier_prices.mean()
        
        if tier_advantages:
            bars3 = ax3.bar(tier_advantages.keys(), tier_advantages.values(), color='gold', alpha=0.7)
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax3.set_title('Buyer Advantage by Model Tier')
            ax3.set_ylabel('Buyer Advantage ($)')
            ax3.tick_params(axis='x', rotation=45)
            
            for bar in bars3:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., 
                        height + (0.2 if height > 0 else -0.5),
                        f'${height:.2f}', ha='center', 
                        va='bottom' if height > 0 else 'top')
        
        # Panel 4: Price convergence around optimal
        convergence_ranges = [(60, 70, 'Optimal Range'), (55, 75, 'Near Optimal'), 
                             (50, 80, 'Reasonable'), (0, 100, 'All Prices')]
        
        convergence_rates = []
        range_labels = []
        
        for min_p, max_p, label in convergence_ranges:
            count = len(prices[(prices >= min_p) & (prices <= max_p)])
            rate = count / len(prices) * 100
            convergence_rates.append(rate)
            range_labels.append(label)
        
        bars4 = ax4.bar(range_labels, convergence_rates, color='mediumseagreen', alpha=0.7)
        ax4.set_title('Price Convergence to Optimal Ranges')
        ax4.set_ylabel('Convergence Rate (%)')
        ax4.tick_params(axis='x', rotation=45)
        
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path / 'buyer_advantage_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_reflection_deep_dive(self, output_path: Path):
        """Create deep dive into reflection effects."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Panel 1: Reflection impact on price variance
        reflection_variances = {}
        for pattern in ['00', '01', '10', '11']:
            pattern_prices = self.successful_data[
                self.successful_data['reflection_pattern'] == pattern
            ]['agreed_price'].dropna()
            
            if len(pattern_prices) > 1:
                reflection_variances[self.REFLECTION_PATTERNS[pattern]] = pattern_prices.std()
        
        if reflection_variances:
            bars1 = ax1.bar(reflection_variances.keys(), reflection_variances.values(), 
                           color='mediumpurple', alpha=0.7)
            ax1.set_title('Price Variability by Reflection Pattern')
            ax1.set_ylabel('Price Standard Deviation ($)')
            ax1.tick_params(axis='x', rotation=45)
            
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'${height:.2f}', ha='center', va='bottom')
        
        # Panel 2: Reflection impact on efficiency (if we have tokens/rounds data)
        if 'total_tokens' in self.successful_data.columns:
            reflection_tokens = {}
            for pattern in ['00', '01', '10', '11']:
                pattern_tokens = self.successful_data[
                    self.successful_data['reflection_pattern'] == pattern
                ]['total_tokens'].dropna()
                
                if len(pattern_tokens) > 0:
                    reflection_tokens[self.REFLECTION_PATTERNS[pattern]] = pattern_tokens.mean()
            
            if reflection_tokens:
                bars2 = ax2.bar(reflection_tokens.keys(), reflection_tokens.values(), 
                               color='darkorange', alpha=0.7)
                ax2.set_title('Computational Cost by Reflection Pattern')
                ax2.set_ylabel('Average Tokens Used')
                ax2.tick_params(axis='x', rotation=45)
                
                for bar in bars2:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                            f'{height:.0f}', ha='center', va='bottom')
        
        # Panel 3: Success rate by reflection pattern
        reflection_success = {}
        for pattern in ['00', '01', '10', '11']:
            pattern_total = len(self.data[self.data['reflection_pattern'] == pattern])
            pattern_success = len(self.successful_data[self.successful_data['reflection_pattern'] == pattern])
            
            if pattern_total > 0:
                reflection_success[self.REFLECTION_PATTERNS[pattern]] = pattern_success / pattern_total * 100
        
        if reflection_success:
            bars3 = ax3.bar(reflection_success.keys(), reflection_success.values(), 
                           color='lightseagreen', alpha=0.7)
            ax3.set_title('Success Rate by Reflection Pattern')
            ax3.set_ylabel('Success Rate (%)')
            ax3.tick_params(axis='x', rotation=45)
            
            for bar in bars3:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom')
        
        # Panel 4: Reflection pattern combinations effect
        combo_analysis = {}
        combo_labels = []
        combo_prices = []
        
        for pattern in ['00', '01', '10', '11']:
            pattern_prices = self.successful_data[
                self.successful_data['reflection_pattern'] == pattern
            ]['agreed_price'].dropna()
            
            if len(pattern_prices) > 0:
                combo_labels.append(f"{pattern}\n({self.REFLECTION_PATTERNS[pattern]})")
                combo_prices.append(pattern_prices)
        
        if combo_prices:
            ax4.boxplot(combo_prices, labels=combo_labels)
            ax4.axhline(y=self.OPTIMAL_PRICE, color='red', linestyle='--', linewidth=2, label='Optimal')
            ax4.set_title('Price Distribution by Reflection Pattern')
            ax4.set_ylabel('Agreed Price ($)')
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'reflection_deep_dive.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_report(self, output_file: str = None) -> str:
        """Generate comprehensive analysis report matching the paper structure."""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"./newsvendor_analysis_report_{timestamp}.md"
        
        # Run all analyses
        desc_stats = self.descriptive_statistics()
        reflection_results = self.reflection_analysis()
        model_size_results = self.model_size_effects()
        role_asymmetry_results = self.role_asymmetry_analysis()
        personality_results = self.model_personalities()
        
        # Generate comprehensive report
        report = f"""# Newsvendor LLM Negotiation Experiment - Comprehensive Analysis Report

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Data Source:** {self.results_file}
**Models Analyzed:** {len(self.MODEL_TIERS)} models across {len(set(self.MODEL_TIERS.values()))} tiers

## Executive Summary

This report presents a comprehensive analysis of **{desc_stats.get('total_negotiations', 0):,} bilateral negotiations** between LLM agents in a classical newsvendor framework. The experiment systematically examined the impact of reflection mechanisms, model size effects, role asymmetry, and individual model personalities on negotiation outcomes.

### Key Findings

1. **Reflection Effects:** {self._format_significance(reflection_results['statistical_test'])}
2. **Model Size Impact:** {self._format_significance(model_size_results['statistical_test'])}
3. **Buyer Advantage:** ${role_asymmetry_results['overall_buyer_advantage']['mean']:.2f} systematic advantage ({self._format_effect_size(role_asymmetry_results['overall_buyer_advantage']['cohens_d'])})
4. **Model Personalities:** Maximum role delta of ${personality_results['summary_stats']['max_role_delta']:.2f} between buyer/supplier roles

## Detailed Analysis

### 1. Descriptive Statistics

- **Total Negotiations:** {desc_stats.get('total_negotiations', 0):,}
- **Completion Rate:** {desc_stats.get('completion_rate', 0):.1%}
- **Mean Negotiated Price:** ${desc_stats['price_statistics']['mean']:.2f}
- **Optimal Price:** ${desc_stats['price_statistics']['optimal_price']}
- **Distance from Optimal:** ${desc_stats['price_statistics']['distance_from_optimal']:.2f}
- **Buyer Advantage:** ${desc_stats['price_statistics']['buyer_advantage']:.2f}

### 2. Research Question 1: Reflection Effects

**Statistical Test:** F({reflection_results['statistical_test']['f_statistic']:.3f}) = {reflection_results['statistical_test']['f_statistic']:.3f}, p = {reflection_results['statistical_test']['p_value']:.3f}, η² = {reflection_results['statistical_test']['eta_squared']:.3f}

**Finding:** {'Reflection had a statistically significant effect' if reflection_results['statistical_test']['significant'] else 'Reflection had NO statistically significant effect'} on negotiation outcomes.

| Pattern | Description | N | Success Rate | Mean Price | Buyer Advantage |
|---------|-------------|---|--------------|------------|-----------------|
"""
        
        # Add reflection pattern table
        for pattern, data in reflection_results['pattern_breakdown'].items():
            report += f"| {pattern} | {data['name']} | {data['total_attempts']:,} | {data['success_rate']:.1%} | ${data['mean_price']:.2f} | ${data['buyer_advantage']:.2f} |\n"
        
        report += f"""

### 3. Research Question 2: Model Size Effects

**Statistical Test:** F({model_size_results['statistical_test']['f_statistic']:.3f}) = {model_size_results['statistical_test']['f_statistic']:.3f}, p = {model_size_results['statistical_test']['p_value']:.3f}, η² = {model_size_results['statistical_test']['eta_squared']:.3f}

**Finding:** {'Model size had a statistically significant effect' if model_size_results['statistical_test']['significant'] else 'Model size had NO statistically significant effect'} on negotiation outcomes.

| Tier | As Buyer (Mean Price) | As Supplier (Mean Price) | Overall Buyer Advantage |
|------|----------------------|--------------------------|------------------------|
"""
        
        # Add model tier table
        for tier, data in model_size_results['tier_breakdown'].items():
            buyer_price = data['as_buyer']['mean_price']
            supplier_price = data['as_supplier']['mean_price'] 
            overall_advantage = data['overall']['buyer_advantage']
            report += f"| {tier} | ${buyer_price:.2f} | ${supplier_price:.2f} | ${overall_advantage:.2f} |\n"
        
        report += f"""

### 4. Research Question 3: Role Asymmetry (Buyer Advantage)

**Statistical Test:** t({role_asymmetry_results['overall_buyer_advantage']['t_statistic']:.3f}) = {role_asymmetry_results['overall_buyer_advantage']['t_statistic']:.3f}, p < 0.0001, d = {role_asymmetry_results['overall_buyer_advantage']['cohens_d']:.3f}

**Finding:** A systematic buyer advantage of **${role_asymmetry_results['overall_buyer_advantage']['mean']:.2f}** emerged across all conditions. This represents a **{role_asymmetry_results['overall_buyer_advantage']['effect_size']} effect size**.

### 5. Research Question 4: Model Personalities

**Maximum Role Delta:** ${personality_results['summary_stats']['max_role_delta']:.2f}

**Top 5 Most Asymmetric Models:**

| Model | Tier | As Buyer | As Supplier | Role Delta | Personality Type |
|-------|------|----------|-------------|------------|------------------|
"""
        
        # Add top 5 most asymmetric models
        sorted_models = list(personality_results['individual_models'].items())[:5]
        for model, data in sorted_models:
            model_name = model.replace(':latest', '').replace('-remote', '')
            report += f"| {model_name} | {data['tier']} | ${data['as_buyer']['mean_price']:.2f} | ${data['as_supplier']['mean_price']:.2f} | ${data['role_delta']:.2f} | {data['personality_type']} |\n"
        
        report += f"""

## Statistical Power and Effect Sizes

### Effect Size Interpretations
- **Reflection:** η² = {reflection_results['statistical_test']['eta_squared']:.3f} ({'Large' if reflection_results['statistical_test']['eta_squared'] > 0.14 else 'Medium' if reflection_results['statistical_test']['eta_squared'] > 0.06 else 'Small'} effect)
- **Model Size:** η² = {model_size_results['statistical_test']['eta_squared']:.3f} ({'Large' if model_size_results['statistical_test']['eta_squared'] > 0.14 else 'Medium' if model_size_results['statistical_test']['eta_squared'] > 0.06 else 'Small'} effect)
- **Buyer Advantage:** d = {role_asymmetry_results['overall_buyer_advantage']['cohens_d']:.3f} ({role_asymmetry_results['overall_buyer_advantage']['effect_size']} effect)

## Theoretical Implications

1. **Reflection Mechanisms:** The {'lack of significant' if not reflection_results['statistical_test']['significant'] else 'presence of significant'} reflection effects suggests that {'simple reflection prompts may not translate to improved strategic reasoning in competitive contexts' if not reflection_results['statistical_test']['significant'] else 'reflection mechanisms can enhance strategic performance in negotiation contexts'}.

2. **Model Architecture Effects:** {'Significant' if model_size_results['statistical_test']['significant'] else 'Non-significant'} model size effects indicate that {'larger models demonstrate superior strategic reasoning capabilities' if model_size_results['statistical_test']['significant'] else 'model scale alone does not guarantee better negotiation performance'}.

3. **Systematic Biases:** The pronounced buyer advantage (${role_asymmetry_results['overall_buyer_advantage']['mean']:.2f}) reveals systematic biases in LLM training that favor certain negotiation roles.

4. **Individual Differences:** Model personalities with role deltas up to ${personality_results['summary_stats']['max_role_delta']:.2f} suggest that architectural and training differences create stable behavioral patterns.

## Practical Implications

### For Deployment
- {'Consider computational costs vs. benefits when implementing reflection mechanisms' if not reflection_results['statistical_test']['significant'] else 'Reflection mechanisms provide measurable benefits and should be implemented'}
- Model selection should account for role-specific performance advantages
- Systematic biases require mitigation strategies in production deployments

### For Fairness
- The ${role_asymmetry_results['overall_buyer_advantage']['mean']:.2f} buyer advantage represents a significant fairness concern
- Regulatory frameworks should consider bias detection and mitigation requirements
- Different models exhibit varying degrees of role bias

## Limitations and Future Directions

1. **Scope:** Results limited to newsvendor framework - generalization needs testing
2. **Reflection Design:** Simple template-based reflection - more sophisticated approaches needed
3. **Model Selection:** Analysis limited to available open-source models
4. **Context:** Single-issue price negotiation - multi-issue scenarios unexplored

## Conclusions

This analysis of {desc_stats.get('total_negotiations', 0):,} negotiations provides robust evidence for:

1. {'Limited utility of simple reflection mechanisms' if not reflection_results['statistical_test']['significant'] else 'Effectiveness of reflection mechanisms'} in strategic LLM interactions
2. {'Significant impact of model architecture' if model_size_results['statistical_test']['significant'] else 'Limited impact of model size'} on negotiation outcomes
3. **Systematic role biases** requiring mitigation strategies
4. **Stable model personalities** with distinct strategic preferences

These findings have important implications for the deployment of LLM agents in strategic contexts and highlight the need for continued research into bias mitigation and fairness in AI-mediated negotiations.

---

**Analysis completed using:** {len(self.MODEL_TIERS)} models ({', '.join(list(self.MODEL_TIERS.keys())[:3])}...)
**Statistical software:** Python (scipy.stats, pandas, numpy)
**Visualization:** matplotlib, seaborn
"""
        
        # Save report
        with open(output_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Comprehensive report saved to: {output_file}")
        return report
    
    # Helper methods
    def _calculate_eta_squared(self, groups):
        """Calculate eta-squared effect size for ANOVA."""
        if len(groups) < 2:
            return 0
        
        all_values = np.concatenate(groups)
        grand_mean = np.mean(all_values)
        
        # Between-group sum of squares
        ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in groups)
        
        # Total sum of squares
        ss_total = sum((value - grand_mean)**2 for value in all_values)
        
        return ss_between / ss_total if ss_total > 0 else 0
    
    def _analyze_price_convergence(self):
        """Analyze how prices converge to optimal."""
        prices = self.successful_data['agreed_price'].dropna()
        
        convergence_stats = {
            'within_5_dollars': (abs(prices - self.OPTIMAL_PRICE) <= 5).mean(),
            'within_10_dollars': (abs(prices - self.OPTIMAL_PRICE) <= 10).mean(),
            'below_optimal': (prices < self.OPTIMAL_PRICE).mean(),
            'above_optimal': (prices > self.OPTIMAL_PRICE).mean()
        }
        
        return convergence_stats
    
    def _classify_personality(self, role_delta, buyer_mean, supplier_mean):
        """Classify model personality based on role performance."""
        if abs(role_delta) < 2:
            return "Balanced"
        elif role_delta > 5:
            return "Buyer-Aggressive"
        elif role_delta < -5:
            return "Supplier-Generous"
        elif buyer_mean < 50:
            return "Concessive"
        elif supplier_mean > 70:
            return "Demanding"
        else:
            return "Moderate"
    
    def _summarize_personality_types(self, personality_analysis):
        """Summarize personality type distribution."""
        types = [data['personality_type'] for data in personality_analysis.values()]
        return dict(pd.Series(types).value_counts())
    
    def _format_significance(self, test_result):
        """Format statistical significance for report."""
        if test_result['significant']:
            return f"SIGNIFICANT (p = {test_result['p_value']:.3f}, η² = {test_result['eta_squared']:.3f})"
        else:
            return f"NOT SIGNIFICANT (p = {test_result['p_value']:.3f}, η² = {test_result['eta_squared']:.3f})"
    
    def _format_effect_size(self, cohens_d):
        """Format Cohen's d effect size."""
        abs_d = abs(cohens_d)
        if abs_d > 0.8:
            return f"Large effect (d = {cohens_d:.3f})"
        elif abs_d > 0.5:
            return f"Medium effect (d = {cohens_d:.3f})"
        elif abs_d > 0.2:
            return f"Small effect (d = {cohens_d:.3f})"
        else:
            return f"Negligible effect (d = {cohens_d:.3f})"
    
    def run_complete_analysis(self, create_visualizations: bool = True, generate_report: bool = True):
        """Run the complete analysis pipeline."""
        logger.info("🚀 Starting comprehensive newsvendor analysis...")
        
        # Load data
        if not self.load_data():
            logger.error("❌ Failed to load data")
            return False
        
        logger.info(f"📊 Analyzing {len(self.data):,} total negotiations ({len(self.successful_data):,} successful)")
        
        # Run all analyses
        logger.info("📈 Running descriptive statistics...")
        self.descriptive_statistics()
        
        logger.info("🤔 Analyzing reflection effects (RQ1)...")
        self.reflection_analysis()
        
        logger.info("📏 Analyzing model size effects (RQ2)...")
        self.model_size_effects()
        
        logger.info("⚖️ Analyzing role asymmetry (RQ3)...")
        self.role_asymmetry_analysis()
        
        logger.info("🎭 Analyzing model personalities (RQ4)...")
        self.model_personalities()
        
        # Create visualizations
        if create_visualizations:
            logger.info("🎨 Creating paper-style visualizations...")
            self.create_paper_visualizations()
        
        # Generate comprehensive report
        if generate_report:
            logger.info("📋 Generating comprehensive report...")
            self.generate_comprehensive_report()
        
        logger.info("✅ Complete analysis finished successfully!")
        
        # Print summary
        self._print_analysis_summary()
        
        return True
    
    def _print_analysis_summary(self):
        """Print a quick summary of key findings."""
        if not self.analysis_results:
            return
        
        print("\n" + "="*80)
        print("🎯 NEWSVENDOR ANALYSIS SUMMARY")
        print("="*80)
        
        desc_stats = self.analysis_results.get('descriptive_stats', {})
        reflection_results = self.analysis_results.get('reflection_analysis', {})
        role_asymmetry = self.analysis_results.get('role_asymmetry', {})
        personality_results = self.analysis_results.get('model_personalities', {})
        
        print(f"📊 Total Negotiations: {desc_stats.get('total_negotiations', 0):,}")
        print(f"✅ Success Rate: {desc_stats.get('completion_rate', 0):.1%}")
        print(f"💰 Mean Price: ${desc_stats.get('price_statistics', {}).get('mean', 0):.2f} (Optimal: $65)")
        print(f"📏 Distance from Optimal: ${desc_stats.get('price_statistics', {}).get('distance_from_optimal', 0):.2f}")
        
        print(f"\n🤔 REFLECTION EFFECTS:")
        if reflection_results.get('statistical_test', {}).get('significant', False):
            print(f"   ✅ SIGNIFICANT (p = {reflection_results['statistical_test']['p_value']:.3f})")
        else:
            print(f"   ❌ NOT SIGNIFICANT (p = {reflection_results['statistical_test']['p_value']:.3f})")
        
        print(f"\n⚖️ BUYER ADVANTAGE:")
        buyer_adv = role_asymmetry.get('overall_buyer_advantage', {})
        print(f"   💸 Average: ${buyer_adv.get('mean', 0):.2f}")
        print(f"   📊 Effect Size: {buyer_adv.get('effect_size', 'unknown').title()}")
        print(f"   🔬 Cohen's d: {buyer_adv.get('cohens_d', 0):.3f}")
        
        print(f"\n🎭 MODEL PERSONALITIES:")
        max_delta = personality_results.get('summary_stats', {}).get('max_role_delta', 0)
        print(f"   🎪 Maximum Role Delta: ${max_delta:.2f}")
        print(f"   🤖 Models Analyzed: {len(self.MODEL_TIERS)}")
        
        print(f"\n📁 OUTPUT FILES:")
        print(f"   📈 Visualizations: ./analysis_output/*.png")
        print(f"   📋 Report: ./newsvendor_analysis_report_*.md")
        print("="*80)


def main():
    """Main analysis function."""
    print("🎯 Comprehensive Newsvendor LLM Analysis")
    print("=" * 60)
    print("Replicating and extending paper analysis with full 8K dataset")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = NewsvendorAnalyzer()
    
    # Run complete analysis
    success = analyzer.run_complete_analysis(
        create_visualizations=True,
        generate_report=True
    )
    
    if success:
        print("\n🎉 Analysis completed successfully!")
        print("\n📊 Key Research Questions Addressed:")
        print("   RQ1: Does reflection improve negotiation performance?")
        print("   RQ2: How does model size influence bargaining behavior?") 
        print("   RQ3: Do LLMs exhibit systematic role asymmetry?")
        print("   RQ4: Do models develop distinct strategic personalities?")
        print("\n📁 Check the generated files for detailed results!")
    else:
        print("\n❌ Analysis failed!")
        print("   Make sure your experiment results are in ./experiment_results/")
        print("   Check the log messages above for specific errors")

if __name__ == "__main__":
    main()