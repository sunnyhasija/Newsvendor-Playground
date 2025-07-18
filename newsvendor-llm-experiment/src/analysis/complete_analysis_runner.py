#!/usr/bin/env python3
"""
Comprehensive LLM Negotiation Analysis Suite
============================================

A unified, publication-ready analysis of LLM-to-LLM negotiations in the newsvendor framework.
Combines descriptive statistics, inferential testing, effect size analysis, and advanced visualizations
to provide deep insights into reflection mechanisms, model capabilities, and strategic behaviors.

Author: Research Team
Date: 2025
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from scipy import stats
from scipy.stats import (
    f_oneway, ttest_ind, ttest_1samp, chi2_contingency, 
    mannwhitneyu, kruskal, levene, shapiro, anderson,
    pearsonr, spearmanr, ks_2samp
)
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.power import FTestAnovaPower, TTestPower
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
from itertools import combinations, product
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set sophisticated plotting style
plt.style.use('default')
sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'font.family': 'serif'
})


class ComprehensiveLLMNegotiationAnalyzer:
    """
    Comprehensive analyzer for LLM negotiation experiments.
    
    Provides publication-ready statistical analysis, effect size calculations,
    advanced visualizations, and strategic behavior insights.
    """
    
    def __init__(self, results_file: str = None, config: Dict[str, Any] = None):
        """Initialize the comprehensive analyzer."""
        self.config = config or {}
        self.results_file = results_file
        self.data = None
        self.successful_data = None
        self.analysis_results = {}
        
        # Experimental constants
        self.OPTIMAL_PRICE = 65
        self.RETAIL_PRICE = 100
        self.PRODUCTION_COST = 30
        self.DEMAND_MEAN = 40
        self.DEMAND_STD = 10
        
        # Statistical constants
        self.ALPHA = 0.05
        self.EFFECT_SIZE_THRESHOLDS = {
            'small': {'eta2': 0.01, 'cohens_d': 0.2, 'cohens_f': 0.1},
            'medium': {'eta2': 0.06, 'cohens_d': 0.5, 'cohens_f': 0.25},
            'large': {'eta2': 0.14, 'cohens_d': 0.8, 'cohens_f': 0.4}
        }
        
        # Model classifications
        self.MODEL_TIERS = {
            'qwen2:1.5b': 'Ultra-Compact',
            'gemma2:2b': 'Compact',
            'phi3:mini': 'Compact',
            'llama3.2:latest': 'Compact',
            'mistral:instruct': 'Mid-Range',
            'qwen:7b': 'Mid-Range',
            'qwen3:latest': 'Large',
            'claude-sonnet-4-remote': 'Premium',
            'o3-remote': 'Premium',
            'grok-remote': 'Premium'
        }
        
        self.MODEL_FAMILIES = {
            'qwen2:1.5b': 'Qwen', 'qwen:7b': 'Qwen', 'qwen3:latest': 'Qwen',
            'gemma2:2b': 'Gemma', 'phi3:mini': 'Phi', 'llama3.2:latest': 'Llama',
            'mistral:instruct': 'Mistral', 'claude-sonnet-4-remote': 'Claude',
            'o3-remote': 'GPT', 'grok-remote': 'Grok'
        }
        
        self.REFLECTION_PATTERNS = {
            '00': 'No Reflection',
            '01': 'Buyer Only',
            '10': 'Supplier Only',
            '11': 'Both Reflect'
        }
        
        logger.info("Initialized Comprehensive LLM Negotiation Analyzer")
    
    def load_and_validate_data(self, results_file: str = None) -> bool:
        """Load and comprehensively validate experimental data."""
        if results_file:
            self.results_file = Path(results_file)
        elif not self.results_file:
            results_dir = Path("./experiment_results")
            files = list(results_dir.glob("complete_results_*.json"))
            if not files:
                logger.error("No results files found")
                return False
            self.results_file = max(files, key=lambda f: f.stat().st_mtime)
        
        logger.info(f"Loading data from: {self.results_file}")
        
        try:
            with open(self.results_file, 'r') as f:
                data = json.load(f)
            
            # Handle different data formats
            if 'results' in data:
                results_list = data['results']
            else:
                results_list = data
            
            self.data = pd.DataFrame(results_list)
            
            # Comprehensive data validation and cleaning
            self._validate_and_clean_data()
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def _validate_and_clean_data(self):
        """Comprehensive data validation and cleaning."""
        logger.info("🔍 Comprehensive Data Validation and Cleaning")
        logger.info("=" * 60)
        
        initial_rows = len(self.data)
        logger.info(f"Initial dataset: {initial_rows:,} negotiations")
        
        # Data type conversions and cleaning
        self.data['completed'] = self.data['completed'].astype(bool)
        self.data['agreed_price'] = pd.to_numeric(self.data['agreed_price'], errors='coerce')
        self.data['total_rounds'] = pd.to_numeric(self.data.get('total_rounds', 0), errors='coerce').fillna(0)
        self.data['total_tokens'] = pd.to_numeric(self.data.get('total_tokens', 0), errors='coerce').fillna(0)
        self.data['total_time'] = pd.to_numeric(self.data.get('total_time', 0), errors='coerce').fillna(0)
        
        # Enhanced success criteria
        self.data['has_valid_price'] = (
            pd.notna(self.data['agreed_price']) & 
            (self.data['agreed_price'] > 0) &
            (self.data['agreed_price'] <= 200)  # Reasonable upper bound
        )
        
        self.data['true_success'] = (
            self.data['completed'] & 
            self.data['has_valid_price']
        )
        
        # Create successful negotiations subset
        self.successful_data = self.data[self.data['true_success']].copy()
        
        if len(self.successful_data) == 0:
            logger.error("No successful negotiations found!")
            return
        
        # Add derived variables
        self._add_derived_variables()
        
        # Data quality assessment
        self._assess_data_quality()
        
        # Experimental design validation
        self._validate_experimental_design()
    
    def _add_derived_variables(self):
        """Add derived variables for analysis."""
        # Price-related variables
        self.successful_data['buyer_advantage'] = self.OPTIMAL_PRICE - self.successful_data['agreed_price']
        self.successful_data['supplier_advantage'] = self.successful_data['agreed_price'] - self.PRODUCTION_COST
        self.successful_data['distance_from_optimal'] = abs(self.successful_data['agreed_price'] - self.OPTIMAL_PRICE)
        self.successful_data['price_efficiency'] = 1 - (self.successful_data['distance_from_optimal'] / self.OPTIMAL_PRICE)
        
        # Model classifications
        self.successful_data['buyer_tier'] = self.successful_data['buyer_model'].map(self.MODEL_TIERS)
        self.successful_data['supplier_tier'] = self.successful_data['supplier_model'].map(self.MODEL_TIERS)
        self.successful_data['buyer_family'] = self.successful_data['buyer_model'].map(self.MODEL_FAMILIES)
        self.successful_data['supplier_family'] = self.successful_data['supplier_model'].map(self.MODEL_FAMILIES)
        
        # Negotiation characteristics
        self.successful_data['is_homogeneous'] = (
            self.successful_data['buyer_model'] == self.successful_data['supplier_model']
        )
        self.successful_data['tier_match'] = (
            self.successful_data['buyer_tier'] == self.successful_data['supplier_tier']
        )
        self.successful_data['reflection_name'] = self.successful_data['reflection_pattern'].map(self.REFLECTION_PATTERNS)
        
        # Efficiency metrics
        self.successful_data['tokens_per_round'] = np.where(
            self.successful_data['total_rounds'] > 0,
            self.successful_data['total_tokens'] / self.successful_data['total_rounds'],
            0
        )
        
        self.successful_data['time_per_round'] = np.where(
            self.successful_data['total_rounds'] > 0,
            self.successful_data['total_time'] / self.successful_data['total_rounds'],
            0
        )
        
        # Strategic categories
        self.successful_data['price_category'] = pd.cut(
            self.successful_data['agreed_price'],
            bins=[0, 45, 55, 75, 200],
            labels=['Low (<$45)', 'Below Optimal ($45-55)', 'Above Optimal ($55-75)', 'High (>$75)']
        )
        
        # Profit calculations (using expected demand)
        expected_demand = self.DEMAND_MEAN
        self.successful_data['buyer_profit'] = (
            (self.RETAIL_PRICE - self.successful_data['agreed_price']) * expected_demand
        )
        self.successful_data['supplier_profit'] = (
            (self.successful_data['agreed_price'] - self.PRODUCTION_COST) * expected_demand
        )
        self.successful_data['total_profit'] = (
            self.successful_data['buyer_profit'] + self.successful_data['supplier_profit']
        )
        
        logger.info(f"Added derived variables for {len(self.successful_data):,} successful negotiations")
    
    def _assess_data_quality(self):
        """Assess data quality and completeness."""
        logger.info("📊 Data Quality Assessment")
        logger.info("-" * 40)
        
        # Missing data analysis
        missing_data = self.data.isnull().sum()
        critical_missing = missing_data[missing_data > 0]
        
        if len(critical_missing) > 0:
            logger.warning("Missing data detected:")
            for col, count in critical_missing.items():
                pct = count / len(self.data) * 100
                logger.warning(f"  {col}: {count:,} ({pct:.1f}%)")
        else:
            logger.info("✅ No missing data in critical columns")
        
        # Success rates by key variables
        total_n = len(self.data)
        success_n = len(self.successful_data)
        success_rate = success_n / total_n
        
        logger.info(f"Overall success rate: {success_rate:.1%} ({success_n:,}/{total_n:,})")
        
        if success_rate < 0.3:
            logger.warning("⚠️ Low success rate may affect statistical power")
        elif success_rate < 0.7:
            logger.info("📈 Moderate success rate - adequate for analysis")
        else:
            logger.info("🎉 High success rate - excellent data quality")
        
        # Price distribution validation
        if len(self.successful_data) > 0:
            price_stats = self.successful_data['agreed_price'].describe()
            logger.info(f"\nPrice distribution (successful negotiations):")
            logger.info(f"  Mean: ${price_stats['mean']:.2f}")
            logger.info(f"  Median: ${price_stats['50%']:.2f}")
            logger.info(f"  Range: ${price_stats['min']:.2f} - ${price_stats['max']:.2f}")
            logger.info(f"  Optimal: ${self.OPTIMAL_PRICE}")
    
    def _validate_experimental_design(self):
        """Validate experimental design balance and completeness."""
        logger.info("🔬 Experimental Design Validation")
        logger.info("-" * 40)
        
        # Check factor combinations
        if all(col in self.data.columns for col in ['buyer_model', 'supplier_model', 'reflection_pattern']):
            # Model combinations
            model_combinations = self.data.groupby(['buyer_model', 'supplier_model']).size()
            logger.info(f"Model combinations: {len(model_combinations)}")
            
            # Reflection pattern distribution
            reflection_dist = self.data['reflection_pattern'].value_counts().sort_index()
            logger.info(f"Reflection patterns: {dict(reflection_dist)}")
            
            # Balance assessment
            min_n = model_combinations.min()
            max_n = model_combinations.max()
            balance_ratio = min_n / max_n if max_n > 0 else 0
            
            logger.info(f"Sample size range: {min_n} - {max_n} (balance ratio: {balance_ratio:.2f})")
            
            if balance_ratio < 0.5:
                logger.warning("⚠️ Unbalanced design detected")
            else:
                logger.info("✅ Reasonably balanced design")
            
            # Power analysis preview
            if len(self.successful_data) > 0:
                self._quick_power_assessment()
    
    def _quick_power_assessment(self):
        """Quick power assessment for main effects."""
        # Sample size for smallest group in reflection analysis
        reflection_sizes = self.successful_data['reflection_pattern'].value_counts()
        min_group_size = reflection_sizes.min()
        
        # Estimate effect size from data
        if len(self.successful_data) > 0:
            reflection_means = self.successful_data.groupby('reflection_pattern')['agreed_price'].mean()
            if len(reflection_means) > 1:
                grand_mean = self.successful_data['agreed_price'].mean()
                between_var = np.var(reflection_means)
                total_var = np.var(self.successful_data['agreed_price'])
                eta_squared = between_var / total_var if total_var > 0 else 0
                
                logger.info(f"Reflection analysis: min n={min_group_size}, estimated η²={eta_squared:.3f}")
                
                if min_group_size < 30:
                    logger.warning("⚠️ Small group sizes may limit power")
                elif eta_squared < 0.01:
                    logger.warning("⚠️ Small effect size detected")
                else:
                    logger.info("✅ Adequate power expected")
    
    def comprehensive_descriptive_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive descriptive statistics."""
        logger.info("📈 Comprehensive Descriptive Analysis")
        logger.info("=" * 50)
        
        desc_stats = {}
        
        # Overall sample characteristics
        desc_stats['sample_overview'] = self._analyze_sample_overview()
        
        # Price analysis
        desc_stats['price_analysis'] = self._analyze_price_distributions()
        
        # Model performance
        desc_stats['model_analysis'] = self._analyze_model_performance()
        
        # Reflection effects
        desc_stats['reflection_analysis'] = self._analyze_reflection_patterns()
        
        # Negotiation efficiency
        desc_stats['efficiency_analysis'] = self._analyze_negotiation_efficiency()
        
        # Strategic behaviors
        desc_stats['strategic_analysis'] = self._analyze_strategic_behaviors()
        
        self.analysis_results['descriptive_analysis'] = desc_stats
        return desc_stats
    
    def _analyze_sample_overview(self) -> Dict[str, Any]:
        """Analyze overall sample characteristics."""
        total_n = len(self.data)
        success_n = len(self.successful_data)
        
        overview = {
            'total_negotiations': total_n,
            'successful_negotiations': success_n,
            'success_rate': success_n / total_n,
            'unique_models': len(set(self.data['buyer_model'].unique()) | set(self.data['supplier_model'].unique())),
            'reflection_patterns': len(self.data['reflection_pattern'].unique()),
            'data_collection_span': 'experiment_duration',  # Could calculate from timestamps
        }
        
        if len(self.successful_data) > 0:
            overview['negotiation_characteristics'] = {
                'avg_rounds': float(self.successful_data['total_rounds'].mean()),
                'avg_tokens': float(self.successful_data['total_tokens'].mean()),
                'avg_duration': float(self.successful_data['total_time'].mean()),
                'price_range': (float(self.successful_data['agreed_price'].min()), 
                              float(self.successful_data['agreed_price'].max())),
            }
        
        logger.info(f"Sample overview: {success_n:,}/{total_n:,} successful ({overview['success_rate']:.1%})")
        
        return overview
    
    def _analyze_price_distributions(self) -> Dict[str, Any]:
        """Comprehensive price distribution analysis."""
        if len(self.successful_data) == 0:
            return {}
        
        prices = self.successful_data['agreed_price']
        
        # Central tendency and dispersion
        price_stats = {
            'descriptive_stats': {
                'mean': float(prices.mean()),
                'median': float(prices.median()),
                'mode': float(prices.mode().iloc[0]) if len(prices.mode()) > 0 else None,
                'std': float(prices.std()),
                'variance': float(prices.var()),
                'cv': float(prices.std() / prices.mean()) if prices.mean() != 0 else 0,
                'range': float(prices.max() - prices.min()),
                'iqr': float(prices.quantile(0.75) - prices.quantile(0.25)),
            },
            'distributional_properties': {
                'skewness': float(prices.skew()),
                'kurtosis': float(prices.kurtosis()),
                'normality_test': stats.shapiro(prices[:5000]) if len(prices) <= 5000 else stats.anderson(prices),
            },
            'percentiles': {
                f'p{p}': float(prices.quantile(p/100)) 
                for p in [5, 10, 25, 50, 75, 90, 95]
            },
            'optimal_price_analysis': {
                'optimal_price': self.OPTIMAL_PRICE,
                'distance_from_optimal': {
                    'mean': float(abs(prices - self.OPTIMAL_PRICE).mean()),
                    'median': float(abs(prices - self.OPTIMAL_PRICE).median()),
                    'within_5': float((abs(prices - self.OPTIMAL_PRICE) <= 5).mean()),
                    'within_10': float((abs(prices - self.OPTIMAL_PRICE) <= 10).mean()),
                }
            }
        }
        
        # Price convergence analysis
        price_stats['convergence_analysis'] = {
            'below_optimal': float((prices < self.OPTIMAL_PRICE).mean()),
            'at_optimal': float((prices == self.OPTIMAL_PRICE).mean()),
            'above_optimal': float((prices > self.OPTIMAL_PRICE).mean()),
            'buyer_advantage': float((self.OPTIMAL_PRICE - prices).mean()),
            'supplier_advantage': float((prices - self.PRODUCTION_COST).mean()),
        }
        
        logger.info(f"Price analysis: μ=${price_stats['descriptive_stats']['mean']:.2f}, "
                   f"σ=${price_stats['descriptive_stats']['std']:.2f}, "
                   f"optimal_distance=${price_stats['optimal_price_analysis']['distance_from_optimal']['mean']:.2f}")
        
        return price_stats
    
    def _analyze_model_performance(self) -> Dict[str, Any]:
        """Comprehensive model performance analysis."""
        model_analysis = {}
        
        # Individual model performance
        model_analysis['individual_models'] = {}
        
        for model in self.MODEL_TIERS.keys():
            if model in self.data['buyer_model'].values or model in self.data['supplier_model'].values:
                model_stats = self._analyze_single_model(model)
                if model_stats:
                    model_analysis['individual_models'][model] = model_stats
        
        # Model tier analysis
        model_analysis['tier_analysis'] = self._analyze_model_tiers()
        
        # Model family analysis
        model_analysis['family_analysis'] = self._analyze_model_families()
        
        # Pairing effects
        model_analysis['pairing_effects'] = self._analyze_model_pairings()
        
        logger.info(f"Analyzed {len(model_analysis['individual_models'])} individual models")
        
        return model_analysis
    
    def _analyze_single_model(self, model: str) -> Dict[str, Any]:
        """Analyze performance of a single model."""
        # As buyer
        buyer_data = self.successful_data[self.successful_data['buyer_model'] == model]
        buyer_all = self.data[self.data['buyer_model'] == model]
        
        # As supplier
        supplier_data = self.successful_data[self.successful_data['supplier_model'] == model]
        supplier_all = self.data[self.data['supplier_model'] == model]
        
        if len(buyer_all) == 0 and len(supplier_all) == 0:
            return None
        
        model_stats = {
            'model_info': {
                'tier': self.MODEL_TIERS.get(model, 'Unknown'),
                'family': self.MODEL_FAMILIES.get(model, 'Unknown'),
            },
            'as_buyer': self._calculate_role_stats(buyer_data, buyer_all),
            'as_supplier': self._calculate_role_stats(supplier_data, supplier_all),
        }
        
        # Overall performance
        all_successful = pd.concat([buyer_data, supplier_data])
        all_attempts = pd.concat([buyer_all, supplier_all])
        
        if len(all_attempts) > 0:
            model_stats['overall'] = self._calculate_role_stats(all_successful, all_attempts)
            
            # Role asymmetry
            if len(buyer_data) > 0 and len(supplier_data) > 0:
                buyer_price = buyer_data['agreed_price'].mean()
                supplier_price = supplier_data['agreed_price'].mean()
                model_stats['role_asymmetry'] = {
                    'price_difference': float(buyer_price - supplier_price),
                    'asymmetry_magnitude': float(abs(buyer_price - supplier_price)),
                    'preferred_role': 'buyer' if buyer_price < supplier_price else 'supplier'
                }
        
        return model_stats
    
    def _calculate_role_stats(self, successful_data: pd.DataFrame, all_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics for a model in a specific role."""
        if len(all_data) == 0:
            return {'count': 0}
        
        stats = {
            'count': len(all_data),
            'successful': len(successful_data),
            'success_rate': len(successful_data) / len(all_data),
        }
        
        if len(successful_data) > 0:
            prices = successful_data['agreed_price']
            stats.update({
                'mean_price': float(prices.mean()),
                'median_price': float(prices.median()),
                'std_price': float(prices.std()),
                'buyer_advantage': float((self.OPTIMAL_PRICE - prices).mean()),
                'distance_from_optimal': float(abs(prices - self.OPTIMAL_PRICE).mean()),
                'avg_rounds': float(successful_data['total_rounds'].mean()),
                'avg_tokens': float(successful_data['total_tokens'].mean()),
            })
        
        return stats
    
    def _analyze_model_tiers(self) -> Dict[str, Any]:
        """Analyze performance by model tier."""
        tier_analysis = {}
        
        for tier in set(self.MODEL_TIERS.values()):
            # Combine buyer and supplier data for this tier
            tier_buyer = self.successful_data[self.successful_data['buyer_tier'] == tier]
            tier_supplier = self.successful_data[self.successful_data['supplier_tier'] == tier]
            tier_data = pd.concat([tier_buyer, tier_supplier])
            
            tier_buyer_all = self.data[self.data.get('buyer_tier') == tier] if 'buyer_tier' in self.data else pd.DataFrame()
            tier_supplier_all = self.data[self.data.get('supplier_tier') == tier] if 'supplier_tier' in self.data else pd.DataFrame()
            tier_all = pd.concat([tier_buyer_all, tier_supplier_all])
            
            if len(tier_all) > 0:
                tier_analysis[tier] = self._calculate_role_stats(tier_data, tier_all)
        
        return tier_analysis
    
    def _analyze_model_families(self) -> Dict[str, Any]:
        """Analyze performance by model family."""
        family_analysis = {}
        
        for family in set(self.MODEL_FAMILIES.values()):
            # Combine buyer and supplier data for this family
            family_buyer = self.successful_data[self.successful_data['buyer_family'] == family]
            family_supplier = self.successful_data[self.successful_data['supplier_family'] == family]
            family_data = pd.concat([family_buyer, family_supplier])
            
            # Count all attempts
            family_buyer_all = self.data[self.data['buyer_model'].map(self.MODEL_FAMILIES) == family]
            family_supplier_all = self.data[self.data['supplier_model'].map(self.MODEL_FAMILIES) == family]
            family_all = pd.concat([family_buyer_all, family_supplier_all])
            
            if len(family_all) > 0:
                family_analysis[family] = self._calculate_role_stats(family_data, family_all)
        
        return family_analysis
    
    def _analyze_model_pairings(self) -> Dict[str, Any]:
        """Analyze model pairing effects."""
        pairing_analysis = {
            'homogeneous_vs_heterogeneous': {},
            'tier_matching_effects': {},
            'cross_family_effects': {}
        }
        
        if len(self.successful_data) > 0:
            # Homogeneous vs heterogeneous
            homo_data = self.successful_data[self.successful_data['is_homogeneous']]
            hetero_data = self.successful_data[~self.successful_data['is_homogeneous']]
            
            pairing_analysis['homogeneous_vs_heterogeneous'] = {
                'homogeneous': self._calculate_pairing_stats(homo_data),
                'heterogeneous': self._calculate_pairing_stats(hetero_data)
            }
            
            # Tier matching effects
            tier_match_data = self.successful_data[self.successful_data['tier_match']]
            tier_mismatch_data = self.successful_data[~self.successful_data['tier_match']]
            
            pairing_analysis['tier_matching_effects'] = {
                'matched_tiers': self._calculate_pairing_stats(tier_match_data),
                'mismatched_tiers': self._calculate_pairing_stats(tier_mismatch_data)
            }
        
        return pairing_analysis
    
    def _calculate_pairing_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics for model pairings."""
        if len(data) == 0:
            return {'count': 0}
        
        return {
            'count': len(data),
            'mean_price': float(data['agreed_price'].mean()),
            'std_price': float(data['agreed_price'].std()),
            'buyer_advantage': float(data['buyer_advantage'].mean()),
            'avg_rounds': float(data['total_rounds'].mean()),
            'avg_tokens': float(data['total_tokens'].mean()),
        }
    
    def _analyze_reflection_patterns(self) -> Dict[str, Any]:
        """Comprehensive reflection pattern analysis."""
        reflection_analysis = {}
        
        for pattern, name in self.REFLECTION_PATTERNS.items():
            pattern_data = self.successful_data[self.successful_data['reflection_pattern'] == pattern]
            pattern_all = self.data[self.data['reflection_pattern'] == pattern]
            
            reflection_analysis[pattern] = {
                'name': name,
                'stats': self._calculate_role_stats(pattern_data, pattern_all)
            }
            
            # Additional reflection-specific metrics
            if len(pattern_data) > 0:
                reflection_analysis[pattern]['reflection_effects'] = {
                    'price_variance': float(pattern_data['agreed_price'].var()),
                    'convergence_rate': float((abs(pattern_data['agreed_price'] - self.OPTIMAL_PRICE) <= 5).mean()),
                    'efficiency_score': float(pattern_data['price_efficiency'].mean()),
                }
        
        return reflection_analysis
    
    def _analyze_negotiation_efficiency(self) -> Dict[str, Any]:
        """Analyze negotiation efficiency and resource usage."""
        if len(self.successful_data) == 0:
            return {}
        
        efficiency_analysis = {
            'round_efficiency': {
                'mean_rounds': float(self.successful_data['total_rounds'].mean()),
                'median_rounds': float(self.successful_data['total_rounds'].median()),
                'rounds_distribution': self.successful_data['total_rounds'].value_counts().to_dict(),
            },
            'token_efficiency': {
                'mean_tokens': float(self.successful_data['total_tokens'].mean()),
                'median_tokens': float(self.successful_data['total_tokens'].median()),
                'tokens_per_round': float(self.successful_data['tokens_per_round'].mean()),
            },
            'time_efficiency': {
                'mean_time': float(self.successful_data['total_time'].mean()),
                'median_time': float(self.successful_data['total_time'].median()),
                'time_per_round': float(self.successful_data['time_per_round'].mean()),
            }
        }
        
        # Efficiency correlations
        efficiency_analysis['efficiency_correlations'] = {
            'rounds_vs_price_quality': float(stats.pearsonr(
                self.successful_data['total_rounds'], 
                self.successful_data['price_efficiency']
            )[0]),
            'tokens_vs_success': float(stats.pearsonr(
                self.successful_data['total_tokens'],
                self.successful_data['price_efficiency']
            )[0]),
        }
        
        return efficiency_analysis
    
    def _analyze_strategic_behaviors(self) -> Dict[str, Any]:
        """Analyze strategic behaviors and patterns."""
        if len(self.successful_data) == 0:
            return {}
        
        strategic_analysis = {
            'profit_distribution': {
                'buyer_profit': {
                    'mean': float(self.successful_data['buyer_profit'].mean()),
                    'median': float(self.successful_data['buyer_profit'].median()),
                    'positive_rate': float((self.successful_data['buyer_profit'] > 0).mean()),
                },
                'supplier_profit': {
                    'mean': float(self.successful_data['supplier_profit'].mean()),
                    'median': float(self.successful_data['supplier_profit'].median()),
                    'positive_rate': float((self.successful_data['supplier_profit'] > 0).mean()),
                },
                'total_efficiency': {
                    'mean_total_profit': float(self.successful_data['total_profit'].mean()),
                    'pareto_efficiency': float((self.successful_data['total_profit'] / (self.RETAIL_PRICE - self.PRODUCTION_COST) / self.DEMAND_MEAN).mean()),
                }
            },
            'price_categories': self.successful_data['price_category'].value_counts().to_dict(),
            'bargaining_patterns': {
                'buyer_advantage_distribution': {
                    'mean': float(self.successful_data['buyer_advantage'].mean()),
                    'std': float(self.successful_data['buyer_advantage'].std()),
                    'median': float(self.successful_data['buyer_advantage'].median()),
                    'positive_rate': float((self.successful_data['buyer_advantage'] > 0).mean()),
                }
            }
        }
        
        return strategic_analysis
    
    def inferential_statistical_analysis(self) -> Dict[str, Any]:
        """Comprehensive inferential statistical analysis."""
        logger.info("📊 Inferential Statistical Analysis")
        logger.info("=" * 50)
        
        statistical_results = {}
        
        # Research Question 1: Reflection Effects
        statistical_results['reflection_effects'] = self._test_reflection_effects()
        
        # Research Question 2: Model Size/Tier Effects
        statistical_results['model_effects'] = self._test_model_effects()
        
        # Research Question 3: Role Asymmetry
        statistical_results['role_asymmetry'] = self._test_role_asymmetry()
        
        # Research Question 4: Interaction Effects
        statistical_results['interaction_effects'] = self._test_interaction_effects()
        
        # Additional analyses
        statistical_results['efficiency_effects'] = self._test_efficiency_effects()
        statistical_results['strategic_effects'] = self._test_strategic_effects()
        
        # Power analysis
        statistical_results['power_analysis'] = self._comprehensive_power_analysis()
        
        self.analysis_results['statistical_analysis'] = statistical_results
        return statistical_results
    
    def _test_reflection_effects(self) -> Dict[str, Any]:
        """Test reflection pattern effects on negotiation outcomes."""
        if len(self.successful_data) == 0:
            return {'error': 'No successful negotiations for analysis'}
        
        reflection_results = {}
        
        # ANOVA on agreed prices by reflection pattern
        reflection_groups = []
        reflection_labels = []
        
        for pattern in ['00', '01', '10', '11']:
            group_data = self.successful_data[
                self.successful_data['reflection_pattern'] == pattern
            ]['agreed_price'].dropna()
            
            if len(group_data) > 0:
                reflection_groups.append(group_data)
                reflection_labels.append(pattern)
        
        if len(reflection_groups) >= 2:
            # Parametric ANOVA
            f_stat, p_value = f_oneway(*reflection_groups)
            
            # Non-parametric alternative
            h_stat, h_p_value = kruskal(*reflection_groups)
            
            # Effect size (eta-squared)
            eta_squared = self._calculate_eta_squared(reflection_groups)
            
            # Assumption testing
            assumptions = self._test_anova_assumptions(reflection_groups, reflection_labels)
            
            reflection_results['price_effects'] = {
                'parametric': {'f_statistic': f_stat, 'p_value': p_value},
                'nonparametric': {'h_statistic': h_stat, 'p_value': h_p_value},
                'effect_size': {'eta_squared': eta_squared, 'interpretation': self._interpret_eta_squared(eta_squared)},
                'assumptions': assumptions,
                'group_sizes': [len(group) for group in reflection_groups],
                'group_means': [float(group.mean()) for group in reflection_groups],
                'significant': p_value < self.ALPHA
            }
            
            # Post-hoc analysis if significant
            if p_value < self.ALPHA and len(reflection_groups) > 2:
                reflection_results['posthoc'] = self._posthoc_reflection_analysis(reflection_groups, reflection_labels)
        
        # Additional reflection analyses
        reflection_results.update(self._additional_reflection_tests())
        
        logger.info(f"Reflection effects: F={reflection_results.get('price_effects', {}).get('parametric', {}).get('f_statistic', 0):.3f}, "
                   f"p={reflection_results.get('price_effects', {}).get('parametric', {}).get('p_value', 1):.3f}")
        
        return reflection_results
    
    def _test_model_effects(self) -> Dict[str, Any]:
        """Test model tier and family effects."""
        model_results = {}
        
        # Model tier effects
        tier_groups = []
        tier_labels = []
        
        for tier in set(self.MODEL_TIERS.values()):
            # Combine buyer and supplier data for this tier
            tier_data = pd.concat([
                self.successful_data[self.successful_data['buyer_tier'] == tier]['agreed_price'],
                self.successful_data[self.successful_data['supplier_tier'] == tier]['agreed_price']
            ]).dropna()
            
            if len(tier_data) > 0:
                tier_groups.append(tier_data)
                tier_labels.append(tier)
        
        if len(tier_groups) >= 2:
            f_stat, p_value = f_oneway(*tier_groups)
            h_stat, h_p_value = kruskal(*tier_groups)
            eta_squared = self._calculate_eta_squared(tier_groups)
            
            model_results['tier_effects'] = {
                'parametric': {'f_statistic': f_stat, 'p_value': p_value},
                'nonparametric': {'h_statistic': h_stat, 'p_value': h_p_value},
                'effect_size': {'eta_squared': eta_squared, 'interpretation': self._interpret_eta_squared(eta_squared)},
                'tier_labels': tier_labels,
                'group_means': [float(group.mean()) for group in tier_groups],
                'significant': p_value < self.ALPHA
            }
        
        # Model family effects
        family_results = self._test_model_family_effects()
        if family_results:
            model_results['family_effects'] = family_results
        
        # Individual model analysis
        model_results['individual_model_tests'] = self._test_individual_models()
        
        return model_results
    
    def _test_role_asymmetry(self) -> Dict[str, Any]:
        """Test for systematic role asymmetry (buyer advantage)."""
        if len(self.successful_data) == 0:
            return {}
        
        buyer_advantages = self.successful_data['buyer_advantage']
        
        # One-sample t-test against 0 (no buyer advantage)
        t_stat, p_value = ttest_1samp(buyer_advantages, 0)
        
        # Effect size (Cohen's d)
        cohens_d = buyer_advantages.mean() / buyer_advantages.std() if buyer_advantages.std() > 0 else 0
        
        # Confidence interval
        n = len(buyer_advantages)
        sem = buyer_advantages.std() / np.sqrt(n)
        ci_95 = stats.t.interval(0.95, n-1, buyer_advantages.mean(), sem)
        
        # Non-parametric test
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(buyer_advantages, alternative='two-sided')
        
        role_asymmetry = {
            'buyer_advantage_test': {
                'mean_advantage': float(buyer_advantages.mean()),
                'std_advantage': float(buyer_advantages.std()),
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'effect_interpretation': self._interpret_cohens_d(cohens_d),
                'ci_95': ci_95,
                'significant': p_value < self.ALPHA
            },
            'nonparametric_test': {
                'wilcoxon_statistic': wilcoxon_stat,
                'p_value': wilcoxon_p,
                'significant': wilcoxon_p < self.ALPHA
            },
            'distribution_analysis': {
                'proportion_positive': float((buyer_advantages > 0).mean()),
                'proportion_negative': float((buyer_advantages < 0).mean()),
                'proportion_zero': float((buyer_advantages == 0).mean()),
            }
        }
        
        # Test consistency across conditions
        role_asymmetry['consistency_tests'] = self._test_asymmetry_consistency()
        
        logger.info(f"Role asymmetry: buyer advantage=${buyer_advantages.mean():.2f}, "
                   f"t={t_stat:.3f}, p={p_value:.3f}, d={cohens_d:.3f}")
        
        return role_asymmetry
    
    def _test_interaction_effects(self) -> Dict[str, Any]:
        """Test for interaction effects between factors."""
        interaction_results = {}
        
        if len(self.successful_data) > 100:  # Need sufficient data
            try:
                # Reflection × Model Tier interaction
                factorial_data = self.successful_data.dropna(subset=['agreed_price', 'reflection_pattern', 'buyer_tier'])
                
                if len(factorial_data) > 50:
                    # Two-way ANOVA
                    formula = 'agreed_price ~ C(reflection_pattern) + C(buyer_tier) + C(reflection_pattern):C(buyer_tier)'
                    model = ols(formula, data=factorial_data).fit()
                    anova_table = anova_lm(model, typ=2)
                    
                    interaction_results['reflection_x_tier'] = {
                        'anova_results': {
                            'reflection_main': {
                                'f_stat': anova_table.iloc[0]['F'],
                                'p_value': anova_table.iloc[0]['PR(>F)'],
                                'significant': anova_table.iloc[0]['PR(>F)'] < self.ALPHA
                            },
                            'tier_main': {
                                'f_stat': anova_table.iloc[1]['F'],
                                'p_value': anova_table.iloc[1]['PR(>F)'],
                                'significant': anova_table.iloc[1]['PR(>F)'] < self.ALPHA
                            },
                            'interaction': {
                                'f_stat': anova_table.iloc[2]['F'],
                                'p_value': anova_table.iloc[2]['PR(>F)'],
                                'significant': anova_table.iloc[2]['PR(>F)'] < self.ALPHA
                            }
                        },
                        'model_fit': {
                            'r_squared': model.rsquared,
                            'adjusted_r_squared': model.rsquared_adj,
                            'f_statistic': model.fvalue,
                            'f_p_value': model.f_pvalue
                        }
                    }
                
            except Exception as e:
                logger.warning(f"Interaction analysis failed: {e}")
                interaction_results['error'] = str(e)
        
        return interaction_results
    
    def _test_efficiency_effects(self) -> Dict[str, Any]:
        """Test effects on negotiation efficiency metrics."""
        if len(self.successful_data) == 0:
            return {}
        
        efficiency_results = {}
        
        # Round efficiency by reflection
        round_groups = []
        for pattern in ['00', '01', '10', '11']:
            group_rounds = self.successful_data[
                self.successful_data['reflection_pattern'] == pattern
            ]['total_rounds'].dropna()
            if len(group_rounds) > 0:
                round_groups.append(group_rounds)
        
        if len(round_groups) >= 2:
            f_stat, p_value = f_oneway(*round_groups)
            efficiency_results['rounds_by_reflection'] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < self.ALPHA,
                'group_means': [float(group.mean()) for group in round_groups]
            }
        
        # Token efficiency correlations
        if 'total_tokens' in self.successful_data.columns:
            token_price_corr = stats.pearsonr(
                self.successful_data['total_tokens'],
                self.successful_data['agreed_price']
            )
            
            efficiency_results['token_correlations'] = {
                'tokens_vs_price': {
                    'correlation': token_price_corr[0],
                    'p_value': token_price_corr[1],
                    'significant': token_price_corr[1] < self.ALPHA
                }
            }
        
        return efficiency_results
    
    def _test_strategic_effects(self) -> Dict[str, Any]:
        """Test strategic behavior effects."""
        if len(self.successful_data) == 0:
            return {}
        
        strategic_results = {}
        
        # Homogeneous vs heterogeneous pairing effects
        if 'is_homogeneous' in self.successful_data.columns:
            homo_prices = self.successful_data[self.successful_data['is_homogeneous']]['agreed_price']
            hetero_prices = self.successful_data[~self.successful_data['is_homogeneous']]['agreed_price']
            
            if len(homo_prices) > 0 and len(hetero_prices) > 0:
                t_stat, p_value = ttest_ind(homo_prices, hetero_prices)
                
                strategic_results['pairing_effects'] = {
                    'homogeneous_mean': float(homo_prices.mean()),
                    'heterogeneous_mean': float(hetero_prices.mean()),
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < self.ALPHA,
                    'effect_size': self._calculate_cohens_d(homo_prices, hetero_prices)
                }
        
        # Profit distribution tests
        buyer_profits = self.successful_data['buyer_profit']
        supplier_profits = self.successful_data['supplier_profit']
        
        # Test if profits are significantly different from zero
        buyer_t, buyer_p = ttest_1samp(buyer_profits, 0)
        supplier_t, supplier_p = ttest_1samp(supplier_profits, 0)
        
        strategic_results['profit_tests'] = {
            'buyer_profit_test': {
                'mean_profit': float(buyer_profits.mean()),
                't_statistic': buyer_t,
                'p_value': buyer_p,
                'significant': buyer_p < self.ALPHA
            },
            'supplier_profit_test': {
                'mean_profit': float(supplier_profits.mean()),
                't_statistic': supplier_t,
                'p_value': supplier_p,
                'significant': supplier_p < self.ALPHA
            }
        }
        
        return strategic_results
    
    def _comprehensive_power_analysis(self) -> Dict[str, Any]:
        """Comprehensive statistical power analysis."""
        power_results = {}
        
        # Sample sizes
        total_n = len(self.data)
        successful_n = len(self.successful_data)
        
        # Power for reflection ANOVA
        if len(self.successful_data) > 0:
            reflection_groups = self.successful_data['reflection_pattern'].value_counts()
            min_group_size = reflection_groups.min()
            
            try:
                # Estimate effect size from data
                reflection_means = self.successful_data.groupby('reflection_pattern')['agreed_price'].mean()
                grand_mean = self.successful_data['agreed_price'].mean()
                between_var = np.sum(reflection_groups * (reflection_means - grand_mean)**2)
                total_var = np.sum((self.successful_data['agreed_price'] - grand_mean)**2)
                eta_squared = between_var / total_var if total_var > 0 else 0
                
                cohens_f = np.sqrt(eta_squared / (1 - eta_squared)) if eta_squared < 1 else 0
                
                power_analysis_tool = FTestAnovaPower()
                observed_power = power_analysis_tool.solve_power(
                    effect_size=cohens_f,
                    nobs=min_group_size,
                    k_groups=len(reflection_groups),
                    alpha=self.ALPHA
                ) if cohens_f > 0 else 0
                
                power_results['reflection_anova'] = {
                    'effect_size_eta2': eta_squared,
                    'effect_size_cohens_f': cohens_f,
                    'min_group_size': min_group_size,
                    'num_groups': len(reflection_groups),
                    'observed_power': observed_power,
                    'adequate_power': observed_power >= 0.8
                }
                
            except Exception as e:
                logger.warning(f"Reflection power analysis failed: {e}")
        
        # Power for buyer advantage test
        if len(self.successful_data) > 0:
            buyer_advantages = self.successful_data['buyer_advantage']
            effect_size_d = buyer_advantages.mean() / buyer_advantages.std() if buyer_advantages.std() > 0 else 0
            
            try:
                ttest_power_tool = TTestPower()
                observed_power = ttest_power_tool.solve_power(
                    effect_size=abs(effect_size_d),
                    nobs=len(buyer_advantages),
                    alpha=self.ALPHA
                ) if effect_size_d != 0 else 0
                
                power_results['buyer_advantage'] = {
                    'effect_size_d': effect_size_d,
                    'sample_size': len(buyer_advantages),
                    'observed_power': observed_power,
                    'adequate_power': observed_power >= 0.8
                }
                
            except Exception as e:
                logger.warning(f"Buyer advantage power analysis failed: {e}")
        
        # Sample size recommendations
        power_results['recommendations'] = self._power_recommendations(power_results)
        
        return power_results
    
    def advanced_visualizations(self, output_dir: str = "./analysis"):
        """Create comprehensive advanced visualizations."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info("🎨 Creating Advanced Visualizations")
        logger.info("=" * 50)
        
        # Set style for publication quality
        plt.style.use('default')
        sns.set_style("whitegrid")
        plt.rcParams.update({
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'font.family': 'serif'
        })
        
        # Main analysis figures
        self._create_executive_dashboard(output_path)
        self._create_reflection_analysis_figure(output_path)
        self._create_model_performance_figure(output_path)
        self._create_strategic_behavior_figure(output_path)
        self._create_efficiency_analysis_figure(output_path)
        self._create_statistical_summary_figure(output_path)
        
        # Interactive visualizations
        self._create_interactive_dashboard(output_path)
        
        logger.info(f"All visualizations saved to: {output_path}")
    
    def _create_executive_dashboard(self, output_path: Path):
        """Create executive summary dashboard."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        if len(self.successful_data) == 0:
            return
        
        # Panel 1: Overall success rates
        success_data = self.data.groupby('reflection_pattern').agg({
            'true_success': ['count', 'sum', 'mean']
        })
        success_data.columns = ['Total', 'Successful', 'Rate']
        
        reflection_names = [self.REFLECTION_PATTERNS.get(p, p) for p in success_data.index]
        bars1 = axes[0,0].bar(reflection_names, success_data['Rate'], color='steelblue', alpha=0.8)
        axes[0,0].set_title('Success Rate by Reflection Pattern')
        axes[0,0].set_ylabel('Success Rate')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, rate in zip(bars1, success_data['Rate']):
            axes[0,0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                          f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # Panel 2: Price distribution
        axes[0,1].hist(self.successful_data['agreed_price'], bins=30, alpha=0.7, 
                      color='darkgreen', edgecolor='black')
        axes[0,1].axvline(x=self.OPTIMAL_PRICE, color='red', linestyle='--', 
                         linewidth=2, label=f'Optimal (${self.OPTIMAL_PRICE})')
        axes[0,1].axvline(x=self.successful_data['agreed_price'].mean(), color='blue', 
                         linestyle='--', linewidth=2, 
                         label=f'Mean (${self.successful_data["agreed_price"].mean():.1f})')
        axes[0,1].set_title('Price Distribution (Successful Negotiations)')
        axes[0,1].set_xlabel('Agreed Price ($)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].legend()
        
        # Panel 3: Buyer advantage by reflection
        buyer_adv_by_reflection = self.successful_data.groupby('reflection_pattern')['buyer_advantage'].mean()
        reflection_names = [self.REFLECTION_PATTERNS.get(p, p) for p in buyer_adv_by_reflection.index]
        
        colors = ['red' if x > 0 else 'blue' for x in buyer_adv_by_reflection.values]
        bars3 = axes[0,2].bar(reflection_names, buyer_adv_by_reflection.values, 
                             color=colors, alpha=0.7)
        axes[0,2].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[0,2].set_title('Buyer Advantage by Reflection')
        axes[0,2].set_ylabel('Buyer Advantage ($)')
        axes[0,2].tick_params(axis='x', rotation=45)
        
        # Panel 4: Model tier performance
        tier_performance = {}
        for tier in set(self.MODEL_TIERS.values()):
            tier_data = pd.concat([
                self.successful_data[self.successful_data['buyer_tier'] == tier],
                self.successful_data[self.successful_data['supplier_tier'] == tier]
            ])
            if len(tier_data) > 0:
                tier_performance[tier] = tier_data['agreed_price'].mean()
        
        if tier_performance:
            bars4 = axes[1,0].bar(tier_performance.keys(), tier_performance.values(), 
                                 color='orange', alpha=0.7)
            axes[1,0].axhline(y=self.OPTIMAL_PRICE, color='red', linestyle='--', linewidth=2)
            axes[1,0].set_title('Average Price by Model Tier')
            axes[1,0].set_ylabel('Average Price ($)')
            axes[1,0].tick_params(axis='x', rotation=45)
        
        # Panel 5: Negotiation efficiency
        efficiency_by_reflection = self.successful_data.groupby('reflection_pattern')['total_rounds'].mean()
        reflection_names = [self.REFLECTION_PATTERNS.get(p, p) for p in efficiency_by_reflection.index]
        
        bars5 = axes[1,1].bar(reflection_names, efficiency_by_reflection.values, 
                             color='purple', alpha=0.7)
        axes[1,1].set_title('Average Rounds by Reflection')
        axes[1,1].set_ylabel('Average Rounds')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        # Panel 6: Key statistics summary
        axes[1,2].axis('off')
        
        # Calculate key statistics
        total_negotiations = len(self.data)
        successful_negotiations = len(self.successful_data)
        success_rate = successful_negotiations / total_negotiations
        mean_price = self.successful_data['agreed_price'].mean()
        mean_buyer_advantage = self.successful_data['buyer_advantage'].mean()
        optimal_distance = abs(self.successful_data['agreed_price'] - self.OPTIMAL_PRICE).mean()
        
        stats_text = f"""Key Statistics
        
Total Negotiations: {total_negotiations:,}
Successful: {successful_negotiations:,} ({success_rate:.1%})

Mean Price: ${mean_price:.2f}
Optimal Price: ${self.OPTIMAL_PRICE}
Distance from Optimal: ${optimal_distance:.2f}

Buyer Advantage: ${mean_buyer_advantage:.2f}
{'Significant bias toward buyers' if mean_buyer_advantage > 2 else 'Balanced outcomes'}

Models Tested: {len(self.MODEL_TIERS)}
Reflection Patterns: {len(self.REFLECTION_PATTERNS)}"""
        
        axes[1,2].text(0.1, 0.9, stats_text, transform=axes[1,2].transAxes, 
                      fontsize=11, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path / 'executive_dashboard.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_path / 'executive_dashboard.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_reflection_analysis_figure(self, output_path: Path):
        """Create detailed reflection analysis figure."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        if len(self.successful_data) == 0:
            return
        
        # Panel 1: Price effects with error bars
        reflection_stats = self.successful_data.groupby('reflection_pattern')['agreed_price'].agg(['mean', 'std', 'count'])
        reflection_names = [self.REFLECTION_PATTERNS.get(p, p) for p in reflection_stats.index]
        
        # Calculate standard errors
        reflection_stats['sem'] = reflection_stats['std'] / np.sqrt(reflection_stats['count'])
        
        bars1 = axes[0,0].bar(reflection_names, reflection_stats['mean'], 
                             yerr=reflection_stats['sem'], capsize=5, alpha=0.7, color='skyblue')
        axes[0,0].axhline(y=self.OPTIMAL_PRICE, color='red', linestyle='--', linewidth=2,
                         label=f'Optimal (${self.OPTIMAL_PRICE})')
        axes[0,0].set_title('Price Effects by Reflection Pattern')
        axes[0,0].set_ylabel('Mean Agreed Price ($)')
        axes[0,0].legend()
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Add value labels and sample sizes
        for bar, mean_val, n in zip(bars1, reflection_stats['mean'], reflection_stats['count']):
            axes[0,0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                          f'${mean_val:.1f}\n(n={n})', ha='center', va='bottom', fontsize=9)
        
        # Panel 2: Distribution comparison
        reflection_data = [
            self.successful_data[self.successful_data['reflection_pattern'] == pattern]['agreed_price']
            for pattern in ['00', '01', '10', '11']
        ]
        reflection_labels = [self.REFLECTION_PATTERNS[p] for p in ['00', '01', '10', '11']]
        
        box_plot = axes[0,1].boxplot([data.values for data in reflection_data if len(data) > 0], 
                                    labels=[label for data, label in zip(reflection_data, reflection_labels) if len(data) > 0],
                                    patch_artist=True)
        
        # Color the boxes
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        for patch, color in zip(box_plot['boxes'], colors[:len(box_plot['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[0,1].axhline(y=self.OPTIMAL_PRICE, color='red', linestyle='--', linewidth=2)
        axes[0,1].set_title('Price Distribution by Reflection Pattern')
        axes[0,1].set_ylabel('Agreed Price ($)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Panel 3: Efficiency effects
        efficiency_stats = self.successful_data.groupby('reflection_pattern').agg({
            'total_rounds': ['mean', 'std'],
            'total_tokens': ['mean', 'std']
        })
        
        # Normalize rounds and tokens for comparison
        rounds_norm = efficiency_stats[('total_rounds', 'mean')] / efficiency_stats[('total_rounds', 'mean')].max()
        tokens_norm = efficiency_stats[('total_tokens', 'mean')] / efficiency_stats[('total_tokens', 'mean')].max()
        
        x = np.arange(len(efficiency_stats.index))
        width = 0.35
        
        reflection_names = [self.REFLECTION_PATTERNS.get(p, p) for p in efficiency_stats.index]
        
        bars3a = axes[1,0].bar(x - width/2, rounds_norm, width, label='Rounds (normalized)', alpha=0.8, color='blue')
        bars3b = axes[1,0].bar(x + width/2, tokens_norm, width, label='Tokens (normalized)', alpha=0.8, color='green')
        
        axes[1,0].set_title('Efficiency by Reflection Pattern')
        axes[1,0].set_ylabel('Normalized Efficiency Score')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(reflection_names, rotation=45)
        axes[1,0].legend()
        
        # Panel 4: Statistical significance visualization
        # Run ANOVA and show results
        reflection_groups = [
            self.successful_data[self.successful_data['reflection_pattern'] == pattern]['agreed_price'].dropna()
            for pattern in ['00', '01', '10', '11']
        ]
        reflection_groups = [group for group in reflection_groups if len(group) > 0]
        
        if len(reflection_groups) >= 2:
            f_stat, p_value = f_oneway(*reflection_groups)
            eta_squared = self._calculate_eta_squared(reflection_groups)
            
            # Create significance visualization
            axes[1,1].axis('off')
            
            sig_text = f"""Reflection Effects Analysis
            
ANOVA Results:
F-statistic: {f_stat:.3f}
p-value: {p_value:.3f}
Effect size (η²): {eta_squared:.3f}

Interpretation:
{'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'} effect
{self._interpret_eta_squared(eta_squared)} effect size

Sample Sizes:
{chr(10).join([f'{self.REFLECTION_PATTERNS.get(pattern, pattern)}: {len(group)}' for pattern, group in zip(['00', '01', '10', '11'], reflection_groups)])}

Conclusion:
{'Reflection mechanisms show measurable impact on negotiation outcomes.' if p_value < 0.05 else 'No significant evidence that reflection affects outcomes.'}"""
            
            axes[1,1].text(0.1, 0.9, sig_text, transform=axes[1,1].transAxes,
                          fontsize=10, verticalalignment='top', fontfamily='monospace',
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path / 'reflection_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_path / 'reflection_analysis.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_model_performance_figure(self, output_path: Path):
        """Create comprehensive model performance figure."""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        if len(self.successful_data) == 0:
            return
        
        # Panel 1: Model tier comparison
        tier_stats = {}
        for tier in set(self.MODEL_TIERS.values()):
            tier_data = pd.concat([
                self.successful_data[self.successful_data['buyer_tier'] == tier],
                self.successful_data[self.successful_data['supplier_tier'] == tier]
            ])
            if len(tier_data) > 0:
                tier_stats[tier] = {
                    'mean_price': tier_data['agreed_price'].mean(),
                    'std_price': tier_data['agreed_price'].std(),
                    'count': len(tier_data),
                    'buyer_advantage': tier_data['buyer_advantage'].mean()
                }
        
        if tier_stats:
            tier_names = list(tier_stats.keys())
            tier_means = [tier_stats[tier]['mean_price'] for tier in tier_names]
            tier_stds = [tier_stats[tier]['std_price'] for tier in tier_names]
            tier_counts = [tier_stats[tier]['count'] for tier in tier_names]
            
            bars1 = axes[0,0].bar(tier_names, tier_means, yerr=tier_stds, capsize=5, alpha=0.7, color='orange')
            axes[0,0].axhline(y=self.OPTIMAL_PRICE, color='red', linestyle='--', linewidth=2)
            axes[0,0].set_title('Performance by Model Tier')
            axes[0,0].set_ylabel('Mean Agreed Price ($)')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # Add sample sizes
            for bar, count in zip(bars1, tier_counts):
                axes[0,0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                              f'n={count}', ha='center', va='bottom', fontsize=9)
        
        # Panel 2: Individual model heatmap
        model_performance = {}
        for model in self.MODEL_TIERS.keys():
            buyer_data = self.successful_data[self.successful_data['buyer_model'] == model]
            supplier_data = self.successful_data[self.successful_data['supplier_model'] == model]
            
            buyer_mean = buyer_data['agreed_price'].mean() if len(buyer_data) > 0 else np.nan
            supplier_mean = supplier_data['agreed_price'].mean() if len(supplier_data) > 0 else np.nan
            
            model_performance[model] = {'buyer': buyer_mean, 'supplier': supplier_mean}
        
        # Create heatmap data
        models = list(model_performance.keys())
        buyer_scores = [model_performance[model]['buyer'] for model in models]
        supplier_scores = [model_performance[model]['supplier'] for model in models]
        
        heatmap_data = np.array([buyer_scores, supplier_scores])
        
        # Clean model names for display
        model_names_clean = [model.replace(':latest', '').replace('-remote', '') for model in models]
        
        im = axes[0,1].imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto')
        axes[0,1].set_xticks(range(len(models)))
        axes[0,1].set_xticklabels(model_names_clean, rotation=45, ha='right')
        axes[0,1].set_yticks([0, 1])
        axes[0,1].set_yticklabels(['As Buyer', 'As Supplier'])
        axes[0,1].set_title('Model Performance Heatmap\n(Average Agreed Prices)')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[0,1])
        cbar.set_label('Agreed Price ($)')
        
        # Add text annotations
        for i in range(2):
            for j in range(len(models)):
                if not np.isnan(heatmap_data[i, j]):
                    axes[0,1].text(j, i, f'${heatmap_data[i, j]:.0f}', 
                                  ha='center', va='center', fontsize=8, color='white')
        
        # Panel 3: Role asymmetry by model
        role_asymmetry = {}
        for model in self.MODEL_TIERS.keys():
            buyer_data = self.successful_data[self.successful_data['buyer_model'] == model]
            supplier_data = self.successful_data[self.successful_data['supplier_model'] == model]
            
            if len(buyer_data) > 0 and len(supplier_data) > 0:
                buyer_mean = buyer_data['agreed_price'].mean()
                supplier_mean = supplier_data['agreed_price'].mean()
                asymmetry = buyer_mean - supplier_mean
                role_asymmetry[model] = asymmetry
        
        if role_asymmetry:
            model_names = [model.replace(':latest', '').replace('-remote', '') for model in role_asymmetry.keys()]
            asymmetry_values = list(role_asymmetry.values())
            colors = ['red' if x > 0 else 'blue' for x in asymmetry_values]
            
            bars3 = axes[0,2].bar(model_names, asymmetry_values, color=colors, alpha=0.7)
            axes[0,2].axhline(y=0, color='black', linestyle='-', linewidth=1)
            axes[0,2].set_title('Role Asymmetry by Model\n(Buyer Price - Supplier Price)')
            axes[0,2].set_ylabel('Price Difference ($)')
            axes[0,2].tick_params(axis='x', rotation=45)
        
        # Panel 4: Success rates by model
        model_success_rates = {}
        for model in self.MODEL_TIERS.keys():
            total_buyer = len(self.data[self.data['buyer_model'] == model])
            total_supplier = len(self.data[self.data['supplier_model'] == model])
            success_buyer = len(self.successful_data[self.successful_data['buyer_model'] == model])
            success_supplier = len(self.successful_data[self.successful_data['supplier_model'] == model])
            
            total_all = total_buyer + total_supplier
            success_all = success_buyer + success_supplier
            
            if total_all > 0:
                model_success_rates[model] = success_all / total_all
        
        if model_success_rates:
            model_names = [model.replace(':latest', '').replace('-remote', '') for model in model_success_rates.keys()]
            success_rates = list(model_success_rates.values())
            
            bars4 = axes[1,0].bar(model_names, success_rates, alpha=0.7, color='green')
            axes[1,0].set_title('Success Rate by Model')
            axes[1,0].set_ylabel('Success Rate')
            axes[1,0].tick_params(axis='x', rotation=45)
            
            # Add percentage labels
            for bar, rate in zip(bars4, success_rates):
                axes[1,0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                              f'{rate:.1%}', ha='center', va='bottom', fontsize=9)
        
        # Panel 5: Model family comparison
        family_stats = {}
        for family in set(self.MODEL_FAMILIES.values()):
            family_buyer = self.successful_data[self.successful_data['buyer_family'] == family]
            family_supplier = self.successful_data[self.successful_data['supplier_family'] == family]
            family_data = pd.concat([family_buyer, family_supplier])
            
            if len(family_data) > 0:
                family_stats[family] = {
                    'mean_price': family_data['agreed_price'].mean(),
                    'count': len(family_data),
                    'buyer_advantage': family_data['buyer_advantage'].mean()
                }
        
        if family_stats:
            family_names = list(family_stats.keys())
            family_means = [family_stats[family]['mean_price'] for family in family_names]
            family_counts = [family_stats[family]['count'] for family in family_names]
            
            bars5 = axes[1,1].bar(family_names, family_means, alpha=0.7, color='purple')
            axes[1,1].axhline(y=self.OPTIMAL_PRICE, color='red', linestyle='--', linewidth=2)
            axes[1,1].set_title('Performance by Model Family')
            axes[1,1].set_ylabel('Mean Agreed Price ($)')
            axes[1,1].tick_params(axis='x', rotation=45)
            
            # Add sample sizes
            for bar, count in zip(bars5, family_counts):
                axes[1,1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                              f'n={count}', ha='center', va='bottom', fontsize=9)
        
        # Panel 6: Statistical summary
        axes[1,2].axis('off')
        
        # Run statistical tests
        if tier_stats and len(tier_stats) > 1:
            tier_groups = []
            for tier in tier_stats.keys():
                tier_data = pd.concat([
                    self.successful_data[self.successful_data['buyer_tier'] == tier],
                    self.successful_data[self.successful_data['supplier_tier'] == tier]
                ])['agreed_price']
                tier_groups.append(tier_data)
            
            if len(tier_groups) >= 2:
                f_stat, p_value = f_oneway(*tier_groups)
                eta_squared = self._calculate_eta_squared(tier_groups)
                
                model_stats_text = f"""Model Effects Analysis
                
Tier ANOVA Results:
F-statistic: {f_stat:.3f}
p-value: {p_value:.3f}
Effect size (η²): {eta_squared:.3f}

Interpretation:
{'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'} tier effects
{self._interpret_eta_squared(eta_squared)} effect size

Model Tiers Tested:
{chr(10).join([f'{tier}: {stats["count"]} negotiations' for tier, stats in tier_stats.items()])}

Conclusion:
{'Model tier significantly affects negotiation outcomes.' if p_value < 0.05 else 'No significant tier-based differences detected.'}"""
                
                axes[1,2].text(0.1, 0.9, model_stats_text, transform=axes[1,2].transAxes,
                              fontsize=10, verticalalignment='top', fontfamily='monospace',
                              bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path / 'model_performance.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_path / 'model_performance.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_strategic_behavior_figure(self, output_path: Path):
        """Create strategic behavior analysis figure."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        if len(self.successful_data) == 0:
            return
        
        # Panel 1: Buyer advantage distribution
        buyer_advantages = self.successful_data['buyer_advantage']
        
        axes[0,0].hist(buyer_advantages, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0,0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Advantage')
        axes[0,0].axvline(x=buyer_advantages.mean(), color='orange', linestyle='--', linewidth=2,
                         label=f'Mean (${buyer_advantages.mean():.2f})')
        axes[0,0].set_title('Buyer Advantage Distribution')
        axes[0,0].set_xlabel('Buyer Advantage ($)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].legend()
        
        # Add statistical test results
        t_stat, p_value = ttest_1samp(buyer_advantages, 0)
        axes[0,0].text(0.02, 0.98, f't = {t_stat:.3f}\np < 0.001' if p_value < 0.001 else f't = {t_stat:.3f}\np = {p_value:.3f}',
                      transform=axes[0,0].transAxes, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Panel 2: Profit analysis
        buyer_profits = self.successful_data['buyer_profit']
        supplier_profits = self.successful_data['supplier_profit']
        
        x = np.arange(2)
        profit_means = [buyer_profits.mean(), supplier_profits.mean()]
        profit_stds = [buyer_profits.std(), supplier_profits.std()]
        
        bars2 = axes[0,1].bar(['Buyer Profit', 'Supplier Profit'], profit_means, 
                             yerr=profit_stds, capsize=5, alpha=0.7, 
                             color=['lightblue', 'lightcoral'])
        axes[0,1].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[0,1].set_title('Average Profits by Role')
        axes[0,1].set_ylabel('Profit ($)')
        
        # Add value labels
        for bar, mean_val in zip(bars2, profit_means):
            axes[0,1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 10,
                          f'${mean_val:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Panel 3: Price convergence analysis
        distances = self.successful_data['distance_from_optimal']
        convergence_ranges = [
            ('Within $5', (distances <= 5).sum()),
            ('$5-10', ((distances > 5) & (distances <= 10)).sum()),
            ('$10-15', ((distances > 10) & (distances <= 15)).sum()),
            ('Over $15', (distances > 15).sum())
        ]
        
        range_labels, range_counts = zip(*convergence_ranges)
        range_percentages = [count / len(distances) * 100 for count in range_counts]
        
        colors = ['darkgreen', 'green', 'orange', 'red']
        bars3 = axes[1,0].bar(range_labels, range_percentages, color=colors, alpha=0.7)
        axes[1,0].set_title('Price Convergence to Optimal')
        axes[1,0].set_ylabel('Percentage of Negotiations')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Add percentage labels
        for bar, pct in zip(bars3, range_percentages):
            axes[1,0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                          f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Panel 4: Homogeneous vs Heterogeneous outcomes
        homo_data = self.successful_data[self.successful_data['is_homogeneous']]
        hetero_data = self.successful_data[~self.successful_data['is_homogeneous']]
        
        if len(homo_data) > 0 and len(hetero_data) > 0:
            pairing_data = [homo_data['agreed_price'], hetero_data['agreed_price']]
            pairing_labels = [f'Homogeneous\n(n={len(homo_data)})', f'Heterogeneous\n(n={len(hetero_data)})']
            
            box_plot = axes[1,1].boxplot(pairing_data, labels=pairing_labels, patch_artist=True)
            
            # Color the boxes
            box_plot['boxes'][0].set_facecolor('lightblue')
            box_plot['boxes'][1].set_facecolor('lightgreen')
            for box in box_plot['boxes']:
                box.set_alpha(0.7)
            
            axes[1,1].axhline(y=self.OPTIMAL_PRICE, color='red', linestyle='--', linewidth=2)
            axes[1,1].set_title('Homogeneous vs Heterogeneous Pairings')
            axes[1,1].set_ylabel('Agreed Price ($)')
            
            # Statistical test
            t_stat, p_value = ttest_ind(homo_data['agreed_price'], hetero_data['agreed_price'])
            axes[1,1].text(0.02, 0.98, f't = {t_stat:.3f}\np = {p_value:.3f}',
                          transform=axes[1,1].transAxes, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path / 'strategic_behavior.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_path / 'strategic_behavior.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_efficiency_analysis_figure(self, output_path: Path):
        """Create negotiation efficiency analysis figure."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        if len(self.successful_data) == 0:
            return
        
        # Panel 1: Rounds vs Price Quality
        price_efficiency = self.successful_data['price_efficiency']
        total_rounds = self.successful_data['total_rounds']
        
        axes[0,0].scatter(total_rounds, price_efficiency, alpha=0.6, color='blue')
        
        # Add trend line
        if len(total_rounds) > 1:
            z = np.polyfit(total_rounds, price_efficiency, 1)
            p = np.poly1d(z)
            axes[0,0].plot(sorted(total_rounds), p(sorted(total_rounds)), "r--", alpha=0.8)
            
            # Correlation
            corr, p_val = stats.pearsonr(total_rounds, price_efficiency)
            axes[0,0].text(0.02, 0.98, f'r = {corr:.3f}\np = {p_val:.3f}',
                          transform=axes[0,0].transAxes, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        axes[0,0].set_title('Negotiation Length vs Price Quality')
        axes[0,0].set_xlabel('Total Rounds')
        axes[0,0].set_ylabel('Price Efficiency')
        
        # Panel 2: Token usage by reflection
        if 'total_tokens' in self.successful_data.columns:
            token_stats = self.successful_data.groupby('reflection_pattern')['total_tokens'].agg(['mean', 'std'])
            reflection_names = [self.REFLECTION_PATTERNS.get(p, p) for p in token_stats.index]
            
            bars2 = axes[0,1].bar(reflection_names, token_stats['mean'], 
                                 yerr=token_stats['std'], capsize=5, alpha=0.7, color='orange')
            axes[0,1].set_title('Token Usage by Reflection Pattern')
            axes[0,1].set_ylabel('Average Tokens Used')
            axes[0,1].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, mean_val in zip(bars2, token_stats['mean']):
                axes[0,1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
                              f'{mean_val:.0f}', ha='center', va='bottom', fontsize=9)
        
        # Panel 3: Efficiency distribution
        round_counts = self.successful_data['total_rounds'].value_counts().sort_index()
        
        bars3 = axes[1,0].bar(round_counts.index, round_counts.values, alpha=0.7, color='green')
        axes[1,0].set_title('Distribution of Negotiation Lengths')
        axes[1,0].set_xlabel('Number of Rounds')
        axes[1,0].set_ylabel('Number of Negotiations')
        
        # Add percentage labels for high bars
        total_negotiations = round_counts.sum()
        for i, (rounds, count) in enumerate(round_counts.items()):
            if count / total_negotiations > 0.05:  # Only label if >5%
                axes[1,0].text(rounds, count + total_negotiations * 0.01,
                              f'{count/total_negotiations:.1%}', ha='center', va='bottom', fontsize=8)
        
        # Panel 4: Efficiency vs Outcome Quality
        # Create efficiency score combining rounds and tokens
        if 'total_tokens' in self.successful_data.columns:
            # Normalize metrics (lower is better for efficiency)
            rounds_norm = (self.successful_data['total_rounds'].max() - self.successful_data['total_rounds']) / self.successful_data['total_rounds'].max()
            tokens_norm = (self.successful_data['total_tokens'].max() - self.successful_data['total_tokens']) / self.successful_data['total_tokens'].max()
            efficiency_score = (rounds_norm + tokens_norm) / 2
            
            axes[1,1].scatter(efficiency_score, price_efficiency, alpha=0.6, color='purple')
            
            # Correlation
            if len(efficiency_score) > 1:
                corr, p_val = stats.pearsonr(efficiency_score, price_efficiency)
                axes[1,1].text(0.02, 0.98, f'r = {corr:.3f}\np = {p_val:.3f}',
                              transform=axes[1,1].transAxes, verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            axes[1,1].set_title('Process Efficiency vs Outcome Quality')
            axes[1,1].set_xlabel('Process Efficiency Score')
            axes[1,1].set_ylabel('Price Efficiency')
        else:
            axes[1,1].axis('off')
            axes[1,1].text(0.5, 0.5, 'Token data not available', ha='center', va='center',
                          transform=axes[1,1].transAxes, fontsize=14)
        
        plt.tight_layout()
        plt.savefig(output_path / 'efficiency_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_path / 'efficiency_analysis.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_statistical_summary_figure(self, output_path: Path):
        """Create comprehensive statistical summary figure."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Panel 1: Effect sizes comparison
        effect_sizes = []
        effect_labels = []
        
        if 'statistical_analysis' in self.analysis_results:
            stats_results = self.analysis_results['statistical_analysis']
            
            # Reflection effects
            if 'reflection_effects' in stats_results and 'price_effects' in stats_results['reflection_effects']:
                eta2 = stats_results['reflection_effects']['price_effects']['effect_size']['eta_squared']
                effect_sizes.append(eta2)
                effect_labels.append('Reflection\n(η²)')
            
            # Model effects
            if 'model_effects' in stats_results and 'tier_effects' in stats_results['model_effects']:
                eta2 = stats_results['model_effects']['tier_effects']['effect_size']['eta_squared']
                effect_sizes.append(eta2)
                effect_labels.append('Model Tier\n(η²)')
            
            # Role asymmetry
            if 'role_asymmetry' in stats_results and 'buyer_advantage_test' in stats_results['role_asymmetry']:
                cohens_d = abs(stats_results['role_asymmetry']['buyer_advantage_test']['cohens_d'])
                # Convert Cohen's d to approximate eta² for comparison
                eta2_equiv = cohens_d**2 / (cohens_d**2 + 4)
                effect_sizes.append(eta2_equiv)
                effect_labels.append('Buyer Advantage\n(d→η²)')
        
        if effect_sizes:
            colors = ['blue', 'green', 'red'][:len(effect_sizes)]
            bars1 = axes[0,0].bar(effect_labels, effect_sizes, color=colors, alpha=0.7)
            
            # Add effect size interpretation lines
            axes[0,0].axhline(y=0.01, color='gray', linestyle=':', alpha=0.7, label='Small (0.01)')
            axes[0,0].axhline(y=0.06, color='gray', linestyle='--', alpha=0.7, label='Medium (0.06)')
            axes[0,0].axhline(y=0.14, color='gray', linestyle='-', alpha=0.7, label='Large (0.14)')
            
            axes[0,0].set_title('Effect Sizes Comparison')
            axes[0,0].set_ylabel('Effect Size (η² equivalent)')
            axes[0,0].legend(loc='upper right')
            
            # Add value labels
            for bar, value in zip(bars1, effect_sizes):
                axes[0,0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                              f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Panel 2: P-values summary
        p_values = []
        p_labels = []
        
        if 'statistical_analysis' in self.analysis_results:
            stats_results = self.analysis_results['statistical_analysis']
            
            # Collect p-values
            if 'reflection_effects' in stats_results and 'price_effects' in stats_results['reflection_effects']:
                p_val = stats_results['reflection_effects']['price_effects']['parametric']['p_value']
                p_values.append(p_val)
                p_labels.append('Reflection')
            
            if 'model_effects' in stats_results and 'tier_effects' in stats_results['model_effects']:
                p_val = stats_results['model_effects']['tier_effects']['parametric']['p_value']
                p_values.append(p_val)
                p_labels.append('Model Tier')
            
            if 'role_asymmetry' in stats_results and 'buyer_advantage_test' in stats_results['role_asymmetry']:
                p_val = stats_results['role_asymmetry']['buyer_advantage_test']['p_value']
                p_values.append(p_val)
                p_labels.append('Buyer Advantage')
        
        if p_values:
            # Convert to -log10 for better visualization
            neg_log_p = [-np.log10(max(p, 1e-10)) for p in p_values]  # Avoid log(0)
            
            colors = ['red' if p < 0.05 else 'gray' for p in p_values]
            bars2 = axes[0,1].bar(p_labels, neg_log_p, color=colors, alpha=0.7)
            
            # Add significance line
            axes[0,1].axhline(y=-np.log10(0.05), color='red', linestyle='--', linewidth=2, 
                             label='p = 0.05')
            axes[0,1].axhline(y=-np.log10(0.01), color='red', linestyle='-', linewidth=2, 
                             label='p = 0.01')
            
            axes[0,1].set_title('Statistical Significance')
            axes[0,1].set_ylabel('-log₁₀(p-value)')
            axes[0,1].legend()
            
            # Add p-value labels
            for bar, p_val in zip(bars2, p_values):
                label = f'p<0.001' if p_val < 0.001 else f'p={p_val:.3f}'
                axes[0,1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                              label, ha='center', va='bottom', fontsize=9, rotation=45)
        
        # Panel 3: Sample sizes and power
        if 'statistical_analysis' in self.analysis_results and 'power_analysis' in self.analysis_results['statistical_analysis']:
            power_results = self.analysis_results['statistical_analysis']['power_analysis']
            
            power_data = []
            power_labels = []
            
            if 'reflection_anova' in power_results:
                power_data.append(power_results['reflection_anova']['observed_power'])
                power_labels.append('Reflection\nANOVA')
            
            if 'buyer_advantage' in power_results:
                power_data.append(power_results['buyer_advantage']['observed_power'])
                power_labels.append('Buyer\nAdvantage')
            
            if power_data:
                colors = ['green' if p >= 0.8 else 'orange' if p >= 0.6 else 'red' for p in power_data]
                bars3 = axes[1,0].bar(power_labels, power_data, color=colors, alpha=0.7)
                
                axes[1,0].axhline(y=0.8, color='green', linestyle='--', linewidth=2, 
                                 label='Adequate Power (0.8)')
                axes[1,0].set_title('Statistical Power Analysis')
                axes[1,0].set_ylabel('Observed Power')
                axes[1,0].set_ylim(0, 1)
                axes[1,0].legend()
                
                # Add power values
                for bar, power in zip(bars3, power_data):
                    axes[1,0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                                  f'{power:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Panel 4: Summary statistics table
        axes[1,1].axis('off')
        
        # Create summary table
        summary_stats = self._generate_summary_statistics()
        
        # Format as table
        table_data = []
        for key, value in summary_stats.items():
            if isinstance(value, float):
                if 'rate' in key.lower() or 'percentage' in key.lower():
                    formatted_value = f'{value:.1%}'
                elif 'price' in key.lower() or 'advantage' in key.lower():
                    formatted_value = f'${value:.2f}'
                else:
                    formatted_value = f'{value:.3f}'
            else:
                formatted_value = str(value)
            
            table_data.append([key.replace('_', ' ').title(), formatted_value])
        
        # Create table
        table = axes[1,1].table(cellText=table_data,
                               colLabels=['Statistic', 'Value'],
                               cellLoc='left',
                               loc='center',
                               colWidths=[0.6, 0.4])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Style the table
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # Header row
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        axes[1,1].set_title('Summary Statistics', pad=20, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path / 'statistical_summary.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_path / 'statistical_summary.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_interactive_dashboard(self, output_path: Path):
        """Create interactive Plotly dashboard."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import plotly.express as px
            
            if len(self.successful_data) == 0:
                return
            
            # Create subplot figure
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Price Distribution by Reflection Pattern',
                    'Model Performance Heatmap',
                    'Negotiation Efficiency',
                    'Strategic Behavior Analysis'
                ),
                specs=[[{'secondary_y': False}, {'secondary_y': False}],
                       [{'secondary_y': False}, {'secondary_y': False}]]
            )
            
            # Panel 1: Box plot of prices by reflection
            reflection_patterns = ['00', '01', '10', '11']
            for i, pattern in enumerate(reflection_patterns):
                pattern_data = self.successful_data[
                    self.successful_data['reflection_pattern'] == pattern
                ]['agreed_price']
                
                if len(pattern_data) > 0:
                    fig.add_trace(
                        go.Box(
                            y=pattern_data,
                            name=self.REFLECTION_PATTERNS[pattern],
                            boxpoints='outliers',
                            jitter=0.3,
                            pointpos=-1.8
                        ),
                        row=1, col=1
                    )
            
            # Add optimal price line
            fig.add_hline(
                y=self.OPTIMAL_PRICE, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Optimal (${self.OPTIMAL_PRICE})",
                row=1, col=1
            )
            
            # Panel 2: Heatmap would go here (simplified for space)
            # Panel 3: Scatter plot of rounds vs price efficiency
            fig.add_trace(
                go.Scatter(
                    x=self.successful_data['total_rounds'],
                    y=self.successful_data['price_efficiency'],
                    mode='markers',
                    marker=dict(
                        color=self.successful_data['buyer_advantage'],
                        colorscale='RdBu',
                        showscale=True,
                        colorbar=dict(title="Buyer Advantage ($)")
                    ),
                    text=[f"Price: ${p:.0f}<br>Rounds: {r}<br>Buyer Adv: ${ba:.1f}" 
                          for p, r, ba in zip(
                              self.successful_data['agreed_price'],
                              self.successful_data['total_rounds'],
                              self.successful_data['buyer_advantage']
                          )],
                    hovertemplate='%{text}<extra></extra>',
                    name='Negotiations'
                ),
                row=2, col=1
            )
            
            # Panel 4: Profit comparison
            profit_data = {
                'Role': ['Buyer', 'Supplier'],
                'Average_Profit': [
                    self.successful_data['buyer_profit'].mean(),
                    self.successful_data['supplier_profit'].mean()
                ]
            }
            
            fig.add_trace(
                go.Bar(
                    x=profit_data['Role'],
                    y=profit_data['Average_Profit'],
                    marker_color=['lightblue', 'lightcoral'],
                    name='Average Profit'
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title_text="Interactive LLM Negotiation Analysis Dashboard",
                showlegend=False,
                height=800,
                font=dict(size=12)
            )
            
            # Update axis labels
            fig.update_xaxes(title_text="Reflection Pattern", row=1, col=1)
            fig.update_yaxes(title_text="Agreed Price ($)", row=1, col=1)
            fig.update_xaxes(title_text="Total Rounds", row=2, col=1)
            fig.update_yaxes(title_text="Price Efficiency", row=2, col=1)
            fig.update_xaxes(title_text="Role", row=2, col=2)
            fig.update_yaxes(title_text="Average Profit ($)", row=2, col=2)
            
            # Save interactive plot
            fig.write_html(output_path / 'interactive_dashboard.html')
            
            logger.info("Created interactive dashboard")
            
        except ImportError:
            logger.warning("Plotly not available - skipping interactive dashboard")
        except Exception as e:
            logger.warning(f"Error creating interactive dashboard: {e}")
    
    # Helper methods for statistical calculations
    def _calculate_eta_squared(self, groups: List[np.ndarray]) -> float:
        """Calculate eta-squared effect size for ANOVA."""
        if len(groups) < 2:
            return 0.0
        
        all_values = np.concatenate(groups)
        grand_mean = np.mean(all_values)
        
        # Between-group sum of squares
        ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in groups)
        
        # Total sum of squares
        ss_total = sum((value - grand_mean)**2 for value in all_values)
        
        return ss_between / ss_total if ss_total > 0 else 0.0
    
    def _calculate_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size for two groups."""
        n1, n2 = len(group1), len(group2)
        if n1 <= 1 or n2 <= 1:
            return 0.0
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + 
                             (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def _interpret_eta_squared(self, eta_squared: float) -> str:
        """Interpret eta-squared effect size."""
        if eta_squared < 0.01:
            return "Negligible"
        elif eta_squared < 0.06:
            return "Small"
        elif eta_squared < 0.14:
            return "Medium"
        else:
            return "Large"
    
    def _interpret_cohens_d(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "Negligible"
        elif abs_d < 0.5:
            return "Small"
        elif abs_d < 0.8:
            return "Medium"
        else:
            return "Large"
    
    def _test_anova_assumptions(self, groups: List[np.ndarray], labels: List[str]) -> Dict[str, Any]:
        """Test ANOVA assumptions."""
        assumptions = {}
        
        # Normality testing
        normality_results = {}
        for group, label in zip(groups, labels):
            if len(group) > 3:
                if len(group) <= 5000:
                    stat, p_val = shapiro(group)
                    test_name = "Shapiro-Wilk"
                else:
                    stat, crit_vals, sig_levels = anderson(group, dist='norm')
                    p_val = 0.05 if stat > crit_vals[2] else 0.1  # Approximate
                    test_name = "Anderson-Darling"
                
                normality_results[label] = {
                    'statistic': stat,
                    'p_value': p_val,
                    'is_normal': p_val > 0.05,
                    'test': test_name
                }
        
        assumptions['normality'] = normality_results
        
        # Homogeneity of variance
        if len(groups) > 1:
            stat, p_val = levene(*groups)
            assumptions['homogeneity'] = {
                'statistic': stat,
                'p_value': p_val,
                'homogeneous': p_val > 0.05,
                'test': 'Levene'
            }
        
        return assumptions
    
    def _posthoc_reflection_analysis(self, groups: List[np.ndarray], labels: List[str]) -> Dict[str, Any]:
        """Perform post-hoc analysis for reflection patterns."""
        posthoc_results = {}
        
        try:
            # Tukey HSD
            all_data = []
            all_labels = []
            
            for group, label in zip(groups, labels):
                all_data.extend(group.tolist())
                all_labels.extend([label] * len(group))
            
            tukey_result = pairwise_tukeyhsd(all_data, all_labels, alpha=0.05)
            
            # Extract significant pairs
            significant_pairs = []
            for i, row in enumerate(tukey_result.summary().data[1:]):
                if row[5] == 'True':  # reject null hypothesis
                    significant_pairs.append({
                        'group1': row[0],
                        'group2': row[1],
                        'meandiff': float(row[2]),
                        'p_adj': float(row[4])
                    })
            
            posthoc_results['tukey_hsd'] = {
                'significant_pairs': significant_pairs,
                'summary': str(tukey_result)
            }
            
        except Exception as e:
            logger.warning(f"Post-hoc analysis failed: {e}")
            posthoc_results['error'] = str(e)
        
        return posthoc_results
    
    def _additional_reflection_tests(self) -> Dict[str, Any]:
        """Additional tests for reflection effects."""
        additional_tests = {}
        
        # Test variance effects
        reflection_variances = {}
        for pattern in ['00', '01', '10', '11']:
            pattern_data = self.successful_data[
                self.successful_data['reflection_pattern'] == pattern
            ]['agreed_price'].dropna()
            
            if len(pattern_data) > 1:
                reflection_variances[pattern] = pattern_data.var()
        
        if len(reflection_variances) > 1:
            additional_tests['variance_effects'] = {
                'variances': reflection_variances,
                'interpretation': 'Reflection patterns show different price variance'
            }
        
        # Test efficiency effects
        if 'total_rounds' in self.successful_data.columns:
            round_groups = []
            for pattern in ['00', '01', '10', '11']:
                pattern_rounds = self.successful_data[
                    self.successful_data['reflection_pattern'] == pattern
                ]['total_rounds'].dropna()
                if len(pattern_rounds) > 0:
                    round_groups.append(pattern_rounds)
            
            if len(round_groups) >= 2:
                f_stat, p_value = f_oneway(*round_groups)
                additional_tests['efficiency_effects'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        return additional_tests
    
    def _test_model_family_effects(self) -> Dict[str, Any]:
        """Test model family effects."""
        family_groups = []
        family_labels = []
        
        for family in set(self.MODEL_FAMILIES.values()):
            family_data = pd.concat([
                self.successful_data[self.successful_data['buyer_family'] == family]['agreed_price'],
                self.successful_data[self.successful_data['supplier_family'] == family]['agreed_price']
            ]).dropna()
            
            if len(family_data) > 0:
                family_groups.append(family_data)
                family_labels.append(family)
        
        if len(family_groups) >= 2:
            f_stat, p_value = f_oneway(*family_groups)
            eta_squared = self._calculate_eta_squared(family_groups)
            
            return {
                'f_statistic': f_stat,
                'p_value': p_value,
                'eta_squared': eta_squared,
                'interpretation': self._interpret_eta_squared(eta_squared),
                'family_labels': family_labels,
                'significant': p_value < 0.05
            }
        
        return {}
    
    def _test_individual_models(self) -> Dict[str, Any]:
        """Test individual model performance differences."""
        individual_tests = {}
        
        # Compare each model as buyer vs as supplier
        for model in self.MODEL_TIERS.keys():
            buyer_data = self.successful_data[self.successful_data['buyer_model'] == model]['agreed_price']
            supplier_data = self.successful_data[self.successful_data['supplier_model'] == model]['agreed_price']
            
            if len(buyer_data) > 5 and len(supplier_data) > 5:
                t_stat, p_value = ttest_ind(buyer_data, supplier_data)
                cohens_d = self._calculate_cohens_d(buyer_data.values, supplier_data.values)
                
                individual_tests[model] = {
                    'buyer_mean': float(buyer_data.mean()),
                    'supplier_mean': float(supplier_data.mean()),
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'significant': p_value < 0.05,
                    'interpretation': self._interpret_cohens_d(cohens_d)
                }
        
        return individual_tests
    
    def _test_asymmetry_consistency(self) -> Dict[str, Any]:
        """Test if buyer advantage is consistent across conditions."""
        consistency_tests = {}
        
        # Test by reflection pattern
        reflection_advantages = []
        for pattern in ['00', '01', '10', '11']:
            pattern_advantage = self.successful_data[
                self.successful_data['reflection_pattern'] == pattern
            ]['buyer_advantage']
            if len(pattern_advantage) > 0:
                reflection_advantages.append(pattern_advantage)
        
        if len(reflection_advantages) >= 2:
            f_stat, p_value = f_oneway(*reflection_advantages)
            consistency_tests['across_reflection'] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'consistent': p_value >= 0.05,  # Non-significant = consistent
                'interpretation': 'Buyer advantage is consistent across reflection patterns' if p_value >= 0.05 else 'Buyer advantage varies by reflection pattern'
            }
        
        # Test by model tier
        tier_advantages = []
        for tier in set(self.MODEL_TIERS.values()):
            tier_data = pd.concat([
                self.successful_data[self.successful_data['buyer_tier'] == tier],
                self.successful_data[self.successful_data['supplier_tier'] == tier]
            ])['buyer_advantage']
            
            if len(tier_data) > 0:
                tier_advantages.append(tier_data)
        
        if len(tier_advantages) >= 2:
            f_stat, p_value = f_oneway(*tier_advantages)
            consistency_tests['across_tiers'] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'consistent': p_value >= 0.05,
                'interpretation': 'Buyer advantage is consistent across model tiers' if p_value >= 0.05 else 'Buyer advantage varies by model tier'
            }
        
        return consistency_tests
    
    def _power_recommendations(self, power_results: Dict[str, Any]) -> List[str]:
        """Generate power analysis recommendations."""
        recommendations = []
        
        for test_name, test_results in power_results.items():
            if isinstance(test_results, dict) and 'observed_power' in test_results:
                power = test_results['observed_power']
                
                if power < 0.5:
                    recommendations.append(f"{test_name}: Very low power ({power:.2f}) - results unreliable")
                elif power < 0.8:
                    recommendations.append(f"{test_name}: Marginal power ({power:.2f}) - consider larger sample")
                else:
                    recommendations.append(f"{test_name}: Adequate power ({power:.2f}) - results reliable")
        
        return recommendations
    
    def _generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics for the table."""
        summary = {}
        
        if len(self.successful_data) > 0:
            summary['sample_size'] = len(self.data)
            summary['success_rate'] = len(self.successful_data) / len(self.data)
            summary['mean_price'] = self.successful_data['agreed_price'].mean()
            summary['price_std'] = self.successful_data['agreed_price'].std()
            summary['buyer_advantage'] = self.successful_data['buyer_advantage'].mean()
            summary['optimal_distance'] = abs(self.successful_data['agreed_price'] - self.OPTIMAL_PRICE).mean()
            summary['avg_rounds'] = self.successful_data['total_rounds'].mean()
            
            if 'total_tokens' in self.successful_data.columns:
                summary['avg_tokens'] = self.successful_data['total_tokens'].mean()
            
            # Convergence rates
            summary['within_5_optimal'] = (abs(self.successful_data['agreed_price'] - self.OPTIMAL_PRICE) <= 5).mean()
            summary['within_10_optimal'] = (abs(self.successful_data['agreed_price'] - self.OPTIMAL_PRICE) <= 10).mean()
        
        return summary
    
    def generate_comprehensive_report(self, output_file: str = None) -> str:
        """Generate comprehensive analysis report."""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"./analysis/comprehensive_llm_negotiation_report_{timestamp}.md"
        
        # Ensure output directory exists
        Path(output_file).parent.mkdir(exist_ok=True)
        
        # Run all analyses if not already done
        if 'descriptive_analysis' not in self.analysis_results:
            self.comprehensive_descriptive_analysis()
        
        if 'statistical_analysis' not in self.analysis_results:
            self.inferential_statistical_analysis()
        
        # Generate report content
        report = self._generate_report_content()
        
        # Save report
        with open(output_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Comprehensive report saved to: {output_file}")
        return output_file
    
    def _generate_report_content(self) -> str:
        """Generate the comprehensive report content."""
        desc_stats = self.analysis_results.get('descriptive_analysis', {})
        stat_results = self.analysis_results.get('statistical_analysis', {})
        
        report = f"""# Comprehensive Analysis of LLM-to-LLM Negotiations in the Newsvendor Framework

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Analyzer Version:** Comprehensive LLM Negotiation Analyzer v2.0

## Executive Summary

This report presents a comprehensive analysis of **{len(self.data):,} bilateral negotiations** between large language model (LLM) agents in a classical newsvendor framework. The study systematically examines the impact of reflection mechanisms, model capabilities, and strategic behaviors on negotiation outcomes through rigorous statistical analysis and advanced visualizations.

### Key Findings

1. **Success Rate:** {len(self.successful_data):,}/{len(self.data):,} negotiations completed successfully ({len(self.successful_data)/len(self.data):.1%})
2. **Price Outcomes:** Mean agreed price of ${self.successful_data['agreed_price'].mean():.2f} (optimal: ${self.OPTIMAL_PRICE})
3. **Buyer Advantage:** Systematic bias of ${self.successful_data['buyer_advantage'].mean():.2f} favoring buyers
4. **Reflection Effects:** {'Significant' if stat_results.get('reflection_effects', {}).get('price_effects', {}).get('significant', False) else 'Non-significant'} impact on negotiation outcomes
5. **Model Effects:** {'Significant' if stat_results.get('model_effects', {}).get('tier_effects', {}).get('significant', False) else 'Non-significant'} differences between model tiers

## Methodology

### Experimental Design
- **Total Sample:** {len(self.data):,} negotiations
- **Model Architectures:** {len(self.MODEL_TIERS)} models across {len(set(self.MODEL_TIERS.values()))} performance tiers
- **Reflection Conditions:** 4 patterns (No Reflection, Buyer Only, Supplier Only, Both Reflect)
- **Outcome Measures:** Agreed prices, negotiation efficiency, strategic behaviors

### Newsvendor Framework Parameters
- **Optimal Wholesale Price:** ${self.OPTIMAL_PRICE}
- **Retail Price:** ${self.RETAIL_PRICE}
- **Production Cost:** ${self.PRODUCTION_COST}
- **Demand Distribution:** Normal(μ={self.DEMAND_MEAN}, σ={self.DEMAND_STD})

### Statistical Approach
- **Significance Level:** α = {self.ALPHA}
- **Effect Size Measures:** η² for ANOVA, Cohen's d for t-tests
- **Multiple Comparisons:** Bonferroni and Tukey HSD corrections
- **Assumption Testing:** Normality (Shapiro-Wilk/Anderson-Darling), Homogeneity (Levene's test)
- **Robustness:** Non-parametric alternatives, bootstrapping

## Descriptive Analysis

### Sample Characteristics
"""
        
        # Add sample overview
        if 'sample_overview' in desc_stats:
            overview = desc_stats['sample_overview']
            report += f"""
- **Total Negotiations:** {overview.get('total_negotiations', 0):,}
- **Successful Negotiations:** {overview.get('successful_negotiations', 0):,}
- **Success Rate:** {overview.get('success_rate', 0):.1%}
- **Unique Models:** {overview.get('unique_models', 0)}
- **Reflection Patterns:** {overview.get('reflection_patterns', 0)}
"""
            
            if 'negotiation_characteristics' in overview:
                chars = overview['negotiation_characteristics']
                report += f"""
### Negotiation Characteristics
- **Average Rounds:** {chars.get('avg_rounds', 0):.1f}
- **Average Tokens:** {chars.get('avg_tokens', 0):,.0f}
- **Average Duration:** {chars.get('avg_duration', 0):.1f} seconds
- **Price Range:** ${chars.get('price_range', [0, 0])[0]:.0f} - ${chars.get('price_range', [0, 0])[1]:.0f}
"""
        
        # Add price analysis
        if 'price_analysis' in desc_stats:
            price_stats = desc_stats['price_analysis']['descriptive_stats']
            optimal_analysis = desc_stats['price_analysis']['optimal_price_analysis']
            convergence = desc_stats['price_analysis']['convergence_analysis']
            
            report += f"""
### Price Distribution Analysis

**Central Tendency:**
- Mean: ${price_stats['mean']:.2f}
- Median: ${price_stats['median']:.2f}
- Standard Deviation: ${price_stats['std']:.2f}
- Coefficient of Variation: {price_stats['cv']:.3f}

**Convergence to Optimal Price:**
- Mean Distance from Optimal: ${optimal_analysis['distance_from_optimal']['mean']:.2f}
- Within $5 of Optimal: {optimal_analysis['distance_from_optimal']['within_5']:.1%}
- Within $10 of Optimal: {optimal_analysis['distance_from_optimal']['within_10']:.1%}

**Market Dynamics:**
- Below Optimal Price: {convergence['below_optimal']:.1%}
- Above Optimal Price: {convergence['above_optimal']:.1%}
- Buyer Advantage: ${convergence['buyer_advantage']:.2f}
- Supplier Advantage: ${convergence['supplier_advantage']:.2f}
"""
        
        # Add model analysis
        report += self._add_model_analysis_section(desc_stats)
        
        # Add reflection analysis
        report += self._add_reflection_analysis_section(desc_stats)
        
        # Add statistical analysis
        report += self._add_statistical_analysis_section(stat_results)
        
        # Add conclusions and implications
        report += self._add_conclusions_section(desc_stats, stat_results)
        
        return report
    
    def _add_model_analysis_section(self, desc_stats: Dict[str, Any]) -> str:
        """Add model analysis section to report."""
        section = "\n## Model Performance Analysis\n"
        
        if 'model_analysis' in desc_stats:
            model_data = desc_stats['model_analysis']
            
            # Individual model performance
            if 'individual_models' in model_data:
                section += "\n### Individual Model Performance\n\n"
                section += "| Model | Tier | As Buyer | As Supplier | Role Asymmetry |\n"
                section += "|-------|------|----------|-------------|----------------|\n"
                
                for model, stats in model_data['individual_models'].items():
                    model_name = model.replace(':latest', '').replace('-remote', '')
                    tier = stats['model_info']['tier']
                    
                    buyer_price = stats['as_buyer'].get('mean_price', 0)
                    supplier_price = stats['as_supplier'].get('mean_price', 0)
                    
                    asymmetry = stats.get('role_asymmetry', {}).get('price_difference', 0)
                    
                    section += f"| {model_name} | {tier} | ${buyer_price:.1f} | ${supplier_price:.1f} | ${asymmetry:.1f} |\n"
            
            # Tier analysis
            if 'tier_analysis' in model_data:
                section += "\n### Performance by Model Tier\n\n"
                section += "| Tier | Count | Success Rate | Mean Price | Buyer Advantage |\n"
                section += "|------|-------|--------------|------------|------------------|\n"
                
                for tier, stats in model_data['tier_analysis'].items():
                    count = stats.get('count', 0)
                    success_rate = stats.get('success_rate', 0)
                    mean_price = stats.get('mean_price', 0)
                    buyer_advantage = stats.get('buyer_advantage', 0)
                    
                    section += f"| {tier} | {count} | {success_rate:.1%} | ${mean_price:.1f} | ${buyer_advantage:.1f} |\n"
            
            # Pairing effects
            if 'pairing_effects' in model_data:
                pairing = model_data['pairing_effects']
                section += "\n### Model Pairing Effects\n\n"
                
                if 'homogeneous_vs_heterogeneous' in pairing:
                    homo = pairing['homogeneous_vs_heterogeneous']['homogeneous']
                    hetero = pairing['homogeneous_vs_heterogeneous']['heterogeneous']
                    
                    section += f"**Homogeneous Pairings:** {homo.get('count', 0)} negotiations, ${homo.get('mean_price', 0):.1f} average price\n"
                    section += f"**Heterogeneous Pairings:** {hetero.get('count', 0)} negotiations, ${hetero.get('mean_price', 0):.1f} average price\n\n"
        
        return section
    
    def _add_reflection_analysis_section(self, desc_stats: Dict[str, Any]) -> str:
        """Add reflection analysis section to report."""
        section = "\n## Reflection Mechanism Analysis\n"
        
        if 'reflection_analysis' in desc_stats:
            reflection_data = desc_stats['reflection_analysis']
            
            section += "\n### Performance by Reflection Pattern\n\n"
            section += "| Pattern | Description | Count | Success Rate | Mean Price | Buyer Advantage |\n"
            section += "|---------|-------------|-------|--------------|------------|------------------|\n"
            
            for pattern, data in reflection_data.items():
                name = data.get('name', pattern)
                stats = data.get('stats', {})
                
                count = stats.get('count', 0)
                success_rate = stats.get('success_rate', 0)
                mean_price = stats.get('mean_price', 0)
                buyer_advantage = stats.get('buyer_advantage', 0)
                
                section += f"| {pattern} | {name} | {count} | {success_rate:.1%} | ${mean_price:.1f} | ${buyer_advantage:.1f} |\n"
            
            # Reflection effects analysis
            section += "\n### Reflection Mechanism Effects\n\n"
            
            for pattern, data in reflection_data.items():
                if 'reflection_effects' in data:
                    effects = data['reflection_effects']
                    name = data.get('name', pattern)
                    
                    section += f"**{name}:**\n"
                    section += f"- Price Variance: ${effects.get('price_variance', 0):.1f}\n"
                    section += f"- Convergence Rate: {effects.get('convergence_rate', 0):.1%}\n"
                    section += f"- Efficiency Score: {effects.get('efficiency_score', 0):.3f}\n\n"
        
        return section
    
    def _add_statistical_analysis_section(self, stat_results: Dict[str, Any]) -> str:
        """Add statistical analysis section to report."""
        section = "\n## Statistical Analysis\n"
        
        # Reflection effects
        if 'reflection_effects' in stat_results:
            section += "\n### Research Question 1: Reflection Effects\n\n"
            
            if 'price_effects' in stat_results['reflection_effects']:
                price_effects = stat_results['reflection_effects']['price_effects']
                
                f_stat = price_effects['parametric']['f_statistic']
                p_value = price_effects['parametric']['p_value']
                eta_squared = price_effects['effect_size']['eta_squared']
                interpretation = price_effects['effect_size']['interpretation']
                
                section += f"**ANOVA Results:**\n"
                section += f"- F-statistic: {f_stat:.3f}\n"
                section += f"- p-value: {p_value:.3f}\n"
                section += f"- Effect size (η²): {eta_squared:.3f} ({interpretation})\n"
                section += f"- Significant: {'Yes' if price_effects['significant'] else 'No'}\n\n"
                
                # Non-parametric confirmation
                h_stat = price_effects['nonparametric']['h_statistic']
                h_p = price_effects['nonparametric']['p_value']
                section += f"**Non-parametric Confirmation (Kruskal-Wallis):**\n"
                section += f"- H-statistic: {h_stat:.3f}\n"
                section += f"- p-value: {h_p:.3f}\n\n"
                
                # Assumptions testing
                if 'assumptions' in price_effects:
                    section += f"**Assumption Testing:**\n"
                    assumptions = price_effects['assumptions']
                    
                    if 'normality' in assumptions:
                        for group, norm_test in assumptions['normality'].items():
                            section += f"- {group} normality: {'✓' if norm_test['is_normal'] else '✗'} (p={norm_test['p_value']:.3f})\n"
                    
                    if 'homogeneity' in assumptions:
                        homo = assumptions['homogeneity']
                        section += f"- Homogeneity of variance: {'✓' if homo['homogeneous'] else '✗'} (p={homo['p_value']:.3f})\n"
                    
                    section += "\n"
        
        # Model effects
        if 'model_effects' in stat_results:
            section += "\n### Research Question 2: Model Effects\n\n"
            
            if 'tier_effects' in stat_results['model_effects']:
                tier_effects = stat_results['model_effects']['tier_effects']
                
                f_stat = tier_effects['parametric']['f_statistic']
                p_value = tier_effects['parametric']['p_value']
                eta_squared = tier_effects['effect_size']['eta_squared']
                interpretation = tier_effects['effect_size']['interpretation']
                
                section += f"**Model Tier ANOVA:**\n"
                section += f"- F-statistic: {f_stat:.3f}\n"
                section += f"- p-value: {p_value:.3f}\n"
                section += f"- Effect size (η²): {eta_squared:.3f} ({interpretation})\n"
                section += f"- Significant: {'Yes' if tier_effects['significant'] else 'No'}\n\n"
            
            if 'family_effects' in stat_results['model_effects']:
                family_effects = stat_results['model_effects']['family_effects']
                
                if family_effects:
                    section += f"**Model Family Effects:**\n"
                    section += f"- F-statistic: {family_effects['f_statistic']:.3f}\n"
                    section += f"- p-value: {family_effects['p_value']:.3f}\n"
                    section += f"- Effect size (η²): {family_effects['eta_squared']:.3f} ({family_effects['interpretation']})\n\n"
        
        # Role asymmetry
        if 'role_asymmetry' in stat_results:
            section += "\n### Research Question 3: Role Asymmetry (Buyer Advantage)\n\n"
            
            if 'buyer_advantage_test' in stat_results['role_asymmetry']:
                buyer_test = stat_results['role_asymmetry']['buyer_advantage_test']
                
                mean_adv = buyer_test['mean_advantage']
                t_stat = buyer_test['t_statistic']
                p_value = buyer_test['p_value']
                cohens_d = buyer_test['cohens_d']
                interpretation = buyer_test['effect_interpretation']
                ci_95 = buyer_test['ci_95']
                
                section += f"**One-sample t-test (H₀: buyer advantage = 0):**\n"
                section += f"- Mean buyer advantage: ${mean_adv:.2f}\n"
                section += f"- t-statistic: {t_stat:.3f}\n"
                section += f"- p-value: {p_value:.3f}\n"
                section += f"- Effect size (Cohen's d): {cohens_d:.3f} ({interpretation})\n"
                section += f"- 95% CI: [${ci_95[0]:.2f}, ${ci_95[1]:.2f}]\n"
                section += f"- Significant: {'Yes' if buyer_test['significant'] else 'No'}\n\n"
            
            # Distribution analysis
            if 'distribution_analysis' in stat_results['role_asymmetry']:
                dist = stat_results['role_asymmetry']['distribution_analysis']
                
                section += f"**Distribution of Buyer Advantages:**\n"
                section += f"- Positive (buyer favored): {dist['proportion_positive']:.1%}\n"
                section += f"- Negative (supplier favored): {dist['proportion_negative']:.1%}\n"
                section += f"- Zero (equal): {dist['proportion_zero']:.1%}\n\n"
            
            # Consistency tests
            if 'consistency_tests' in stat_results['role_asymmetry']:
                consistency = stat_results['role_asymmetry']['consistency_tests']
                
                section += f"**Consistency Across Conditions:**\n"
                
                if 'across_reflection' in consistency:
                    refl_cons = consistency['across_reflection']
                    section += f"- Across reflection patterns: {'✓ Consistent' if refl_cons['consistent'] else '✗ Varies'} (p={refl_cons['p_value']:.3f})\n"
                
                if 'across_tiers' in consistency:
                    tier_cons = consistency['across_tiers']
                    section += f"- Across model tiers: {'✓ Consistent' if tier_cons['consistent'] else '✗ Varies'} (p={tier_cons['p_value']:.3f})\n"
                
                section += "\n"
        
        # Interaction effects
        if 'interaction_effects' in stat_results:
            section += "\n### Interaction Effects\n\n"
            
            if 'reflection_x_tier' in stat_results['interaction_effects']:
                interaction = stat_results['interaction_effects']['reflection_x_tier']
                
                if 'anova_results' in interaction:
                    anova = interaction['anova_results']
                    
                    section += f"**Two-way ANOVA (Reflection × Model Tier):**\n"
                    section += f"- Reflection main effect: F={anova['reflection_main']['f_stat']:.3f}, p={anova['reflection_main']['p_value']:.3f}\n"
                    section += f"- Model tier main effect: F={anova['tier_main']['f_stat']:.3f}, p={anova['tier_main']['p_value']:.3f}\n"
                    section += f"- Interaction effect: F={anova['interaction']['f_stat']:.3f}, p={anova['interaction']['p_value']:.3f}\n"
                    section += f"- Model R²: {interaction['model_fit']['r_squared']:.3f}\n\n"
        
        # Power analysis
        if 'power_analysis' in stat_results:
            section += "\n### Statistical Power Analysis\n\n"
            power = stat_results['power_analysis']
            
            if 'reflection_anova' in power:
                refl_power = power['reflection_anova']
                section += f"**Reflection ANOVA Power:**\n"
                section += f"- Observed power: {refl_power['observed_power']:.3f}\n"
                section += f"- Effect size (η²): {refl_power['effect_size_eta2']:.3f}\n"
                section += f"- Min group size: {refl_power['min_group_size']}\n"
                section += f"- Adequate power: {'✓ Yes' if refl_power['adequate_power'] else '✗ No'}\n\n"
            
            if 'buyer_advantage' in power:
                buyer_power = power['buyer_advantage']
                section += f"**Buyer Advantage Test Power:**\n"
                section += f"- Observed power: {buyer_power['observed_power']:.3f}\n"
                section += f"- Effect size (Cohen's d): {buyer_power['effect_size_d']:.3f}\n"
                section += f"- Sample size: {buyer_power['sample_size']}\n"
                section += f"- Adequate power: {'✓ Yes' if buyer_power['adequate_power'] else '✗ No'}\n\n"
            
            if 'recommendations' in power:
                section += f"**Power Recommendations:**\n"
                for rec in power['recommendations']:
                    section += f"- {rec}\n"
                section += "\n"
        
        return section
    
    def _add_conclusions_section(self, desc_stats: Dict[str, Any], stat_results: Dict[str, Any]) -> str:
        """Add conclusions and implications section to report."""
        section = "\n## Conclusions and Implications\n"
        
        # Research questions conclusions
        section += "\n### Research Question Conclusions\n\n"
        
        # RQ1: Reflection effects
        reflection_significant = stat_results.get('reflection_effects', {}).get('price_effects', {}).get('significant', False)
        reflection_effect = stat_results.get('reflection_effects', {}).get('price_effects', {}).get('effect_size', {}).get('interpretation', 'unknown')
        
        section += f"**RQ1: Do reflection mechanisms improve negotiation outcomes?**\n"
        if reflection_significant:
            section += f"✅ **YES** - Reflection mechanisms show a {reflection_effect.lower()} but statistically significant effect on negotiation prices.\n"
        else:
            section += f"❌ **NO** - No statistically significant evidence that reflection mechanisms affect negotiation outcomes.\n"
        section += "\n"
        
        # RQ2: Model effects
        model_significant = stat_results.get('model_effects', {}).get('tier_effects', {}).get('significant', False)
        model_effect = stat_results.get('model_effects', {}).get('tier_effects', {}).get('effect_size', {}).get('interpretation', 'unknown')
        
        section += f"**RQ2: Do model capabilities affect negotiation performance?**\n"
        if model_significant:
            section += f"✅ **YES** - Model tier shows a {model_effect.lower()} effect on negotiation outcomes.\n"
        else:
            section += f"❌ **NO** - No significant differences detected between model tiers in negotiation performance.\n"
        section += "\n"
        
        # RQ3: Role asymmetry
        asymmetry_significant = stat_results.get('role_asymmetry', {}).get('buyer_advantage_test', {}).get('significant', False)
        buyer_advantage = stat_results.get('role_asymmetry', {}).get('buyer_advantage_test', {}).get('mean_advantage', 0)
        asymmetry_effect = stat_results.get('role_asymmetry', {}).get('buyer_advantage_test', {}).get('effect_interpretation', 'unknown')
        
        section += f"**RQ3: Is there systematic role asymmetry in LLM negotiations?**\n"
        if asymmetry_significant:
            section += f"✅ **YES** - Strong evidence for systematic buyer advantage of ${buyer_advantage:.2f} ({asymmetry_effect.lower()} effect).\n"
        else:
            section += f"❌ **NO** - No significant role asymmetry detected.\n"
        section += "\n"
        
        # Key insights
        section += "\n### Key Insights\n\n"
        
        # Calculate key metrics
        success_rate = len(self.successful_data) / len(self.data)
        mean_price = self.successful_data['agreed_price'].mean()
        price_efficiency = 1 - (abs(self.successful_data['agreed_price'] - self.OPTIMAL_PRICE).mean() / self.OPTIMAL_PRICE)
        
        section += f"1. **Negotiation Success:** {success_rate:.1%} of negotiations reached agreement, indicating {'high' if success_rate > 0.8 else 'moderate' if success_rate > 0.5 else 'low'} effectiveness of LLM-to-LLM negotiations.\n\n"
        
        section += f"2. **Price Discovery:** Mean agreed price of ${mean_price:.2f} vs. optimal ${self.OPTIMAL_PRICE} shows {'excellent' if price_efficiency > 0.9 else 'good' if price_efficiency > 0.8 else 'moderate'} price discovery (efficiency: {price_efficiency:.1%}).\n\n"
        
        if asymmetry_significant:
            section += f"3. **Systematic Bias:** Buyer advantage of ${buyer_advantage:.2f} indicates significant fairness concerns for AI-mediated negotiations.\n\n"
        
        if reflection_significant:
            section += f"4. **Reflection Value:** Statistical evidence supports the value of reflection mechanisms in strategic AI interactions.\n\n"
        else:
            section += f"4. **Reflection Limitations:** Simple reflection prompts may not provide sufficient strategic advantage to justify computational costs.\n\n"
        
        if model_significant:
            section += f"5. **Model Capabilities:** Clear evidence that model architecture and scale affect strategic reasoning capabilities.\n\n"
        else:
            section += f"5. **Model Equivalence:** Surprising finding that model scale may not determine negotiation effectiveness in this domain.\n\n"
        
        # Practical implications
        section += "\n### Practical Implications\n\n"
        
        section += "**For AI Deployment:**\n"
        if asymmetry_significant:
            section += f"- Role-specific biases require mitigation strategies before production deployment\n"
        if reflection_significant:
            section += f"- Reflection mechanisms provide measurable benefits and should be implemented\n"
        else:
            section += f"- Simple reflection approaches may not justify computational overhead\n"
        if model_significant:
            section += f"- Model selection should consider strategic reasoning capabilities\n"
        section += "\n"
        
        section += "**For Research:**\n"
        section += f"- Need for bias detection and mitigation frameworks\n"
        section += f"- Exploration of advanced reflection architectures\n"
        section += f"- Cross-domain validation of findings\n"
        section += f"- Human-AI negotiation benchmarking\n\n"
        
        section += "**For Policy:**\n"
        if asymmetry_significant:
            section += f"- Regulatory frameworks should address AI negotiation biases\n"
        section += f"- Standards for fairness in AI-mediated transactions\n"
        section += f"- Transparency requirements for AI negotiation systems\n\n"
        
        # Limitations
        section += "\n### Limitations\n\n"
        section += "1. **Domain Specificity:** Results limited to newsvendor framework - generalization requires testing\n"
        section += "2. **Reflection Design:** Simple template-based reflection - more sophisticated approaches may yield different results\n"
        section += "3. **Model Selection:** Analysis limited to available models at time of study\n"
        section += "4. **Static Evaluation:** Models don't learn or adapt during negotiations\n"
        section += "5. **Cultural Context:** English-language negotiations may not generalize globally\n\n"
        
        # Future directions
        section += "\n### Future Research Directions\n\n"
        section += "1. **Advanced Reflection:** Tree-of-thought, chain-of-thought, and constitutional AI approaches\n"
        section += "2. **Multi-Issue Negotiations:** Beyond single-price bargaining to complex multi-attribute negotiations\n"
        section += "3. **Dynamic Learning:** Adaptive strategies and learning from negotiation experience\n"
        section += "4. **Human Benchmarking:** Direct comparison with human negotiator performance\n"
        section += "5. **Bias Mitigation:** Systematic approaches to reducing role-specific biases\n"
        section += "6. **Cross-Cultural Validation:** Testing across different languages and cultural contexts\n"
        section += "7. **Real-World Deployment:** Field studies in actual business negotiation contexts\n\n"
        
        # Final summary
        section += "\n### Summary\n\n"
        section += f"This comprehensive analysis of {len(self.data):,} LLM negotiations provides robust evidence for "
        
        conclusions = []
        if reflection_significant:
            conclusions.append("measurable reflection benefits")
        if model_significant:
            conclusions.append("model capability effects")
        if asymmetry_significant:
            conclusions.append("systematic role biases")
        
        if conclusions:
            section += f"{', '.join(conclusions[:-1])}"
            if len(conclusions) > 1:
                section += f", and {conclusions[-1]}"
            else:
                section += conclusions[0]
        else:
            section += "limited effects of reflection and model differences but important baseline insights"
        
        section += f". These findings have immediate implications for the responsible deployment of AI agents in strategic contexts and highlight critical areas for continued research.\n\n"
        
        # Technical appendix note
        section += "---\n\n"
        section += "**Technical Note:** Complete statistical outputs, effect size calculations, assumption tests, and replication materials are available in the accompanying analysis files.\n\n"
        section += f"**Generated by:** Comprehensive LLM Negotiation Analyzer v2.0\n"
        section += f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        section += f"**Total Runtime:** Complete pipeline with {len(self.data):,} negotiations\n"
        
        return section
    
    def run_complete_analysis(self, output_dir: str = "./analysis") -> Dict[str, str]:
        """Run the complete analysis pipeline."""
        logger.info("🚀 Starting Comprehensive LLM Negotiation Analysis")
        logger.info("=" * 80)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        results = {}
        
        try:
            # Step 1: Load and validate data
            logger.info("📊 Loading and validating data...")
            if not self.load_and_validate_data():
                logger.error("❌ Data loading failed")
                return {}
            
            # Step 2: Descriptive analysis
            logger.info("📈 Running comprehensive descriptive analysis...")
            self.comprehensive_descriptive_analysis()
            
            # Step 3: Inferential statistics
            logger.info("🔬 Running inferential statistical analysis...")
            self.inferential_statistical_analysis()
            
            # Step 4: Advanced visualizations
            logger.info("🎨 Creating advanced visualizations...")
            self.advanced_visualizations(output_dir)
            
            # Step 5: Generate comprehensive report
            logger.info("📋 Generating comprehensive report...")
            report_file = self.generate_comprehensive_report(
                str(output_path / f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
            )
            results['report'] = report_file
            
            # Step 6: Save analysis results as JSON
            results_file = output_path / f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                # Convert numpy types to Python types for JSON serialization
                json_results = self._convert_for_json(self.analysis_results)
                json.dump(json_results, f, indent=2, default=str)
            results['data'] = str(results_file)
            
            # Step 7: Create summary dashboard
            self._create_summary_dashboard(output_path)
            results['dashboard'] = str(output_path / 'executive_dashboard.png')
            
            logger.info("✅ Comprehensive analysis completed successfully!")
            
            # Print executive summary
            self._print_executive_summary()
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            raise e
    
    def _convert_for_json(self, obj):
        """Convert analysis results for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def _create_summary_dashboard(self, output_path: Path):
        """Create a summary dashboard image."""
        # This creates a single-page summary (already implemented in _create_executive_dashboard)
        pass
    
    def _print_executive_summary(self):
        """Print executive summary to console."""
        print("\n" + "=" * 80)
        print("🎯 COMPREHENSIVE LLM NEGOTIATION ANALYSIS - EXECUTIVE SUMMARY")
        print("=" * 80)
        
        total_n = len(self.data)
        success_n = len(self.successful_data)
        success_rate = success_n / total_n
        
        print(f"📊 SAMPLE OVERVIEW:")
        print(f"   Total Negotiations: {total_n:,}")
        print(f"   Successful: {success_n:,} ({success_rate:.1%})")
        print(f"   Models Tested: {len(self.MODEL_TIERS)}")
        print(f"   Reflection Patterns: {len(self.REFLECTION_PATTERNS)}")
        
        if success_n > 0:
            mean_price = self.successful_data['agreed_price'].mean()
            buyer_advantage = self.successful_data['buyer_advantage'].mean()
            optimal_distance = abs(self.successful_data['agreed_price'] - self.OPTIMAL_PRICE).mean()
            
            print(f"\n💰 PRICE OUTCOMES:")
            print(f"   Mean Price: ${mean_price:.2f} (Optimal: ${self.OPTIMAL_PRICE})")
            print(f"   Distance from Optimal: ${optimal_distance:.2f}")
            print(f"   Buyer Advantage: ${buyer_advantage:.2f}")
            
            # Statistical results
            stat_results = self.analysis_results.get('statistical_analysis', {})
            
            print(f"\n🔬 KEY STATISTICAL FINDINGS:")
            
            # Reflection effects
            if 'reflection_effects' in stat_results:
                refl_sig = stat_results['reflection_effects'].get('price_effects', {}).get('significant', False)
                refl_eta = stat_results['reflection_effects'].get('price_effects', {}).get('effect_size', {}).get('eta_squared', 0)
                print(f"   Reflection Effects: {'✅ SIGNIFICANT' if refl_sig else '❌ NOT SIGNIFICANT'} (η²={refl_eta:.3f})")
            
            # Model effects
            if 'model_effects' in stat_results:
                model_sig = stat_results['model_effects'].get('tier_effects', {}).get('significant', False)
                model_eta = stat_results['model_effects'].get('tier_effects', {}).get('effect_size', {}).get('eta_squared', 0)
                print(f"   Model Tier Effects: {'✅ SIGNIFICANT' if model_sig else '❌ NOT SIGNIFICANT'} (η²={model_eta:.3f})")
            
            # Role asymmetry
            if 'role_asymmetry' in stat_results:
                asym_sig = stat_results['role_asymmetry'].get('buyer_advantage_test', {}).get('significant', False)
                asym_d = stat_results['role_asymmetry'].get('buyer_advantage_test', {}).get('cohens_d', 0)
                print(f"   Buyer Advantage: {'✅ SIGNIFICANT' if asym_sig else '❌ NOT SIGNIFICANT'} (d={asym_d:.3f})")
            
            # Power analysis
            if 'power_analysis' in stat_results:
                power_data = stat_results['power_analysis']
                print(f"\n⚡ STATISTICAL POWER:")
                
                if 'reflection_anova' in power_data:
                    refl_power = power_data['reflection_anova']['observed_power']
                    print(f"   Reflection Analysis: {refl_power:.3f} ({'✅ ADEQUATE' if refl_power >= 0.8 else '⚠️ MARGINAL' if refl_power >= 0.6 else '❌ LOW'})")
                
                if 'buyer_advantage' in power_data:
                    buyer_power = power_data['buyer_advantage']['observed_power']
                    print(f"   Buyer Advantage Test: {buyer_power:.3f} ({'✅ ADEQUATE' if buyer_power >= 0.8 else '⚠️ MARGINAL' if buyer_power >= 0.6 else '❌ LOW'})")
        
        print(f"\n📁 OUTPUT FILES:")
        print(f"   📊 Executive Dashboard: ./analysis/executive_dashboard.png")
        print(f"   📈 Reflection Analysis: ./analysis/reflection_analysis.png")
        print(f"   🤖 Model Performance: ./analysis/model_performance.png")
        print(f"   ⚖️ Strategic Behavior: ./analysis/strategic_behavior.png")
        print(f"   📋 Comprehensive Report: ./analysis/comprehensive_report_*.md")
        print(f"   🌐 Interactive Dashboard: ./analysis/interactive_dashboard.html")
        
        print(f"\n💡 KEY RECOMMENDATIONS:")
        
        if success_rate < 0.5:
            print(f"   ⚠️ Low success rate indicates need for negotiation protocol improvements")
        
        if 'role_asymmetry' in stat_results and stat_results['role_asymmetry'].get('buyer_advantage_test', {}).get('significant', False):
            print(f"   🚨 Systematic buyer bias requires mitigation before deployment")
        
        if 'reflection_effects' in stat_results and stat_results['reflection_effects'].get('price_effects', {}).get('significant', False):
            print(f"   ✅ Reflection mechanisms provide measurable benefits")
        else:
            print(f"   💭 Simple reflection may not justify computational costs")
        
        if 'model_effects' in stat_results and stat_results['model_effects'].get('tier_effects', {}).get('significant', False):
            print(f"   🎯 Model selection should consider strategic capabilities")
        
        print("=" * 80)


def main():
    """Main function to run comprehensive analysis."""
    print("🎯 Comprehensive LLM Negotiation Analysis Suite")
    print("=" * 60)
    print("Publication-ready statistical analysis with advanced visualizations")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = ComprehensiveLLMNegotiationAnalyzer()
    
    # Run complete analysis
    try:
        results = analyzer.run_complete_analysis()
        
        if results:
            print("\n🎉 Analysis completed successfully!")
            print("\n📊 Generated Files:")
            for file_type, file_path in results.items():
                print(f"   {file_type}: {file_path}")
            
            print("\n📈 Analysis Features:")
            print("   ✅ Comprehensive descriptive statistics")
            print("   ✅ Rigorous inferential testing")
            print("   ✅ Effect size calculations")
            print("   ✅ Power analysis")
            print("   ✅ Advanced visualizations")
            print("   ✅ Interactive dashboard")
            print("   ✅ Publication-ready report")
            
            print("\n🔬 Statistical Methods:")
            print("   • ANOVA with assumption testing")
            print("   • Non-parametric alternatives")
            print("   • Post-hoc analysis with corrections")
            print("   • Bootstrap confidence intervals")
            print("   • Interaction effect testing")
            print("   • Comprehensive power analysis")
            
        else:
            print("\n❌ Analysis failed!")
            print("   Check data files and logs for details")
            
    except Exception as e:
        print(f"\n💥 Error during analysis: {e}")
        print("   Check the logs for detailed error information")


if __name__ == "__main__":
    main()