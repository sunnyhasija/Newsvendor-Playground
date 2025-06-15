"""
Statistical Tests for Newsvendor Experiment

Comprehensive statistical analysis including ANOVA, t-tests, chi-squared tests,
and effect size calculations for hypothesis testing.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, normaltest, levene
from typing import Dict, List, Any, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StatisticalTest:
    """Container for statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float]
    interpretation: str
    significant: bool
    details: Dict[str, Any]


@dataclass
class HypothesisTest:
    """Container for hypothesis test results."""
    hypothesis: str
    supported: bool
    confidence_level: float
    effect_size: float
    statistical_tests: List[StatisticalTest]
    interpretation: str


class StatisticalAnalyzer:
    """Comprehensive statistical analysis for newsvendor experiment."""
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize statistical analyzer.
        
        Args:
            alpha: Significance level for hypothesis testing
        """
        self.alpha = alpha
        self.optimal_price = 65.0
        
    def test_reflection_benefits(self, results: pd.DataFrame) -> HypothesisTest:
        """Test H1: Reflection provides negotiation benefits."""
        
        successful = results[results['completed'] == True].copy()
        
        # Group by reflection patterns
        groups = {
            'no_reflection': successful[successful['reflection_pattern'] == '00']['agreed_price'].dropna(),
            'buyer_reflection': successful[successful['reflection_pattern'] == '01']['agreed_price'].dropna(),
            'supplier_reflection': successful[successful['reflection_pattern'] == '10']['agreed_price'].dropna(),
            'both_reflection': successful[successful['reflection_pattern'] == '11']['agreed_price'].dropna()
        }
        
        # Remove empty groups
        groups = {k: v for k, v in groups.items() if len(v) > 0}
        
        if len(groups) < 2:
            return HypothesisTest(
                hypothesis="H1: Reflection Benefits",
                supported=False,
                confidence_level=0.0,
                effect_size=0.0,
                statistical_tests=[],
                interpretation="Insufficient data for analysis"
            )
        
        statistical_tests = []
        
        # 1. ANOVA test for overall differences
        if len(groups) > 2:
            group_data = list(groups.values())
            f_stat, p_value = stats.f_oneway(*group_data)
            
            # Calculate eta-squared (effect size for ANOVA)
            grand_mean = np.mean(np.concatenate(group_data))
            ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in group_data)
            ss_total = sum(sum((x - grand_mean)**2 for x in group) for group in group_data)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0.0
            
            statistical_tests.append(StatisticalTest(
                test_name="One-way ANOVA",
                statistic=f_stat,
                p_value=p_value,
                effect_size=eta_squared,
                interpretation=self._interpret_anova(f_stat, p_value, eta_squared),
                significant=p_value < self.alpha,
                details={
                    'groups': len(groups),
                    'total_n': sum(len(group) for group in group_data),
                    'group_means': {k: np.mean(v) for k, v in groups.items()}
                }
            ))
        
        # 2. Pairwise t-tests comparing reflection vs no reflection
        if 'no_reflection' in groups:
            baseline = groups['no_reflection']
            
            for condition, group_data in groups.items():
                if condition != 'no_reflection' and len(group_data) > 0:
                    t_stat, p_value = stats.ttest_ind(baseline, group_data, equal_var=False)
                    
                    # Cohen's d effect size
                    pooled_std = np.sqrt((np.var(baseline, ddof=1) + np.var(group_data, ddof=1)) / 2)
                    cohens_d = (np.mean(group_data) - np.mean(baseline)) / pooled_std if pooled_std > 0 else 0.0
                    
                    statistical_tests.append(StatisticalTest(
                        test_name=f"T-test: {condition} vs no_reflection",
                        statistic=t_stat,
                        p_value=p_value,
                        effect_size=abs(cohens_d),
                        interpretation=self._interpret_ttest(t_stat, p_value, cohens_d),
                        significant=p_value < self.alpha,
                        details={
                            'mean_baseline': np.mean(baseline),
                            'mean_treatment': np.mean(group_data),
                            'difference': np.mean(group_data) - np.mean(baseline),
                            'cohens_d': cohens_d
                        }
                    ))
        
        # Determine overall hypothesis support
        significant_tests = [t for t in statistical_tests if t.significant]
        overall_support = len(significant_tests) > 0
        
        # Calculate overall effect size (average of significant effects)
        if significant_tests:
            overall_effect = np.mean([t.effect_size for t in significant_tests])
            confidence = 1 - min(t.p_value for t in significant_tests)
        else:
            overall_effect = 0.0
            confidence = 0.0
        
        return HypothesisTest(
            hypothesis="H1: Reflection Benefits",
            supported=overall_support,
            confidence_level=confidence,
            effect_size=overall_effect,
            statistical_tests=statistical_tests,
            interpretation=self._interpret_h1_results(statistical_tests)
        )
    
    def test_role_asymmetry(self, results: pd.DataFrame) -> HypothesisTest:
        """Test H3: Reflection provides greater benefits to buyers than suppliers."""
        
        successful = results[results['completed'] == True].copy()
        
        # Compare buyer vs supplier reflection benefits
        no_reflection = successful[successful['reflection_pattern'] == '00']['agreed_price'].dropna()
        buyer_reflection = successful[successful['reflection_pattern'] == '01']['agreed_price'].dropna()
        supplier_reflection = successful[successful['reflection_pattern'] == '10']['agreed_price'].dropna()
        
        if len(no_reflection) == 0 or (len(buyer_reflection) == 0 and len(supplier_reflection) == 0):
            return HypothesisTest(
                hypothesis="H3: Role Asymmetry",
                supported=False,
                confidence_level=0.0,
                effect_size=0.0,
                statistical_tests=[],
                interpretation="Insufficient data for analysis"
            )
        
        statistical_tests = []
        
        # Calculate benefits (negative = favorable to buyers, positive = favorable to suppliers)
        baseline_mean = np.mean(no_reflection)
        buyer_benefit = np.mean(buyer_reflection) - baseline_mean if len(buyer_reflection) > 0 else 0.0
        supplier_benefit = np.mean(supplier_reflection) - baseline_mean if len(supplier_reflection) > 0 else 0.0
        
        # Test 1: Buyer reflection benefit
        if len(buyer_reflection) > 0:
            t_stat, p_value = stats.ttest_ind(no_reflection, buyer_reflection, equal_var=False)
            pooled_std = np.sqrt((np.var(no_reflection, ddof=1) + np.var(buyer_reflection, ddof=1)) / 2)
            cohens_d = buyer_benefit / pooled_std if pooled_std > 0 else 0.0
            
            statistical_tests.append(StatisticalTest(
                test_name="Buyer Reflection Benefit",
                statistic=t_stat,
                p_value=p_value,
                effect_size=abs(cohens_d),
                interpretation=f"Buyer reflection {'decreases' if buyer_benefit < 0 else 'increases'} prices by ${abs(buyer_benefit):.2f}",
                significant=p_value < self.alpha,
                details={
                    'benefit': buyer_benefit,
                    'direction': 'buyer_favorable' if buyer_benefit < 0 else 'supplier_favorable'
                }
            ))
        
        # Test 2: Supplier reflection benefit
        if len(supplier_reflection) > 0:
            t_stat, p_value = stats.ttest_ind(no_reflection, supplier_reflection, equal_var=False)
            pooled_std = np.sqrt((np.var(no_reflection, ddof=1) + np.var(supplier_reflection, ddof=1)) / 2)
            cohens_d = supplier_benefit / pooled_std if pooled_std > 0 else 0.0
            
            statistical_tests.append(StatisticalTest(
                test_name="Supplier Reflection Benefit",
                statistic=t_stat,
                p_value=p_value,
                effect_size=abs(cohens_d),
                interpretation=f"Supplier reflection {'decreases' if supplier_benefit < 0 else 'increases'} prices by ${abs(supplier_benefit):.2f}",
                significant=p_value < self.alpha,
                details={
                    'benefit': supplier_benefit,
                    'direction': 'buyer_favorable' if supplier_benefit < 0 else 'supplier_favorable'
                }
            ))
        
        # Test 3: Direct comparison of benefits
        if len(buyer_reflection) > 0 and len(supplier_reflection) > 0:
            t_stat, p_value = stats.ttest_ind(buyer_reflection, supplier_reflection, equal_var=False)
            pooled_std = np.sqrt((np.var(buyer_reflection, ddof=1) + np.var(supplier_reflection, ddof=1)) / 2)
            cohens_d = (np.mean(buyer_reflection) - np.mean(supplier_reflection)) / pooled_std if pooled_std > 0 else 0.0
            
            statistical_tests.append(StatisticalTest(
                test_name="Direct Comparison: Buyer vs Supplier Reflection",
                statistic=t_stat,
                p_value=p_value,
                effect_size=abs(cohens_d),
                interpretation=f"{'Buyer' if cohens_d < 0 else 'Supplier'} reflection achieves {'lower' if cohens_d < 0 else 'higher'} prices",
                significant=p_value < self.alpha,
                details={
                    'buyer_mean': np.mean(buyer_reflection),
                    'supplier_mean': np.mean(supplier_reflection),
                    'difference': np.mean(buyer_reflection) - np.mean(supplier_reflection)
                }
            ))
        
        # Determine hypothesis support
        # H3 is supported if buyer reflection shows greater benefit (more negative buyer_benefit or lower prices)
        role_asymmetry_exists = abs(buyer_benefit) > abs(supplier_benefit) if len(buyer_reflection) > 0 and len(supplier_reflection) > 0 else False
        buyer_advantage = buyer_benefit < supplier_benefit if len(buyer_reflection) > 0 and len(supplier_reflection) > 0 else False
        
        h3_supported = role_asymmetry_exists and buyer_advantage
        
        # Calculate confidence and effect size
        significant_tests = [t for t in statistical_tests if t.significant]
        if significant_tests:
            overall_effect = np.mean([t.effect_size for t in significant_tests])
            confidence = 1 - min(t.p_value for t in significant_tests)
        else:
            overall_effect = abs(buyer_benefit - supplier_benefit) / max(abs(buyer_benefit), abs(supplier_benefit), 1)
            confidence = 0.5
        
        return HypothesisTest(
            hypothesis="H3: Role Asymmetry (Buyer Advantage)",
            supported=h3_supported,
            confidence_level=confidence,
            effect_size=overall_effect,