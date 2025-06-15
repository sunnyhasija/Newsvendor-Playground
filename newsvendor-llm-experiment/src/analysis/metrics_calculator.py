"""
Metrics Calculator for Newsvendor Experiment

Calculates key performance metrics for negotiation analysis including
convergence rates, price optimality, efficiency, and reflection benefits.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExperimentMetrics:
    """Container for experiment-wide metrics."""
    total_negotiations: int
    success_rate: float
    convergence_rate: float
    avg_price: float
    price_std: float
    avg_rounds: float
    avg_tokens: float
    token_efficiency: float
    distance_from_optimal: float
    optimal_convergence_rate: float


@dataclass
class ReflectionMetrics:
    """Container for reflection-specific metrics."""
    pattern: str
    count: int
    success_rate: float
    avg_price: float
    price_std: float
    avg_rounds: float
    avg_tokens: float
    distance_from_optimal: float
    price_range: Tuple[float, float]


@dataclass
class ModelMetrics:
    """Container for model-specific metrics."""
    model_name: str
    as_buyer: Dict[str, float]
    as_supplier: Dict[str, float]
    overall_performance: float
    efficiency_score: float


class MetricsCalculator:
    """Calculate comprehensive metrics for newsvendor negotiation analysis."""
    
    def __init__(self, optimal_price: float = 65.0, efficiency_target: int = 2000):
        """
        Initialize metrics calculator.
        
        Args:
            optimal_price: Theoretical optimal price for newsvendor problem
            efficiency_target: Target tokens per negotiation for efficiency scoring
        """
        self.optimal_price = optimal_price
        self.efficiency_target = efficiency_target
        self.price_tolerance = 8.0  # ±$8 from optimal considered "good"
        
    def calculate_experiment_metrics(self, results: pd.DataFrame) -> ExperimentMetrics:
        """Calculate overall experiment metrics."""
        
        # Filter successful negotiations
        successful = results[results['completed'] == True].copy()
        
        if len(successful) == 0:
            logger.warning("No successful negotiations found")
            return ExperimentMetrics(
                total_negotiations=len(results),
                success_rate=0.0,
                convergence_rate=0.0,
                avg_price=0.0,
                price_std=0.0,
                avg_rounds=0.0,
                avg_tokens=0.0,
                token_efficiency=0.0,
                distance_from_optimal=float('inf'),
                optimal_convergence_rate=0.0
            )
        
        # Calculate basic metrics
        total_negotiations = len(results)
        success_rate = len(successful) / total_negotiations
        
        # Price metrics
        prices = successful['agreed_price'].dropna()
        avg_price = prices.mean()
        price_std = prices.std()
        
        # Distance from optimal
        distances = np.abs(prices - self.optimal_price)
        avg_distance = distances.mean()
        
        # Optimal convergence (within tolerance)
        optimal_count = sum(distances <= self.price_tolerance)
        optimal_convergence_rate = optimal_count / len(prices) if len(prices) > 0 else 0.0
        
        # Efficiency metrics
        avg_rounds = successful['total_rounds'].mean()
        avg_tokens = successful['total_tokens'].mean()
        token_efficiency = self.efficiency_target / avg_tokens if avg_tokens > 0 else 0.0
        
        # Convergence rate (agreements / total)
        convergence_rate = success_rate  # Since all successful negotiations converged
        
        return ExperimentMetrics(
            total_negotiations=total_negotiations,
            success_rate=success_rate,
            convergence_rate=convergence_rate,
            avg_price=avg_price,
            price_std=price_std,
            avg_rounds=avg_rounds,
            avg_tokens=avg_tokens,
            token_efficiency=token_efficiency,
            distance_from_optimal=avg_distance,
            optimal_convergence_rate=optimal_convergence_rate
        )
    
    def calculate_reflection_metrics(self, results: pd.DataFrame) -> List[ReflectionMetrics]:
        """Calculate metrics by reflection pattern."""
        
        reflection_metrics = []
        
        for pattern in ['00', '01', '10', '11']:
            pattern_data = results[results['reflection_pattern'] == pattern]
            successful = pattern_data[pattern_data['completed'] == True]
            
            if len(pattern_data) == 0:
                continue
                
            prices = successful['agreed_price'].dropna()
            
            metrics = ReflectionMetrics(
                pattern=pattern,
                count=len(pattern_data),
                success_rate=len(successful) / len(pattern_data),
                avg_price=prices.mean() if len(prices) > 0 else 0.0,
                price_std=prices.std() if len(prices) > 0 else 0.0,
                avg_rounds=successful['total_rounds'].mean() if len(successful) > 0 else 0.0,
                avg_tokens=successful['total_tokens'].mean() if len(successful) > 0 else 0.0,
                distance_from_optimal=np.abs(prices - self.optimal_price).mean() if len(prices) > 0 else float('inf'),
                price_range=(prices.min(), prices.max()) if len(prices) > 0 else (0.0, 0.0)
            )
            
            reflection_metrics.append(metrics)
        
        return reflection_metrics
    
    def calculate_model_metrics(self, results: pd.DataFrame) -> List[ModelMetrics]:
        """Calculate metrics by model performance."""
        
        model_metrics = []
        successful = results[results['completed'] == True].copy()
        
        # Get unique models
        all_models = set(results['buyer_model'].unique()) | set(results['supplier_model'].unique())
        
        for model in all_models:
            # As buyer performance
            as_buyer = successful[successful['buyer_model'] == model]
            buyer_metrics = {
                'count': len(as_buyer),
                'success_rate': len(as_buyer) / len(results[results['buyer_model'] == model]) if len(results[results['buyer_model'] == model]) > 0 else 0.0,
                'avg_price': as_buyer['agreed_price'].mean() if len(as_buyer) > 0 else 0.0,
                'avg_rounds': as_buyer['total_rounds'].mean() if len(as_buyer) > 0 else 0.0,
                'avg_tokens': as_buyer['total_tokens'].mean() if len(as_buyer) > 0 else 0.0
            }
            
            # As supplier performance
            as_supplier = successful[successful['supplier_model'] == model]
            supplier_metrics = {
                'count': len(as_supplier),
                'success_rate': len(as_supplier) / len(results[results['supplier_model'] == model]) if len(results[results['supplier_model'] == model]) > 0 else 0.0,
                'avg_price': as_supplier['agreed_price'].mean() if len(as_supplier) > 0 else 0.0,
                'avg_rounds': as_supplier['total_rounds'].mean() if len(as_supplier) > 0 else 0.0,
                'avg_tokens': as_supplier['total_tokens'].mean() if len(as_supplier) > 0 else 0.0
            }
            
            # Overall performance score (closer to optimal = better)
            all_model_data = successful[
                (successful['buyer_model'] == model) | 
                (successful['supplier_model'] == model)
            ]
            
            if len(all_model_data) > 0:
                avg_distance = np.abs(all_model_data['agreed_price'] - self.optimal_price).mean()
                overall_performance = max(0, 1 - (avg_distance / self.optimal_price))
                
                avg_tokens = all_model_data['total_tokens'].mean()
                efficiency_score = min(1.0, self.efficiency_target / avg_tokens) if avg_tokens > 0 else 0.0
            else:
                overall_performance = 0.0
                efficiency_score = 0.0
            
            model_metrics.append(ModelMetrics(
                model_name=model,
                as_buyer=buyer_metrics,
                as_supplier=supplier_metrics,
                overall_performance=overall_performance,
                efficiency_score=efficiency_score
            ))
        
        return sorted(model_metrics, key=lambda x: x.overall_performance, reverse=True)
    
    def calculate_reflection_benefits(self, results: pd.DataFrame) -> Dict[str, Any]:
        """Calculate specific reflection benefit metrics."""
        
        successful = results[results['completed'] == True].copy()
        
        # Group by reflection conditions
        no_reflection = successful[successful['reflection_pattern'] == '00']
        buyer_only = successful[successful['reflection_pattern'] == '01']
        supplier_only = successful[successful['reflection_pattern'] == '10']
        both_reflection = successful[successful['reflection_pattern'] == '11']
        
        # Calculate benefits
        benefits = {
            'baseline_price': no_reflection['agreed_price'].mean() if len(no_reflection) > 0 else 0.0,
            'buyer_reflection_benefit': {
                'price_change': (buyer_only['agreed_price'].mean() - no_reflection['agreed_price'].mean()) if len(buyer_only) > 0 and len(no_reflection) > 0 else 0.0,
                'efficiency_change': (no_reflection['total_rounds'].mean() - buyer_only['total_rounds'].mean()) if len(buyer_only) > 0 and len(no_reflection) > 0 else 0.0,
                'avg_price': buyer_only['agreed_price'].mean() if len(buyer_only) > 0 else 0.0
            },
            'supplier_reflection_benefit': {
                'price_change': (supplier_only['agreed_price'].mean() - no_reflection['agreed_price'].mean()) if len(supplier_only) > 0 and len(no_reflection) > 0 else 0.0,
                'efficiency_change': (no_reflection['total_rounds'].mean() - supplier_only['total_rounds'].mean()) if len(supplier_only) > 0 and len(no_reflection) > 0 else 0.0,
                'avg_price': supplier_only['agreed_price'].mean() if len(supplier_only) > 0 else 0.0
            },
            'mutual_reflection_benefit': {
                'price_change': (both_reflection['agreed_price'].mean() - no_reflection['agreed_price'].mean()) if len(both_reflection) > 0 and len(no_reflection) > 0 else 0.0,
                'efficiency_change': (no_reflection['total_rounds'].mean() - both_reflection['total_rounds'].mean()) if len(both_reflection) > 0 and len(no_reflection) > 0 else 0.0,
                'avg_price': both_reflection['agreed_price'].mean() if len(both_reflection) > 0 else 0.0
            }
        }
        
        # Role asymmetry analysis
        buyer_refl_conditions = successful[successful['reflection_pattern'].isin(['01', '11'])]
        supplier_refl_conditions = successful[successful['reflection_pattern'].isin(['10', '11'])]
        
        benefits['role_asymmetry'] = {
            'buyer_advantage': (no_reflection['agreed_price'].mean() - buyer_refl_conditions['agreed_price'].mean()) if len(buyer_refl_conditions) > 0 and len(no_reflection) > 0 else 0.0,
            'supplier_advantage': (supplier_refl_conditions['agreed_price'].mean() - no_reflection['agreed_price'].mean()) if len(supplier_refl_conditions) > 0 and len(no_reflection) > 0 else 0.0
        }
        
        return benefits
    
    def calculate_efficiency_metrics(self, results: pd.DataFrame) -> Dict[str, Any]:
        """Calculate efficiency metrics across different dimensions."""
        
        successful = results[results['completed'] == True].copy()
        
        if len(successful) == 0:
            return {"error": "No successful negotiations"}
        
        efficiency_metrics = {
            'overall': {
                'avg_rounds': successful['total_rounds'].mean(),
                'avg_tokens': successful['total_tokens'].mean(),
                'tokens_per_round': successful['total_tokens'].mean() / successful['total_rounds'].mean() if successful['total_rounds'].mean() > 0 else 0.0,
                'efficiency_score': (successful['total_tokens'] <= self.efficiency_target).mean()
            },
            'by_reflection': {},
            'by_model_tier': {},
            'by_price_range': {}
        }
        
        # Efficiency by reflection pattern
        for pattern in ['00', '01', '10', '11']:
            pattern_data = successful[successful['reflection_pattern'] == pattern]
            if len(pattern_data) > 0:
                efficiency_metrics['by_reflection'][pattern] = {
                    'avg_rounds': pattern_data['total_rounds'].mean(),
                    'avg_tokens': pattern_data['total_tokens'].mean(),
                    'efficiency_score': (pattern_data['total_tokens'] <= self.efficiency_target).mean()
                }
        
        # Efficiency by price outcome
        price_ranges = [
            ('very_low', 0, 45),
            ('low', 45, 55),
            ('optimal', 55, 75),
            ('high', 75, 100)
        ]
        
        for range_name, min_price, max_price in price_ranges:
            range_data = successful[
                (successful['agreed_price'] >= min_price) & 
                (successful['agreed_price'] < max_price)
            ]
            if len(range_data) > 0:
                efficiency_metrics['by_price_range'][range_name] = {
                    'count': len(range_data),
                    'avg_rounds': range_data['total_rounds'].mean(),
                    'avg_tokens': range_data['total_tokens'].mean()
                }
        
        return efficiency_metrics
    
    def generate_summary_report(self, results: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive summary report."""
        
        experiment_metrics = self.calculate_experiment_metrics(results)
        reflection_metrics = self.calculate_reflection_metrics(results)
        model_metrics = self.calculate_model_metrics(results)
        reflection_benefits = self.calculate_reflection_benefits(results)
        efficiency_metrics = self.calculate_efficiency_metrics(results)
        
        return {
            'experiment_overview': experiment_metrics.__dict__,
            'reflection_analysis': [m.__dict__ for m in reflection_metrics],
            'model_performance': [m.__dict__ for m in model_metrics],
            'reflection_benefits': reflection_benefits,
            'efficiency_analysis': efficiency_metrics,
            'key_findings': self._extract_key_findings(experiment_metrics, reflection_metrics, model_metrics),
            'research_implications': self._generate_research_implications(reflection_benefits)
        }
    
    def _extract_key_findings(self, experiment: ExperimentMetrics, reflection: List[ReflectionMetrics], models: List[ModelMetrics]) -> Dict[str, Any]:
        """Extract key findings for research publication."""
        
        # Best reflection pattern for different outcomes
        best_buyer_pattern = min(reflection, key=lambda x: x.avg_price)
        most_efficient_pattern = min(reflection, key=lambda x: x.avg_rounds)
        best_optimal_pattern = min(reflection, key=lambda x: x.distance_from_optimal)
        
        # Top performing models
        top_model = models[0] if models else None
        
        return {
            'overall_success_rate': experiment.success_rate,
            'price_convergence': {
                'average_price': experiment.avg_price,
                'distance_from_optimal': experiment.distance_from_optimal,
                'optimal_convergence_rate': experiment.optimal_convergence_rate
            },
            'reflection_insights': {
                'best_buyer_outcome': {
                    'pattern': best_buyer_pattern.pattern,
                    'avg_price': best_buyer_pattern.avg_price
                },
                'most_efficient': {
                    'pattern': most_efficient_pattern.pattern,
                    'avg_rounds': most_efficient_pattern.avg_rounds
                },
                'closest_to_optimal': {
                    'pattern': best_optimal_pattern.pattern,
                    'distance': best_optimal_pattern.distance_from_optimal
                }
            },
            'model_insights': {
                'top_performer': top_model.model_name if top_model else None,
                'performance_score': top_model.overall_performance if top_model else 0.0
            },
            'efficiency': {
                'avg_rounds': experiment.avg_rounds,
                'avg_tokens': experiment.avg_tokens,
                'token_efficiency': experiment.token_efficiency
            }
        }
    
    def _generate_research_implications(self, benefits: Dict[str, Any]) -> List[str]:
        """Generate research implications based on findings."""
        
        implications = []
        
        # H1: Reflection Benefit
        if any(abs(benefits[key]['price_change']) > 2 for key in ['buyer_reflection_benefit', 'supplier_reflection_benefit']):
            implications.append("H1 SUPPORTED: Reflection provides measurable benefits in negotiation outcomes")
        
        # H3: Role Asymmetry
        if abs(benefits['role_asymmetry']['buyer_advantage']) > abs(benefits['role_asymmetry']['supplier_advantage']):
            implications.append("H3 SUPPORTED: Reflection provides greater benefits to buyers than suppliers")
        elif abs(benefits['role_asymmetry']['supplier_advantage']) > abs(benefits['role_asymmetry']['buyer_advantage']):
            implications.append("H3 PARTIAL: Reflection provides greater benefits to suppliers than buyers")
        
        # Efficiency implications
        if benefits['buyer_reflection_benefit']['efficiency_change'] > 0:
            implications.append("Buyer reflection improves negotiation efficiency")
        
        return implications