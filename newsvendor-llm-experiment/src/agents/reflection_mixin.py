"""
Reflection Mixin for Negotiation Agents

Provides self-reflection capabilities for LLM agents to improve
negotiation performance through strategic thinking and self-monitoring.
"""

import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ReflectionResult:
    """Result of reflection analysis."""
    reflection_text: str
    key_insights: List[str]
    strategic_recommendation: str
    confidence_score: float
    price_analysis: Dict[str, Any]


class ReflectionMixin:
    """Mixin class providing reflection capabilities for negotiation agents."""
    
    def __init__(self, role: str, model_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize reflection capabilities.
        
        Args:
            role: Agent role ('buyer' or 'supplier')
            model_name: Name of the underlying LLM model
            config: Configuration dictionary
        """
        self.role = role
        self.model_name = model_name
        self.config = config or {}
        
        # Reflection configuration
        self.reflection_enabled = True
        self.max_reflection_tokens = self._get_reflection_token_limit()
        self.structured_template = self.config.get('structured_template', True)
        
        # Track reflection history
        self.reflection_history: List[ReflectionResult] = []
        
        logger.debug(f"Initialized ReflectionMixin for {role} using {model_name}")
    
    def _get_reflection_token_limit(self) -> int:
        """Get appropriate token limit for reflection based on model tier."""
        # Ultra-compact models get shorter reflections to prevent truncation
        model_tier = self._get_model_tier()
        
        if model_tier == "ultra":
            return 80
        elif model_tier == "compact":
            return 120
        else:
            return 150
    
    def _get_model_tier(self) -> str:
        """Determine model tier for reflection configuration."""
        # Ultra-compact models
        if any(name in self.model_name for name in ["tinyllama", "qwen2:1.5b"]):
            return "ultra"
        # Compact models
        elif any(name in self.model_name for name in ["gemma2:2b", "phi3:mini", "llama3.2"]):
            return "compact"
        # Mid-range models
        elif any(name in self.model_name for name in ["mistral", "qwen:7b"]):
            return "mid"
        # Large models
        else:
            return "large"
    
    def generate_reflection(
        self, 
        negotiation_history: List[Dict[str, Any]], 
        round_number: int,
        max_rounds: int
    ) -> str:
        """
        Generate reflection content for the current negotiation state.
        
        Args:
            negotiation_history: List of previous negotiation turns
            round_number: Current round number
            max_rounds: Maximum rounds allowed
            
        Returns:
            Formatted reflection text for inclusion in prompt
        """
        try:
            if not self.reflection_enabled:
                return ""
            
            # Analyze current situation
            analysis = self._analyze_negotiation_state(negotiation_history, round_number, max_rounds)
            
            # Generate reflection based on model tier and role
            if self._get_model_tier() == "ultra":
                reflection_text = self._generate_compact_reflection(analysis)
            else:
                reflection_text = self._generate_full_reflection(analysis)
            
            # Validate reflection content
            if self._validate_reflection(reflection_text):
                # Store reflection result
                reflection_result = ReflectionResult(
                    reflection_text=reflection_text,
                    key_insights=analysis.get('insights', []),
                    strategic_recommendation=analysis.get('strategy', 'continue'),
                    confidence_score=analysis.get('confidence', 0.5),
                    price_analysis=analysis.get('price_analysis', {})
                )
                
                self.reflection_history.append(reflection_result)
                return reflection_text
            else:
                logger.warning(f"Generated reflection failed validation for {self.role}")
                return self._generate_fallback_reflection(analysis)
                
        except Exception as e:
            logger.error(f"Error generating reflection for {self.role}: {e}")
            return self._generate_fallback_reflection({})
    
    def _analyze_negotiation_state(
        self, 
        history: List[Dict[str, Any]], 
        round_number: int,
        max_rounds: int
    ) -> Dict[str, Any]:
        """Analyze current negotiation state for reflection."""
        
        analysis = {
            'round_info': {
                'current_round': round_number,
                'max_rounds': max_rounds,
                'rounds_remaining': max_rounds - round_number,
                'urgency_level': 'high' if max_rounds - round_number <= 2 else 'medium' if max_rounds - round_number <= 4 else 'low'
            },
            'price_analysis': {},
            'insights': [],
            'strategy': 'continue',
            'confidence': 0.5
        }
        
        if not history:
            analysis['insights'] = ['Opening negotiation', 'Need to establish initial position']
            analysis['strategy'] = 'make_opening_offer'
            return analysis
        
        # Extract price information
        my_offers = []
        opponent_offers = []
        
        for turn in history:
            price = turn.get('price')
            speaker = turn.get('speaker')
            
            if price and speaker == self.role:
                my_offers.append(price)
            elif price and speaker != self.role:
                opponent_offers.append(price)
        
        analysis['price_analysis'] = {
            'my_offers': my_offers,
            'opponent_offers': opponent_offers,
            'my_last_offer': my_offers[-1] if my_offers else None,
            'opponent_last_offer': opponent_offers[-1] if opponent_offers else None,
            'price_gap': abs(my_offers[-1] - opponent_offers[-1]) if my_offers and opponent_offers else None
        }
        
        # Generate insights based on role
        if self.role == "buyer":
            analysis.update(self._analyze_buyer_position(analysis))
        else:
            analysis.update(self._analyze_supplier_position(analysis))
        
        return analysis
    
    def _analyze_buyer_position(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze negotiation from buyer's perspective."""
        price_analysis = analysis['price_analysis']
        insights = []
        strategy = 'continue'
        confidence = 0.5
        
        opponent_last = price_analysis.get('opponent_last_offer')
        my_last = price_analysis.get('my_last_offer')
        
        if opponent_last:
            # Analyze supplier's price
            if opponent_last <= 50:
                insights.append('Supplier offering reasonable price')
                strategy = 'consider_acceptance'
                confidence = 0.8
            elif opponent_last <= 70:
                insights.append('Supplier price in negotiable range')
                strategy = 'counter_offer'
                confidence = 0.6
            else:
                insights.append('Supplier price too high')
                strategy = 'aggressive_counter'
                confidence = 0.4
            
            # Estimate supplier's cost
            estimated_cost = max(20, opponent_last - 25)  # Assume they want some profit
            insights.append(f'Estimated supplier cost: ~${estimated_cost}')
        
        # Urgency factor
        if analysis['round_info']['urgency_level'] == 'high':
            insights.append('Time pressure - consider accepting reasonable offers')
            if strategy == 'aggressive_counter':
                strategy = 'counter_offer'  # Be less aggressive near deadline
        
        return {
            'insights': insights,
            'strategy': strategy,
            'confidence': confidence
        }
    
    def _analyze_supplier_position(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze negotiation from supplier's perspective."""
        price_analysis = analysis['price_analysis']
        insights = []
        strategy = 'continue'
        confidence = 0.5
        
        opponent_last = price_analysis.get('opponent_last_offer')
        my_last = price_analysis.get('my_last_offer')
        
        if opponent_last:
            # Analyze buyer's offer (supplier's perspective)
            profit = opponent_last - 30  # Cost is $30
            
            if profit >= 25:
                insights.append('Buyer offering good profit margin')
                strategy = 'consider_acceptance'
                confidence = 0.8
            elif profit >= 15:
                insights.append('Buyer offer provides decent profit')
                strategy = 'counter_offer'
                confidence = 0.6
            else:
                insights.append('Buyer offer too low for good profit')
                strategy = 'hold_firm'
                confidence = 0.4
            
            insights.append(f'Profit from their offer: ${profit}')
            
            # Estimate buyer's max value
            estimated_max = opponent_last + 15  # They probably have some room
            insights.append(f'Estimated buyer max: ~${estimated_max}')
        
        # Urgency factor
        if analysis['round_info']['urgency_level'] == 'high':
            insights.append('Time pressure - consider accepting marginal offers')
            if strategy == 'hold_firm':
                strategy = 'counter_offer'  # Be more flexible near deadline
        
        return {
            'insights': insights,
            'strategy': strategy,
            'confidence': confidence
        }
    
    def _generate_compact_reflection(self, analysis: Dict[str, Any]) -> str:
        """Generate compact reflection for ultra-compact models."""
        round_info = analysis['round_info']
        price_analysis = analysis['price_analysis']
        
        round_text = f"Round {round_info['current_round']}: "
        
        if self.role == "buyer":
            opponent_offer = price_analysis.get('opponent_last_offer', 'none')
            my_profit = f"${100 - opponent_offer}" if opponent_offer != 'none' else 'unknown'
            strategy = analysis.get('strategy', 'counter')
            
            return f"""<think>
{round_text}Supplier offered ${opponent_offer}
My profit: {my_profit}
Strategy: {strategy}
</think>"""
        
        else:  # supplier
            opponent_offer = price_analysis.get('opponent_last_offer', 'none')
            my_profit = f"${opponent_offer - 30}" if opponent_offer != 'none' else 'unknown'
            strategy = analysis.get('strategy', 'counter')
            
            return f"""<think>
{round_text}Buyer offered ${opponent_offer}
My profit: {my_profit}
Strategy: {strategy}
</think>"""
    
    def _generate_full_reflection(self, analysis: Dict[str, Any]) -> str:
        """Generate full reflection for larger models."""
        round_info = analysis['round_info']
        price_analysis = analysis['price_analysis']
        insights = analysis.get('insights', [])
        strategy = analysis.get('strategy', 'continue')
        
        situation_text = f"Round {round_info['current_round']}/{round_info['max_rounds']}"
        
        # Build reflection content
        reflection_parts = [
            f"SITUATION: {situation_text}",
        ]
        
        # Add price information
        if self.role == "buyer":
            supplier_offer = price_analysis.get('opponent_last_offer', 'none')
            my_offer = price_analysis.get('my_last_offer', 'none')
            reflection_parts.extend([
                f"LAST OFFER: Supplier offered ${supplier_offer}",
                f"MY POSITION: Need good profit margin"
            ])
        else:
            buyer_offer = price_analysis.get('opponent_last_offer', 'none')
            my_offer = price_analysis.get('my_last_offer', 'none')
            reflection_parts.extend([
                f"LAST OFFER: Buyer offered ${buyer_offer}",
                f"MY POSITION: Need profit above ${30}"  # cost
            ])
        
        # Add analysis
        if insights:
            reflection_parts.append("ANALYSIS:")
            reflection_parts.extend([f"- {insight}" for insight in insights[:3]])  # Limit insights
        
        # Add strategy
        strategy_text = self._format_strategy(strategy)
        reflection_parts.extend([
            "STRATEGY:",
            f"- Next move: {strategy_text}"
        ])
        
        return f"<think>\n{chr(10).join(reflection_parts)}\n</think>"
    
    def _format_strategy(self, strategy: str) -> str:
        """Format strategy for human-readable output."""
        strategy_map = {
            'make_opening_offer': 'Make strong opening offer',
            'consider_acceptance': 'Consider accepting this offer',
            'counter_offer': 'Make reasonable counter-offer',
            'aggressive_counter': 'Counter aggressively',
            'hold_firm': 'Hold firm on price',
            'continue': 'Continue negotiation'
        }
        
        return strategy_map.get(strategy, strategy.replace('_', ' ').title())
    
    def _generate_fallback_reflection(self, analysis: Dict[str, Any]) -> str:
        """Generate simple fallback reflection if main generation fails."""
        round_num = analysis.get('round_info', {}).get('current_round', 1)
        
        if self.role == "buyer":
            return f"""<think>
Round {round_num}: Analyze supplier's offer
Target: Get lowest possible price
Strategy: Counter if too high
</think>"""
        else:
            return f"""<think>
Round {round_num}: Analyze buyer's offer
Target: Get good profit margin
Strategy: Counter if too low
</think>"""
    
    def _validate_reflection(self, reflection_text: str) -> bool:
        """Validate that reflection contains required elements."""
        if not reflection_text.strip():
            return False
        
        # Check for proper think block structure
        if not (reflection_text.startswith('<think>') and reflection_text.endswith('</think>')):
            return False
        
        # Check for minimum required content
        required_elements = [
            r'round|Round',  # Round information
            r'offer|price|\$\d+',  # Price/offer mention
            r'strategy|Strategy|next|move',  # Strategic thinking
        ]
        
        content = reflection_text.lower()
        required_found = sum(1 for pattern in required_elements if re.search(pattern, content, re.IGNORECASE))
        
        return required_found >= 2  # At least 2 of 3 required elements
    
    def get_reflection_stats(self) -> Dict[str, Any]:
        """Get statistics about reflection usage and quality."""
        if not self.reflection_history:
            return {"total_reflections": 0}
        
        total = len(self.reflection_history)
        avg_confidence = sum(r.confidence_score for r in self.reflection_history) / total
        
        strategies_used = {}
        for reflection in self.reflection_history:
            strategy = reflection.strategic_recommendation
            strategies_used[strategy] = strategies_used.get(strategy, 0) + 1
        
        return {
            "total_reflections": total,
            "average_confidence": avg_confidence,
            "strategies_used": strategies_used,
            "reflection_enabled": self.reflection_enabled,
            "max_tokens": self.max_reflection_tokens,
            "model_tier": self._get_model_tier()
        }