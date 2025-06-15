"""
Buyer Agent for Newsvendor Negotiation

Implements the retailer's negotiation behavior with role-specific
prompts, reflection capabilities, and strategic reasoning.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .reflection_mixin import ReflectionMixin
from ..core.model_manager import OptimizedModelManager, GenerationResponse

logger = logging.getLogger(__name__)


@dataclass
class BuyerState:
    """Current state of buyer's negotiation strategy."""
    target_price: int
    max_acceptable_price: int
    min_acceptable_price: int
    estimated_supplier_cost: Optional[int]
    negotiation_strategy: str  # "aggressive", "moderate", "cooperative"
    confidence_level: float
    rounds_remaining: int


class BuyerAgent:
    """Retailer agent that wants the LOWEST possible wholesale price."""
    
    def __init__(
        self, 
        model_name: str, 
        model_manager: OptimizedModelManager,
        reflection_enabled: bool = False,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize buyer agent.
        
        Args:
            model_name: Name of the LLM model to use
            model_manager: Model manager instance
            reflection_enabled: Whether to use reflection
            config: Configuration dictionary
        """
        self.model_name = model_name
        self.model_manager = model_manager
        self.reflection_enabled = reflection_enabled
        self.config = config or {}
        
        # Buyer's private information (from newsvendor setup)
        self.selling_price = self.config.get('selling_price', 100)
        self.demand_mean = self.config.get('demand_mean', 40)
        self.demand_std = self.config.get('demand_std', 10)
        
        # Strategic parameters
        self.target_profit_margin = self.config.get('target_profit_margin', 35)
        self.min_profit_margin = self.config.get('min_profit_margin', 20)
        
        # Initialize reflection mixin if enabled
        if reflection_enabled:
            self.reflection = ReflectionMixin(
                role="buyer",
                model_name=model_name,
                config=config
            )
        
        # Track negotiation state
        self.state = BuyerState(
            target_price=self.selling_price - self.target_profit_margin,  # $65
            max_acceptable_price=self.selling_price - self.min_profit_margin,  # $80
            min_acceptable_price=30,  # Above supplier's likely cost
            estimated_supplier_cost=None,
            negotiation_strategy="moderate",
            confidence_level=0.5,
            rounds_remaining=10
        )
        
        logger.info(f"Initialized BuyerAgent with model {model_name}, reflection={reflection_enabled}")
    
    async def generate_response(
        self, 
        context: str, 
        negotiation_history: List[Dict[str, Any]],
        round_number: int,
        max_rounds: int
    ) -> GenerationResponse:
        """
        Generate buyer's negotiation response.
        
        Args:
            context: Current negotiation context
            negotiation_history: List of previous turns
            round_number: Current round number
            max_rounds: Maximum rounds allowed
            
        Returns:
            GenerationResponse with buyer's message
        """
        try:
            # Update state based on context
            self._update_state(negotiation_history, round_number, max_rounds)
            
            # Build prompt
            if self.reflection_enabled:
                prompt = self._build_reflection_prompt(context, negotiation_history, round_number)
            else:
                prompt = self._build_standard_prompt(context)
            
            # Generate response
            response = await self.model_manager.generate_response(
                model_name=self.model_name,
                prompt=prompt,
                max_tokens=self._get_token_limit()
            )
            
            if response.success:
                logger.debug(f"Buyer ({self.model_name}) generated: {response.text[:50]}...")
            else:
                logger.error(f"Buyer generation failed: {response.error}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in buyer response generation: {e}")
            return GenerationResponse(
                text="I offer $50",  # Fallback response
                tokens_used=0,
                generation_time=0.0,
                success=False,
                error=str(e),
                model_name=self.model_name
            )
    
    def _update_state(
        self, 
        history: List[Dict[str, Any]], 
        round_number: int, 
        max_rounds: int
    ) -> None:
        """Update buyer's strategic state based on negotiation progress."""
        self.state.rounds_remaining = max_rounds - round_number
        
        # Estimate supplier's cost from their offers
        supplier_offers = [
            turn.get('price') for turn in history 
            if turn.get('speaker') == 'supplier' and turn.get('price')
        ]
        
        if supplier_offers:
            # Assume supplier wants at least 20% profit margin
            latest_offer = supplier_offers[-1]
            self.state.estimated_supplier_cost = max(20, latest_offer - 20)
            
            # Adjust strategy based on supplier's behavior
            if len(supplier_offers) >= 2:
                price_trend = supplier_offers[-1] - supplier_offers[-2]
                if price_trend < 0:  # Supplier lowering price
                    self.state.negotiation_strategy = "aggressive"
                elif price_trend > 0:  # Supplier raising price
                    self.state.negotiation_strategy = "cooperative"
        
        # Adjust urgency as rounds decrease
        if self.state.rounds_remaining <= 2:
            self.state.negotiation_strategy = "cooperative"
            # Raise acceptable price if running out of time
            self.state.max_acceptable_price = min(85, self.state.max_acceptable_price + 5)
    
    def _build_standard_prompt(self, context: str) -> str:
        """Build standard prompt without reflection."""
        return f"""You are a retailer negotiating wholesale price with a supplier. You want the LOWEST possible price.

YOUR PRIVATE INFO (do not reveal):
- You sell at: ${self.selling_price} per unit
- Demand: Normal distribution, mean {self.demand_mean} units, std {self.demand_std}
- Your profit = ({self.selling_price} - wholesale_price) × units_sold

RULES:
- Give SHORT responses only
- Make offers like "I offer $45" or "How about $38?"
- Accept good offers by saying "I accept $X"
- NO explanations, stories, or reasoning
- Price range: $1-99 only

Current situation: {context}

Your response (keep it under 15 words):"""
    
    def _build_reflection_prompt(
        self, 
        context: str, 
        history: List[Dict[str, Any]], 
        round_number: int
    ) -> str:
        """Build prompt with reflection capabilities."""
        
        # Generate reflection content
        reflection_content = self._generate_reflection(history, round_number)
        
        return f"""{reflection_content}

You are a retailer negotiating wholesale price with a supplier. You want the LOWEST possible price.

YOUR PRIVATE INFO (do not reveal):
- You sell at: ${self.selling_price} per unit
- Demand: Normal distribution, mean {self.demand_mean} units, std {self.demand_std}
- Your profit = ({self.selling_price} - wholesale_price) × units_sold

RULES:
- Give SHORT responses only
- Make offers like "I offer $45" or "How about $38?"
- Accept good offers by saying "I accept $X"
- NO explanations, stories, or reasoning
- Price range: $1-99 only

Current situation: {context}

Your response (keep it under 15 words):"""
    
    def _generate_reflection(self, history: List[Dict[str, Any]], round_number: int) -> str:
        """Generate reflection content for strategic thinking."""
        
        # Get latest supplier offer
        last_supplier_offer = "none"
        my_last_offer = "none"
        
        for turn in reversed(history):
            if turn.get('speaker') == 'supplier' and turn.get('price') and last_supplier_offer == "none":
                last_supplier_offer = f"${turn['price']}"
            elif turn.get('speaker') == 'buyer' and turn.get('price') and my_last_offer == "none":
                my_last_offer = f"${turn['price']}"
        
        # Calculate strategic insights
        estimated_cost = f"${self.state.estimated_supplier_cost}" if self.state.estimated_supplier_cost else "unknown"
        
        # Determine strategy
        if self.state.negotiation_strategy == "aggressive":
            strategy = "push harder for lower price"
        elif self.state.negotiation_strategy == "cooperative":
            strategy = "move toward acceptable middle ground"
        else:
            strategy = "make reasonable counter-offer"
        
        return f"""<think>
Current negotiation status:
- Last supplier offer: {last_supplier_offer}
- My last offer: {my_last_offer}
- Round: {round_number}/10

Quick analysis:
- Their offer suggests cost around: {estimated_cost}
- My target profit margin: ~${self.target_profit_margin} per unit
- Should I: {strategy}?

Strategy: {strategy}
</think>"""
    
    def _get_token_limit(self) -> int:
        """Get appropriate token limit for this model."""
        model_configs = self.model_manager.model_configs
        config = model_configs.get(self.model_name, {})
        
        if self.reflection_enabled:
            # Add extra tokens for reflection
            base_limit = config.get('token_limit', 256)
            reflection_tokens = 100 if config.get('tier') == 'ultra' else 150
            return base_limit + reflection_tokens
        else:
            return config.get('token_limit', 256)
    
    def should_accept_offer(self, offered_price: int) -> bool:
        """Determine if buyer should accept the offered price."""
        
        # Never accept if above maximum acceptable price
        if offered_price > self.state.max_acceptable_price:
            return False
        
        # Always accept if at or below target price
        if offered_price <= self.state.target_price:
            return True
        
        # Probabilistic acceptance based on how close to target
        # and how many rounds remain
        price_gap = offered_price - self.state.target_price
        max_gap = self.state.max_acceptable_price - self.state.target_price
        
        # Higher probability if fewer rounds remain
        urgency_factor = (10 - self.state.rounds_remaining) / 10
        
        # Calculate acceptance probability
        base_prob = 1 - (price_gap / max_gap)
        adjusted_prob = base_prob + (urgency_factor * 0.3)
        
        return adjusted_prob > 0.7  # 70% threshold
    
    def get_strategic_insights(self) -> Dict[str, Any]:
        """Get current strategic insights for analysis."""
        return {
            "role": "buyer",
            "model_name": self.model_name,
            "reflection_enabled": self.reflection_enabled,
            "target_price": self.state.target_price,
            "max_acceptable_price": self.state.max_acceptable_price,
            "estimated_supplier_cost": self.state.estimated_supplier_cost,
            "negotiation_strategy": self.state.negotiation_strategy,
            "confidence_level": self.state.confidence_level,
            "rounds_remaining": self.state.rounds_remaining,
            "private_info": {
                "selling_price": self.selling_price,
                "demand_mean": self.demand_mean,
                "target_profit_margin": self.target_profit_margin
            }
        }