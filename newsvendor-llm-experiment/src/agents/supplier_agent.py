"""
Supplier Agent for Newsvendor Negotiation

Implements the supplier's negotiation behavior with role-specific
prompts, reflection capabilities, and strategic reasoning.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .reflection_mixin import ReflectionMixin
from ..core.model_manager import OptimizedModelManager, GenerationResponse

logger = logging.getLogger(__name__)


@dataclass
class SupplierState:
    """Current state of supplier's negotiation strategy."""
    target_price: int
    min_acceptable_price: int
    max_reasonable_price: int
    estimated_buyer_value: Optional[int]
    negotiation_strategy: str  # "aggressive", "moderate", "cooperative"
    confidence_level: float
    rounds_remaining: int


class SupplierAgent:
    """Supplier agent that wants the HIGHEST possible wholesale price."""
    
    def __init__(
        self, 
        model_name: str, 
        model_manager: OptimizedModelManager,
        reflection_enabled: bool = False,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize supplier agent.
        
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
        
        # Supplier's private information (from newsvendor setup)
        self.production_cost = self.config.get('production_cost', 30)
        
        # Strategic parameters
        self.target_profit_margin = self.config.get('target_profit_margin', 35)
        self.min_profit_margin = self.config.get('min_profit_margin', 15)
        
        # Initialize reflection mixin if enabled
        if reflection_enabled:
            self.reflection = ReflectionMixin(
                role="supplier",
                model_name=model_name,
                config=config
            )
        
        # Track negotiation state
        self.state = SupplierState(
            target_price=self.production_cost + self.target_profit_margin,  # $65
            min_acceptable_price=self.production_cost + self.min_profit_margin,  # $45
            max_reasonable_price=85,  # Don't price too aggressively
            estimated_buyer_value=None,
            negotiation_strategy="moderate",
            confidence_level=0.5,
            rounds_remaining=10
        )
        
        logger.info(f"Initialized SupplierAgent with model {model_name}, reflection={reflection_enabled}")
    
    async def generate_response(
        self, 
        context: str, 
        negotiation_history: List[Dict[str, Any]],
        round_number: int,
        max_rounds: int
    ) -> GenerationResponse:
        """
        Generate supplier's negotiation response.
        
        Args:
            context: Current negotiation context
            negotiation_history: List of previous turns
            round_number: Current round number
            max_rounds: Maximum rounds allowed
            
        Returns:
            GenerationResponse with supplier's message
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
                logger.debug(f"Supplier ({self.model_name}) generated: {response.text[:50]}...")
            else:
                logger.error(f"Supplier generation failed: {response.error}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in supplier response generation: {e}")
            return GenerationResponse(
                text="I want $65",  # Fallback response
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
        """Update supplier's strategic state based on negotiation progress."""
        self.state.rounds_remaining = max_rounds - round_number
        
        # Estimate buyer's valuation from their offers
        buyer_offers = [
            turn.get('price') for turn in history 
            if turn.get('speaker') == 'buyer' and turn.get('price')
        ]
        
        if buyer_offers:
            # Assume buyer's max acceptable price is ~10-20% above their highest offer
            highest_offer = max(buyer_offers)
            self.state.estimated_buyer_value = highest_offer + 10
            
            # Adjust strategy based on buyer's behavior
            if len(buyer_offers) >= 2:
                price_trend = buyer_offers[-1] - buyer_offers[-2]
                if price_trend > 0:  # Buyer raising offers
                    self.state.negotiation_strategy = "aggressive"
                elif price_trend < 0:  # Buyer lowering offers
                    self.state.negotiation_strategy = "cooperative"
        
        # Adjust urgency as rounds decrease
        if self.state.rounds_remaining <= 2:
            self.state.negotiation_strategy = "cooperative"
            # Lower acceptable price if running out of time
            self.state.min_acceptable_price = max(40, self.state.min_acceptable_price - 3)
    
    def _build_standard_prompt(self, context: str) -> str:
        """Build standard prompt without reflection."""
        return f"""You are a supplier negotiating wholesale price with a retailer. You want the HIGHEST possible price above your costs.

YOUR PRIVATE INFO (do not reveal):
- Production cost: ${self.production_cost} per unit
- Your profit = (wholesale_price - {self.production_cost}) × units_sold

RULES:
- Give SHORT responses only
- Make offers like "I want $65" or "How about $58?"
- Accept good offers by saying "I accept $X"
- NO explanations, stories, or reasoning
- Price range: $31-200 only

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

You are a supplier negotiating wholesale price with a retailer. You want the HIGHEST possible price above your costs.

YOUR PRIVATE INFO (do not reveal):
- Production cost: ${self.production_cost} per unit
- Your profit = (wholesale_price - {self.production_cost}) × units_sold

RULES:
- Give SHORT responses only
- Make offers like "I want $65" or "How about $58?"
- Accept good offers by saying "I accept $X"
- NO explanations, stories, or reasoning
- Price range: $31-200 only

Current situation: {context}

Your response (keep it under 15 words):"""
    
    def _generate_reflection(self, history: List[Dict[str, Any]], round_number: int) -> str:
        """Generate reflection content for strategic thinking."""
        
        # Get latest buyer offer
        last_buyer_offer = "none"
        my_last_offer = "none"
        
        for turn in reversed(history):
            if turn.get('speaker') == 'buyer' and turn.get('price') and last_buyer_offer == "none":
                last_buyer_offer = f"${turn['price']}"
            elif turn.get('speaker') == 'supplier' and turn.get('price') and my_last_offer == "none":
                my_last_offer = f"${turn['price']}"
        
        # Calculate profit from buyer's offer
        my_profit = "unknown"
        if last_buyer_offer != "none":
            try:
                offer_amount = int(last_buyer_offer.replace('$', ''))
                profit = offer_amount - self.production_cost
                my_profit = f"${profit}"
            except:
                pass
        
        # Estimate market value
        market_estimate = f"${self.state.estimated_buyer_value}" if self.state.estimated_buyer_value else "unknown"
        
        # Determine strategy
        if self.state.negotiation_strategy == "aggressive":
            strategy = "hold firm for higher price"
        elif self.state.negotiation_strategy == "cooperative":
            strategy = "move toward acceptable deal"
        else:
            strategy = "make reasonable counter-offer"
        
        return f"""<think>
Current negotiation status:
- Last buyer offer: {last_buyer_offer}
- My last offer: {my_last_offer}
- Round: {round_number}/10

Quick analysis:
- Their offer gives me profit of: {my_profit}
- Market seems to value around: {market_estimate}
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
        """Determine if supplier should accept the offered price."""
        
        # Never accept if below minimum acceptable price
        if offered_price < self.state.min_acceptable_price:
            return False
        
        # Always accept if at or above target price
        if offered_price >= self.state.target_price:
            return True
        
        # Probabilistic acceptance based on how close to target
        # and how many rounds remain
        price_gap = self.state.target_price - offered_price
        max_gap = self.state.target_price - self.state.min_acceptable_price
        
        # Higher probability if fewer rounds remain
        urgency_factor = (10 - self.state.rounds_remaining) / 10
        
        # Calculate acceptance probability
        base_prob = 1 - (price_gap / max_gap)
        adjusted_prob = base_prob + (urgency_factor * 0.3)
        
        return adjusted_prob > 0.7  # 70% threshold
    
    def get_strategic_insights(self) -> Dict[str, Any]:
        """Get current strategic insights for analysis."""
        return {
            "role": "supplier",
            "model_name": self.model_name,
            "reflection_enabled": self.reflection_enabled,
            "target_price": self.state.target_price,
            "min_acceptable_price": self.state.min_acceptable_price,
            "estimated_buyer_value": self.state.estimated_buyer_value,
            "negotiation_strategy": self.state.negotiation_strategy,
            "confidence_level": self.state.confidence_level,
            "rounds_remaining": self.state.rounds_remaining,
            "private_info": {
                "production_cost": self.production_cost,
                "target_profit_margin": self.target_profit_margin
            }
        }