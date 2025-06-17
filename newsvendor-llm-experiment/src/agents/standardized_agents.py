#!/usr/bin/env python3
"""
src/agents/standardized_agents.py
Standardized Agents for Newsvendor Experiment - integrates with existing architecture
Uses unified reflection prompts across all 10 models (local + remote)
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Current state of agent's negotiation strategy."""
    target_price: int
    min_acceptable_price: int
    max_acceptable_price: int
    estimated_partner_cost: Optional[int]
    negotiation_strategy: str
    confidence_level: float
    rounds_remaining: int


class StandardizedBuyerAgent:
    """Buyer agent with standardized reflection prompts for all models."""
    
    def __init__(
        self, 
        model_name: str, 
        model_manager,
        reflection_enabled: bool = False,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize standardized buyer agent."""
        self.model_name = model_name
        self.model_manager = model_manager
        self.reflection_enabled = reflection_enabled
        self.config = config or {}
        
        # Buyer's private information (corrected parameters)
        self.selling_price = self.config.get('selling_price', 100)
        self.demand_mean = self.config.get('demand_mean', 40)  # Corrected: Normal(40, 10)
        self.demand_std = self.config.get('demand_std', 10)   # Corrected: Normal(40, 10)
        
        # Strategic parameters
        self.target_profit_margin = self.config.get('target_profit_margin', 35)
        self.min_profit_margin = self.config.get('min_profit_margin', 20)
        
        # Track negotiation state
        self.state = AgentState(
            target_price=self.selling_price - self.target_profit_margin,  # $65
            max_acceptable_price=self.selling_price - self.min_profit_margin,  # $80
            min_acceptable_price=30,  # Above supplier's likely cost
            estimated_partner_cost=None,
            negotiation_strategy="moderate",
            confidence_level=0.5,
            rounds_remaining=10
        )
        
        logger.info(f"Initialized StandardizedBuyerAgent with model {model_name}, reflection={reflection_enabled}")
    
    async def generate_response(
        self, 
        context: str, 
        negotiation_history: List[Dict[str, Any]],
        round_number: int,
        max_rounds: int
    ):
        """Generate buyer's negotiation response with standardized prompts."""
        try:
            # Update state based on context
            self._update_state(negotiation_history, round_number, max_rounds)
            
            # Build standardized prompt
            prompt = self._build_standardized_prompt(context, negotiation_history, round_number)
            
            # Generate response with appropriate parameters
            if self.model_name == 'o3-remote':
                response = await self.model_manager.generate_response(
                    model_name=self.model_name,
                    prompt=prompt,
                    max_completion_tokens=2000,
                    reasoning_effort='high'
                )
            else:
                response = await self.model_manager.generate_response(
                    model_name=self.model_name,
                    prompt=prompt,
                    max_tokens=2000
                )
            
            if response.success:
                logger.debug(f"Buyer ({self.model_name}) generated: {response.text[:50]}...")
            else:
                logger.error(f"Buyer generation failed: {response.error}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in buyer response generation: {e}")
            # Create fallback response
            fallback_response = type('Response', (), {
                'text': "I offer $50",
                'tokens_used': 0,
                'generation_time': 0.0,
                'success': False,
                'error': str(e),
                'model_name': self.model_name,
                'timestamp': 0.0,
                'cost_estimate': 0.0
            })()
            return fallback_response
    
    def _build_standardized_prompt(
        self, 
        context: str, 
        history: List[Dict[str, Any]], 
        round_number: int
    ) -> str:
        """Build standardized prompt with optional reflection for all models."""
        
        # Base negotiation prompt (same for all models)
        base_prompt = f"""You are a retailer negotiating wholesale price with a supplier. You want the LOWEST possible price.

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
        
        if self.reflection_enabled:
            # Add standardized reflection block for ALL models
            reflection_content = self._generate_standardized_reflection(history, round_number)
            return f"""{reflection_content}

{base_prompt}"""
        else:
            return base_prompt
    
    def _generate_standardized_reflection(self, history: List[Dict[str, Any]], round_number: int) -> str:
        """Generate standardized reflection content for ALL models."""
        
        # Get latest supplier offer
        last_supplier_offer = "none"
        my_last_offer = "none"
        
        for turn in reversed(history):
            if turn.get('speaker') == 'supplier' and turn.get('price') and last_supplier_offer == "none":
                last_supplier_offer = f"${turn['price']}"
            elif turn.get('speaker') == 'buyer' and turn.get('price') and my_last_offer == "none":
                my_last_offer = f"${turn['price']}"
        
        # Calculate strategic insights
        estimated_cost = f"${self.state.estimated_partner_cost}" if self.state.estimated_partner_cost else "unknown"
        
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
            self.state.estimated_partner_cost = max(20, latest_offer - 20)
            
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
            self.state.max_acceptable_price = min(85, self.state.max_acceptable_price + 5)


class StandardizedSupplierAgent:
    """Supplier agent with standardized reflection prompts for all models."""
    
    def __init__(
        self, 
        model_name: str, 
        model_manager,
        reflection_enabled: bool = False,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize standardized supplier agent."""
        self.model_name = model_name
        self.model_manager = model_manager
        self.reflection_enabled = reflection_enabled
        self.config = config or {}
        
        # Supplier's private information
        self.production_cost = self.config.get('production_cost', 30)
        
        # Strategic parameters
        self.target_profit_margin = self.config.get('target_profit_margin', 35)
        self.min_profit_margin = self.config.get('min_profit_margin', 15)
        
        # Track negotiation state
        self.state = AgentState(
            target_price=self.production_cost + self.target_profit_margin,  # $65
            min_acceptable_price=self.production_cost + self.min_profit_margin,  # $45
            max_acceptable_price=85,  # Don't price too aggressively
            estimated_partner_cost=None,
            negotiation_strategy="moderate",
            confidence_level=0.5,
            rounds_remaining=10
        )
        
        logger.info(f"Initialized StandardizedSupplierAgent with model {model_name}, reflection={reflection_enabled}")
    
    async def generate_response(
        self, 
        context: str, 
        negotiation_history: List[Dict[str, Any]],
        round_number: int,
        max_rounds: int
    ):
        """Generate supplier's negotiation response with standardized prompts."""
        try:
            # Update state based on context
            self._update_state(negotiation_history, round_number, max_rounds)
            
            # Build standardized prompt
            prompt = self._build_standardized_prompt(context, negotiation_history, round_number)
            
            # Generate response with appropriate parameters
            if self.model_name == 'o3-remote':
                response = await self.model_manager.generate_response(
                    model_name=self.model_name,
                    prompt=prompt,
                    max_completion_tokens=2000,
                    reasoning_effort='high'
                )
            else:
                response = await self.model_manager.generate_response(
                    model_name=self.model_name,
                    prompt=prompt,
                    max_tokens=2000
                )
            
            if response.success:
                logger.debug(f"Supplier ({self.model_name}) generated: {response.text[:50]}...")
            else:
                logger.error(f"Supplier generation failed: {response.error}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in supplier response generation: {e}")
            # Create fallback response
            fallback_response = type('Response', (), {
                'text': "I want $65",
                'tokens_used': 0,
                'generation_time': 0.0,
                'success': False,
                'error': str(e),
                'model_name': self.model_name,
                'timestamp': 0.0,
                'cost_estimate': 0.0
            })()
            return fallback_response
    
    def _build_standardized_prompt(
        self, 
        context: str, 
        history: List[Dict[str, Any]], 
        round_number: int
    ) -> str:
        """Build standardized prompt with optional reflection for all models."""
        
        # Base negotiation prompt (same for all models)
        base_prompt = f"""You are a supplier negotiating wholesale price with a retailer. You want the HIGHEST possible price above your costs.

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
        
        if self.reflection_enabled:
            # Add standardized reflection block for ALL models
            reflection_content = self._generate_standardized_reflection(history, round_number)
            return f"""{reflection_content}

{base_prompt}"""
        else:
            return base_prompt
    
    def _generate_standardized_reflection(self, history: List[Dict[str, Any]], round_number: int) -> str:
        """Generate standardized reflection content for ALL models."""
        
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
        market_estimate = f"${self.state.estimated_partner_cost}" if self.state.estimated_partner_cost else "unknown"
        
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
            self.state.estimated_partner_cost = highest_offer + 10
            
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
            self.state.min_acceptable_price = max(40, self.state.min_acceptable_price - 3)