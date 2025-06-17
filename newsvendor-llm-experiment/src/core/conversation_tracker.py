"""
Conversation State Management for Newsvendor Negotiations

Tracks negotiation rounds, speaker alternation, termination conditions,
and maintains conversation history with bulletproof state management.
"""

import time
import logging
from collections import deque
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from parsing.acceptance_detector import AcceptanceDetector, TerminationType
from parsing.price_extractor import RobustPriceExtractor
logger = logging.getLogger(__name__)


@dataclass
class NegotiationTurn:
    """Single turn in a negotiation."""
    round_number: int
    speaker: str
    message: str
    price: Optional[int]
    reflection: Optional[str]
    timestamp: float
    tokens_used: int = 0
    generation_time: float = 0.0
    confidence: float = 0.0


@dataclass
class NegotiationResult:
    """Complete negotiation outcome."""
    negotiation_id: str
    buyer_model: str
    supplier_model: str
    reflection_pattern: str  # "00", "01", "10", "11"
    completed: bool
    agreed_price: Optional[int]
    termination_type: TerminationType
    total_rounds: int
    total_tokens: int
    total_time: float
    buyer_profit: Optional[float]
    supplier_profit: Optional[float]
    distance_from_optimal: Optional[float]
    turns: List[NegotiationTurn]
    metadata: Dict[str, Any]


class ConversationTracker:
    """Bulletproof conversation state tracking."""
    
    def __init__(self, negotiation_id: str, buyer_model: str, supplier_model: str, 
                 reflection_pattern: str, config: Optional[Dict[str, Any]] = None):
        """Initialize conversation tracker."""
        self.negotiation_id = negotiation_id
        self.buyer_model = buyer_model
        self.supplier_model = supplier_model
        self.reflection_pattern = reflection_pattern
        self.config = config or {}
        
        # Conversation state
        self.turns: List[NegotiationTurn] = []
        self.current_speaker = "buyer"  # Always starts with buyer
        self.round_number = 0
        
        # Termination state
        self.termination_type: Optional[TerminationType] = None
        self.agreed_price: Optional[int] = None
        self.completed = False
        
        # Price tracking for convergence detection
        self.last_prices = deque(maxlen=5)  # Track recent prices
        self.last_speakers = deque(maxlen=5)  # Track speakers for prices
        
        # Configuration
        self.max_rounds = self.config.get('max_rounds', 10)
        self.optimal_price = self.config.get('optimal_price', 65)
        self.selling_price = self.config.get('selling_price', 100)
        self.production_cost = self.config.get('production_cost', 30)
        
        # Components
        self.price_extractor = RobustPriceExtractor(config)
        self.acceptance_detector = AcceptanceDetector(config)
        
        # Timing
        self.start_time = time.time()
        self.last_activity = self.start_time
        
        logger.info(f"Initialized negotiation {negotiation_id}: {buyer_model} vs {supplier_model}")
    
    def add_turn(
        self, 
        speaker: str, 
        message: str, 
        reflection: Optional[str] = None,
        tokens_used: int = 0,
        generation_time: float = 0.0
    ) -> bool:
        """
        Add a turn and check for termination conditions.
        
        Args:
            speaker: 'buyer' or 'supplier'
            message: The negotiation message
            reflection: Optional reflection content
            tokens_used: Number of tokens used in generation
            generation_time: Time taken for generation
            
        Returns:
            True if turn was added successfully, False if invalid
        """
        try:
            # Validate speaker alternation
            if speaker != self.current_speaker:
                logger.warning(
                    f"Turn order violation in {self.negotiation_id}: "
                    f"expected {self.current_speaker}, got {speaker}"
                )
                return False
            
            # Check if negotiation already terminated
            if self.completed:
                logger.warning(f"Attempt to add turn to completed negotiation {self.negotiation_id}")
                return False
            
            # Increment round number
            self.round_number += 1
            
            # Extract price from message
            price = self.price_extractor.extract_price(
                message, 
                previous_context=[turn.message for turn in self.turns[-3:]],
                speaker_role=speaker
            )
            
            # Create turn record
            turn = NegotiationTurn(
                round_number=self.round_number,
                speaker=speaker,
                message=message,
                price=price,
                reflection=reflection,
                timestamp=time.time(),
                tokens_used=tokens_used,
                generation_time=generation_time,
                confidence=0.0  # Could be filled by price extractor
            )
            
            # Add to conversation history
            self.turns.append(turn)
            
            # Update price tracking
            if price is not None:
                self.last_prices.append(price)
                self.last_speakers.append(speaker)
            
            # Check termination conditions
            self._check_termination(message, price)
            
            # Switch speakers for next turn
            self.current_speaker = "supplier" if speaker == "buyer" else "buyer"
            
            # Update activity timestamp
            self.last_activity = time.time()
            
            logger.debug(f"Added turn {self.round_number} for {speaker}: price={price}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding turn to {self.negotiation_id}: {e}")
            return False
    
    def _check_termination(self, message: str, price: Optional[int]) -> None:
        """Check all termination conditions."""
        if self.completed:
            return
        
        # Check for explicit acceptance
        acceptance_result = self.acceptance_detector.detect_acceptance(
            message, price, list(self.last_prices)
        )
        
        if acceptance_result.is_acceptance:
            self.termination_type = acceptance_result.termination_type
            self.agreed_price = acceptance_result.accepted_price or price
            self.completed = True
            logger.info(f"Negotiation {self.negotiation_id} terminated by acceptance at ${self.agreed_price}")
            return
        
        # Check for price convergence (requires alternating speakers)
        if len(self.last_prices) >= 2 and len(self.last_speakers) >= 2:
            convergence_result = self.acceptance_detector.detect_convergence(
                list(self.last_prices), list(self.last_speakers)
            )
            
            if convergence_result.is_acceptance:
                self.termination_type = convergence_result.termination_type
                self.agreed_price = convergence_result.accepted_price
                self.completed = True
                logger.info(f"Negotiation {self.negotiation_id} terminated by convergence at ${self.agreed_price}")
                return
        
        # Check for round limit
        if self.round_number >= self.max_rounds:
            self.termination_type = TerminationType.TIMEOUT
            self.completed = True
            logger.info(f"Negotiation {self.negotiation_id} terminated by timeout after {self.round_number} rounds")
            return
    
    def force_termination(self, reason: str = "forced") -> None:
        """Force termination of negotiation."""
        if not self.completed:
            self.termination_type = TerminationType.FAILURE
            self.completed = True
            logger.warning(f"Forced termination of {self.negotiation_id}: {reason}")
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current negotiation state summary."""
        return {
            "negotiation_id": self.negotiation_id,
            "round_number": self.round_number,
            "current_speaker": self.current_speaker,
            "completed": self.completed,
            "termination_type": self.termination_type.value if self.termination_type else None,
            "agreed_price": self.agreed_price,
            "total_turns": len(self.turns),
            "recent_prices": list(self.last_prices),
            "recent_speakers": list(self.last_speakers),
            "elapsed_time": time.time() - self.start_time
        }
    
    def get_conversation_history(self) -> str:
        """Get formatted conversation history for context."""
        if not self.turns:
            return "This is a new negotiation. Make your opening offer."
        
        history_lines = []
        for turn in self.turns:
            role = turn.speaker.capitalize()
            if turn.price:
                history_lines.append(f"Round {turn.round_number}: {role} offered ${turn.price}")
            else:
                # Truncate long messages for history
                message_preview = turn.message[:50] + "..." if len(turn.message) > 50 else turn.message
                history_lines.append(f"Round {turn.round_number}: {role} said \"{message_preview}\"")
        
        # Add latest message for context
        if self.turns:
            latest = self.turns[-1]
            partner_role = "Supplier" if latest.speaker == "supplier" else "Buyer"
            history_lines.append(f"\nLatest: {partner_role} said \"{latest.message}\"")
        
        # Add final round warning if approaching limit
        if self.round_number >= self.max_rounds - 1:
            history_lines.append(f"\nFINAL ROUND ({self.round_number + 1}/{self.max_rounds}). This is your last chance to make a deal!")
        
        return "\n".join(history_lines)
    
    def calculate_profits(self) -> Tuple[Optional[float], Optional[float]]:
        """Calculate buyer and supplier profits if deal was reached."""
        if not self.agreed_price:
            return None, None
        
        # Buyer profit = (selling_price - wholesale_price) * expected_demand
        # Using expected demand of 40 units (mean of Normal(40, 10))
        expected_demand = 40
        buyer_profit = (self.selling_price - self.agreed_price) * expected_demand
        
        # Supplier profit = (wholesale_price - production_cost) * expected_demand
        supplier_profit = (self.agreed_price - self.production_cost) * expected_demand
        
        return buyer_profit, supplier_profit
    
    def get_final_result(self) -> NegotiationResult:
        """Get complete negotiation result."""
        buyer_profit, supplier_profit = self.calculate_profits()
        
        # Calculate distance from optimal price
        distance_from_optimal = None
        if self.agreed_price:
            distance_from_optimal = abs(self.agreed_price - self.optimal_price)
        
        # Calculate totals
        total_tokens = sum(turn.tokens_used for turn in self.turns)
        total_time = time.time() - self.start_time
        
        # Generate metadata
        metadata = {
            "price_extraction_stats": self.price_extractor.get_extraction_stats(),
            "final_state": self.get_current_state(),
            "conversation_length": len(self.turns),
            "avg_tokens_per_turn": total_tokens / len(self.turns) if self.turns else 0,
            "prices_offered": list(self.last_prices),
            "negotiation_range": {
                "min_price": min(self.last_prices) if self.last_prices else None,
                "max_price": max(self.last_prices) if self.last_prices else None,
                "price_variance": None  # Could calculate variance
            }
        }
        
        return NegotiationResult(
            negotiation_id=self.negotiation_id,
            buyer_model=self.buyer_model,
            supplier_model=self.supplier_model,
            reflection_pattern=self.reflection_pattern,
            completed=self.completed,
            agreed_price=self.agreed_price,
            termination_type=self.termination_type or TerminationType.FAILURE,
            total_rounds=self.round_number,
            total_tokens=total_tokens,
            total_time=total_time,
            buyer_profit=buyer_profit,
            supplier_profit=supplier_profit,
            distance_from_optimal=distance_from_optimal,
            turns=self.turns.copy(),
            metadata=metadata
        )
    
    def export_conversation(self) -> Dict[str, Any]:
        """Export conversation for analysis."""
        result = self.get_final_result()
        return asdict(result)