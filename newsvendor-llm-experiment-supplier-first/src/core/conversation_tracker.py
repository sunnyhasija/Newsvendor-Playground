"""
Conversation State Management for Newsvendor Negotiations v0.6

Enhanced version with CONFIGURABLE TURN ORDER for literature bias vs anchoring research.
Tracks negotiation rounds, speaker alternation, termination conditions,
and maintains conversation history with bulletproof state management.

v0.6 KEY FEATURE: Configurable turn order via context assignment only.
Identical prompts are used - only the context assignment changes.
"""

import time
import logging
import random
from collections import deque
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# Import enhanced price extractor
try:
    from src.parsing.enhanced_price_extractor import create_enhanced_price_extractor
    ENHANCED_EXTRACTION_AVAILABLE = True
except ImportError:
    # Fallback to original extractor
    from src.parsing.price_extractor import RobustPriceExtractor
    ENHANCED_EXTRACTION_AVAILABLE = False

from src.parsing.acceptance_detector import AcceptanceDetector, TerminationType

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
    """Complete negotiation outcome with v0.6 turn order tracking."""
    negotiation_id: str
    buyer_model: str
    supplier_model: str
    reflection_pattern: str  # "00", "01", "10", "11"
    turn_order_strategy: str  # v0.6: "buyer_first", "supplier_first", "random"
    first_speaker: str  # v0.6: tracks who actually went first
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
    """Enhanced conversation state tracking with configurable turn order (v0.6)."""
    
    def __init__(self, negotiation_id: str, buyer_model: str, supplier_model: str, 
                 reflection_pattern: str, turn_order_strategy: str = "buyer_first",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize conversation tracker with configurable turn order (v0.6).
        
        Args:
            negotiation_id: Unique identifier for this negotiation
            buyer_model: Model name for buyer role
            supplier_model: Model name for supplier role  
            reflection_pattern: "00", "01", "10", "11"
            turn_order_strategy: "buyer_first", "supplier_first", "random"
            config: Additional configuration parameters
        """
        self.negotiation_id = negotiation_id
        self.buyer_model = buyer_model
        self.supplier_model = supplier_model
        self.reflection_pattern = reflection_pattern
        self.turn_order_strategy = turn_order_strategy
        self.config = config or {}
        
        # v0.6: Determine initial speaker based on strategy
        self.first_speaker = self._determine_initial_speaker(turn_order_strategy)
        self.current_speaker = self.first_speaker
        
        # Conversation state
        self.turns: List[NegotiationTurn] = []
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
        
        # Initialize enhanced price extraction with fallback capability
        self.price_extractor = self._initialize_price_extractor()
        
        # Acceptance detector (unchanged)
        self.acceptance_detector = AcceptanceDetector(config)
        
        # Timing
        self.start_time = time.time()
        self.last_activity = self.start_time
        
        logger.info(f"Initialized negotiation {negotiation_id}: {buyer_model} vs {supplier_model}")
        logger.info(f"v0.6 Turn order: {turn_order_strategy} -> {self.first_speaker} goes first")
        if ENHANCED_EXTRACTION_AVAILABLE:
            logger.debug("Using enhanced price extraction with local model fallback")
        else:
            logger.debug("Using traditional price extraction (enhanced extractor not available)")
    
    def _determine_initial_speaker(self, strategy: str) -> str:
        """
        Determine who goes first based on turn order strategy.
        
        Args:
            strategy: "buyer_first", "supplier_first", "random"
            
        Returns:
            "buyer" or "supplier"
        """
        if strategy == "buyer_first":
            return "buyer"
        elif strategy == "supplier_first":
            return "supplier"
        elif strategy == "random":
            return random.choice(["buyer", "supplier"])
        else:
            logger.warning(f"Unknown turn_order_strategy '{strategy}', defaulting to buyer_first")
            return "buyer"
    
    def _initialize_price_extractor(self):
        """Initialize price extractor with fallback capability if available."""
        
        if not ENHANCED_EXTRACTION_AVAILABLE:
            # Use original extractor as fallback
            logger.info("Enhanced price extractor not available, using traditional extraction")
            return RobustPriceExtractor(self.config)
        
        try:
            # Try to initialize enhanced extractor with local model fallback
            import ollama
            
            # Create Ollama client for fallback
            ollama_client = ollama.Client()
            
            # Test if a suitable fallback model is available
            fallback_model = None
            test_models = ["llama3.2:latest", "llama3:latest", "mistral:instruct", "gemma2:2b", "tinyllama:latest"]
            
            for model in test_models:
                try:
                    # Quick test to see if model is available
                    ollama_client.generate(
                        model=model,
                        prompt="test",
                        options={'num_predict': 1}
                    )
                    fallback_model = model
                    logger.info(f"Using {model} for price extraction fallback")
                    break
                except Exception:
                    continue
            
            if fallback_model:
                return create_enhanced_price_extractor(
                    config=self.config,
                    ollama_client=ollama_client,
                    fallback_model=fallback_model
                )
            else:
                logger.warning("No suitable local model found for price extraction fallback")
                return create_enhanced_price_extractor(self.config, None)  # No fallback
                
        except Exception as e:
            logger.warning(f"Could not initialize enhanced price extractor: {e}")
            logger.info("Falling back to traditional price extraction")
            return RobustPriceExtractor(self.config)
    
    async def add_turn(
        self, 
        speaker: str, 
        message: str, 
        reflection: Optional[str] = None,
        tokens_used: int = 0,
        generation_time: float = 0.0
    ) -> bool:
        """
        Add a turn with enhanced price extraction.
        
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
            
            # Extract price with enhanced extractor (async if available)
            if ENHANCED_EXTRACTION_AVAILABLE and hasattr(self.price_extractor, 'extract_price'):
                # Enhanced extractor - async call
                price = await self.price_extractor.extract_price(
                    message, 
                    previous_context=[turn.message for turn in self.turns[-3:]],
                    speaker_role=speaker
                )
            else:
                # Traditional extractor - sync call
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
        """Get current negotiation state summary with v0.6 turn order info."""
        return {
            "negotiation_id": self.negotiation_id,
            "round_number": self.round_number,
            "current_speaker": self.current_speaker,
            "turn_order_strategy": self.turn_order_strategy,  # v0.6
            "first_speaker": self.first_speaker,  # v0.6
            "completed": self.completed,
            "termination_type": self.termination_type.value if self.termination_type else None,
            "agreed_price": self.agreed_price,
            "total_turns": len(self.turns),
            "recent_prices": list(self.last_prices),
            "recent_speakers": list(self.last_speakers),
            "elapsed_time": time.time() - self.start_time
        }
    
    def get_conversation_history(self) -> str:
        """
        Get formatted conversation history for context with v0.6 configurable prompts.
        
        v0.6 CRITICAL: This is where turn order control happens!
        We use IDENTICAL prompt text, only changing the context assignment.
        """
        if not self.turns:
            # Opening context - depends on current speaker (determined by turn order strategy)
            return "This is a new negotiation. Make your opening offer."
        
        # Build conversation history
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
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get enhanced price extraction statistics."""
        if ENHANCED_EXTRACTION_AVAILABLE and hasattr(self.price_extractor, 'get_extraction_stats'):
            base_stats = self.price_extractor.get_extraction_stats()
        else:
            # Traditional extractor stats
            base_stats = {
                "total_attempts": len(self.turns),
                "successful_extractions": len([t for t in self.turns if t.price is not None]),
                "fallback_enabled": False
            }
            if base_stats["total_attempts"] > 0:
                base_stats["success_rate"] = base_stats["successful_extractions"] / base_stats["total_attempts"]
            else:
                base_stats["success_rate"] = 0
        
        # Add negotiation-specific stats
        base_stats.update({
            "negotiation_id": self.negotiation_id,
            "total_turns": len(self.turns),
            "turns_with_prices": len([t for t in self.turns if t.price is not None]),
            "price_extraction_rate": len([t for t in self.turns if t.price is not None]) / len(self.turns) if self.turns else 0,
            "enhanced_extraction_available": ENHANCED_EXTRACTION_AVAILABLE
        })
        
        return base_stats
    
    def get_final_result(self) -> NegotiationResult:
        """Get complete negotiation result with v0.6 turn order tracking."""
        buyer_profit, supplier_profit = self.calculate_profits()
        
        # Calculate distance from optimal price
        distance_from_optimal = None
        if self.agreed_price:
            distance_from_optimal = abs(self.agreed_price - self.optimal_price)
        
        # Calculate totals
        total_tokens = sum(turn.tokens_used for turn in self.turns)
        total_time = time.time() - self.start_time
        
        # Generate enhanced metadata with extraction stats and v0.6 turn order info
        metadata = {
            "price_extraction_stats": self.get_extraction_stats(),
            "final_state": self.get_current_state(),
            "conversation_length": len(self.turns),
            "avg_tokens_per_turn": total_tokens / len(self.turns) if self.turns else 0,
            "prices_offered": list(self.last_prices),
            "negotiation_range": {
                "min_price": min(self.last_prices) if self.last_prices else None,
                "max_price": max(self.last_prices) if self.last_prices else None,
                "price_variance": None  # Could calculate variance
            },
            "extraction_method": "enhanced" if ENHANCED_EXTRACTION_AVAILABLE else "traditional",
            # v0.6: Turn order research tracking
            "turn_order_analysis": {
                "strategy": self.turn_order_strategy,
                "first_speaker": self.first_speaker,
                "opening_advantage": self.first_speaker == "buyer" if self.agreed_price else None,
                "anchoring_effect": self.turns[0].price if self.turns and self.turns[0].price else None
            }
        }
        
        return NegotiationResult(
            negotiation_id=self.negotiation_id,
            buyer_model=self.buyer_model,
            supplier_model=self.supplier_model,
            reflection_pattern=self.reflection_pattern,
            turn_order_strategy=self.turn_order_strategy,  # v0.6
            first_speaker=self.first_speaker,  # v0.6
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