"""
Acceptance Detection for Newsvendor Negotiation

Detects when negotiating parties have reached agreement through
explicit acceptance statements or implicit convergence.
"""

import re
import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TerminationType(Enum):
    """Types of negotiation termination."""
    ACCEPTANCE = "acceptance"
    CONVERGENCE = "convergence"
    TIMEOUT = "timeout"
    FAILURE = "failure"


@dataclass
class AcceptanceResult:
    """Result of acceptance detection."""
    is_acceptance: bool
    confidence: float
    pattern_matched: str
    accepted_price: Optional[int]
    termination_type: TerminationType


class AcceptanceDetector:
    """Detect explicit and implicit agreement in negotiations."""
    
    # Acceptance patterns with confidence scores
    ACCEPTANCE_PATTERNS = [
        (r'\bI accept\b', 0.95, "explicit_accept"),
        (r'\baccept\b.*\$?(\d+)', 0.90, "accept_with_price"),
        (r'\bdeal\b', 0.85, "deal_statement"),
        (r'\bagreed?\b', 0.80, "agreed_statement"),
        (r'\bokay?\b.*\$?(\d+)', 0.75, "okay_with_price"),
        (r'\bfine\b.*\$?(\d+)', 0.70, "fine_with_price"),
        (r'\byes\b.*\$?(\d+)', 0.65, "yes_with_price"),
        (r'\bsounds good\b', 0.60, "sounds_good"),
        (r'\bthat works\b', 0.65, "that_works"),
        (r'\blet\'s do it\b', 0.70, "lets_do_it"),
    ]
    
    # Rejection patterns (negative indicators)
    REJECTION_PATTERNS = [
        r'\bno\b',
        r'\breject\b',
        r'\brefuse\b',
        r'\bcan\'t accept\b',
        r'\btoo low\b',
        r'\btoo high\b',
        r'\bnot acceptable\b',
    ]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize acceptance detector with configuration."""
        self.config = config or {}
        self.debug_mode = self.config.get('debug', False)
        self.min_confidence = self.config.get('min_confidence', 0.6)
        
    def detect_acceptance(
        self, 
        message: str, 
        current_price: Optional[int] = None,
        previous_prices: Optional[List[int]] = None
    ) -> AcceptanceResult:
        """
        Detect if message indicates acceptance of an offer.
        
        Args:
            message: The negotiation message to analyze
            current_price: Price mentioned in current message
            previous_prices: Recent prices in negotiation history
            
        Returns:
            AcceptanceResult with detection details
        """
        try:
            # Clean message (remove reflection blocks)
            clean_message = self._remove_reflection_blocks(message)
            
            if self.debug_mode:
                logger.debug(f"Detecting acceptance in: '{clean_message}'")
            
            # Check for explicit rejection first
            if self._is_rejection(clean_message):
                return AcceptanceResult(
                    is_acceptance=False,
                    confidence=0.95,
                    pattern_matched="rejection",
                    accepted_price=None,
                    termination_type=TerminationType.FAILURE
                )
            
            # Check for acceptance patterns
            best_confidence = 0.0
            best_pattern = ""
            accepted_price = None
            
            for pattern, confidence, pattern_name in self.ACCEPTANCE_PATTERNS:
                match = re.search(pattern, clean_message, re.IGNORECASE)
                if match:
                    # Extract price if captured in pattern
                    price_from_pattern = None
                    if match.groups():
                        try:
                            price_from_pattern = int(match.group(1))
                        except (ValueError, IndexError):
                            pass
                    
                    # Determine accepted price
                    price = price_from_pattern or current_price
                    
                    # Adjust confidence based on context
                    adjusted_confidence = self._adjust_acceptance_confidence(
                        confidence, clean_message, price, previous_prices
                    )
                    
                    if adjusted_confidence > best_confidence:
                        best_confidence = adjusted_confidence
                        best_pattern = pattern_name
                        accepted_price = price
            
            # Determine if this constitutes acceptance
            is_acceptance = best_confidence >= self.min_confidence
            
            if is_acceptance and self.debug_mode:
                logger.debug(f"Acceptance detected: {best_pattern} (confidence: {best_confidence:.2f})")
            
            return AcceptanceResult(
                is_acceptance=is_acceptance,
                confidence=best_confidence,
                pattern_matched=best_pattern,
                accepted_price=accepted_price,
                termination_type=TerminationType.ACCEPTANCE if is_acceptance else TerminationType.FAILURE
            )
            
        except Exception as e:
            logger.error(f"Acceptance detection error: {e}")
            return AcceptanceResult(
                is_acceptance=False,
                confidence=0.0,
                pattern_matched="error",
                accepted_price=None,
                termination_type=TerminationType.FAILURE
            )
    
    def detect_convergence(
        self, 
        recent_prices: List[int], 
        recent_speakers: List[str]
    ) -> AcceptanceResult:
        """
        Detect convergence when both parties offer the same price.
        
        Args:
            recent_prices: List of recent prices (at least 2)
            recent_speakers: List of speakers for each price
            
        Returns:
            AcceptanceResult indicating convergence detection
        """
        try:
            if len(recent_prices) < 2 or len(recent_speakers) < 2:
                return AcceptanceResult(
                    is_acceptance=False,
                    confidence=0.0,
                    pattern_matched="insufficient_data",
                    accepted_price=None,
                    termination_type=TerminationType.FAILURE
                )
            
            # Check if last two prices are the same from different speakers
            if (recent_prices[-1] == recent_prices[-2] and 
                recent_speakers[-1] != recent_speakers[-2]):
                
                converged_price = recent_prices[-1]
                
                # High confidence for exact price convergence
                confidence = 0.95
                
                if self.debug_mode:
                    logger.debug(f"Price convergence detected at ${converged_price}")
                
                return AcceptanceResult(
                    is_acceptance=True,
                    confidence=confidence,
                    pattern_matched="price_convergence",
                    accepted_price=converged_price,
                    termination_type=TerminationType.CONVERGENCE
                )
            
            # Check for near-convergence (within $1-2)
            if len(recent_prices) >= 3:
                price_diff = abs(recent_prices[-1] - recent_prices[-2])
                if price_diff <= 2 and recent_speakers[-1] != recent_speakers[-2]:
                    # Lower confidence for near-convergence
                    confidence = 0.75
                    agreed_price = round((recent_prices[-1] + recent_prices[-2]) / 2)
                    
                    if self.debug_mode:
                        logger.debug(f"Near-convergence detected around ${agreed_price}")
                    
                    return AcceptanceResult(
                        is_acceptance=True,
                        confidence=confidence,
                        pattern_matched="near_convergence",
                        accepted_price=agreed_price,
                        termination_type=TerminationType.CONVERGENCE
                    )
            
            return AcceptanceResult(
                is_acceptance=False,
                confidence=0.0,
                pattern_matched="no_convergence",
                accepted_price=None,
                termination_type=TerminationType.FAILURE
            )
            
        except Exception as e:
            logger.error(f"Convergence detection error: {e}")
            return AcceptanceResult(
                is_acceptance=False,
                confidence=0.0,
                pattern_matched="error",
                accepted_price=None,
                termination_type=TerminationType.FAILURE
            )
    
    def _remove_reflection_blocks(self, text: str) -> str:
        """Remove reflection content before analysis."""
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<think>.*', '', text, flags=re.IGNORECASE)
        return text.strip()
    
    def _is_rejection(self, message: str) -> bool:
        """Check if message contains rejection indicators."""
        message_lower = message.lower().strip()
        
        for pattern in self.REJECTION_PATTERNS:
            if re.search(pattern, message_lower):
                return True
        return False
    
    def _adjust_acceptance_confidence(
        self, 
        base_confidence: float, 
        message: str, 
        price: Optional[int],
        previous_prices: Optional[List[int]]
    ) -> float:
        """Adjust acceptance confidence based on context."""
        confidence = base_confidence
        
        # Boost confidence if acceptance is at start of message
        if re.match(r'^\s*(I\s+)?accept', message, re.IGNORECASE):
            confidence *= 1.2
        
        # Boost confidence if price is mentioned with acceptance
        if price is not None and re.search(rf'\$?{price}', message):
            confidence *= 1.1
        
        # Reduce confidence for very short messages (might be truncated)
        if len(message.strip()) < 5:
            confidence *= 0.8
        
        # Boost confidence if price is in reasonable range
        if price and 40 <= price <= 90:
            confidence *= 1.1
        
        # Context-based adjustments with previous prices
        if previous_prices and price:
            # If accepting a price close to recent offers, boost confidence
            for prev_price in previous_prices[-3:]:  # Last 3 prices
                if abs(price - prev_price) <= 5:
                    confidence *= 1.05
                    break
        
        return min(confidence, 1.0)  # Cap at 1.0
    
    def validate_test_cases(self) -> Dict[str, bool]:
        """Run validation on acceptance detection test cases."""
        test_cases = [
            # (message, current_price, expected_acceptance, expected_price)
            ("I accept $50", 50, True, 50),
            ("I accept", 45, True, 45),
            ("Deal!", None, True, None),
            ("That sounds good", 42, True, 42),
            ("No, too high", 60, False, None),
            ("How about $55?", 55, False, None),
            ("<think>Good price</think> I accept", 48, True, 48),
            ("Agreed at $52", 52, True, 52),
            ("Fine, $47", 47, True, 47),
            ("I reject your offer", 50, False, None),
        ]
        
        results = {}
        for message, price, expected_acceptance, expected_price in test_cases:
            result = self.detect_acceptance(message, price)
            
            acceptance_correct = result.is_acceptance == expected_acceptance
            price_correct = result.accepted_price == expected_price
            
            results[message] = acceptance_correct and price_correct
            
        return results