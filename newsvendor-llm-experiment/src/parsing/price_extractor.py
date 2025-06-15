"""
Robust Price Extraction Engine for Newsvendor Experiment

Handles multiple extraction strategies with fallback logic to parse
prices from diverse LLM outputs while filtering reflection content.
"""

import re
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of price extraction attempt."""
    price: Optional[int]
    confidence: float
    pattern_used: str
    raw_text: str
    cleaned_text: str


class RobustPriceExtractor:
    """Multi-strategy price extraction with fallback logic."""
    
    # Extraction patterns in order of specificity/reliability
    EXTRACTION_PATTERNS = [
        # Explicit offers (highest confidence)
        (r'(?:offer|propose|suggest|how about)\s*\$?(\d{1,3})\b', 0.95, "explicit_offer"),
        
        # Acceptance with price
        (r'(?:accept|agreed?|deal)\s*\$?(\d{1,3})\b', 0.90, "acceptance_with_price"),
        
        # Standalone currency prices
        (r'\$(\d{1,3})\b', 0.85, "currency_standalone"),
        
        # Let's do/make it phrases
        (r'(?:let\'s do|make it|how about)\s*\$?(\d{1,3})\b', 0.80, "suggestion"),
        
        # Counter-offers
        (r'(?:counter|instead)\s*\$?(\d{1,3})\b', 0.75, "counter_offer"),
        
        # Want/need statements
        (r'(?:want|need|require)\s*\$?(\d{1,3})\b', 0.70, "want_statement"),
        
        # Numerical-only responses (lower confidence)
        (r'^\s*(\d{1,3})\s*$', 0.60, "number_only"),
        
        # Price with dollar word
        (r'(\d{1,3})\s*dollars?', 0.65, "dollar_word"),
    ]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize price extractor with optional configuration."""
        self.config = config or {}
        self.price_range = self.config.get('price_range', {'min': 1, 'max': 200})
        self.debug_mode = self.config.get('debug', False)
        self.extraction_history: List[ExtractionResult] = []
    
    def extract_price(
        self, 
        text: str, 
        previous_context: Optional[List[str]] = None,
        speaker_role: Optional[str] = None
    ) -> Optional[int]:
        """
        Extract price with context-aware validation.
        
        Args:
            text: Raw response text from LLM
            previous_context: Previous messages for context validation
            speaker_role: 'buyer' or 'supplier' for role-specific validation
            
        Returns:
            Extracted price as integer, or None if no valid price found
        """
        try:
            # Remove reflection blocks first
            cleaned_text = self._remove_reflection_blocks(text)
            
            if self.debug_mode:
                logger.debug(f"Extracting price from: '{text}' -> '{cleaned_text}'")
            
            best_result = None
            best_confidence = 0.0
            
            # Try each pattern in order of confidence
            for pattern, base_confidence, pattern_name in self.EXTRACTION_PATTERNS:
                matches = re.findall(pattern, cleaned_text, re.IGNORECASE)
                
                if matches:
                    # Take the last match (most likely the actual offer)
                    price_str = matches[-1]
                    try:
                        price = int(price_str)
                        
                        # Validate price
                        if self._validate_price(price, previous_context, speaker_role):
                            # Adjust confidence based on context
                            adjusted_confidence = self._adjust_confidence(
                                base_confidence, price, cleaned_text, speaker_role
                            )
                            
                            result = ExtractionResult(
                                price=price,
                                confidence=adjusted_confidence,
                                pattern_used=pattern_name,
                                raw_text=text,
                                cleaned_text=cleaned_text
                            )
                            
                            # Keep track of best result
                            if adjusted_confidence > best_confidence:
                                best_result = result
                                best_confidence = adjusted_confidence
                                
                    except ValueError:
                        continue
            
            # Store extraction history for analysis
            if best_result:
                self.extraction_history.append(best_result)
                if self.debug_mode:
                    logger.debug(f"Extracted price {best_result.price} with confidence {best_result.confidence:.2f}")
                return best_result.price
            else:
                # Record failed extraction
                failed_result = ExtractionResult(
                    price=None,
                    confidence=0.0,
                    pattern_used="none",
                    raw_text=text,
                    cleaned_text=cleaned_text
                )
                self.extraction_history.append(failed_result)
                if self.debug_mode:
                    logger.debug(f"No valid price found in: '{cleaned_text}'")
                return None
                
        except Exception as e:
            logger.error(f"Price extraction error: {e}")
            return None
    
    def _remove_reflection_blocks(self, text: str) -> str:
        """Remove all reflection content before parsing."""
        # Remove complete <think> blocks
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove incomplete <think> blocks (everything after <think>)
        text = re.sub(r'<think>.*', '', text, flags=re.IGNORECASE)
        
        # Also remove any stray closing tags
        text = re.sub(r'</think>', '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _validate_price(
        self, 
        price: int, 
        context: Optional[List[str]] = None, 
        speaker_role: Optional[str] = None
    ) -> bool:
        """Validate price reasonableness and role-specific constraints."""
        
        # Basic range check
        if not (self.price_range['min'] <= price <= self.price_range['max']):
            if self.debug_mode:
                logger.debug(f"Price {price} outside valid range {self.price_range}")
            return False
        
        # Role-specific validation
        if speaker_role == 'buyer':
            # Buyers shouldn't offer above retail price
            if price >= 100:  # retail price is $100
                if self.debug_mode:
                    logger.debug(f"Buyer price {price} too high (>= retail price)")
                return False
        elif speaker_role == 'supplier':
            # Suppliers shouldn't offer below cost
            if price <= 30:  # production cost is $30
                if self.debug_mode:
                    logger.debug(f"Supplier price {price} too low (<= production cost)")
                return False
        
        # Additional context-based validation could go here
        # For example, checking if price is reasonable given recent offers
        
        return True
    
    def _adjust_confidence(
        self, 
        base_confidence: float, 
        price: int, 
        text: str, 
        speaker_role: Optional[str]
    ) -> float:
        """Adjust confidence based on contextual factors."""
        confidence = base_confidence
        
        # Boost confidence for prices in reasonable negotiation range
        if 35 <= price <= 85:  # Good negotiation range
            confidence *= 1.1
        
        # Boost confidence if price appears in structured format
        if re.search(r'(I offer|How about|I accept)\s*\$?' + str(price), text, re.IGNORECASE):
            confidence *= 1.2
        
        # Reduce confidence for edge case prices
        if price <= 10 or price >= 150:
            confidence *= 0.8
        
        # Role-specific confidence adjustments
        if speaker_role == 'buyer' and price > 80:
            confidence *= 0.9  # Buyers offering high prices less likely
        elif speaker_role == 'supplier' and price < 40:
            confidence *= 0.9  # Suppliers offering low prices less likely
        
        return min(confidence, 1.0)  # Cap at 1.0
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get statistics about extraction performance."""
        if not self.extraction_history:
            return {"total_attempts": 0}
        
        total = len(self.extraction_history)
        successful = sum(1 for r in self.extraction_history if r.price is not None)
        
        patterns_used = {}
        confidences = []
        
        for result in self.extraction_history:
            if result.price is not None:
                patterns_used[result.pattern_used] = patterns_used.get(result.pattern_used, 0) + 1
                confidences.append(result.confidence)
        
        return {
            "total_attempts": total,
            "successful_extractions": successful,
            "success_rate": successful / total if total > 0 else 0,
            "patterns_used": patterns_used,
            "average_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "confidence_range": (min(confidences), max(confidences)) if confidences else (0, 0)
        }
    
    def validate_test_cases(self) -> Dict[str, bool]:
        """Run validation on predefined test cases."""
        test_cases = [
            ("I offer $45 for this deal", 45),
            ("How about 67?", 67),
            ("Let's make it $52", 52),
            ("<think>Maybe 40?</think> I propose 50", 50),
            ("I accept your offer of 55", 55),
            ("$42", 42),
            ("No, I want 38 dollars", 38),
            ("I accept", None),  # Should return None (no price)
            ("Considering market conditions...", None),  # Should return None
        ]
        
        results = {}
        for test_input, expected in test_cases:
            extracted = self.extract_price(test_input)
            results[test_input] = (extracted == expected)
            
            # Debug failed test cases
            if extracted != expected:
                logger.debug(f"FAILED: '{test_input}' -> expected {expected}, got {extracted}")
        
        return results