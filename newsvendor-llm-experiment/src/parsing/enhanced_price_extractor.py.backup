"""
Enhanced Price Extraction Engine with Local Model Fallback

Extends the robust price extractor with a local model fallback for edge cases
where traditional regex patterns fail but a language model could interpret the intent.
"""

import re
import logging
import asyncio
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

try:
    import ollama
except ImportError:
    print("Please install ollama: pip install ollama")
    ollama = None

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of price extraction attempt."""
    price: Optional[int]
    confidence: float
    pattern_used: str
    raw_text: str
    cleaned_text: str
    used_fallback: bool = False
    fallback_reasoning: Optional[str] = None


class EnhancedPriceExtractor:
    """Multi-strategy price extraction with local model fallback for edge cases."""
    
    # Extraction patterns in order of specificity/reliability
    EXTRACTION_PATTERNS = [
        # Explicit offers (highest confidence)
        (r'(?:offer|propose|suggest)\s*\$?(\d{1,3})\b', 0.95, "explicit_offer"),
        
        # How about patterns
        (r'how about\s*\$?(\d{1,3})\b', 0.95, "how_about"),
        
        # Acceptance with price - handles "of 55" pattern
        (r'(?:accept|agreed?|deal).*?(?:of\s+)?\$?(\d{1,3})\b', 0.90, "acceptance_with_price"),
        
        # Standalone currency prices
        (r'\$(\d{1,3})\b', 0.85, "currency_standalone"),
        
        # Let's make it phrases
        (r'(?:let\'s\s+(?:make\s+it|do)|make\s+it)\s*\$?(\d{1,3})\b', 0.80, "suggestion"),
        
        # Counter-offers
        (r'(?:counter|instead)\s*\$?(\d{1,3})\b', 0.75, "counter_offer"),
        
        # Want/need statements
        (r'(?:want|need|require)\s*\$?(\d{1,3})\b', 0.70, "want_statement"),
        
        # Price with dollar word
        (r'(\d{1,3})\s*dollars?\b', 0.65, "dollar_word"),
        
        # Numerical-only responses (lower confidence)
        (r'^\s*(\d{1,3})\s*$', 0.60, "number_only"),
        
        # General number extraction (lowest confidence)
        (r'\b(\d{1,3})\b', 0.50, "general_number"),
    ]
    
    # Indicators that suggest a price might be present (for fallback decision)
    PRICE_INDICATORS = [
        r'\$', r'dollar', r'offer', r'accept', r'want', r'need', r'propose',
        r'deal', r'price', r'cost', r'how about', r'I\'ll take', r'pay',
        r'\d+', r'thirty', r'forty', r'fifty', r'sixty', r'seventy', r'eighty', r'ninety'
    ]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, ollama_client=None, fallback_model: str = "llama3.2:latest"):
        """Initialize enhanced price extractor with fallback capability."""
        self.config = config or {}
        self.price_range = self.config.get('price_range', {'min': 1, 'max': 200})
        self.debug_mode = self.config.get('debug', False)
        self.extraction_history: List[ExtractionResult] = []
        
        # Fallback configuration
        self.ollama_client = ollama_client
        self.fallback_model = fallback_model
        self.fallback_enabled = ollama_client is not None
        self.fallback_used_count = 0
        self.fallback_success_count = 0
        
        if self.fallback_enabled:
            logger.info(f"Enhanced price extractor initialized with {fallback_model} fallback")
        else:
            logger.warning("Fallback model not available - using traditional extraction only")
    
    async def extract_price(
        self, 
        text: str, 
        previous_context: Optional[List[str]] = None,
        speaker_role: Optional[str] = None
    ) -> Optional[int]:
        """
        Extract price with context-aware validation and local model fallback.
        
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
            
            # Try traditional pattern matching first
            result = await self._traditional_extraction(cleaned_text, previous_context, speaker_role, text)
            
            # If traditional extraction failed and we have fallback capability
            if result.price is None and self.fallback_enabled and self._should_use_fallback(cleaned_text):
                if self.debug_mode:
                    logger.debug(f"Traditional extraction failed, trying fallback for: '{cleaned_text}'")
                
                fallback_result = await self._fallback_extraction(cleaned_text, text)
                if fallback_result.price is not None:
                    # Validate fallback result
                    if self._validate_price(fallback_result.price, previous_context, speaker_role):
                        result = fallback_result
                        self.fallback_success_count += 1
                        if self.debug_mode:
                            logger.debug(f"Fallback extraction succeeded: ${fallback_result.price}")
            
            # Store extraction history for analysis
            self.extraction_history.append(result)
            
            if result.price and self.debug_mode:
                method = "fallback" if result.used_fallback else "traditional"
                logger.debug(f"Extracted price ${result.price} using {method} method (confidence: {result.confidence:.2f})")
            elif self.debug_mode:
                logger.debug(f"No valid price found in: '{cleaned_text}'")
            
            return result.price
                
        except Exception as e:
            logger.error(f"Price extraction error: {e}")
            return None
    
    async def _traditional_extraction(
        self, 
        cleaned_text: str, 
        previous_context: Optional[List[str]], 
        speaker_role: Optional[str],
        original_text: str
    ) -> ExtractionResult:
        """Perform traditional regex-based extraction."""
        
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
                            raw_text=original_text,
                            cleaned_text=cleaned_text,
                            used_fallback=False
                        )
                        
                        # Keep track of best result
                        if adjusted_confidence > best_confidence:
                            best_result = result
                            best_confidence = adjusted_confidence
                            
                except ValueError:
                    continue
        
        # Return best result or failed result
        if best_result:
            return best_result
        else:
            return ExtractionResult(
                price=None,
                confidence=0.0,
                pattern_used="none",
                raw_text=original_text,
                cleaned_text=cleaned_text,
                used_fallback=False
            )
    
    def _should_use_fallback(self, text: str) -> bool:
        """Determine if we should try fallback extraction."""
        if not self.fallback_enabled:
            return False
        
        # Check if text contains price indicators
        text_lower = text.lower()
        has_indicators = any(re.search(pattern, text_lower) for pattern in self.PRICE_INDICATORS)
        
        # Also check text length (avoid very short or very long texts)
        reasonable_length = 5 <= len(text.strip()) <= 500
        
        return has_indicators and reasonable_length
    
    async def _fallback_extraction(self, cleaned_text: str, original_text: str) -> ExtractionResult:
        """Use local model to extract price from ambiguous text."""
        
        if not self.fallback_enabled:
            return ExtractionResult(
                price=None, confidence=0.0, pattern_used="fallback_unavailable",
                raw_text=original_text, cleaned_text=cleaned_text, used_fallback=True
            )
        
        self.fallback_used_count += 1
        
        try:
            # Create focused prompt for price extraction
            prompt = f"""You are helping extract negotiation prices from text. 

Text: "{cleaned_text}"

Task: Find any price offer, acceptance, or negotiation amount in this text.

Look for:
- Direct offers like "I offer $45" 
- Acceptances like "I accept 50"
- Numbers with context like "how about forty-five"
- Word numbers like "thirty-eight dollars"

If you find a clear price, respond with just the number (no $ symbol).
If no clear price exists, respond with "NONE".

Examples:
"I offer $45" → 45
"How about thirty-eight" → 38
"Let's negotiate further" → NONE
"I accept" → NONE

Price:"""
            
            # Call local model
            response = await asyncio.to_thread(
                self.ollama_client.generate,
                model=self.fallback_model,
                prompt=prompt,
                options={
                    'num_predict': 30,
                    'temperature': 0.1,
                    'top_p': 0.9,
                    'stop': ['\n', 'Explanation:', 'Note:']
                }
            )
            
            response_text = response.get('response', '').strip()
            
            # Parse the response
            price = self._parse_fallback_response(response_text)
            
            confidence = 0.75 if price else 0.0  # Medium confidence for fallback
            
            return ExtractionResult(
                price=price,
                confidence=confidence,
                pattern_used="local_model_fallback",
                raw_text=original_text,
                cleaned_text=cleaned_text,
                used_fallback=True,
                fallback_reasoning=response_text
            )
            
        except Exception as e:
            logger.error(f"Fallback extraction error: {e}")
            return ExtractionResult(
                price=None,
                confidence=0.0,
                pattern_used="fallback_error",
                raw_text=original_text,
                cleaned_text=cleaned_text,
                used_fallback=True,
                fallback_reasoning=f"Error: {str(e)}"
            )
    
    def _parse_fallback_response(self, response_text: str) -> Optional[int]:
        """Parse the response from the fallback model."""
        response_text = response_text.strip().upper()
        
        if response_text == "NONE" or not response_text:
            return None
        
        # Try to extract number from response
        numbers = re.findall(r'\b(\d{1,3})\b', response_text)
        if numbers:
            try:
                price = int(numbers[0])
                # Basic validation
                if 1 <= price <= 200:
                    return price
            except ValueError:
                pass
        
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
        
        # Role-specific validation (relaxed for research)
        if speaker_role == 'buyer':
            # Buyers shouldn't offer above retail price
            if price >= 100:  # retail price is $100
                if self.debug_mode:
                    logger.debug(f"Buyer price {price} too high (>= retail price)")
                return False
        elif speaker_role == 'supplier':
            # Suppliers shouldn't offer below cost (but allow some edge cases for research)
            if price <= 25:  # Well below production cost of $30
                if self.debug_mode:
                    logger.debug(f"Supplier price {price} extremely low (<= $25)")
                return False
        
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
        """Get statistics about extraction performance including fallback usage."""
        if not self.extraction_history:
            return {"total_attempts": 0}
        
        total = len(self.extraction_history)
        successful = sum(1 for r in self.extraction_history if r.price is not None)
        fallback_attempts = sum(1 for r in self.extraction_history if r.used_fallback)
        fallback_successes = sum(1 for r in self.extraction_history if r.used_fallback and r.price is not None)
        
        patterns_used = {}
        confidences = []
        
        for result in self.extraction_history:
            if result.price is not None:
                patterns_used[result.pattern_used] = patterns_used.get(result.pattern_used, 0) + 1
                confidences.append(result.confidence)
        
        stats = {
            "total_attempts": total,
            "successful_extractions": successful,
            "success_rate": successful / total if total > 0 else 0,
            "patterns_used": patterns_used,
            "average_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "confidence_range": (min(confidences), max(confidences)) if confidences else (0, 0),
            "fallback_enabled": self.fallback_enabled,
            "fallback_attempts": fallback_attempts,
            "fallback_successes": fallback_successes,
            "fallback_success_rate": fallback_successes / fallback_attempts if fallback_attempts > 0 else 0,
            "traditional_vs_fallback": {
                "traditional_successes": successful - fallback_successes,
                "fallback_successes": fallback_successes,
                "fallback_usage_rate": fallback_attempts / total if total > 0 else 0
            }
        }
        
        return stats


# Factory function for easy integration
def create_enhanced_price_extractor(
    config: Optional[Dict[str, Any]] = None,
    ollama_client=None,
    fallback_model: str = "llama3.2:latest"
) -> EnhancedPriceExtractor:
    """Create an enhanced price extractor with optional fallback capability."""
    
    if ollama_client is None:
        try:
            import ollama
            ollama_client = ollama.Client()
            logger.info("Created Ollama client for fallback extraction")
        except ImportError:
            logger.warning("Ollama not available - fallback extraction disabled")
            ollama_client = None
    
    return EnhancedPriceExtractor(
        config=config,
        ollama_client=ollama_client,
        fallback_model=fallback_model
    )