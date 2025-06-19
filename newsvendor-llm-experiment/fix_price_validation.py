#!/usr/bin/env python3
"""Fix the price validation bug"""

import re

# Read the current file
with open('src/parsing/enhanced_price_extractor.py', 'r') as f:
    content = f.read()

print("🔧 Applying price validation fix...")

# 1. Update _validate_price method signature and logic
old_validate_method = '''    def _validate_price(
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
        
        return True'''

new_validate_method = '''    def _validate_price(
        self, 
        price: int, 
        context: Optional[List[str]] = None, 
        speaker_role: Optional[str] = None,
        message_text: str = ""
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
            # Check if this is an acceptance vs an offer
            is_acceptance = bool(re.search(r'\\b(accept|agree|deal|yes|ok|fine)\\b', message_text, re.IGNORECASE))
            
            if price <= 25:  # Well below production cost of $30
                if is_acceptance:
                    # Allow suppliers to accept low prices (they're echoing buyer's offer)
                    if self.debug_mode:
                        logger.debug(f"Supplier accepting low price ${price} - allowed as acceptance")
                    return True
                else:
                    # Still reject unrealistic supplier offers
                    if self.debug_mode:
                        logger.debug(f"Supplier offering extremely low price ${price} - rejected")
                    return False
        
        return True'''

# Replace the method
content = content.replace(old_validate_method, new_validate_method)

# 2. Update the call in _traditional_extraction
content = content.replace(
    'if self._validate_price(price, previous_context, speaker_role):',
    'if self._validate_price(price, previous_context, speaker_role, original_text):'
)

# 3. Update the call in extract_price fallback
content = content.replace(
    'if self._validate_price(fallback_result.price, previous_context, speaker_role):',
    'if self._validate_price(fallback_result.price, previous_context, speaker_role, text):'
)

# Write the fixed file
with open('src/parsing/enhanced_price_extractor.py', 'w') as f:
    f.write(content)

print("✅ Price validation fix applied!")
