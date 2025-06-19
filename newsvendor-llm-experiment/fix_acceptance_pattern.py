#!/usr/bin/env python3
"""Fix the acceptance detection pattern"""

# Read the current file
with open('src/parsing/enhanced_price_extractor.py', 'r') as f:
    content = f.read()

print("🔧 Improving acceptance detection pattern...")

# Replace the simple pattern with a more comprehensive one
old_pattern = r'\\b(accept|agree|deal|yes|ok|fine)\\b'
new_pattern = r'\\b(accept|agree[ds]?|deal|yes|ok|fine)\\b'

content = content.replace(old_pattern, new_pattern)

# Write the updated file
with open('src/parsing/enhanced_price_extractor.py', 'w') as f:
    f.write(content)

print("✅ Acceptance pattern improved!")
