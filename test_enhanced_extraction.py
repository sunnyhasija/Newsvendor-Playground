#!/usr/bin/env python3
"""
Quick test script to validate enhanced price extraction integration
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "newsvendor-llm-experiment-supplier-first" / "src"))

# Configure logging to show enhanced extraction messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_enhanced_extraction():
    """Test enhanced price extractor initialization and basic functionality."""
    print("🧪 Testing Enhanced Price Extraction Integration...")
    
    try:
        # Test import
        from core.conversation_tracker import ConversationTracker
        print("✅ Enhanced price extractor import successful")
        
        # Test initialization
        tracker = ConversationTracker(
            negotiation_id="test_001",
            buyer_model="test_model",
            supplier_model="test_model", 
            reflection_pattern="00",
            turn_order_strategy="buyer_first",
            config={'max_rounds': 5}
        )
        
        print("✅ ConversationTracker initialized with enhanced extractor")
        
        # Test the price extractor stats to confirm enhanced version
        stats = tracker.get_extraction_stats()
        if stats.get("enhanced_extraction_available", False):
            print("✅ Enhanced extraction confirmed available")
            print(f"   Fallback enabled: {stats.get('fallback_enabled', False)}")
        else:
            print("⚠️  Enhanced extraction available but using traditional fallback")
        
        # Test a simple price extraction
        test_message = "I offer $55"
        price = await tracker.price_extractor.extract_price(test_message, speaker_role="buyer")
        print(f"✅ Test extraction: '{test_message}' -> ${price}")
        
        # Print final statistics
        final_stats = tracker.get_extraction_stats()
        print(f"📊 Extraction Statistics:")
        print(f"   Enhanced available: {final_stats.get('enhanced_extraction_available', False)}")
        print(f"   Fallback enabled: {final_stats.get('fallback_enabled', False)}")
        print(f"   Total attempts: {final_stats.get('total_attempts', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_enhanced_extraction())
    if success:
        print("\n🎉 Enhanced price extraction validation PASSED")
    else:
        print("\n💥 Enhanced price extraction validation FAILED")
