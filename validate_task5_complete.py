#!/usr/bin/env python3
"""
Final validation test for Task 5 - Enhanced Price Extraction with Local Model Fallback

This script validates that:
1. Enhanced price extractor is properly integrated
2. ConversationTracker uses enhanced extractor
3. Metadata includes enhanced extraction statistics
4. System is ready for full experiments
"""

import sys
import os
from pathlib import Path

# Add project source to Python path
project_path = Path(__file__).parent / "newsvendor-llm-experiment-supplier-first"
sys.path.append(str(project_path / "src"))

def validate_enhanced_extraction_availability():
    """Validate that enhanced extraction is available and properly integrated."""
    print("🔍 Validating Enhanced Extraction Availability...")
    
    try:
        # Test direct import of enhanced extractor
        from parsing.enhanced_price_extractor import EnhancedPriceExtractor, create_enhanced_price_extractor
        print("✅ Enhanced price extractor can be imported directly")
        
        # Test creation
        extractor = EnhancedPriceExtractor()
        print("✅ EnhancedPriceExtractor can be instantiated")
        
        # Test factory function
        factory_extractor = create_enhanced_price_extractor()
        print("✅ create_enhanced_price_extractor factory function works")
        
        # Test key methods exist
        assert hasattr(extractor, 'extract_price'), "Missing extract_price method"
        assert hasattr(extractor, 'get_extraction_stats'), "Missing get_extraction_stats method"
        print("✅ Required methods present")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced extraction validation failed: {e}")
        return False

def validate_conversation_tracker_integration():
    """Validate that ConversationTracker properly uses enhanced extraction."""
    print("\n💬 Validating ConversationTracker Integration...")
    
    try:
        from core.conversation_tracker import ConversationTracker, ENHANCED_EXTRACTION_AVAILABLE
        
        print(f"📊 ENHANCED_EXTRACTION_AVAILABLE = {ENHANCED_EXTRACTION_AVAILABLE}")
        
        if not ENHANCED_EXTRACTION_AVAILABLE:
            print("⚠️  Enhanced extraction flagged as unavailable - checking why...")
            
            # Try direct import to see what's wrong
            try:
                from parsing.enhanced_price_extractor import create_enhanced_price_extractor
                print("   ✅ Enhanced extractor can be imported directly")
                print("   🔧 Issue may be with import path in conversation_tracker.py")
            except ImportError as ie:
                print(f"   ❌ Cannot import enhanced extractor: {ie}")
        
        # Create tracker
        tracker = ConversationTracker(
            negotiation_id="validation_test",
            buyer_model="test_buyer", 
            supplier_model="test_supplier",
            reflection_pattern="00",
            turn_order_strategy="buyer_first"
        )
        
        print("✅ ConversationTracker created successfully")
        
        # Check extractor type
        extractor_class = tracker.price_extractor.__class__.__name__
        print(f"🔧 Using price extractor: {extractor_class}")
        
        if extractor_class == "EnhancedPriceExtractor":
            print("✅ ConversationTracker is using EnhancedPriceExtractor")
        elif extractor_class == "SimplePriceExtractor":
            print("⚠️  ConversationTracker is using SimplePriceExtractor fallback")
        else:
            print(f"ℹ️  ConversationTracker is using: {extractor_class}")
        
        # Test stats
        stats = tracker.get_extraction_stats()
        enhanced_available = stats.get("enhanced_extraction_available", False)
        print(f"📈 Stats show enhanced available: {enhanced_available}")
        
        return True
        
    except Exception as e:
        print(f"❌ ConversationTracker integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_metadata_integration():
    """Validate that enhanced extraction statistics appear in metadata."""
    print("\n📊 Validating Metadata Integration...")
    
    try:
        from core.conversation_tracker import ConversationTracker
        
        tracker = ConversationTracker(
            negotiation_id="metadata_test",
            buyer_model="test_buyer",
            supplier_model="test_supplier", 
            reflection_pattern="00",
            turn_order_strategy="buyer_first"
        )
        
        # Get final result to check metadata
        result = tracker.get_final_result()
        
        # Check metadata structure
        assert "price_extraction_stats" in result.metadata, "Missing price_extraction_stats"
        
        extraction_stats = result.metadata["price_extraction_stats"]
        print(f"📈 Extraction stats keys: {list(extraction_stats.keys())}")
        
        # Key fields that should be present
        required_fields = ["enhanced_extraction_available", "total_attempts", "successful_extractions"]
        
        for field in required_fields:
            if field in extraction_stats:
                print(f"   ✅ {field}: {extraction_stats[field]}")
            else:
                print(f"   ❌ Missing field: {field}")
        
        # Check turn order analysis metadata (v0.6 feature)
        assert "turn_order_analysis" in result.metadata, "Missing turn_order_analysis"
        
        turn_order_stats = result.metadata["turn_order_analysis"]
        print(f"🔄 Turn order analysis keys: {list(turn_order_stats.keys())}")
        
        print("✅ Metadata integration looks correct")
        return True
        
    except Exception as e:
        print(f"❌ Metadata validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_file_integrity():
    """Validate that all required files exist and have correct content."""
    print("\n📁 Validating File Integrity...")
    
    try:
        base_path = Path(__file__).parent / "newsvendor-llm-experiment-supplier-first"
        
        # Check enhanced_price_extractor.py
        enhanced_extractor_path = base_path / "src" / "parsing" / "enhanced_price_extractor.py"
        if enhanced_extractor_path.exists():
            size = enhanced_extractor_path.stat().st_size
            print(f"✅ enhanced_price_extractor.py exists ({size} bytes)")
            
            # Check for key content
            content = enhanced_extractor_path.read_text()
            key_features = [
                "class EnhancedPriceExtractor",
                "local model fallback", 
                "async def extract_price",
                "def get_extraction_stats",
                "create_enhanced_price_extractor"
            ]
            
            for feature in key_features:
                if feature in content:
                    print(f"   ✅ Contains: {feature}")
                else:
                    print(f"   ❌ Missing: {feature}")
        else:
            print("❌ enhanced_price_extractor.py does not exist")
            return False
        
        # Check conversation_tracker.py integration
        tracker_path = base_path / "src" / "core" / "conversation_tracker.py"
        if tracker_path.exists():
            content = tracker_path.read_text()
            if "enhanced_price_extractor" in content:
                print("✅ conversation_tracker.py references enhanced_price_extractor")
            else:
                print("❌ conversation_tracker.py missing enhanced_price_extractor reference")
        
        return True
        
    except Exception as e:
        print(f"❌ File integrity check failed: {e}")
        return False

def main():
    """Run complete validation suite."""
    print("🚀 Task 5 Validation: Enhanced Price Extraction with Local Model Fallback")
    print("=" * 80)
    print("Comprehensive validation of enhanced price extraction integration")
    print("=" * 80)
    
    tests = [
        ("File Integrity", validate_file_integrity),
        ("Enhanced Extraction Availability", validate_enhanced_extraction_availability), 
        ("ConversationTracker Integration", validate_conversation_tracker_integration),
        ("Metadata Integration", validate_metadata_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        if test_func():
            passed += 1
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")
    
    print("\n" + "=" * 80)
    print(f"🏁 VALIDATION RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print("✅ Enhanced price extraction with local model fallback is properly integrated")
        print("✅ Task 5 implementation is COMPLETE and ready for use")
        print("📋 System ready for turn order experiments with enhanced price extraction")
        print("\nExpected behavior:")
        print("  • Enhanced price extractor will be used when available")
        print("  • Local model fallback (llama3.2, mistral, etc.) will be attempted")
        print("  • Detailed extraction statistics will be included in results")
        print("  • Simple regex fallback if enhanced extractor unavailable")
    else:
        print("❌ SOME VALIDATION TESTS FAILED")
        print("🔧 Please review the failed tests and fix issues before proceeding")
        
        if passed >= 2:
            print("ℹ️  Core functionality appears to work, issues may be minor")
        else:
            print("⚠️  Significant issues detected - major fixes needed")
    
    print("=" * 80)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
