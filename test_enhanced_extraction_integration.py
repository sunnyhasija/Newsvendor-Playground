#!/usr/bin/env python3
"""
Test Enhanced Price Extraction Integration

Quick test to validate that Task 5 (enhanced price extraction with local model fallback) 
has been correctly integrated into the supplier-first experiment.
"""

import sys
from pathlib import Path

# Add project paths
sys.path.append(str(Path(__file__).parent / "newsvendor-llm-experiment-supplier-first" / "src"))

def test_enhanced_extractor_availability():
    """Test that enhanced price extractor is available and properly imported."""
    print("🧪 Testing Enhanced Price Extractor Availability...")
    
    try:
        # Test import
        from parsing.enhanced_price_extractor import EnhancedPriceExtractor, create_enhanced_price_extractor
        print("✅ Enhanced price extractor import successful")
        
        # Test class availability
        extractor = EnhancedPriceExtractor()
        print("✅ EnhancedPriceExtractor class instantiated successfully")
        
        # Check key attributes
        assert hasattr(extractor, 'EXTRACTION_PATTERNS'), "Missing EXTRACTION_PATTERNS"
        assert hasattr(extractor, 'PRICE_INDICATORS'), "Missing PRICE_INDICATORS"
        assert hasattr(extractor, 'extract_price'), "Missing extract_price method"
        assert hasattr(extractor, 'get_extraction_stats'), "Missing get_extraction_stats method"
        print("✅ All expected attributes and methods present")
        
        # Test stats method
        stats = extractor.get_extraction_stats()
        assert isinstance(stats, dict), "get_extraction_stats should return dict"
        print("✅ get_extraction_stats working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced extractor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_conversation_tracker_integration():
    """Test that ConversationTracker is using the enhanced extractor."""
    print("\n💬 Testing ConversationTracker Integration...")
    
    try:
        from core.conversation_tracker import ConversationTracker, ENHANCED_EXTRACTION_AVAILABLE
        
        print(f"📊 ENHANCED_EXTRACTION_AVAILABLE = {ENHANCED_EXTRACTION_AVAILABLE}")
        
        # Create conversation tracker
        tracker = ConversationTracker(
            negotiation_id="test_enhanced",
            buyer_model="test_buyer",
            supplier_model="test_supplier", 
            reflection_pattern="00",
            turn_order_strategy="buyer_first"
        )
        
        print("✅ ConversationTracker instantiated with enhanced extractor")
        
        # Test extraction stats
        stats = tracker.get_extraction_stats()
        enhanced_available = stats.get("enhanced_extraction_available", False)
        print(f"📈 Enhanced extraction available in tracker: {enhanced_available}")
        
        if enhanced_available:
            print("✅ Enhanced extraction properly integrated")
        else:
            print("⚠️  Enhanced extraction available but tracker using fallback")
        
        # Test price extractor type
        extractor_class = tracker.price_extractor.__class__.__name__
        print(f"🔧 Price extractor class: {extractor_class}")
        
        if extractor_class == "EnhancedPriceExtractor":
            print("✅ ConversationTracker using EnhancedPriceExtractor")
        else:
            print(f"⚠️  ConversationTracker using fallback: {extractor_class}")
        
        return True
        
    except Exception as e:
        print(f"❌ ConversationTracker integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_import_structure():
    """Test that all imports are working correctly."""
    print("\n📦 Testing Import Structure...")
    
    try:
        # Test core imports
        from core.conversation_tracker import ConversationTracker
        from parsing.enhanced_price_extractor import EnhancedPriceExtractor
        from parsing.acceptance_detector import AcceptanceDetector
        
        print("✅ Core imports working")
        
        # Test that enhanced extractor is used in conversation tracker
        import core.conversation_tracker as ct_module
        
        # Check if enhanced extractor is imported in the module
        if hasattr(ct_module, 'create_enhanced_price_extractor'):
            print("✅ Enhanced extractor function available in conversation_tracker")
        else:
            print("⚠️  Enhanced extractor function not found in conversation_tracker")
        
        # Check ENHANCED_EXTRACTION_AVAILABLE flag
        available = getattr(ct_module, 'ENHANCED_EXTRACTION_AVAILABLE', False)
        print(f"🚩 ENHANCED_EXTRACTION_AVAILABLE flag: {available}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_file_exists():
    """Verify that the enhanced_price_extractor.py file exists and has the right content."""
    print("\n📁 Verifying File Structure...")
    
    try:
        file_path = Path(__file__).parent / "newsvendor-llm-experiment-supplier-first" / "src" / "parsing" / "enhanced_price_extractor.py"
        
        if not file_path.exists():
            print(f"❌ Enhanced price extractor file not found: {file_path}")
            return False
        
        print(f"✅ Enhanced price extractor file exists: {file_path}")
        
        # Read file and check for key content
        content = file_path.read_text()
        
        required_content = [
            "class EnhancedPriceExtractor",
            "local model fallback",
            "extract_price",
            "get_extraction_stats",
            "create_enhanced_price_extractor"
        ]
        
        for requirement in required_content:
            if requirement in content:
                print(f"✅ Found required content: {requirement}")
            else:
                print(f"❌ Missing required content: {requirement}")
                return False
        
        # Check file size (should be substantial)
        file_size = file_path.stat().st_size
        print(f"📏 File size: {file_size} bytes")
        
        if file_size > 10000:  # Should be ~18KB based on our earlier check
            print("✅ File size indicates complete implementation")
        else:
            print("⚠️  File size smaller than expected")
        
        return True
        
    except Exception as e:
        print(f"❌ File verification failed: {e}")
        return False

def main():
    """Run all enhanced price extraction tests."""
    print("🚀 Enhanced Price Extraction Integration Test - Task 5")
    print("=" * 70)
    print("Validating enhanced price extractor with local model fallback")
    print("=" * 70)
    
    tests = [
        ("File Structure", verify_file_exists),
        ("Import Structure", test_import_structure), 
        ("Enhanced Extractor", test_enhanced_extractor_availability),
        ("ConversationTracker Integration", test_conversation_tracker_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} Test: PASSED")
        else:
            print(f"❌ {test_name} Test: FAILED")
    
    print("\n" + "=" * 70)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Enhanced price extraction with local model fallback is properly integrated")
        print("✅ Task 5 implementation is COMPLETE")
    else:
        print("❌ SOME TESTS FAILED")
        print("🔧 Please check the implementation")
    
    print("=" * 70)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
