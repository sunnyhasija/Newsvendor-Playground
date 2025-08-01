#!/usr/bin/env python3
"""
Quick validation test for Task 1: Smart Throttling System
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

# Test imports and basic functionality
def test_throttling_implementation():
    print("🔧 Testing Task 1: Smart Throttling System Implementation")
    print("=" * 60)
    
    try:
        # Test import
        from run_turn_order_experiment import TurnOrderExperimentRunner
        print("✅ Successfully imported TurnOrderExperimentRunner")
        
        # Test initialization
        runner = TurnOrderExperimentRunner(max_concurrent=1)
        print("✅ Successfully initialized runner")
        
        # Test throttling configuration
        assert hasattr(runner, 'throttle_delays'), "Missing throttle_delays"
        assert hasattr(runner, 'throttle_counts'), "Missing throttle_counts"
        assert hasattr(runner, 'throttle_multipliers'), "Missing throttle_multipliers"
        assert hasattr(runner, 'last_call_times'), "Missing last_call_times"
        print("✅ Throttling configuration properly initialized")
        
        # Test throttling methods
        assert hasattr(runner, '_apply_smart_throttling'), "Missing _apply_smart_throttling method"
        assert hasattr(runner, '_handle_throttling_event'), "Missing _handle_throttling_event method"
        assert hasattr(runner, '_generate_with_throttling_retry'), "Missing _generate_with_throttling_retry method"
        assert hasattr(runner, '_analyze_throttling_impact'), "Missing _analyze_throttling_impact method"
        print("✅ All throttling methods are present")
        
        # Test throttling configuration values
        expected_models = ['claude-sonnet-4-remote', 'o3-remote', 'grok-remote', 'default']
        for model in expected_models:
            assert model in runner.throttle_delays, f"Missing throttle delay for {model}"
        print("✅ Throttling delays configured for all expected models")
        
        # Test throttling event handling
        runner._handle_throttling_event('test-model')
        assert 'test-model' in runner.throttle_counts, "Throttling event not recorded"
        assert runner.throttle_counts['test-model'] == 1, "Throttling count incorrect"
        print("✅ Throttling event handling works correctly")
        
        # Test throttling analysis
        from core.conversation_tracker import NegotiationResult
        from parsing.acceptance_detector import TerminationType
        
        # Create mock results for analysis
        mock_results = [
            NegotiationResult(
                negotiation_id="test1",
                buyer_model="test-model",
                supplier_model="test-model2",
                reflection_pattern="00",
                turn_order_strategy="buyer_first",
                first_speaker="buyer",
                completed=True,
                agreed_price=60.0,
                termination_type=TerminationType.PRICE_AGREEMENT,
                total_rounds=3,
                total_tokens=100,
                total_time=10.0,
                buyer_profit=None,
                supplier_profit=None,
                distance_from_optimal=None,
                turns=[],
                metadata={}
            )
        ]
        
        analysis = runner._analyze_throttling_impact(mock_results)
        assert 'summary' in analysis, "Missing throttling analysis summary"
        assert 'by_model' in analysis, "Missing by_model throttling analysis"
        assert 'impact_on_performance' in analysis, "Missing performance impact analysis"
        print("✅ Throttling analysis works correctly")
        
        print("\n🎉 Task 1: Smart Throttling System - ALL TESTS PASSED!")
        print("🚀 Ready for validation with actual negotiations")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_throttling_implementation()
    sys.exit(0 if success else 1)
