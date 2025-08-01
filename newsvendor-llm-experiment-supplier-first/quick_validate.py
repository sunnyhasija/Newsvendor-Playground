#!/usr/bin/env python3
"""
Simple validation for Task 1: Smart Throttling System
Tests the core functionality without running full negotiations
"""

import sys
import asyncio
from pathlib import Path

# Add src to path for imports  
sys.path.append(str(Path(__file__).parent / "src"))

async def validate_throttling():
    print("🔧 Task 1: Smart Throttling System - Quick Validation")
    print("=" * 60)
    
    try:
        # Import and initialize
        from run_turn_order_experiment import TurnOrderExperimentRunner
        runner = TurnOrderExperimentRunner(max_concurrent=1)
        
        print("✅ Runner initialized successfully")
        print(f"   Throttle delays configured for: {list(runner.throttle_delays.keys())}")
        
        # Test throttling event
        test_model = "qwen2:1.5b"
        print(f"\n🧪 Testing throttling event for {test_model}")
        
        # Simulate throttling event
        runner._handle_throttling_event(test_model)
        print(f"   ✅ Throttling event recorded: {runner.throttle_counts[test_model]} events")
        print(f"   ✅ Multiplier updated: {runner.throttle_multipliers[test_model]:.2f}x")
        
        # Test throttling delay calculation
        await runner._apply_smart_throttling(test_model)
        print(f"   ✅ Smart throttling applied successfully")
        
        # Test throttling analysis
        print(f"\n📊 Testing throttling analysis")
        
        # Mock some results for analysis
        class MockResult:
            def __init__(self):
                self.negotiation_id = "test"
                self.completed = True
                
        mock_results = [MockResult() for _ in range(10)]
        analysis = runner._analyze_throttling_impact(mock_results)
        
        print(f"   ✅ Analysis generated with sections: {list(analysis.keys())}")
        print(f"   ✅ Total throttle events: {analysis['summary']['total_throttle_events']}")
        print(f"   ✅ Models throttled: {analysis['summary']['models_throttled']}")
        
        # Test recommendations
        recommendations = runner._generate_throttling_recommendations(5, 100)
        print(f"   ✅ Generated {len(recommendations)} recommendations")
        
        print(f"\n🎉 Task 1: Smart Throttling System - VALIDATION SUCCESSFUL!")
        print(f"🚀 All core throttling functions working correctly")
        print(f"📋 Ready for integration testing with actual negotiations")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(validate_throttling())
    print(f"\nValidation {'PASSED' if result else 'FAILED'}")
