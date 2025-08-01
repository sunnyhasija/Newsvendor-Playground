#!/usr/bin/env python3
"""
Validation script for Task 6 - Enhanced Error Handling
Tests robust failure recovery, richer metadata, and clearer logging.
"""

import asyncio
import logging
import json
from pathlib import Path
from run_turn_order_experiment import TurnOrderExperimentRunner

# Set up logging to see all error messages
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S"
)

async def test_enhanced_error_handling():
    """Test enhanced error handling with invalid model to force failures."""
    
    print("🧪 Testing Enhanced Error Handling (Task 6)")
    print("=" * 60)
    
    runner = TurnOrderExperimentRunner(max_concurrent=1)
    
    try:
        # This should work (initialization doesn't validate all models)
        await runner.initialize()
        print("✅ Runner initialized successfully")
        
        # Generate experiment plan with invalid model to force failures
        print("\n📋 Generating experiment plan with invalid model...")
        experiment_plan = runner.generate_experiment_plan(
            models_subset=["not_a_model", "definitely_not_a_model"],  # Invalid models
            strategies_subset=["buyer_first"],
            patterns_subset=["00"],
            replications=2
        )
        print(f"✅ Generated {len(experiment_plan)} configurations")
        
        # Run experiment - this should handle failures gracefully
        print("\n🚀 Running experiment with invalid models (expecting failures)...")
        analysis = await runner.run_turn_order_experiment(experiment_plan, save_results=False)
        
        # Analyze results
        print("\n📊 ANALYSIS RESULTS:")
        print("=" * 40)
        
        summary = analysis.get("experiment_summary", {})  
        failure_summary = analysis.get("failure_summary", {})
        
        print(f"Total negotiations: {summary.get('total_negotiations', 0)}")
        print(f"Successful negotiations: {summary.get('successful_negotiations', 0)}")
        print(f"Success rate: {summary.get('success_rate', 0)*100:.1f}%")
        print(f"Total failures: {failure_summary.get('total_failures', 0)}")
        
        # Check failure breakdown
        by_error_type = failure_summary.get('by_error_type', {})
        print(f"\n🔍 Failure breakdown by error type:")
        for error_type, count in by_error_type.items():
            print(f"  {error_type}: {count}")
        
        # Validation checks
        print(f"\n✅ VALIDATION RESULTS:")
        print("=" * 40)
        
        total_negotiations = summary.get('total_negotiations', 0)
        total_failures = failure_summary.get('total_failures', 0)
        
        # Check 1: Script should not crash
        print("✅ Script completed without crashing")
        
        # Check 2: Should have structured failure results  
        if total_failures > 0:
            print("✅ Structured failure results generated")
        else:
            print("❌ Expected failures but got none")
        
        # Check 3: Should have failure metadata
        if by_error_type:
            print("✅ Error types tracked in failure metadata")
        else:
            print("❌ No error type breakdown found")
        
        # Check 4: Should have failure summary in analysis
        if 'failure_summary' in analysis:
            print("✅ Failure summary included in analysis")
        else:
            print("❌ Failure summary missing from analysis")
        
        # Check 5: All negotiations should be accounted for
        accounted_negotiations = summary.get('successful_negotiations', 0) + total_failures
        if accounted_negotiations == total_negotiations:
            print("✅ All negotiations properly accounted for")
        else:
            print(f"❌ Negotiation accounting mismatch: {accounted_negotiations} vs {total_negotiations}")
        
        print(f"\n🎯 TASK 6 IMPLEMENTATION STATUS:")
        if total_failures > 0 and by_error_type and 'failure_summary' in analysis:
            print("✅ Enhanced error handling successfully implemented!")
            print("   - Failures converted to structured results")
            print("   - Rich metadata captured")
            print("   - Error types tracked and analyzed")
        else:
            print("❌ Enhanced error handling implementation needs review")
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await runner.shutdown()

if __name__ == "__main__":
    asyncio.run(test_enhanced_error_handling())
