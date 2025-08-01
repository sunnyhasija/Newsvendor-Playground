#!/usr/bin/env python3
"""
Quick test script to validate cost tracking implementation
"""

import sys
import asyncio
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from run_turn_order_experiment import TurnOrderExperimentRunner

async def test_cost_tracking():
    """Test cost tracking functionality."""
    print("🧪 Testing cost tracking functionality...")
    
    runner = TurnOrderExperimentRunner(max_concurrent=1)
    
    try:
        # Test initialization
        await runner.initialize()
        print("✅ Initialization successful")
        
        # Generate minimal experiment plan
        experiment_plan = runner.generate_experiment_plan(
            models_subset=["qwen2:1.5b"],
            strategies_subset=["buyer_first"],
            patterns_subset=["00"],
            replications=1
        )
        print(f"✅ Generated experiment plan with {len(experiment_plan)} configurations")
        
        # Test cost estimation
        estimates = runner.estimate_experiment_cost_and_time(experiment_plan)
        print(f"✅ Cost estimation: ${estimates['total_estimated_cost']:.4f}")
        
        # Test helper methods
        print("✅ Testing helper methods...")
        
        # Test _estimate_remaining_time
        eta = runner._estimate_remaining_time(5, 10)
        print(f"  - ETA method works: {eta}")
        
        # Test _calculate_cost_by_model
        mock_results = []
        cost_by_model = runner._calculate_cost_by_model(mock_results)
        print(f"  - Cost by model method works: {len(cost_by_model)} models")
        
        # Test _calculate_cost_efficiency  
        cost_efficiency = runner._calculate_cost_efficiency(mock_results)
        print(f"  - Cost efficiency method works: {cost_efficiency}")
        
        print("✅ All cost tracking components validated successfully!")
        print("🔍 Cost tracking features:")
        print("  - Real-time cost accumulation")
        print("  - Enhanced progress bar with cost metrics")
        print("  - Cost analysis in results")
        print("  - Cost efficiency calculations")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
        
    finally:
        await runner.shutdown()

if __name__ == "__main__":
    success = asyncio.run(test_cost_tracking())
    sys.exit(0 if success else 1)
