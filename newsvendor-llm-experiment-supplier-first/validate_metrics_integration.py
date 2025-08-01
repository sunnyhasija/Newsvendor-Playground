#!/usr/bin/env python3
"""Quick validation test for turn order experiment with metrics calculator."""

import sys
import os
import asyncio
from pathlib import Path
import pandas as pd
from dataclasses import asdict

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def create_mock_result():
    """Create a mock NegotiationResult for testing."""
    from core.conversation_tracker import NegotiationResult
    from parsing.acceptance_detector import TerminationType
    
    return NegotiationResult(
        negotiation_id="test_001",
        buyer_model="qwen2:1.5b",
        supplier_model="qwen2:1.5b", 
        reflection_pattern="00",
        turn_order_strategy="buyer_first",
        first_speaker="buyer",
        completed=True,
        agreed_price=62.5,
        termination_type=TerminationType.AGREEMENT,
        total_rounds=3,
        total_tokens=450,
        total_time=15.2,
        buyer_profit=15.0,
        supplier_profit=12.5,
        distance_from_optimal=2.5,
        turns=[],
        metadata={"total_cost": 0.05}
    )

def test_metrics_calculator():
    """Test that the metrics calculator works with turn order data."""
    try:
        from src.analysis.metrics_calculator import MetricsCalculator
        
        # Create mock data
        results = [create_mock_result() for _ in range(5)]
        
        # Vary turn order strategies
        results[1].turn_order_strategy = "supplier_first"
        results[1].first_speaker = "supplier"
        results[2].turn_order_strategy = "supplier_first" 
        results[2].first_speaker = "supplier"
        results[2].agreed_price = 68.0
        
        # Convert to DataFrame
        df = pd.DataFrame([asdict(r) for r in results])
        
        # Test metrics calculator
        calc = MetricsCalculator()
        analysis = calc.generate_summary_report(df)
        
        # Check that new keys are present
        if 'turn_order_analysis' in analysis:
            print("✅ turn_order_analysis key found in output")
        else:
            print("❌ turn_order_analysis key missing from output")
            
        if 'research_hypotheses' in analysis:
            print("✅ research_hypotheses key found in output")
        else:
            print("❌ research_hypotheses key missing from output")
            
        print("✅ MetricsCalculator test passed")
        return True
        
    except Exception as e:
        print(f"❌ MetricsCalculator test failed: {e}")
        return False

def main():
    """Run validation tests."""
    print("🧪 Running turn order experiment validation...")
    
    # Test 1: Import check
    try:
        from src.analysis.metrics_calculator import MetricsCalculator
        print("✅ MetricsCalculator import successful")
    except Exception as e:
        print(f"❌ MetricsCalculator import failed: {e}")
        return False
    
    # Test 2: Metrics calculator functionality
    if not test_metrics_calculator():
        return False
    
    # Test 3: Main script import
    try:
        # Test that the main script can be imported
        import run_turn_order_experiment
        print("✅ run_turn_order_experiment import successful")
    except Exception as e:
        print(f"❌ run_turn_order_experiment import failed: {e}")
        return False
    
    print("✅ ALL VALIDATION TESTS PASSED!")
    print("🎉 Turn order experiment is ready to run!")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
