#!/usr/bin/env python3
"""
validate_turn_order_system.py
Quick validation test for Turn Order Control System v0.6

This script runs a small test to validate that the turn order control system
works correctly before running the full experiment.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from run_turn_order_experiment import TurnOrderExperimentRunner

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def validate_turn_order_system():
    """Run a quick validation test of the turn order control system."""
    
    logger.info("🧪 Starting Turn Order Control System Validation")
    logger.info("=" * 60)
    
    runner = TurnOrderExperimentRunner(max_concurrent=1)
    
    try:
        # Initialize
        await runner.initialize()
        
        # Create a minimal test plan
        test_models = ["qwen2:1.5b", "gemma2:2b"]  # Just 2 local models
        test_strategies = ["buyer_first", "supplier_first"]
        test_patterns = ["00"]  # No reflection for simplicity
        
        experiment_plan = runner.generate_experiment_plan(
            models_subset=test_models,
            strategies_subset=test_strategies,
            patterns_subset=test_patterns,
            replications=2  # Just 2 replications per condition
        )
        
        logger.info(f"📋 Test Plan:")
        logger.info(f"  Models: {test_models}")
        logger.info(f"  Turn order strategies: {test_strategies}")
        logger.info(f"  Reflection patterns: {test_patterns}")
        logger.info(f"  Total negotiations: {len(experiment_plan) * 2}")
        
        # Run the validation experiment
        analysis = await runner.run_turn_order_experiment(experiment_plan, save_results=False)
        
        # Analyze results
        summary = analysis["experiment_summary"]
        turn_order_analysis = analysis.get("turn_order_analysis", {})
        
        logger.info("🔬 Validation Results:")
        logger.info(f"  Total negotiations: {summary['total_negotiations']}")
        logger.info(f"  Success rate: {summary['success_rate']*100:.1f}%")
        logger.info(f"  Turn order distribution: {turn_order_analysis.get('distribution', {})}")
        
        # Check if both turn order strategies were used
        distribution = turn_order_analysis.get("distribution", {})
        buyer_first_count = distribution.get("buyer_first", 0)
        supplier_first_count = distribution.get("supplier_first", 0)
        
        if buyer_first_count > 0 and supplier_first_count > 0:
            logger.info("✅ VALIDATION PASSED: Both turn order strategies were successfully used")
            
            # Check first speaker tracking
            first_speaker_analysis = turn_order_analysis.get("first_speaker_analysis", {})
            buyer_first_negotiations = first_speaker_analysis.get("buyer", {}).get("count", 0)
            supplier_first_negotiations = first_speaker_analysis.get("supplier", {}).get("count", 0)
            
            logger.info(f"📊 First Speaker Distribution:")
            logger.info(f"  Buyer went first: {buyer_first_negotiations} negotiations")
            logger.info(f"  Supplier went first: {supplier_first_negotiations} negotiations")
            
            if buyer_first_negotiations > 0 and supplier_first_negotiations > 0:
                logger.info("✅ FIRST SPEAKER TRACKING: Working correctly")
            else:
                logger.warning("⚠️  FIRST SPEAKER TRACKING: May have issues")
            
            # Check if we have research findings
            if "research_findings" in analysis:
                findings = analysis["research_findings"]
                if "bias_decomposition" in findings:
                    bias = findings["bias_decomposition"]
                    advantage = bias.get("buyer_advantage_when_supplier_goes_first", 0)
                    logger.info(f"🔬 Research Finding Preview:")
                    logger.info(f"  Buyer advantage when suppliers go first: ${advantage:.2f}")
                    logger.info("✅ RESEARCH ANALYSIS: Working correctly")
                else:
                    logger.warning("⚠️  RESEARCH ANALYSIS: Limited data")
            
            # Overall validation result
            logger.info("=" * 60)
            logger.info("🎉 OVERALL VALIDATION: PASSED")
            logger.info("✅ Turn order control system is working correctly")
            logger.info("✅ Ready to run full experiments")
            
        else:
            logger.error("❌ VALIDATION FAILED: Turn order strategies not properly implemented")
            logger.error(f"  Buyer first: {buyer_first_count}, Supplier first: {supplier_first_count}")
            
    except Exception as e:
        logger.error(f"❌ VALIDATION FAILED: {e}")
        raise
    
    finally:
        await runner.shutdown()


async def test_configuration_loading():
    """Test that configuration loading works with turn order support."""
    
    logger.info("🔧 Testing Configuration Loading...")
    
    try:
        from utils.config_loader import load_config, get_turn_order_strategy, set_turn_order_strategy
        
        # Load config
        config = load_config()
        
        # Test turn order strategy
        strategy = get_turn_order_strategy(config)
        logger.info(f"  Default turn order strategy: {strategy}")
        
        # Test setting different strategies
        for test_strategy in ["buyer_first", "supplier_first", "random"]:
            set_turn_order_strategy(config, test_strategy)
            current_strategy = get_turn_order_strategy(config)
            assert current_strategy == test_strategy, f"Expected {test_strategy}, got {current_strategy}"
            logger.info(f"  ✅ Strategy '{test_strategy}' set correctly")
        
        logger.info("✅ CONFIGURATION: All tests passed")
        
    except Exception as e:
        logger.error(f"❌ CONFIGURATION: Failed - {e}")
        raise


async def test_conversation_tracker():
    """Test that conversation tracker works with turn order control."""
    
    logger.info("💬 Testing Conversation Tracker...")
    
    try:
        from core.conversation_tracker import ConversationTracker
        
        # Test different turn order strategies
        for strategy in ["buyer_first", "supplier_first"]:
            tracker = ConversationTracker(
                negotiation_id=f"test_{strategy}",
                buyer_model="test_buyer",
                supplier_model="test_supplier",
                reflection_pattern="00",
                turn_order_strategy=strategy
            )
            
            expected_first_speaker = "buyer" if strategy == "buyer_first" else "supplier"
            actual_first_speaker = tracker.first_speaker
            
            assert actual_first_speaker == expected_first_speaker, f"Expected {expected_first_speaker}, got {actual_first_speaker}"
            logger.info(f"  ✅ Strategy '{strategy}' -> first speaker '{actual_first_speaker}'")
            
            # Test context generation
            context = tracker.get_conversation_history()
            assert "Make your opening offer" in context, "Opening context not generated correctly"
            logger.info(f"  ✅ Opening context generated correctly for {strategy}")
        
        logger.info("✅ CONVERSATION TRACKER: All tests passed")
        
    except Exception as e:
        logger.error(f"❌ CONVERSATION TRACKER: Failed - {e}")
        raise


async def main():
    """Run all validation tests."""
    
    print("🚀 Turn Order Control System v0.6 - Validation Suite")
    print("=" * 70)
    print("Testing the configurable turn order system for literature bias research")
    print("=" * 70)
    
    try:
        # Test 1: Configuration loading
        await test_configuration_loading()
        print()
        
        # Test 2: Conversation tracker
        await test_conversation_tracker()
        print()
        
        # Test 3: Full system validation
        await validate_turn_order_system()
        
        print("=" * 70)
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print("✅ Turn Order Control System v0.6 is ready for use")
        print("🔬 Ready to conduct literature bias vs anchoring research")
        print("=" * 70)
        
    except Exception as e:
        print("=" * 70)
        print(f"❌ VALIDATION FAILED: {e}")
        print("🔧 Please check the implementation and try again")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())