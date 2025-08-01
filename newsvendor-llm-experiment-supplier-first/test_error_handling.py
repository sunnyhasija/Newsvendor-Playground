#!/usr/bin/env python3
"""
Simple validation test for Task 6 - Enhanced Error Handling
Tests the _create_failed_result method and error metadata generation.
"""

import sys
from datetime import datetime
from pathlib import Path
from dataclasses import asdict

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

# Import the enhanced classes
from run_turn_order_experiment import TurnOrderExperimentRunner, TurnOrderExperimentConfig
from parsing.acceptance_detector import TerminationType

def test_enhanced_error_handling():
    """Test that enhanced error handling creates proper structured failure results."""
    
    print("🧪 Testing Enhanced Error Handling Implementation")
    print("=" * 60)
    
    # Create a runner instance
    runner = TurnOrderExperimentRunner(max_concurrent=1)
    
    # Create a test configuration
    test_config = TurnOrderExperimentConfig(
        buyer_model="test_buyer_model",
        supplier_model="test_supplier_model",
        reflection_pattern="00",
        turn_order_strategy="buyer_first",
        replications=1
    )
    
    # Test creating failed results with different error types
    test_cases = [
        ("Connection timeout", ConnectionError("Network timeout")),
        ("Invalid model", ValueError("Model not found")),
        ("API rate limit", Exception("Rate limit exceeded")),
        ("Generic error", "Simple string error"),
    ]
    
    all_tests_passed = True
    
    for test_name, error in test_cases:
        print(f"\n🔍 Testing: {test_name}")
        
        # Create a failed result
        negotiation_id = f"test_{test_name.replace(' ', '_').lower()}"
        failed_result = runner._create_failed_result(test_config, negotiation_id, error)
        
        # Convert to dict for easier inspection
        result_dict = asdict(failed_result)
        
        # Validate structure
        tests = [
            ("negotiation_id matches", failed_result.negotiation_id == negotiation_id),
            ("completed is False", failed_result.completed == False),
            ("termination_type is FAILURE", failed_result.termination_type == TerminationType.FAILURE),
            ("metadata contains error", "error" in failed_result.metadata),
            ("metadata contains error_type", "error_type" in failed_result.metadata),
            ("metadata contains timestamp", "timestamp" in failed_result.metadata),
            ("metadata contains phase", "phase" in failed_result.metadata),
            ("metadata contains models", "models" in failed_result.metadata),
            ("metadata contains reflection_pattern", "reflection_pattern" in failed_result.metadata),
            ("metadata contains total_cost", "total_cost" in failed_result.metadata),
        ]
        
        case_passed = True
        for test_desc, test_result in tests:
            if test_result:
                print(f"  ✅ {test_desc}")
            else:
                print(f"  ❌ {test_desc}")
                case_passed = False
        
        # Check error type extraction
        expected_error_type = type(error).__name__ if isinstance(error, Exception) else "unknown"
        actual_error_type = failed_result.metadata.get("error_type")
        
        if actual_error_type == expected_error_type:
            print(f"  ✅ error_type correctly identified: {actual_error_type}")
        else:
            print(f"  ❌ error_type mismatch: expected {expected_error_type}, got {actual_error_type}")
            case_passed = False
        
        # Check metadata completeness
        required_metadata = [
            "error", "error_type", "timestamp", "phase", 
            "models", "reflection_pattern", "total_cost"
        ]
        
        missing_metadata = [key for key in required_metadata if key not in failed_result.metadata]
        if not missing_metadata:
            print(f"  ✅ All required metadata present")
        else:
            print(f"  ❌ Missing metadata: {missing_metadata}")
            case_passed = False
        
        if not case_passed:
            all_tests_passed = False
        
        print(f"  📊 Sample metadata: {list(failed_result.metadata.keys())}")
    
    # Test failure analysis functionality  
    print(f"\n🔍 Testing failure analysis functionality")
    
    # Create mock results with failures
    mock_results = []
    error_types = ["ConnectionError", "ValueError", "TimeoutError", "ConnectionError", "ValueError"]
    
    for i, error_type in enumerate(error_types):
        # Create a mock failed result
        mock_result = runner._create_failed_result(
            test_config, 
            negotiation_id=f"mock_negotiation_{i}",
            error=Exception("Mock error")
        )
        # Manually set error type for testing
        mock_result.metadata["error_type"] = error_type
        mock_result.completed = False
        mock_results.append(mock_result)
    
    # Add some successful results
    for i in range(3):
        mock_result = runner._create_failed_result(
            test_config,
            negotiation_id=f"mock_success_{i}",
            error=""
        )
        mock_result.completed = True  # Mark as successful
        mock_results.append(mock_result)
    
    # Test the analysis method
    try:
        analysis = runner._analyze_turn_order_results(mock_results)
        
        failure_summary = analysis.get("failure_summary", {})
        total_failures = failure_summary.get("total_failures", 0)
        by_error_type = failure_summary.get("by_error_type", {})
        
        print(f"  ✅ Analysis completed successfully")
        print(f"  📊 Total failures detected: {total_failures}")
        print(f"  📊 Error type breakdown: {dict(by_error_type)}")
        
        # Validate analysis results
        expected_total_failures = len([r for r in mock_results if not r.completed])
        if total_failures == expected_total_failures:
            print(f"  ✅ Failure count correct: {total_failures}")
        else:
            print(f"  ❌ Failure count mismatch: expected {expected_total_failures}, got {total_failures}")
            all_tests_passed = False
        
        # Check error type aggregation
        expected_counts = {"ConnectionError": 2, "ValueError": 2, "TimeoutError": 1}
        for error_type, expected_count in expected_counts.items():
            actual_count = by_error_type.get(error_type, 0)
            if actual_count == expected_count:
                print(f"  ✅ {error_type} count correct: {actual_count}")
            else:
                print(f"  ❌ {error_type} count mismatch: expected {expected_count}, got {actual_count}")
                all_tests_passed = False
        
    except Exception as e:
        print(f"  ❌ Analysis failed with error: {e}")
        all_tests_passed = False
    
    # Final validation summary
    print(f"\n🎯 VALIDATION SUMMARY")
    print("=" * 40)
    
    if all_tests_passed:
        print("✅ ALL TESTS PASSED!")
        print("✅ Enhanced error handling successfully implemented")
        print("   - Failed results have rich metadata")
        print("   - Error types are properly extracted and categorized")
        print("   - Failure analysis includes error type breakdown")
        print("   - All required metadata fields are present")
        
        print(f"\n📋 IMPLEMENTATION CHECKLIST:")
        print("✅ Enhanced _create_failed_result with rich metadata")
        print("✅ Error type extraction and categorization")
        print("✅ Failure summary in analysis results")
        print("✅ Publication-grade error tracking")
        
        return True
    else:
        print("❌ SOME TESTS FAILED!")
        print("❌ Enhanced error handling needs review")
        return False

if __name__ == "__main__":
    success = test_enhanced_error_handling()
    sys.exit(0 if success else 1)
