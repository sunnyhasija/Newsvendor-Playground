#!/usr/bin/env python3
"""
Debug script to identify failing test cases in parsing components.
"""

import sys
import os
import logging

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.parsing.price_extractor import RobustPriceExtractor
from src.parsing.acceptance_detector import AcceptanceDetector

# Configure logging to see debug messages
logging.basicConfig(level=logging.DEBUG)

def debug_price_extractor():
    """Debug price extractor test cases."""
    print("=== DEBUGGING PRICE EXTRACTOR ===")
    
    extractor = RobustPriceExtractor({'debug': True})
    test_cases = [
        ("I offer $45 for this deal", 45),
        ("How about 67?", 67),
        ("Let's make it $52", 52),
        ("<think>Maybe 40?</think> I propose 50", 50),
        ("I accept your offer of 55", 55),
        ("$42", 42),
        ("No, I want 38 dollars", 38),
        ("I accept", None),  # Should return None (no price)
        ("Considering market conditions...", None),  # Should return None
    ]
    
    failed_tests = []
    
    for test_input, expected in test_cases:
        extracted = extractor.extract_price(test_input)
        passed = (extracted == expected)
        
        print(f"\nTest: '{test_input}'")
        print(f"  Expected: {expected}")
        print(f"  Extracted: {extracted}")
        print(f"  PASSED: {passed}")
        
        if not passed:
            failed_tests.append((test_input, expected, extracted))
    
    print(f"\n=== PRICE EXTRACTOR SUMMARY ===")
    print(f"Total tests: {len(test_cases)}")
    print(f"Failed tests: {len(failed_tests)}")
    
    if failed_tests:
        print("\nFailed test details:")
        for test_input, expected, extracted in failed_tests:
            print(f"  '{test_input}' -> expected {expected}, got {extracted}")
    
    return len(failed_tests) == 0


def debug_acceptance_detector():
    """Debug acceptance detector test cases."""
    print("\n=== DEBUGGING ACCEPTANCE DETECTOR ===")
    
    detector = AcceptanceDetector({'debug': True})
    test_cases = [
        # (message, current_price, expected_acceptance, expected_price)
        ("I accept $50", 50, True, 50),
        ("I accept", 45, True, 45),
        ("Deal!", None, True, None),
        ("That sounds good", 42, True, 42),
        ("No, too high", 60, False, None),
        ("How about $55?", 55, False, None),
        ("<think>Good price</think> I accept", 48, True, 48),
        ("Agreed at $52", 52, True, 52),
        ("Fine, $47", 47, True, 47),
        ("I reject your offer", 50, False, None),
    ]
    
    failed_tests = []
    
    for message, price, expected_acceptance, expected_price in test_cases:
        result = detector.detect_acceptance(message, price)
        
        acceptance_correct = result.is_acceptance == expected_acceptance
        
        # For price comparison, handle None cases carefully
        if expected_price is None:
            price_correct = result.accepted_price is None
        else:
            price_correct = result.accepted_price == expected_price
        
        test_passed = acceptance_correct and price_correct
        
        print(f"\nTest: '{message}' (price={price})")
        print(f"  Expected: accept={expected_acceptance}, price={expected_price}")
        print(f"  Got: accept={result.is_acceptance}, price={result.accepted_price}")
        print(f"  Pattern: {result.pattern_matched}, Confidence: {result.confidence:.2f}")
        print(f"  PASSED: {test_passed}")
        
        if not test_passed:
            failed_tests.append((message, price, expected_acceptance, expected_price, result))
    
    print(f"\n=== ACCEPTANCE DETECTOR SUMMARY ===")
    print(f"Total tests: {len(test_cases)}")
    print(f"Failed tests: {len(failed_tests)}")
    
    if failed_tests:
        print("\nFailed test details:")
        for message, price, expected_acceptance, expected_price, result in failed_tests:
            print(f"  '{message}' -> expected accept={expected_acceptance}, price={expected_price}")
            print(f"                 got accept={result.is_acceptance}, price={result.accepted_price}")
    
    return len(failed_tests) == 0


if __name__ == "__main__":
    price_ok = debug_price_extractor()
    acceptance_ok = debug_acceptance_detector()
    
    print(f"\n=== OVERALL RESULTS ===")
    print(f"Price Extractor: {'PASS' if price_ok else 'FAIL'}")
    print(f"Acceptance Detector: {'PASS' if acceptance_ok else 'FAIL'}")
    print(f"Overall: {'PASS' if price_ok and acceptance_ok else 'FAIL'}")