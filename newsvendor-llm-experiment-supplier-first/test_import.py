#!/usr/bin/env python3
"""Test script to verify metrics calculator import works correctly."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.analysis.metrics_calculator import MetricsCalculator
    print("✅ Import successful: MetricsCalculator imported without errors")
    
    # Test instantiation
    calc = MetricsCalculator()
    print("✅ Instantiation successful: MetricsCalculator created")
    
    # Test that new methods exist
    if hasattr(calc, 'calculate_turn_order_metrics'):
        print("✅ Method check: calculate_turn_order_metrics exists")
    else:
        print("❌ Method check: calculate_turn_order_metrics missing")
        
    if hasattr(calc, 'calculate_research_hypotheses'):
        print("✅ Method check: calculate_research_hypotheses exists")
    else:
        print("❌ Method check: calculate_research_hypotheses missing")
        
    if hasattr(calc, '_calculate_cohens_d'):
        print("✅ Method check: _calculate_cohens_d exists")
    else:
        print("❌ Method check: _calculate_cohens_d missing")
    
    print("✅ ALL IMPORT TESTS PASSED")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)
