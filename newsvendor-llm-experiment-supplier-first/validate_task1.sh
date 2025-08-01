#!/bin/bash

echo "🔧 Task 1: Smart Throttling System - Validation Test"
echo "===================================================="

cd "/Users/pod/Library/CloudStorage/Dropbox/School/My Papers in Progress/LLMs in SCM/Newsvendor and LLMs/Playground/newsvendor-llm-experiment-supplier-first/"

echo "📍 Current directory: $(pwd)"
echo ""

echo "🧪 Running Throttling System Validation (from Development Bible):"
echo "Command: python run_turn_order_experiment.py --models qwen2:1.5b --strategies buyer_first --patterns 00 --replications 2"
echo ""

python3 run_turn_order_experiment.py --models qwen2:1.5b --strategies buyer_first --patterns 00 --replications 2

echo ""
echo "✅ Validation test completed!"
echo "Expected: Log shows 'Smart throttling enabled' and throttling events tracked"
