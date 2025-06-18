#!/bin/bash
# Newsvendor Cleanup Script

echo "🧹 Starting newsvendor codebase cleanup..."

# 1. Remove redundant core files
echo "Removing legacy core files..."
rm -f src/agents/buyer_agent.py
rm -f src/agents/supplier_agent.py  
rm -f src/agents/reflection_mixin.py
rm -f src/core/model_manager.py
rm -f src/core/negotiation_engine.py
rm -f src/parsing/price_extractor.py
rm -f src/parsing/validation.py

# 2. Remove redundant experiment runners
echo "Removing redundant experiment runners..."
rm -f run_full_experiment_updated.py
rm -f run_validation_updated.py
rm -f setup_updated_experiment.py

# 3. Remove backup files
echo "Removing backup files..."
rm -f *.backup_*

# 4. Remove old log files (keep most recent)
echo "Removing old log files..."
rm -f newsvendor_experiment.log
rm -f newsvendor_validation_enhanced.log
rm -f newsvendor_validation_no_tinyllama.log
rm -f newsvendor_validation_updated.log
rm -f newsvendor_validation_with_grok.log

# 5. Remove old validation directories
echo "Removing old validation results..."
rm -rf validation_results_enhanced
rm -rf validation_results_improved  
rm -rf validation_results_updated
rm -rf validation_results_with_grok

# 6. Clean old outputs (keep newest 5)
echo "Cleaning old outputs (keeping newest 5)..."
cd outputs
ls -t | tail -n +6 | head -20 | xargs rm -rf 2>/dev/null || true
cd ..

# 7. Remove Python cache
echo "Removing Python cache files..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

echo "✅ Cleanup complete!"
echo "📊 Checking space saved..."
du -sh . 2>/dev/null || echo "Directory size check completed"