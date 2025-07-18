# Newsvendor LLM Experiment v0.6 - Turn Order Control

Research into literature bias vs anchoring effects in LLM negotiations through configurable turn order control.

## 🎯 Research Question

**Does the $14.25 buyer advantage stem from literature bias in training data, or from anchoring effects due to buyers always going first?**

## 🔬 Methodology

- **Identical Prompts**: Zero prompt engineering confounds
- **Context Assignment Control**: Only opening vs response context changes  
- **2×2 Factorial Design**: Turn Order × Reflection
- **Perfect Experimental Control**: Same models, game, parameters

## 🚀 Quick Start

### Validation Test:
```bash
python validate_turn_order_system.py
```

### Small Test Experiment:
```bash  
python run_turn_order_experiment.py --models qwen2:1.5b,gemma2:2b --replications 3
```

### Dry Run (Plan Only):
```bash
python run_turn_order_experiment.py --dry-run --full-experiment
```

## 📊 Expected Research Outcomes

- **Literature Bias**: Buyer advantage persists when suppliers go first
- **Anchoring Bias**: Buyer advantage disappears when suppliers go first  
- **Mixed Effects**: Both biases contribute to original finding

## 🔧 Key Files

- `run_turn_order_experiment.py` - Main experiment runner
- `src/core/conversation_tracker.py` - Turn order control implementation
- `config/experiment.yaml` - Turn order configuration
- `validate_turn_order_system.py` - Validation test suite

## 📖 Documentation

See `LLM_NAVIGATION_GUIDE_v0.6_IMPLEMENTED.md` for complete implementation details.

## ✅ Status

**IMPLEMENTATION COMPLETE** - Ready for research deployment