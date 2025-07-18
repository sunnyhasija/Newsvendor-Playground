# LLM Project Navigation Guide: Newsvendor Negotiation Experiment

## 🎯 **Project Overview**

This is a sophisticated research project studying **LLM negotiation capabilities** in a buyer-supplier relationship using the classic **newsvendor problem** from operations research. The key discovery is a **systematic $14.25 buyer advantage**, suggesting LLMs inherit academic perspective biases from training literature.

### **Core Research Question**
Do LLMs inherit discipline-specific biases from academic literature, and can metacognitive reflection help overcome these biases?

### **Key Finding**
LLMs systematically favor buyers by $14.25 on average, likely because operations research literature is inherently buyer-centric while supplier strategies are covered in marketing literature.

---

## 🏗️ **Project Architecture**

### **Experimental Design**
- **10 models** × **10 models** × **4 reflection patterns** × **20 replications** = **8,000 negotiations**
- **Game Setup**: Buyer knows selling price ($100) and demand; Supplier knows cost ($30); Optimal fair price is $65
- **Reflection Patterns**: 00 (none), 01 (buyer only), 10 (supplier only), 11 (both)

### **Directory Structure**
```
src/
├── core/          # Main orchestration engine
├── agents/        # Buyer/supplier negotiation agents  
├── parsing/       # Price extraction & acceptance detection
├── analysis/      # Statistical analysis & visualization
└── utils/         # Configuration & data management

config/            # YAML configuration files
tests/             # Unit and integration tests
analysis/          # Generated reports and outputs
data/              # Experimental data storage
```

---

## 📋 **File Importance Classification**

### **🔥 CRITICAL CORE FILES** (Start Here)

1. **`src/core/unified_model_manager.py`** ⭐⭐⭐
   - **Purpose**: Manages both local (Ollama) and remote (Claude, O3, Grok) models
   - **Key Feature**: Smart throttling with progressive backoff for API rate limits
   - **Important**: Removed TinyLlama, added Grok-3-mini via Azure AI
   - **Generous Token Limits**: No artificial restrictions - lets models express naturally

2. **`src/core/conversation_tracker.py`** ⭐⭐⭐
   - **Purpose**: Bulletproof conversation state management
   - **Features**: Tracks rounds, detects termination, calculates profits
   - **Enhanced**: Fallback price extraction with local model assistance

3. **`src/agents/standardized_agents.py`** ⭐⭐⭐
   - **Purpose**: Buyer and supplier agents with standardized reflection prompts
   - **Key Feature**: Uniform reflection across all 10 models (including premium ones)
   - **Strategic**: Manages private information and role-specific constraints

### **🔧 ESSENTIAL PARSING & DETECTION**

4. **`src/parsing/enhanced_price_extractor.py`** ⭐⭐⭐
   - **Purpose**: Advanced price extraction with local model fallback
   - **Innovation**: When regex fails, uses local LLM to interpret ambiguous responses
   - **Robust**: Handles reflection blocks, validates role-specific constraints

5. **`src/parsing/acceptance_detector.py`** ⭐⭐
   - **Purpose**: Detects explicit acceptance and implicit convergence
   - **Patterns**: "I accept", "deal", convergence detection, rejection patterns

### **⚙️ MAIN EXECUTION FILES**

6. **`run_full_experiment_with_throttling.py`** ⭐⭐⭐
   - **Purpose**: Main experiment runner with smart API throttling
   - **Features**: Progress tracking, cost estimation, progressive backoff
   - **Scale**: Handles 8,000 negotiations with concurrent execution

7. **`unified_analysis_runner.py`** ⭐⭐
   - **Purpose**: Comprehensive analysis suite combining statistical tests
   - **Output**: Publication-quality figures and detailed statistical reports

### **📊 ANALYSIS & METRICS**

8. **`src/analysis/metrics_calculator.py`** ⭐⭐
   - **Purpose**: Calculate key performance metrics (convergence, efficiency, reflection benefits)
   - **Research Focus**: Tests 4 main hypotheses about reflection and model effects

### **⚙️ CONFIGURATION FILES**

9. **`config/models.yaml`** ⭐⭐
   - **Models**: qwen2:1.5b through qwen3:latest + Claude/O3/Grok remotes
   - **Important**: TinyLlama removed, Grok added, generous token limits
   - **Tiers**: Ultra-compact → Compact → Mid → Large → Premium

10. **`config/experiment.yaml`** ⭐⭐
    - **Game Parameters**: Selling price $100, cost $30, optimal $65
    - **Corrected**: Demand distribution changed from Uniform[50,150] to Normal(40,10)

11. **`config/prompts.yaml`** ⭐⭐
    - **Standardized Prompts**: Uniform across all models with reflection variants
    - **Anti-chattiness**: Special handling for verbose models

---

## 🔍 **Key Changes During Development**

### **Model Changes**
- **❌ Removed**: `tinyllama:latest` (poor performance)
- **✅ Added**: `grok-remote` (Azure AI Services integration)
- **🔧 Fixed**: Game parameters (corrected demand distribution)

### **Technical Enhancements**
- **Smart Throttling**: Progressive backoff for API rate limits
- **Enhanced Extraction**: Local model fallback for difficult price parsing
- **Generous Limits**: Removed artificial token restrictions
- **Unified Reflection**: Standardized prompts across all model tiers

### **Bug Fixes Identified**
- `fix_acceptance_pattern.py` - Fixed acceptance detection edge cases
- `fix_price_validation.py` - Corrected price validation logic
- Multiple enhancement iterations in price extraction

---

## 🧪 **Experimental vs Production Files**

### **📈 PRODUCTION/RESEARCH FILES** (Core functionality)
- All files in `src/core/`, `src/agents/`, `src/parsing/`
- Main experiment runners
- Configuration files
- Analysis calculators

### **🔬 EXPERIMENTAL/DEBUG FILES** (Skip for core understanding)
- `add_missing_methods.py` - Method additions during development
- `debug_runner.py` - Testing utilities
- `test_grok.py` - Grok integration testing
- Files in `analysis/` directory - Generated reports (not source code)
- `fix_*` files - Bug hunting and fixes
- `success_analysis.py` - Ad-hoc analysis scripts

### **📊 GENERATED OUTPUTS** (Results, not source)
- `analysis/`, `analysis_output/`, `analysis_results/` - Generated reports
- `data/` - Experimental data
- `outputs/` - Visualization outputs
- `*.log` files - Execution logs

---

## 🎯 **Quick Start for LLM Understanding**

### **If you want to understand THE CORE LOGIC:**
1. Read `src/core/conversation_tracker.py` - See how negotiations flow
2. Read `src/agents/standardized_agents.py` - See how agents think and respond
3. Check `config/experiment.yaml` - Understand the game setup

### **If you want to understand THE TECHNICAL IMPLEMENTATION:**
1. Read `src/core/unified_model_manager.py` - See how models are managed
2. Read `src/parsing/enhanced_price_extractor.py` - See how responses are parsed
3. Check `config/models.yaml` - See available models and configurations

### **If you want to understand THE RESEARCH:**
1. Read `coauthor_memo.md` - Research context and findings
2. Read `README.md` - Full experimental design
3. Check `src/analysis/metrics_calculator.py` - How results are analyzed

### **If you want to RUN THE EXPERIMENT:**
1. Use `run_full_experiment_with_throttling.py` - Main execution
2. Check `config/` files for parameters
3. Use `unified_analysis_runner.py` for analysis

---

## 🚀 **Key Innovation Points**

1. **Reflection Mechanism**: Standardized `<think>` blocks across all models
2. **Multi-Model Support**: Seamless local (Ollama) + remote (Claude/O3/Grok) integration
3. **Smart Throttling**: Adaptive API rate limiting with progressive backoff
4. **Enhanced Parsing**: LLM-assisted price extraction when regex fails
5. **Research Design**: Tests specific hypotheses about bias inheritance and metacognition

---

## ⚠️ **Important Context**

- **Status**: Production-ready research codebase (v0.5)
- **Scale**: Designed for 8,000+ negotiations with cost management
- **Quality**: Extensive validation, error handling, and logging
- **Research Impact**: Advancing understanding of AI negotiation capabilities and bias inheritance

This codebase represents a comprehensive academic research framework that could be adapted for other negotiation scenarios or extended to test different AI capabilities.