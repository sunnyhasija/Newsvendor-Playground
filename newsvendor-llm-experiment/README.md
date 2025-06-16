# Newsvendor LLM Experiment Repository

A comprehensive experimental framework for studying negotiation capabilities of different language models in the classical newsvendor problem setting. This repository implements a systematic study comparing reflection mechanisms, model sizes, and negotiation strategies across 8 different LLM models.

## 🎯 Research Overview

This experiment investigates how self-reflection capabilities enable smaller, efficient models to achieve negotiation outcomes comparable to larger reasoning models in supply chain negotiations.

### Key Research Questions
- **H1 (Reflection Benefit):** Do models with reflection achieve higher convergence rates and better price optimality?
- **H2 (Size-Efficiency Trade-off):** Do mid-range models with reflection outperform large models without reflection on token efficiency?
- **H3 (Role Asymmetry):** Does reflection provide greater benefits to buyers than suppliers?
- **H4 (Mixed Pairing Synergy):** Do heterogeneous model pairings achieve better outcomes than homogeneous pairings?

### Game Setup
- **Selling Price:** $100 (known to buyer only)
- **Production Cost:** $30 (known to supplier only)
- **Demand Distribution:** Normal(μ=40, σ=10) (known to buyer only)
- **Optimal Price:** $65 (fair split-the-difference)
- **Total Negotiations:** 2,840+ across all conditions

## 📁 Repository Structure

### Core Engine (`src/core/`)

#### `negotiation_engine.py`
**Purpose:** Main orchestrator for conducting negotiations between two LLM agents.
- Manages conversation flow and experimental protocol
- Implements adaptive replication strategy (50-40-30-20 based on model computational cost)
- Handles timeout and termination conditions
- Validates experimental setup and model availability

#### `model_manager.py`
**Purpose:** Optimized LLM model loading and generation management.
- Handles Ollama model loading/unloading with memory optimization
- Supports concurrent model operations with configurable limits
- Implements automatic cleanup when memory limits are reached
- Tracks performance statistics (tokens/second, generation times)

#### `conversation_tracker.py`
**Purpose:** Bulletproof conversation state management.
- Tracks negotiation rounds and speaker alternation
- Detects termination conditions (acceptance, convergence, timeout)
- Maintains conversation history with price extraction
- Calculates profits and distance from optimal price

### Negotiation Agents (`src/agents/`)

#### `buyer_agent.py`
**Purpose:** Implements retailer negotiation behavior.
- Wants LOWEST possible wholesale price
- Has private information about selling price ($100) and demand distribution
- Can use reflection for strategic thinking
- Implements role-specific validation (won't offer above retail price)

#### `supplier_agent.py`
**Purpose:** Implements supplier negotiation behavior.
- Wants HIGHEST possible wholesale price above production cost
- Has private information about production cost ($30)
- Can use reflection for strategic reasoning
- Implements role-specific validation (won't offer below cost)

#### `reflection_mixin.py`
**Purpose:** Provides self-reflection capabilities for negotiation agents.
- Generates strategic thinking content in `<think>` blocks
- Adapts reflection complexity based on model tier (ultra/compact/mid/large)
- Analyzes negotiation state and suggests strategies
- Validates reflection content for required elements

### Price Parsing (`src/parsing/`)

#### `price_extractor.py`
**Purpose:** Robust price extraction from diverse LLM outputs.
- Multi-strategy extraction with fallback logic
- Handles reflection content removal before parsing
- Role-specific price validation (buyers ≤$99, suppliers ≥$31)
- Confidence scoring based on pattern specificity

#### `acceptance_detector.py`
**Purpose:** Detects explicit acceptance and implicit convergence.
- Pattern matching for acceptance statements ("I accept", "deal", etc.)
- Convergence detection when both parties offer same price
- Confidence scoring and rejection pattern detection
- Handles various termination types

### Analysis Framework (`src/analysis/`)

#### `complete_analysis_runner.py`
**Purpose:** Orchestrates comprehensive analysis including metrics, statistics, and visualizations.
- Generates executive summaries and detailed reports
- Coordinates between metrics calculator, statistical analyzer, and visualizer
- Creates publication-ready figures and infographics

#### `metrics_calculator.py`
**Purpose:** Calculates key performance metrics for negotiation analysis.
- Convergence rates, price optimality, efficiency metrics
- Reflection benefits analysis across patterns (00, 01, 10, 11)
- Model performance scoring and ranking
- Research implications generation

#### `statistical_tests.py`
**Purpose:** Comprehensive statistical analysis including ANOVA, t-tests, and effect sizes.
- Hypothesis testing for all four research questions
- Power analysis and confidence intervals
- Pairwise comparisons with multiple testing corrections

#### `visualizations.py`
**Purpose:** Creates publication-quality visualizations and dashboards.
- Comprehensive dashboards with 12+ visualization types
- Model performance heatmaps and efficiency comparisons
- Publication figures for academic papers
- Interactive visualizations and summary infographics

#### `conversation_analyzer.py`
**Purpose:** Turn-by-turn behavioral analysis of negotiation conversations.
- Opening bid strategy analysis by model
- Concession pattern detection and measurement
- Language pattern analysis (aggressive vs cooperative)
- Convergence speed analysis

### Experiment Runners (`src/experiments/`)

#### `run_full_experiment.py`
**Purpose:** Main experiment runner with three-phase protocol.
- **Phase 1:** Validation (2 hours, 4 test negotiations)
- **Phase 2:** Statistical Power (6 hours, 3 reps per condition)
- **Phase 3:** Full Dataset (12+ hours, complete replication matrix)
- Progress tracking with tqdm progress bars
- Automatic result saving and analysis

#### `run_validation_suite.py`
**Purpose:** Comprehensive validation before running experiments.
- Dry-run validation (configuration, imports, Ollama availability)
- Full validation with model loading and test negotiations
- System resource checking (RAM, CPU, disk space)
- Parsing component validation

### Utilities (`src/utils/`)

#### `config_loader.py`
**Purpose:** Configuration management with intelligent defaults and environment overrides.
- YAML configuration loading with deep merging
- Environment variable overrides (NEWSVENDOR_GAME__MAX_ROUNDS=15)
- Configuration validation and error checking
- Default configuration generation

#### `data_exporter.py`
**Purpose:** Multi-format data export and storage management.
- Exports to CSV, JSON, Parquet formats
- Conversation transcript export
- Performance dashboard creation
- Automatic backup and compression

#### `debug_data.py`
**Purpose:** Data debugging and sample data generation.
- Validates data file structure and content
- Creates realistic sample data for testing
- Checks for missing columns and null values
- Provides diagnostic information

## 🔧 Configuration Settings

### Game Parameters (`config/experiment.yaml`)

```yaml
game:
  selling_price: 100          # Retail price (buyer's private info)
  production_cost: 30         # Supplier's cost (supplier's private info)
  demand_mean: 40            # Mean demand (buyer's private info)
  demand_std: 10             # Demand std dev (buyer's private info)
  optimal_wholesale_price: 65 # Fair split-the-difference target
  max_rounds: 10             # Maximum negotiation rounds
  timeout_seconds: 60        # Per-negotiation timeout
```

### Model Configuration (`config/models.yaml`)

```yaml
models:
  tinyllama:latest:
    tier: "ultra"              # Model computational tier
    token_limit: 256           # Maximum tokens per response
    temperature: 0.3           # Generation randomness
    top_p: 0.8                # Nucleus sampling threshold
    
  qwen3:latest:
    tier: "large" 
    token_limit: 512
    temperature: 0.6
    top_p: 0.95
```

### Technical Settings

```yaml
technical:
  max_concurrent_models: 1    # Concurrent model loading limit
  memory_limit_gb: 40        # Memory threshold for cleanup
  retry_attempts: 2          # Failed generation retries
  validation_checks: true    # Enable pre-experiment validation
```

### Success Metrics

```yaml
metrics:
  target_completion_rate: 0.95    # 95% successful negotiations
  target_convergence_rate: 0.85   # 85% reach agreement
  target_price_optimality: 8      # ≤$8 gap from optimal
  target_efficiency: 2000         # <2000 tokens/negotiation
```

### Environment Variables

```bash
# Model configuration
export NEWSVENDOR_TECHNICAL__MAX_CONCURRENT_MODELS=2
export NEWSVENDOR_TECHNICAL__MEMORY_LIMIT_GB=32

# Game parameters override
export NEWSVENDOR_GAME__MAX_ROUNDS=15
export NEWSVENDOR_GAME__TIMEOUT_SECONDS=90

# Output configuration
export NEWSVENDOR_STORAGE__OUTPUT_DIR="./my_results"
export NEWSVENDOR_STORAGE__COMPRESSION_ENABLED=true
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running
- At least 16GB RAM (32GB+ recommended)

### Installation
```bash
git clone <repository>
cd newsvendor-llm-experiment
poetry install

# Download required models
ollama pull tinyllama:latest
ollama pull qwen2:1.5b
ollama pull gemma2:2b
ollama pull phi3:mini
ollama pull llama3.2:latest
ollama pull mistral:instruct
ollama pull qwen:7b
ollama pull qwen3:latest
```

### Running Experiments

**Validation:**
```bash
poetry run newsvendor-validate --dry-run
poetry run newsvendor-experiment --phase validation
```

**Full Experiment:**
```bash
poetry run newsvendor-experiment --phase all --output ./results
```

**Custom Model Subset:**
```bash
poetry run newsvendor-experiment --models "tinyllama:latest,qwen2:1.5b" --concurrent 2
```

### Analysis Scripts

**Complete Analysis:**
```bash
python comprehensive_analysis.py  # Statistical analysis with 15+ visualizations
python final_comprehensive_analysis.py  # Academic-quality analysis
python quick_analysis.py  # Fast overview with key insights
```

**Debugging:**
```bash
python debug_analysis.py  # Fix analysis pipeline issues
python debug_reflection.py  # Check reflection pattern data
python debug_parsing_script.py  # Test price extraction components
```

## 📊 Expected Outputs

### Data Files
- **Raw transcripts:** `data/raw/` - Complete negotiation conversations
- **Processed datasets:** `data/processed/` - Clean CSV/Parquet for analysis
- **Analysis results:** `data/analysis/` - Statistical outputs and metrics

### Visualizations
- **Comprehensive dashboard:** 12+ visualization types in single figure
- **Publication figures:** High-quality individual plots for papers
- **Model performance heatmaps:** Pairing effectiveness matrices
- **Efficiency analysis:** Token usage and round distributions

### Reports
- **Executive summary:** Key findings and hypothesis test results
- **Detailed analysis:** Complete statistical analysis with interpretations
- **Conversation analysis:** Turn-by-turn behavioral insights

## 🔬 Research Contributions

This repository provides:
1. **Largest systematic study** of LLM negotiation behavior (2,840+ negotiations)
2. **Novel reflection mechanisms** for improving negotiation outcomes
3. **Comprehensive model comparison** across computational tiers
4. **Empirical evidence** for role asymmetry in AI negotiations
5. **Open-source framework** for replicating and extending research

## 🧪 Advanced Features

### Adaptive Replication Strategy
- **Ultra-compact models:** 50 replications (fast execution)
- **Compact models:** 40 replications (balanced)
- **Mid-range models:** 30 replications (adequate power)
- **Large models:** 20 replications (minimum viable)

### Robust Error Handling
- Automatic model cleanup on memory pressure
- Graceful degradation with partial results
- Comprehensive logging and debugging tools
- Validation at every pipeline stage

### Performance Optimization
- Concurrent model loading where supported
- Memory-efficient data structures
- Streaming analysis for large datasets
- Compression and backup systems

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Install development dependencies: `poetry install --with dev`
2. Set up pre-commit hooks: `pre-commit install`
3. Run tests: `pytest tests/`
4. Follow code formatting: `black src/ && isort src/`

## 📞 Support

- Check troubleshooting section in individual script documentation
- Run validation: `poetry run newsvendor-validate --dry-run`
- Check logs: `./newsvendor_experiment.log`
- Open issues with experiment logs and system info

---

**Status:** Production Ready v0.5 with Corrected Parameters  
**Last Updated:** Jun 15, 2025   
**Research Impact:** Advancing understanding of AI negotiation capabilities