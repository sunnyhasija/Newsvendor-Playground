# LLM Agent Newsvendor Negotiation Experiment v0.5

A comprehensive experimental framework for studying negotiation capabilities of different language models in the classical newsvendor problem setting.

## 🎯 Overview

This experiment investigates whether self-reflection enables smaller, efficient models to achieve negotiation outcomes comparable to larger reasoning models in the newsvendor problem context.

### Key Research Questions

- **H1 (Reflection Benefit):** Models with reflection achieve higher convergence rates and better price optimality than without reflection
- **H2 (Size-Efficiency Trade-off):** Mid-range models with reflection outperform large models without reflection on token efficiency
- **H3 (Role Asymmetry):** Reflection provides greater benefits to suppliers than buyers due to information asymmetry
- **H4 (Mixed Pairing Synergy):** Heterogeneous model pairings achieve better outcomes than homogeneous pairings

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running
- At least 16GB RAM (32GB+ recommended for concurrent models)

### Installation

1. **Clone and install dependencies:**
```bash
git clone <repository>
cd newsvendor-llm-experiment
poetry install
```

2. **Download required models:**
```bash
# Ultra-compact models
ollama pull tinyllama:latest
ollama pull qwen2:1.5b

# Compact models  
ollama pull gemma2:2b
ollama pull phi3:mini
ollama pull llama3.2:latest

# Mid-range models
ollama pull mistral:instruct
ollama pull qwen:7b

# Large model
ollama pull qwen3:latest
```

3. **Validate setup:**
```bash
poetry run newsvendor-validate --dry-run
```

### Running Experiments

**Quick validation:**
```bash
poetry run newsvendor-experiment --phase validation
```

**Full experiment:**
```bash
poetry run newsvendor-experiment --phase all --output ./results
```

**Custom model subset:**
```bash
poetry run newsvendor-experiment --models "tinyllama:latest,qwen2:1.5b,gemma2:2b" --concurrent 2
```

## 📊 Experimental Design

### Game Setup (CORRECTED)

- **Selling Price:** $100 (known to buyer only)
- **Production Cost:** $30 (known to supplier only)  
- **Demand Distribution:** **Normal(μ=40, σ=10)** (known to buyer only)
- **Optimal Price:** $65 (fair split-the-difference)

### Model Selection (8 Models)

| Tier | Model | Size | Token Limit | Research Purpose |
|------|-------|------|-------------|------------------|
| **Ultra-Compact** | tinyllama:latest | 637 MB | 256 | Efficiency baseline |
| | qwen2:1.5b | 934 MB | 256 | Modern ultra-compact |
| **Compact** | gemma2:2b | 1.6 GB | 384 | Google's latest 2B |
| | phi3:mini | 2.2 GB | 384 | Microsoft's mini |
| | llama3.2:latest | 2.0 GB | 384 | Meta's efficient |
| **Mid-Range** | mistral:instruct | 4.1 GB | 512 | Instruction-tuned |
| | qwen:7b | 4.5 GB | 512 | Multilingual standard |
| **Large** | qwen3:latest | 5.2 GB | 512 | Verbose control |

### Adaptive Replication Strategy

- **Ultra-compact models:** 20 replications (fast execution)
- **Compact models:** 15 replications (balanced)
- **Mid-range models:** 10 replications (adequate power)
- **Large models:** 5 replications (minimum viable)

**Total: 1,940 negotiations** across all pairings and reflection conditions.

## 🏗️ Architecture

```
newsvendor_v05/
├── config/                 # Configuration files
├── src/
│   ├── core/              # Core negotiation engine
│   ├── agents/            # Buyer and supplier agents
│   ├── parsing/           # Price extraction and validation
│   ├── analysis/          # Statistical analysis
│   └── utils/             # Utilities and helpers
├── experiments/           # Experiment runners
├── tests/                 # Test suite
└── data/                  # Experimental data
```

## 🔧 Configuration

### Environment Variables

```bash
# Model configuration
export NEWSVENDOR_TECHNICAL__MAX_CONCURRENT_MODELS=2
export NEWSVENDOR_TECHNICAL__MEMORY_LIMIT_GB=32

# Game parameters (use corrected values)
export NEWSVENDOR_GAME__DEMAND_MEAN=40
export NEWSVENDOR_GAME__DEMAND_STD=10

# Output configuration
export NEWSVENDOR_STORAGE__OUTPUT_DIR="./my_results"
```

### Configuration Files

Edit `config/experiment.yaml` for detailed settings:

```yaml
game:
  selling_price: 100
  production_cost: 30
  demand_mean: 40        # CORRECTED
  demand_std: 10         # CORRECTED
  optimal_price: 65      # Fair split
  max_rounds: 10

technical:
  max_concurrent_models: 1
  memory_limit_gb: 40
  timeout_seconds: 60
```

## 📈 Analysis and Results

### Automatic Analysis

The experiment automatically generates:

- **Convergence rates** by model and reflection condition
- **Price optimality** analysis (distance from $65 target)
- **Token efficiency** metrics
- **Statistical tests** (chi-squared, t-tests, ANOVA)
- **Visualizations** (heatmaps, box plots, scatter plots)

### Results Structure

```
data/
├── raw/                   # Raw negotiation transcripts
├── processed/             # Clean CSV/Parquet datasets  
├── analysis/              # Statistical outputs
└── visualizations/        # Charts and figures
```

### Key Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Completion Rate** | >95% | Successful negotiations / attempted |
| **Convergence Rate** | >85% | Agreements / completed negotiations |
| **Price Optimality** | ≤$8 gap | \|agreed_price - 65\| |
| **Token Efficiency** | <2000/negotiation | Total tokens per deal |

## 🧪 Advanced Usage

### Running Specific Phases

```bash
# Phase 1: Quick validation (2 hours)
poetry run newsvendor-experiment --phase validation

# Phase 2: Statistical power (6 hours)  
poetry run newsvendor-experiment --phase power

# Phase 3: Full dataset (12+ hours)
poetry run newsvendor-experiment --phase full
```

### Custom Analysis

```python
from src.experiments.analyze_results import ResultsAnalyzer

# Load and analyze results
analyzer = ResultsAnalyzer('./data/processed/')
results = analyzer.load_complete_dataset()

# Custom analysis
price_analysis = analyzer.analyze_price_convergence()
model_comparison = analyzer.compare_model_performance()
reflection_impact = analyzer.analyze_reflection_benefit()
```

### Testing Model Concurrency

```bash
# Test concurrent model loading
poetry run python -c "
from src.core.model_manager import OptimizedModelManager
import asyncio

async def test():
    manager = OptimizedModelManager(max_concurrent_models=2)
    result = await manager.test_concurrency(['tinyllama:latest', 'qwen2:1.5b'])
    print(result)

asyncio.run(test())
"
```

## 🚨 Troubleshooting

### Common Issues

**Model loading fails:**
```bash
# Check Ollama is running
ollama list

# Verify model exists
ollama pull tinyllama:latest
```

**Memory errors:**
```bash
# Reduce concurrent models
export NEWSVENDOR_TECHNICAL__MAX_CONCURRENT_MODELS=1

# Monitor memory usage
poetry run python -c "import psutil; print(f'RAM: {psutil.virtual_memory().percent}%')"
```

**Price extraction issues:**
```bash
# Test price extraction
poetry run python tests/test_price_extraction.py

# Validate conversation tracking
poetry run python tests/test_conversation_tracker.py
```

### Performance Optimization

**For faster execution:**
- Use fewer models: `--models "tinyllama:latest,qwen2:1.5b"`
- Reduce replications in `config/models.yaml`
- Enable concurrent processing: `--concurrent 2`

**For better quality:**
- Increase token limits in model configs
- Add more reflection patterns
- Use longer timeout values

## 📊 Expected Results

### Anticipated Findings

1. **Reflection consistently improves** negotiation outcomes across model sizes
2. **Mid-range models with reflection** match large model performance at fraction of cost  
3. **Mixed-capability pairings** often outperform homogeneous pairs
4. **Token efficiency** shows diminishing returns beyond certain model sizes

### Deliverables

- **Complete dataset:** 1,940 negotiation transcripts with metadata
- **Analysis report:** Statistical findings with visualizations  
- **Performance benchmarks:** Model comparison across key metrics
- **Best practices guide:** Recommendations for LLM negotiation systems

## 🤝 Contributing

### Development Setup

```bash
# Install development dependencies
poetry install --with dev

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Format code
black src/ tests/
isort src/ tests/
```

### Adding New Models

1. Add model configuration to `config/models.yaml`
2. Test model loading: `poetry run newsvendor-validate --models "new_model:tag"`
3. Update replication matrix in `src/core/negotiation_engine.py`

### Adding New Analysis

1. Create analysis module in `src/analysis/`
2. Add to analysis pipeline in `src/experiments/analyze_results.py`
3. Update visualization generation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Work

- [Newsvendor Model Theory](https://en.wikipedia.org/wiki/Newsvendor_model)
- [Ollama Documentation](https://ollama.ai/docs)
- [LLM Negotiation Research](https://arxiv.org/search/?query=llm+negotiation)

## 📞 Support

For questions or issues:

1. **Check the troubleshooting section** above
2. **Run validation:** `poetry run newsvendor-validate --dry-run`
3. **Check logs:** `./newsvendor_experiment.log`
4. **Open an issue** with experiment logs and system info

---

**Status:** Production Ready v0.5 with Corrected Parameters  
**Last Updated:** December 2024