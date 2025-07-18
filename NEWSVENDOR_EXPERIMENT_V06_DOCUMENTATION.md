# Newsvendor LLM Experiment v0.6 - Complete Technical Documentation

## 📋 **PROJECT OVERVIEW**

### **Purpose**
The Newsvendor LLM Experiment is a research system designed to study negotiation behaviors and biases in Large Language Models (LLMs) through a standardized economic game. Version 0.6 introduces **Turn Order Control** to separate literature bias from anchoring effects in LLM negotiations.

### **Research Innovation (v0.6)**
- **Problem**: Previous experiments showed a $14.25 buyer advantage, but couldn't distinguish between:
  - **Literature Bias**: LLMs favor buyers due to training data
  - **Anchoring Bias**: First-mover advantage (buyers always went first)
- **Solution**: Configurable turn order with identical prompts to isolate bias sources

### **Core Research Question**
*"Does the buyer advantage stem from literature bias in LLM training data, or from anchoring effects due to buyers always going first?"*

---

## 🏗️ **PROJECT ARCHITECTURE**

### **Directory Structure**
```
newsvendor-llm-experiment-supplier-first/
├── README.md                                   # Quick start guide
├── run_turn_order_experiment.py               # Main experiment runner
├── validate_turn_order_system.py              # Validation test suite
├── config/                                     # Configuration files
│   ├── experiment.yaml                        # Experiment parameters
│   ├── models.yaml                            # Model definitions
│   └── prompts.yaml                           # Standardized prompts
├── src/                                        # Core implementation
│   ├── core/                                  # Core negotiation logic
│   │   ├── conversation_tracker.py           # Turn order control system
│   │   └── unified_model_manager.py          # Model interface management
│   ├── agents/                                # LLM negotiation agents
│   │   └── standardized_agents.py            # Buyer/supplier agents
│   ├── parsing/                               # Response processing
│   │   ├── enhanced_price_extractor.py       # Price extraction from text
│   │   └── acceptance_detector.py            # Termination detection
│   ├── utils/                                 # Utility functions
│   │   ├── config_loader.py                  # Configuration management
│   │   └── data_exporter.py                  # Results export and analysis
│   └── analysis/                              # Analysis framework
│       └── __init__.py                        # Analysis modules
└── experiments/                               # Output directory (created at runtime)
```

---

## 🎮 **GAME MECHANICS**

### **The Newsvendor Problem**
A classic operations research scenario where:
- **Buyer (Retailer)**: Wants to buy products at the lowest wholesale price
- **Supplier (Manufacturer)**: Wants to sell at the highest wholesale price
- **Constraint**: Both need to agree on a price to make a deal

### **Game Parameters**
```yaml
game:
  selling_price: 100        # Retail price (buyer's private info)
  production_cost: 30       # Manufacturing cost (supplier's private info)
  demand_mean: 40          # Expected demand (units)
  demand_std: 10           # Demand uncertainty
  optimal_price: 65        # Fair split-the-difference price
```

### **Profit Calculations**
- **Buyer Profit**: `(selling_price - wholesale_price) × units_sold`
- **Supplier Profit**: `(wholesale_price - production_cost) × units_sold`
- **Optimal Solution**: Both parties maximize profit at ~$65 wholesale price

### **Negotiation Rules**
- **Max Rounds**: 10 turns per negotiation
- **Price Ranges**: Buyer ($1-99), Supplier ($31-200)
- **Termination**: Explicit acceptance, price convergence, or timeout
- **Turn Order**: Configurable (v0.6 innovation)

---

## 🔧 **CORE COMPONENTS**

### **1. `run_turn_order_experiment.py` - Main Experiment Runner**

**Purpose**: Orchestrates the complete turn order controlled experiment

**Key Classes**:
- `TurnOrderExperimentConfig`: Configuration for individual experiments
- `TurnOrderExperimentRunner`: Main experiment execution engine

**Core Capabilities**:
- **Experiment Planning**: Generate 2×2 factorial designs
- **Model Management**: Handle 10 different LLMs (local + remote)
- **Concurrent Execution**: Run multiple negotiations simultaneously
- **Cost Estimation**: Predict experiment costs and duration
- **Progress Tracking**: Real-time monitoring with tqdm progress bars
- **Results Analysis**: Research-focused statistical analysis

**Usage Options**:
```bash
# Quick validation test
python run_turn_order_experiment.py --models qwen2:1.5b,gemma2:2b --replications 3

# Full experiment with all models
python run_turn_order_experiment.py --full-experiment

# Custom experiment
python run_turn_order_experiment.py \
  --models claude-sonnet-4-remote,o3-remote \
  --strategies buyer_first,supplier_first \
  --patterns 00,11 \
  --replications 20

# Dry run (planning only)
python run_turn_order_experiment.py --dry-run --full-experiment
```

**Command Line Options**:
- `--models`: Comma-separated model list (default: first 3 models)
- `--strategies`: Turn order strategies (default: buyer_first,supplier_first)
- `--patterns`: Reflection patterns (default: 00,11)
- `--replications`: Number of replications per condition (default: 5)
- `--concurrent`: Max concurrent negotiations (default: 1)
- `--dry-run`: Show plan without execution
- `--full-experiment`: Run complete experiment with all models

### **2. `src/core/conversation_tracker.py` - Turn Order Control System**

**Purpose**: Core innovation for v0.6 - manages conversation state with configurable turn order

**Key Features**:
- **Turn Order Control**: `_determine_initial_speaker()` method
- **Conversation History**: Context-aware prompt generation
- **Termination Detection**: Multiple termination condition checks
- **State Management**: Comprehensive negotiation state tracking
- **Price Extraction**: Enhanced price parsing with fallback
- **Research Metadata**: Turn order tracking for analysis

**Turn Order Strategies**:
```python
def _determine_initial_speaker(self, strategy: str) -> str:
    if strategy == "buyer_first": return "buyer"
    elif strategy == "supplier_first": return "supplier" 
    elif strategy == "random": return random.choice(["buyer", "supplier"])
```

**Context Generation** (v0.6 Critical Innovation):
```python
def get_conversation_history(self) -> str:
    if not self.turns:
        # Opening context - depends on current speaker
        return "This is a new negotiation. Make your opening offer."
    
    # Response context with conversation history
    # ... build history from previous turns
```

**Data Structures**:
- `NegotiationTurn`: Individual turn data
- `NegotiationResult`: Complete negotiation outcome with turn order metadata
- `ConversationTracker`: Main state management class

### **3. `src/agents/standardized_agents.py` - LLM Negotiation Agents**

**Purpose**: Standardized buyer and supplier agents with identical prompt structures

**Key Classes**:
- `StandardizedBuyerAgent`: Retailer role with profit maximization
- `StandardizedSupplierAgent`: Manufacturer role with profit maximization

**Agent Capabilities**:
- **Role-Specific Prompts**: Tailored to buyer/supplier perspectives
- **Reflection Support**: Optional `<think>` blocks for reasoning
- **Context Awareness**: Previous negotiation history integration
- **Price Constraints**: Role-appropriate price range validation
- **Token Efficiency**: Optimized for <15 word responses

**Reflection Patterns**:
- `"00"`: No reflection for either agent
- `"01"`: Supplier reflects, buyer doesn't
- `"10"`: Buyer reflects, supplier doesn't  
- `"11"`: Both agents reflect

### **4. `src/parsing/enhanced_price_extractor.py` - Price Extraction**

**Purpose**: Robust price extraction from natural language negotiation messages

**Features**:
- **Multi-Pattern Matching**: Regex + NLP + fallback model
- **Context Awareness**: Uses conversation history for disambiguation
- **Local Model Fallback**: Uses local LLM for complex cases
- **Confidence Scoring**: Reliability metrics for extractions
- **Performance Tracking**: Success rate monitoring

**Extraction Methods**:
1. **Regex Patterns**: Common price formats ($50, 50 dollars, etc.)
2. **Contextual NLP**: Sentence structure analysis
3. **LLM Fallback**: Local model for complex negotiations
4. **Validation**: Cross-reference with previous offers

### **5. `src/parsing/acceptance_detector.py` - Termination Detection**

**Purpose**: Detect when negotiations should terminate

**Termination Types**:
- `EXPLICIT_ACCEPTANCE`: Direct "I accept" statements
- `PRICE_CONVERGENCE`: Prices within $1-2 of each other
- `TIMEOUT`: Maximum rounds reached
- `FAILURE`: Errors or parsing failures

**Detection Methods**:
- **Keyword Matching**: "accept", "deal", "agreed", etc.
- **Price Analysis**: Convergence pattern detection
- **Context Validation**: Ensure termination is appropriate

### **6. `src/utils/config_loader.py` - Configuration Management**

**Purpose**: Load and manage experiment configuration with turn order support

**Functions**:
- `load_config()`: Load YAML configuration files
- `get_turn_order_strategy()`: Extract turn order setting
- `set_turn_order_strategy()`: Update turn order configuration
- `validate_config()`: Ensure configuration consistency

### **7. `src/utils/data_exporter.py` - Results Export and Analysis**

**Purpose**: Export results and generate research-focused analysis

**Export Formats**:
- **JSON**: Complete negotiation data with metadata
- **CSV**: Flattened data for statistical analysis
- **Parquet**: Efficient storage for large datasets

**Analysis Features**:
- **Turn Order Comparison**: Buyer-first vs supplier-first analysis
- **Bias Decomposition**: Literature bias vs anchoring effect calculation
- **Hypothesis Testing**: H1, H2, H3 statistical testing
- **Research Metrics**: Publication-ready statistics

### **8. `validate_turn_order_system.py` - Validation Test Suite**

**Purpose**: Comprehensive validation of turn order control system

**Test Coverage**:
- **Configuration Loading**: YAML parsing and turn order settings
- **Conversation Tracker**: Turn order strategy implementation
- **Context Generation**: Opening vs response prompt generation
- **System Integration**: End-to-end negotiation with turn order control

**Validation Checks**:
```python
async def test_configuration_loading()     # Config system tests
async def test_conversation_tracker()     # Turn order logic tests  
async def validate_turn_order_system()    # Full system integration test
```

---

## ⚙️ **CONFIGURATION OPTIONS**

### **`config/experiment.yaml` - Main Configuration**

**Turn Order Settings** (v0.6):
```yaml
turn_order:
  strategy: "buyer_first"  # Options: buyer_first, supplier_first, random
  validation_enabled: true
  track_first_speaker: true
```

**Game Parameters**:
```yaml
game:
  selling_price: 100
  production_cost: 30
  demand_distribution:
    type: "normal"
    mean: 40
    std: 10
  optimal_wholesale_price: 65
```

**Negotiation Rules**:
```yaml
negotiation:
  max_rounds: 10
  timeout_seconds: 60
  price_range:
    min: 1
    max: 200
  buyer_price_range:
    min: 1
    max: 99
  supplier_price_range:
    min: 31
    max: 200
```

**Experimental Design**:
```yaml
experimental_design:
  turn_order_experiment:
    turn_orders: ["buyer_first", "supplier_first"]
    reflection_conditions: ["without_reflection", "with_reflection"]
    replications_per_cell: 20
  
  research_hypotheses:
    H1: "Buyer advantage persists in supplier-first conditions (literature bias)"
    H2: "Buyer advantage disappears in supplier-first conditions (anchoring bias only)"
    H3: "Buyer advantage reduces but remains in supplier-first (mixed effects)"
```

### **`config/models.yaml` - Model Configuration**

**Available Models** (10 total):
```yaml
local_models:
  - qwen2:1.5b          # Fast, lightweight
  - gemma2:2b           # Google's Gemma
  - phi3:mini           # Microsoft's Phi-3
  - llama3.2:latest     # Meta's latest Llama
  - mistral:instruct    # Mistral instruction-tuned
  - qwen:7b             # Larger Qwen model
  - qwen3:latest        # Latest Qwen version

remote_models:
  - claude-sonnet-4-remote    # Anthropic's Claude
  - o3-remote                 # OpenAI's O3
  - grok-remote              # xAI's Grok
```

**Model Properties**:
- **Local Models**: Run on local hardware, faster but less capable
- **Remote Models**: API-based, more capable but costlier
- **Cost Estimates**: Built-in cost tracking for budget management

### **`config/prompts.yaml` - Standardized Prompts**

**Buyer Prompts**:
```yaml
buyer_prompts:
  no_reflection: |
    You are a retailer negotiating wholesale price with a supplier.
    You want the LOWEST possible price.
    
    YOUR PRIVATE INFO (do not reveal):
    - You sell at: $100 per unit
    - Demand: Normal distribution, mean 40 units, std 10
    - Your profit = (100 - wholesale_price) × units_sold
    
    Current situation: {context}
    
    Your response (keep it under 15 words):
```

**Supplier Prompts**:
```yaml
supplier_prompts:
  no_reflection: |
    You are a manufacturer negotiating wholesale price with a retailer.
    You want the HIGHEST possible price.
    
    YOUR PRIVATE INFO (do not reveal):
    - Your production cost: $30 per unit
    - Demand: Normal distribution, mean 40 units, std 10
    - Your profit = (wholesale_price - 30) × units_sold
    
    Current situation: {context}
    
    Your response (keep it under 15 words):
```

**Reflection Templates**:
```yaml
reflection_templates:
  buyer_reflection: |
    <think>
    They want ${other_price}. I can profit $(100-wholesale_price)*40 units.
    If I pay ${other_price}, I profit ${(100-other_price)*40}.
    Should I accept, counter, or walk away?
    </think>
```

---

## 🧪 **EXPERIMENTAL DESIGN**

### **2×2 Factorial Design**

**Factors**:
1. **Turn Order**: buyer_first vs supplier_first
2. **Reflection**: without_reflection (00) vs with_reflection (11)

**Experimental Cells**:
1. **buyer_first + no_reflection**: Baseline condition (replicates v0.5)
2. **buyer_first + reflection**: Reflection control
3. **supplier_first + no_reflection**: Turn order test condition
4. **supplier_first + reflection**: Full factorial condition

**Sample Sizes**:
- **Model Pairs**: 10×10 = 100 unique combinations
- **Replications**: 20 per cell for statistical power
- **Total**: 4 cells × 100 model pairs × 20 reps = 8,000 negotiations

### **Research Hypotheses**

**H1 - Literature Bias Only**:
- **Prediction**: Buyer advantage persists when suppliers go first
- **Evidence**: Buyer advantage ≥ $8 in supplier-first conditions
- **Interpretation**: LLM training data systematically favors buyers

**H2 - Anchoring Bias Only**:
- **Prediction**: Buyer advantage disappears when suppliers go first
- **Evidence**: Buyer advantage ≤ $2 in supplier-first conditions
- **Interpretation**: Original finding was pure first-mover advantage

**H3 - Mixed Effects**:
- **Prediction**: Buyer advantage reduces but remains when suppliers go first
- **Evidence**: $2 < buyer advantage < $8 in supplier-first conditions
- **Interpretation**: Both literature bias and anchoring contribute

### **Bias Decomposition Framework**

**Literature Bias Component** = Buyer advantage remaining in supplier-first condition
**Anchoring Bias Component** = Reduction in buyer advantage when suppliers go first
**Total Original Bias** = Literature + Anchoring components

---

## 📊 **OUTPUT AND ANALYSIS**

### **Data Organization**
```
experiments/
├── raw/                                    # Raw negotiation transcripts
├── processed/                              # Flattened analysis-ready data
├── analysis/                               # Statistical analysis results
├── turn_order_analysis/                    # v0.6 Turn order specific analysis
├── buyer_first/                            # Buyer-first condition results
├── supplier_first/                         # Supplier-first condition results
└── comparative_analysis/                   # Between-condition comparisons
```

### **Key Output Files**

**Results Files**:
- `comprehensive_turn_order_analysis_{timestamp}.json` - Complete research analysis
- `turn_order_comparison_{timestamp}.json` - Direct strategy comparison
- `{phase}_{timestamp}_buyer_first.json` - Buyer-first condition data
- `{phase}_{timestamp}_supplier_first.json` - Supplier-first condition data

**Analysis Schema**:
```json
{
  "bias_decomposition": {
    "buyer_first_mean_price": 67.5,
    "supplier_first_mean_price": 65.2,
    "buyer_advantage_when_supplier_goes_first": 2.3,
    "literature_bias_evidence": true,
    "anchoring_effect_size": 2.3
  },
  "hypothesis_testing": {
    "H1_literature_bias": {"evidence": false, "strength": "weak"},
    "H2_anchoring_only": {"evidence": false, "strength": "weak"},
    "H3_mixed_effects": {"evidence": true, "strength": "strong"}
  },
  "interpretation": "Mixed effects detected. Both literature bias and anchoring contribute."
}
```

### **Statistical Analysis Features**

**Built-in Tests**:
- Chi-squared tests for categorical outcomes
- Welch's t-tests for price comparisons
- Mann-Whitney U for non-parametric comparisons
- ANOVA for multi-factor analysis
- Cohen's d for effect size calculation
- Turn order specific tests (v0.6)

**Research Metrics**:
- Success rate by turn order strategy
- Price optimality (distance from $65)
- Negotiation efficiency (tokens used)
- Convergence patterns
- Model-specific performance
- Bias decomposition coefficients

---

## 💰 **COST AND PERFORMANCE**

### **Cost Estimates**

**Remote Model Costs** (per 1000 tokens):
- Claude Sonnet 4: $0.075
- O3: $0.240  
- Grok: $0.020

**Typical Experiment Costs**:
- **Small Test** (3 models, 5 reps): ~$5-10
- **Validation Study** (5 models, 10 reps): ~$25-50
- **Full Research Dataset** (10 models, 20 reps): ~$100-200

### **Performance Estimates**

**Duration** (with API throttling):
- **Small Test**: 15-30 minutes
- **Validation Study**: 2-4 hours
- **Full Dataset**: 12-24 hours

**Resource Requirements**:
- **Local Models**: 8GB+ RAM, modern CPU
- **Remote Models**: Stable internet, API keys
- **Storage**: ~150MB for full dataset

---

## 🔒 **RESEARCH INTEGRITY FEATURES**

### **Perfect Experimental Control**

**Identical Prompts**: 
- Same prompt text in both turn order conditions
- Only context assignment changes (opening vs response)
- Zero prompt engineering confounds

**Deterministic Seeding**:
```python
negotiation_id = f"{buyer_model}_{supplier_model}_{reflection_pattern}_{turn_order_strategy}_rep{rep:02d}"
```

**Reproducible Results**:
- Same random seeds for identical conditions
- Consistent model parameters
- Complete metadata tracking

### **Validation Framework**

**Multi-Level Validation**:
1. **Configuration Validation**: YAML parsing and turn order settings
2. **Component Testing**: Individual module functionality
3. **Integration Testing**: End-to-end system validation
4. **Research Validation**: Turn order effect detection

**Quality Controls**:
- Price extraction validation
- Termination condition verification
- Turn alternation enforcement
- Cost tracking and budget controls

---

## 🚀 **USAGE SCENARIOS**

### **1. Quick System Validation**
```bash
python validate_turn_order_system.py
```
- **Purpose**: Verify system functionality
- **Duration**: 2-3 minutes
- **Output**: Pass/fail validation report

### **2. Research Pilot Study**
```bash
python run_turn_order_experiment.py \
  --models qwen2:1.5b,gemma2:2b,phi3:mini \
  --replications 10
```
- **Purpose**: Preliminary research findings
- **Duration**: 30-60 minutes
- **Output**: Turn order effects analysis

### **3. Full Research Deployment**
```bash
python run_turn_order_experiment.py --full-experiment
```
- **Purpose**: Publication-quality dataset
- **Duration**: 12-24 hours
- **Output**: Complete bias decomposition analysis

### **4. Custom Research Questions**
```bash
python run_turn_order_experiment.py \
  --models claude-sonnet-4-remote,o3-remote \
  --strategies buyer_first,supplier_first \
  --patterns 00 \
  --replications 20
```
- **Purpose**: Targeted hypothesis testing
- **Duration**: Variable based on scope
- **Output**: Focused analysis results

---

## 🎯 **RESEARCH APPLICATIONS**

### **Academic Research**
- **AI Bias Studies**: Systematic bias identification in LLMs
- **Behavioral Economics**: Human-AI negotiation comparisons
- **Operations Research**: Supply chain negotiation modeling
- **Experimental Methodology**: Rigorous AI experimentation templates

### **Industry Applications**
- **Fair AI Systems**: Guidelines for unbiased LLM negotiations
- **Procurement Optimization**: AI-assisted supply chain negotiations
- **Contract Negotiation**: Automated negotiation system design
- **Bias Mitigation**: Targeted interventions for fair outcomes

### **Methodological Contributions**
- **Experimental Control**: Template for rigorous AI bias research
- **Bias Decomposition**: Framework for separating bias sources
- **Turn Order Analysis**: Novel approach to anchoring effect research
- **Reproducible Research**: Complete system for replication studies

---

## 📚 **TECHNICAL IMPLEMENTATION NOTES**

### **Critical Design Decisions**

**1. Context Assignment Strategy**: 
- Preserves identical prompts while controlling turn order
- Eliminates prompt engineering confounds
- Maintains research validity

**2. Deterministic Seeding**:
- Enables perfect reproducibility
- Supports hypothesis testing
- Facilitates meta-analysis

**3. Enhanced Price Extraction**:
- Robust parsing with multiple fallback methods
- Confidence scoring for reliability
- Performance monitoring for quality control

**4. Modular Architecture**:
- Separation of concerns for maintainability
- Pluggable components for extensibility
- Clear interfaces for integration

### **Future Extensions**

**Potential Enhancements**:
- Multi-round reflection patterns
- Dynamic pricing strategies
- Cross-cultural negotiation styles
- Real-time human-AI negotiations
- Advanced statistical modeling

**Research Directions**:
- Industry-specific negotiation scenarios
- Multi-party negotiations
- Long-term relationship modeling
- Fairness metric development
- Bias mitigation techniques

---

## ⚠️ **IMPORTANT USAGE NOTES**

### **Research Integrity Requirements**
1. **Never modify prompts** - Any changes invalidate experimental control
2. **Preserve deterministic seeding** - Maintain exact reproducibility  
3. **Document all changes** - Any modifications must be clearly documented
4. **Validate before research** - Always run validation suite before experiments

### **Best Practices**
1. **Start with validation** - Always run `validate_turn_order_system.py` first
2. **Small tests first** - Begin with 2-3 models and 3-5 replications
3. **Monitor costs** - Track spending with remote models
4. **Backup results** - Save all research data with version control
5. **Document experiments** - Maintain detailed experiment logs

---

**Version**: 0.6 - Turn Order Control for Literature Bias Research  
**Status**: Production Ready - Complete Implementation  
**Last Updated**: Implementation Complete and Validated