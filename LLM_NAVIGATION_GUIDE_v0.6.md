# 🤖 LLM Codebase Navigation Guide: Newsvendor Negotiation Experiment v0.6

## 📋 **DIRECTIONS FOR LLMs**

**Purpose**: This guide helps LLMs quickly understand the structure and purpose of the newsvendor-llm-experiment codebase without getting lost in the complexity.

**How to Use This Guide**:
1. **Start here** if you need to understand what this project does
2. **Use the file ratings** (⭐⭐⭐ system) to prioritize which files to read
3. **Follow the Quick Start paths** based on your specific goal
4. **Skip the experimental/debug files** unless debugging

**Target Audience**: LLMs that need to:
- Understand the research project quickly
- Modify or extend the codebase
- Run experiments or analysis
- Refactor or restructure parts of the code

---

## 🚨 **CRITICAL DISCOVERY: Turn Order Bias (v0.6 Priority)**

### **🔍 Major Finding: Buyers Always Go First**
**File**: `src/core/conversation_tracker.py`, Line 114:
```python
self.current_speaker = "buyer"  # Always starts with buyer
```

### **🎯 Research Impact**
The **$14.25 buyer advantage** finding may be **confounded** by turn order effects:
- **Current Explanation**: Literature bias (operations research is buyer-centric)
- **Alternative Explanation**: First-mover anchoring advantage
- **Reality**: Likely **both factors** contributing

### **⚠️ Methodological Concern**
All 8,000 existing negotiations have **buyers going first**, which means:
1. **Cannot separate** literature bias from anchoring effects
2. **Research conclusions** may be **overstated**
3. **Need controlled turn-order experiment** to validate findings

### **🚀 Solution: Turn Order Control System (v0.6)**
Implement configurable turn order to create **2×2 factorial design**:
- **Turn Order**: Buyer-first vs Supplier-first
- **Reflection**: With vs Without reflection

This will **isolate pure literature bias effects** from negotiation dynamics.

---

## 🔬 **CRITICAL: Prompt Analysis for Turn Order Control (v0.6)**

### **⚠️ Research Integrity Constraint**
**CANNOT CHANGE PROMPTS**: We already collected 8,000 negotiations over 3 days using specific prompts. Changing prompts would confound turn order with prompt engineering effects.

### **✅ Methodologically Sound Approach**
**ONLY CHANGE CONTEXT ASSIGNMENT**: Use identical prompts, only swap which role gets opening vs. response context.

---

## 📋 **DETAILED PROMPT ANALYSIS: Current vs. New**

### **🔍 BUYER PROMPTS - Position Comparison**

#### **BUYER CURRENT (Goes First)**
```yaml
PROMPT TEXT:
You are a retailer negotiating wholesale price with a supplier. You want the LOWEST possible price.

YOUR PRIVATE INFO (do not reveal):
- You sell at: $100 per unit  
- Demand: Normal distribution, mean 40 units, std 10
- Your profit = (100 - wholesale_price) × units_sold

RULES:
- Give SHORT responses only
- Make offers like "I offer $45" or "How about $38?"
- Accept good offers by saying "I accept $X"
- NO explanations, stories, or reasoning
- Price range: $1-99 only

CONTEXT: This is a new negotiation. Make your opening offer.

Your response (keep it under 15 words):
```

#### **BUYER NEW (Goes Second - Responds to Supplier)**
```yaml
PROMPT TEXT:
You are a retailer negotiating wholesale price with a supplier. You want the LOWEST possible price.

YOUR PRIVATE INFO (do not reveal):
- You sell at: $100 per unit
- Demand: Normal distribution, mean 40 units, std 10  
- Your profit = (100 - wholesale_price) × units_sold

RULES:
- Give SHORT responses only
- Make offers like "I offer $45" or "How about $38?"
- Accept good offers by saying "I accept $X"
- NO explanations, stories, or reasoning
- Price range: $1-99 only

CONTEXT: Latest: Supplier said "I want $70"

Your response (keep it under 15 words):
```

**Analysis**: ✅ **IDENTICAL PROMPT TEXT** - Only the context line changes from "Make your opening offer" to "Latest: Supplier said X"

---

### **🔍 SUPPLIER PROMPTS - Position Comparison**

#### **SUPPLIER CURRENT (Goes Second - Responds to Buyer)**
```yaml
PROMPT TEXT:
You are a supplier negotiating wholesale price with a retailer. You want the HIGHEST possible price above your costs.

YOUR PRIVATE INFO (do not reveal):
- Production cost: $30 per unit
- Your profit = (wholesale_price - 30) × units_sold

RULES:
- Give SHORT responses only
- Make offers like "I want $65" or "How about $58?"
- Accept good offers by saying "I accept $X"
- NO explanations, stories, or reasoning
- Price range: $31-200 only

CONTEXT: Latest: Buyer said "I offer $45"

Your response (keep it under 15 words):
```

#### **SUPPLIER NEW (Goes First)**
```yaml
PROMPT TEXT:
You are a supplier negotiating wholesale price with a retailer. You want the HIGHEST possible price above your costs.

YOUR PRIVATE INFO (do not reveal):
- Production cost: $30 per unit
- Your profit = (wholesale_price - 30) × units_sold

RULES:
- Give SHORT responses only
- Make offers like "I want $65" or "How about $58?"
- Accept good offers by saying "I accept $X"
- NO explanations, stories, or reasoning
- Price range: $31-200 only

CONTEXT: This is a new negotiation. Make your opening offer.

Your response (keep it under 15 words):
```

**Analysis**: ✅ **IDENTICAL PROMPT TEXT** - Only the context line changes from "Latest: Buyer said X" to "Make your opening offer"

---

### **🔍 REFLECTION PROMPTS - Position Comparison**

#### **BUYER REFLECTION CURRENT (Goes First)**
```yaml
<think>
Current negotiation status:
- Last supplier offer: none
- My last offer: none
- Round: 1/10

Quick analysis:
- Their offer suggests cost around: unknown
- My target profit margin: ~$30-40 per unit
- Should I: counter/accept/push harder?

Strategy: make opening bid
</think>

[SAME PROMPT TEXT AS ABOVE]
CONTEXT: This is a new negotiation. Make your opening offer.
```

#### **BUYER REFLECTION NEW (Goes Second)**
```yaml
<think>
Current negotiation status:
- Last supplier offer: $70
- My last offer: none
- Round: 2/10

Quick analysis:
- Their offer suggests cost around: $40-50
- My target profit margin: ~$30-40 per unit
- Should I: counter/accept/push harder?

Strategy: counter lower
</think>

[SAME PROMPT TEXT AS ABOVE]
CONTEXT: Latest: Supplier said "I want $70"
```

#### **SUPPLIER REFLECTION CURRENT (Goes Second)**
```yaml
<think>
Current negotiation status:
- Last buyer offer: $45
- My last offer: none
- Round: 2/10

Quick analysis:
- Their offer gives me profit of: $15
- Market seems to value around: $45-50
- Should I: counter/accept/hold firm?

Strategy: counter higher
</think>

[SAME PROMPT TEXT AS ABOVE]
CONTEXT: Latest: Buyer said "I offer $45"
```

#### **SUPPLIER REFLECTION NEW (Goes First)**
```yaml
<think>
Current negotiation status:
- Last buyer offer: none
- My last offer: none
- Round: 1/10

Quick analysis:
- Their offer gives me profit of: unknown
- Market seems to value around: unknown
- Should I: counter/accept/hold firm?

Strategy: make opening bid
</think>

[SAME PROMPT TEXT AS ABOVE]
CONTEXT: This is a new negotiation. Make your opening offer.
```

**Analysis**: ⚠️ **REFLECTION VARIABLES CHANGE** - Same template structure, but different data fills the variables.

---

## 🎯 **SUMMARY: What Changes vs. What Stays the Same**

### **✅ STAYS EXACTLY THE SAME:**
- **Role descriptions** ("You are a retailer..." / "You are a supplier...")
- **Private information** (selling price $100, cost $30, demand distribution)
- **Profit calculations** 
- **Rules text** (length limits, format requirements)
- **Examples** ("I offer $45", "I want $65", "I accept $X")
- **Price ranges** ($1-99 for buyer, $31-200 for supplier)
- **Response length limits** ("keep it under 15 words")
- **Reflection template structure** (`<think>` blocks with same variables)

### **🔄 ONLY CHANGES:**
- **Context line**: 
  - `"This is a new negotiation. Make your opening offer."` 
  - vs. `"Latest: [Partner] said [Offer]"`
- **Reflection variable values**: What specific data fills `{last_offer}`, `{round_number}`, etc.

### **🧠 Psychological Effect (This is the Research Question!)**
- **"Make your opening offer"** → Sets anchor, feels powerful, first-mover advantage
- **"Latest: Partner said X"** → Responds to anchor, feels reactive, second-mover position

**This psychological difference IS the turn order effect we want to measure.**

---

## 💻 **IMPLEMENTATION APPROACH**

### **Current Context Assignment (Buyer-First):**
```python
def get_conversation_history(self) -> str:
    if not self.turns:
        return "This is a new negotiation. Make your opening offer."  # Always to buyer first
    # ... response context to supplier
```

### **New Configurable Context Assignment:**
```python
def get_conversation_history(self, turn_order_strategy="buyer_first") -> str:
    if not self.turns:
        if turn_order_strategy == "supplier_first":
            if self.current_speaker == "supplier":
                return "This is a new negotiation. Make your opening offer."
            # Buyer gets response context in round 2
        else:  # buyer_first (default)
            if self.current_speaker == "buyer":
                return "This is a new negotiation. Make your opening offer."
            # Supplier gets response context in round 2
```

### **Research Integrity Maintained:**
1. **Same prompts** used in both conditions
2. **Pure turn order effect** - only variable changed is who goes first
3. **No prompt confounds** - no risk of introducing new biases
4. **Perfect comparability** with existing 8,000-negotiation dataset

---

## 🔬 **EXPECTED RESEARCH OUTCOMES**

### **Scenario 1: Pure Literature Bias**
- **Prediction**: Buyer advantage **persists** in supplier-first conditions
- **Conclusion**: Strong evidence for literature bias hypothesis
- **Implication**: Operations research training data creates systematic bias

### **Scenario 2: Pure Anchoring Bias**  
- **Prediction**: Buyer advantage **disappears** in supplier-first conditions
- **Conclusion**: Finding was due to turn order, not literature bias
- **Implication**: First-mover advantage is the dominant effect

### **Scenario 3: Mixed Effects (Most Likely)**
- **Prediction**: Buyer advantage **reduces but remains** in supplier-first
- **Conclusion**: Both literature bias AND anchoring effects present
- **Implication**: Multiple bias sources operate simultaneously

### **Quantitative Analysis:**
- **Literature Bias Component** = Buyer advantage remaining in supplier-first condition
- **Anchoring Bias Component** = Reduction in buyer advantage when suppliers go first
- **Total Bias** = Literature + Anchoring components

---

## 🎯 **Project Overview**

This is a sophisticated research project studying **LLM negotiation capabilities** in a buyer-supplier relationship using the classic **newsvendor problem** from operations research. The key discovery is a **systematic $14.25 buyer advantage**, suggesting LLMs inherit academic perspective biases from training literature.

### **Core Research Question**
Do LLMs inherit discipline-specific biases from academic literature, and can metacognitive reflection help overcome these biases?

### **Key Finding (Needs Validation)**
LLMs systematically favor buyers by $14.25 on average, but this could be due to:
1. **Literature bias** (operations research is buyer-centric)
2. **Turn order bias** (buyers always go first in current implementation)
3. **Both factors** working together

---

## 🏗️ **Project Architecture**

### **Experimental Design**
- **10 models** × **10 models** × **4 reflection patterns** × **20 replications** = **8,000 negotiations**
- **Game Setup**: Buyer knows selling price ($100) and demand; Supplier knows cost ($30); Optimal fair price is $65
- **Reflection Patterns**: 00 (none), 01 (buyer only), 10 (supplier only), 11 (both)

### **🧬 Deterministic Seeding System (GENIUS)**
**Location**: `run_full_experiment_with_throttling.py`, lines ~500
```python
negotiation_id = f"{config.buyer_model}_{config.supplier_model}_{config.reflection_pattern}_rep{rep:02d}"
```

**Why It's Brilliant**:
- **Perfect Reproducibility**: Every negotiation has unique, deterministic ID
- **Experimental Control**: Same parameters = same exact negotiation sequence  
- **Research Integrity**: Enables exact replication and validation
- **Debugging Power**: Can trace any individual negotiation precisely

### **v0.6 Extension to Seeding:**
```python
negotiation_id = f"{config.buyer_model}_{config.supplier_model}_{config.reflection_pattern}_{turn_order}_rep{rep:02d}"
```
This maintains reproducibility while adding turn order tracking.

### **Directory Structure**
```
newsvendor-llm-experiment/
├── src/
│   ├── core/          # Main orchestration engine
│   ├── agents/        # Buyer/supplier negotiation agents  
│   ├── parsing/       # Price extraction & acceptance detection
│   ├── analysis/      # Statistical analysis & visualization
│   └── utils/         # Configuration & data management
├── config/            # YAML configuration files
├── tests/             # Unit and integration tests
├── analysis/          # Generated reports and outputs
└── data/              # Experimental data storage
```

---

## 📋 **File Importance Classification**

### **🔥 CRITICAL CORE FILES** (Start Here)

1. **`src/core/unified_model_manager.py`** ⭐⭐⭐
   - **Purpose**: Manages both local (Ollama) and remote (Claude, O3, Grok) models
   - **Key Feature**: Smart throttling with progressive backoff for API rate limits
   - **Important**: Removed TinyLlama, added Grok-3-mini via Azure AI
   - **Generous Token Limits**: No artificial restrictions - lets models express naturally

2. **`src/core/conversation_tracker.py`** ⭐⭐⭐ **[CRITICAL for v0.6]**
   - **Purpose**: Bulletproof conversation state management
   - **TURN ORDER ISSUE**: Line 114 hardcodes `self.current_speaker = "buyer"`
   - **v0.6 Target**: Make turn order configurable via context assignment only
   - **Features**: Tracks rounds, detects termination, calculates profits
   - **Enhanced**: Fallback price extraction with local model assistance

3. **`src/agents/standardized_agents.py`** ⭐⭐⭐ **[NO CHANGES NEEDED for v0.6]**
   - **Purpose**: Buyer and supplier agents with standardized reflection prompts
   - **v0.6 Strategy**: Keep prompts identical, only change context assignment
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

6. **`run_full_experiment_with_throttling.py`** ⭐⭐⭐ **[SEEDING GENIUS HERE]**
   - **Purpose**: Main experiment runner with smart API throttling
   - **SEEDING LOCATION**: Lines ~500 - deterministic negotiation ID generation
   - **Features**: Progress tracking, cost estimation, progressive backoff
   - **Scale**: Handles 8,000 negotiations with concurrent execution

7. **`unified_analysis_runner.py`** ⭐⭐
   - **Purpose**: Comprehensive analysis suite combining statistical tests
   - **Output**: Publication-quality figures and detailed statistical reports

### **📊 ANALYSIS & METRICS**

8. **`src/analysis/metrics_calculator.py`** ⭐⭐ **[UPDATE NEEDED for v0.6]**
   - **Purpose**: Calculate key performance metrics (convergence, efficiency, reflection benefits)
   - **v0.6 Need**: Add turn-order analysis and control variables
   - **Research Focus**: Tests 4 main hypotheses about reflection and model effects

### **⚙️ CONFIGURATION FILES**

9. **`config/models.yaml`** ⭐⭐
   - **Models**: qwen2:1.5b through qwen3:latest + Claude/O3/Grok remotes
   - **Important**: TinyLlama removed, Grok added, generous token limits
   - **Tiers**: Ultra-compact → Compact → Mid → Large → Premium

10. **`config/experiment.yaml`** ⭐⭐ **[NEW CONFIG NEEDED for v0.6]**
    - **Game Parameters**: Selling price $100, cost $30, optimal $65
    - **Corrected**: Demand distribution changed from Uniform[50,150] to Normal(40,10)
    - **v0.6 Addition**: `turn_order_strategy: "buyer_first" | "supplier_first" | "random"`

11. **`config/prompts.yaml`** ⭐⭐ **[NO CHANGES for v0.6]**
    - **Standardized Prompts**: Uniform across all models with reflection variants
    - **Anti-chattiness**: Special handling for verbose models
    - **v0.6 Strategy**: Keep identical, only change context assignment

---

## 🚀 **v0.6 Development Plan: Turn Order Control**

### **📋 Implementation Strategy**

#### **Phase 1: Context Assignment Implementation** (1-2 days)
**Complexity**: Low-Medium (2/5)
**Files to Modify**:
1. **`conversation_tracker.py`**: 
   - Add `turn_order_strategy` parameter to `__init__`
   - Modify `get_conversation_history()` to use configurable context assignment
   - Keep all prompts identical
2. **Testing**: Run 10-20 negotiations to validate basic functionality

#### **Phase 2: Configurable Turn Order System** (2-3 days)  
**Complexity**: Easy-Medium (2/5)
**New Features**:
1. **Config Parameter**: Add `turn_order_strategy` to `experiment.yaml`
2. **Dynamic Selection**:
```python
def _determine_initial_speaker(self, strategy: str) -> str:
    if strategy == "buyer_first":
        return "buyer"
    elif strategy == "supplier_first": 
        return "supplier"
    elif strategy == "random":
        return random.choice(["buyer", "supplier"])
```
3. **Result Tracking**: Add `first_speaker` field to negotiation results
4. **Analysis Updates**: Control for turn order in statistical tests

#### **Phase 3: Validation Experiment** (2-3 days)
**Research Design**:
- **200 Supplier-First Negotiations**: Quick validation test
- **Statistical Comparison**: Compare with buyer-first subset
- **Replications**: 10 per model pair for initial validation

#### **Phase 4: Full 2×2 Factorial** (1 week)
**Comprehensive Design**:
- **Turn Order**: Buyer-first vs Supplier-first  
- **Reflection**: With vs Without
- **Models**: Focus on key representative models
- **Replications**: 20 per condition for statistical power

### **📊 Expected Research Outcomes**

**Scenario 1: Pure Literature Bias**
- Buyer advantage **persists** in supplier-first conditions
- **Conclusion**: Strong evidence for literature bias hypothesis

**Scenario 2: Pure Anchoring Bias**  
- Buyer advantage **disappears** in supplier-first conditions
- **Conclusion**: Finding was due to turn order, not literature bias

**Scenario 3: Mixed Effects** (Most Likely)
- Buyer advantage **reduces but remains** in supplier-first
- **Conclusion**: Both literature bias AND anchoring effects present

---

## 📂 **v0.6 Folder Structure Plan**

### **New Project: "negotiation-game-v0.6"**
```
playground/
├── newsvendor-llm-experiment/     # Original v0.5 (buyer-first only)
└── negotiation-game-v0.6/         # New controlled turn-order version
    ├── src/
    │   ├── core/
    │   │   ├── conversation_tracker.py      # ✅ Turn order configurable via context
    │   │   └── unified_model_manager.py     # ✅ Copy from v0.5
    │   ├── agents/
    │   │   └── standardized_agents.py       # ✅ Identical prompts from v0.5
    │   ├── parsing/                         # ✅ Copy from v0.5
    │   └── analysis/
    │       └── turn_order_analysis.py       # 🆕 Turn order effect analysis
    ├── config/
    │   ├── experiment.yaml                  # 🆕 turn_order_strategy param
    │   ├── models.yaml                      # ✅ Copy from v0.5
    │   └── prompts.yaml                     # ✅ IDENTICAL from v0.5
    ├── experiments/
    │   ├── run_turn_order_validation.py     # 🆕 Quick 200-negotiation test
    │   └── run_factorial_experiment.py      # 🆕 2×2 factorial design
    └── README_v0.6.md                       # 🆕 Turn order experiment docs
```

### **Key v0.6 Innovations**
1. **Preserve v0.5**: Keep original for reference and comparison
2. **Identical Prompts**: Maintain exact same prompt text for perfect control
3. **Context Assignment**: Only change which role gets opening vs. response context
4. **Deterministic Seeding**: Maintain exact same seeding system
5. **Research Ready**: Built for immediate hypothesis testing

---

## 🔍 **Key Changes During Development (Updated for v0.6)**

### **Model Changes**
- **❌ Removed**: `tinyllama:latest` (poor performance)
- **✅ Added**: `grok-remote` (Azure AI Services integration)
- **🔧 Fixed**: Game parameters (corrected demand distribution)

### **Technical Enhancements**
- **Smart Throttling**: Progressive backoff for API rate limits
- **Enhanced Extraction**: Local model fallback for difficult price parsing
- **Generous Limits**: Removed artificial token restrictions
- **Unified Reflection**: Standardized prompts across all model tiers

### **v0.6 Critical Enhancements**
- **🆕 Turn Order Control**: Configurable buyer-first vs supplier-first
- **🆕 Context Assignment**: Swap opening/response context only
- **🆕 Prompt Preservation**: Keep identical prompts for research integrity
- **🆕 Anchoring Analysis**: Separate literature bias from turn order effects  
- **🆕 Factorial Design**: 2×2 experiments for rigorous hypothesis testing
- **🆕 Research Validation**: Control for methodological confounds

### **Bug Fixes Identified**
- `fix_acceptance_pattern.py` - Fixed acceptance detection edge cases
- `fix_price_validation.py` - Corrected price validation logic
- Multiple enhancement iterations in price extraction

---

## 🧪 **Research Priorities for v0.6**

### **🎯 Immediate Priorities**
1. **Validate Turn Order Hypothesis**: Does supplier-first reduce buyer advantage?
2. **Quantify Anchoring vs Literature Effects**: Decompose the $14.25 advantage
3. **Preserve Reproducibility**: Maintain deterministic seeding system
4. **Rapid Prototyping**: Quick 200-negotiation validation tests

### **📊 Analysis Enhancements**
1. **Turn Order Controls**: Add `first_speaker` as statistical control variable
2. **Anchoring Metrics**: Measure first-offer influence on final prices
3. **Interaction Effects**: Test turn order × reflection interactions
4. **Robustness Checks**: Validate findings across different model pairs

### **🔬 Experimental Controls**
1. **Same Models**: Use identical model configurations from v0.5
2. **Same Game**: Keep newsvendor parameters identical
3. **Same Prompts**: Use exact same prompt text
4. **Same Seeding**: Maintain deterministic ID generation
5. **Only Variable**: Change turn order via context assignment

---

## 🎓 **Research Context (Updated)**

This codebase represents a comprehensive academic research framework studying the fundamental question of whether AI systems inherit biases from their training literature. The newsvendor problem serves as a controlled testbed because:

1. **Operations Research literature is buyer-centric** by definition
2. **Supplier strategies** are covered in marketing literature 
3. **LLMs trained on academic literature** might inherit this asymmetry
4. **Result**: Systematic $14.25 advantage for buyers across all models

**v0.6 Extension**: The discovery of turn order bias adds a crucial methodological dimension. By implementing turn order controls, we can:
- **Isolate** pure literature bias effects from anchoring
- **Validate** or **refute** the original literature bias hypothesis  
- **Strengthen** research conclusions with proper experimental controls
- **Advance** understanding of multiple bias sources in AI systems

---

## ⚠️ **Critical Implementation Notes**

### **🧬 Preserve the Seeding Genius**
The deterministic seeding system is **critical for reproducibility**:
```python
negotiation_id = f"{config.buyer_model}_{config.supplier_model}_{config.reflection_pattern}_rep{rep:02d}"
```
**v0.6 Extension**:
```python
negotiation_id = f"{config.buyer_model}_{config.supplier_model}_{config.reflection_pattern}_{turn_order}_rep{rep:02d}"
```
This maintains reproducibility while adding turn order tracking.

### **🔬 Research Integrity**
- **Document Everything**: Track which negotiations used which turn order
- **Version Control**: Maintain clear separation between v0.5 and v0.6 results
- **Statistical Controls**: Always control for turn order in analyses
- **Transparency**: Report both anchoring and literature bias effects
- **Prompt Preservation**: Use identical prompt text for perfect experimental control

---

## 🚀 **Development Timeline Estimate**

### **Week 1: Core Implementation**
- **Days 1-2**: Implement configurable context assignment in conversation tracker
- **Days 3**: Add turn order configuration parameters
- **Days 4-5**: Run 200-negotiation validation test
- **Days 6-7**: Debug and refine implementation

### **Week 2: Validation & Analysis**
- **Days 1-2**: Compare supplier-first vs. buyer-first results
- **Days 3-4**: Statistical analysis of turn order effects
- **Days 5-7**: Document findings and prepare factorial experiment

### **Week 3: Factorial Experiment**
- **Days 1-3**: Design and run 2×2 factorial experiment
- **Days 4-5**: Comprehensive analysis: literature vs. anchoring bias
- **Days 6-7**: Document results and research implications

### **Total Effort**: ~3 weeks for complete turn order control system with validated results

---

**Last Updated**: v0.6 Detailed Implementation Planning with Prompt Analysis
**Next Steps**: Implement context assignment system with identical prompts
**Research Impact**: Could definitively isolate literature bias from anchoring effects