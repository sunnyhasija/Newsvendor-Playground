#!/bin/bash

# LLM Agent Newsvendor Negotiation Experiment v0.5
# Project Structure Setup Script

set -e

echo "🏪 Creating LLM Agent Newsvendor Negotiation Experiment v0.5"
echo "=============================================================="

# Create main project directory
PROJECT_DIR="newsvendor-llm-experiment"
echo "📁 Creating project directory: $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Create main directory structure
echo "📂 Creating directory structure..."

# Root level directories
mkdir -p config
mkdir -p src
mkdir -p experiments  
mkdir -p tests
mkdir -p data/{raw,processed,analysis,visualizations}

# Source code structure
mkdir -p src/core
mkdir -p src/agents
mkdir -p src/parsing
mkdir -p src/analysis
mkdir -p src/utils
mkdir -p src/experiments

# Test structure
mkdir -p tests/unit
mkdir -p tests/integration

echo "✅ Directory structure created"

# Create root level files
echo "📄 Creating root level files..."

touch pyproject.toml
touch README.md
touch Makefile
touch .gitignore
touch LICENSE

# Config files
echo "⚙️ Creating config files..."
touch config/models.yaml
touch config/experiment.yaml
touch config/prompts.yaml

# Source code files
echo "🐍 Creating source code files..."

# Core module
touch src/__init__.py
touch src/core/__init__.py
touch src/core/model_manager.py
touch src/core/negotiation_engine.py
touch src/core/conversation_tracker.py

# Agents module
touch src/agents/__init__.py
touch src/agents/buyer_agent.py
touch src/agents/supplier_agent.py
touch src/agents/reflection_mixin.py

# Parsing module
touch src/parsing/__init__.py
touch src/parsing/price_extractor.py
touch src/parsing/acceptance_detector.py
touch src/parsing/validation.py

# Analysis module
touch src/analysis/__init__.py
touch src/analysis/metrics_calculator.py
touch src/analysis/visualizations.py
touch src/analysis/statistical_tests.py

# Utils module
touch src/utils/__init__.py
touch src/utils/config_loader.py
touch src/utils/data_exporter.py
touch src/utils/logging_config.py
touch src/utils/error_recovery.py
touch src/utils/reproducibility.py

# Experiments
touch src/experiments/__init__.py
touch src/experiments/run_single_negotiation.py
touch src/experiments/run_validation_suite.py
touch src/experiments/run_full_experiment.py
touch src/experiments/analyze_results.py

# Test files
echo "🧪 Creating test files..."
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/integration/__init__.py

touch tests/unit/test_price_extractor.py
touch tests/unit/test_acceptance_detector.py
touch tests/unit/test_conversation_tracker.py
touch tests/unit/test_model_manager.py
touch tests/unit/test_negotiation_engine.py

touch tests/integration/test_full_negotiation.py
touch tests/integration/test_experiment_pipeline.py

# Create .gitignore content
echo "📝 Creating .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project specific
newsvendor_experiment.log
data/
results/
*.csv
*.json
*.parquet
*.gz
backups/

# Temporary files
*.tmp
*.temp
*.bak
EOF

# Create basic LICENSE
echo "📜 Creating LICENSE..."
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2024 Newsvendor Research Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

# Create initial __init__.py files with basic content
echo "🔧 Adding basic __init__.py content..."

cat > src/__init__.py << 'EOF'
"""
LLM Agent Newsvendor Negotiation Experiment v0.5
"""

__version__ = "0.5.0"
EOF

cat > src/core/__init__.py << 'EOF'
"""Core negotiation engine components."""
EOF

cat > src/agents/__init__.py << 'EOF'
"""Negotiation agents (buyer and supplier)."""
EOF

cat > src/parsing/__init__.py << 'EOF'
"""Price extraction and response parsing."""
EOF

cat > src/analysis/__init__.py << 'EOF'
"""Statistical analysis and visualization."""
EOF

cat > src/utils/__init__.py << 'EOF'
"""Utility functions and helpers."""
EOF

cat > src/experiments/__init__.py << 'EOF'
"""Experiment runners and orchestration."""
EOF

cat > tests/__init__.py << 'EOF'
"""Test suite for newsvendor experiment."""
EOF

# Create file structure summary
echo ""
echo "📋 PROJECT STRUCTURE CREATED:"
echo "=============================="

# Show the complete structure
find . -type f | sort | sed 's|./||' | while read file; do
    echo "  $file"
done

echo ""
echo "✅ Project structure setup complete!"
echo ""
echo "📝 NEXT STEPS:"
echo "1. Copy and paste the code content into each file"
echo "2. Initialize poetry: poetry init (or copy pyproject.toml content)"
echo "3. Install dependencies: poetry install"
echo "4. Set up development environment: make setup-dev"
echo "5. Download models: make download-models"
echo "6. Run validation: make run-validation"
echo ""
echo "🎯 Files ready for content:"
echo "  - pyproject.toml (Poetry configuration)"
echo "  - README.md (Project documentation)"
echo "  - Makefile (Development commands)"
echo "  - config/*.yaml (Configuration files)"
echo "  - src/**/*.py (Source code)"
echo "  - tests/**/*.py (Test files)"
echo ""
echo "Happy coding! 🚀"