#!/usr/bin/env python3
"""
setup_updated_experiment.py
Setup script to organize the updated experiment files in your project structure
Run this from the ROOT directory of your project
"""

import shutil
from pathlib import Path
import sys

def setup_experiment_files():
    """Set up the updated experiment files in the correct locations."""
    
    print("🔧 Setting up updated newsvendor experiment files...")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("src").exists():
        print("❌ Error: Please run this script from the ROOT directory of your project")
        print("   (the directory that contains the 'src' folder)")
        sys.exit(1)
    
    # Files to create in src/core/
    core_files = {
        "unified_model_manager.py": """# Content from the 'Updated Unified Model Manager' artifact
# Copy the full content from the first code artifact I provided
"""
    }
    
    # Files to create in src/agents/
    agent_files = {
        "standardized_agents.py": """# Content from the 'Standardized Agents' artifact  
# Copy the full content from the second code artifact I provided
"""
    }
    
    # Files to create in root directory
    root_files = [
        "run_validation_updated.py",
        "run_full_experiment_updated.py"
    ]
    
    print("📁 Creating/updating files:")
    
    # Create core files
    core_dir = Path("src/core")
    for filename, content in core_files.items():
        file_path = core_dir / filename
        print(f"   📝 {file_path}")
        # Note: You'll need to manually copy the actual content
    
    # Create agent files  
    agent_dir = Path("src/agents")
    for filename, content in agent_files.items():
        file_path = agent_dir / filename
        print(f"   📝 {file_path}")
        # Note: You'll need to manually copy the actual content
    
    # List root files to create
    for filename in root_files:
        print(f"   📝 ./{filename}")
    
    print("\n" + "=" * 60)
    print("📋 MANUAL SETUP REQUIRED:")
    print("=" * 60)
    
    print("\n1️⃣ CREATE src/core/unified_model_manager.py")
    print("   Copy the full content from the 'Updated Unified Model Manager' artifact")
    
    print("\n2️⃣ CREATE src/agents/standardized_agents.py") 
    print("   Copy the full content from the 'Standardized Agents' artifact")
    
    print("\n3️⃣ CREATE run_validation_updated.py (in root directory)")
    print("   Copy the full content from the 'Validation Experiment' artifact")
    
    print("\n4️⃣ CREATE run_full_experiment_updated.py (in root directory)")
    print("   Copy the full content from the 'Full Experiment Runner' artifact")
    
    print("\n5️⃣ INSTALL DEPENDENCIES (if needed):")
    print("   pip install tqdm boto3 openai python-dotenv")
    
    print("\n6️⃣ VERIFY .env FILE has credentials:")
    print("   AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY, AWS_REGION")
    print("   AZURE_o3_KEY, AZURE_o3_BASE")
    print("   claude_endpoint")
    
    print("\n✅ THEN RUN:")
    print("   python run_validation_updated.py    # Test all 10 models (~$2-5)")
    print("   python run_full_experiment_updated.py --dry-run  # See experiment plan") 
    print("   python run_full_experiment_updated.py  # Full experiment (~$150-400)")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    setup_experiment_files()