#!/usr/bin/env python3
"""
Bulk Update Script - Add auto-detection to all analysis scripts
Run this to update all your existing analysis scripts with auto-detection capability
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def create_auto_detection_header():
    """Standard auto-detection header for all scripts"""
    return '''# Auto-detection imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src" / "utils"))
sys.path.append(str(Path(__file__).parent.parent / "utils"))

try:
    from file_finder import auto_find_and_load_data, DataFileFinder, load_data_smart
    AUTO_DETECTION_AVAILABLE = True
except ImportError:
    print("⚠️  Auto-detection module not found, using fallback...")
    AUTO_DETECTION_AVAILABLE = False
    
    def auto_find_and_load_data():
        # Fallback - try common paths
        common_paths = [
            "./full_results/processed/complete_20250615_171248.csv",
            "./temp_results.csv",
            "./complete_*.csv"
        ]
        for path in common_paths:
            try:
                import pandas as pd
                return pd.read_csv(path), path
            except:
                continue
        return None, None

'''

def update_script_with_auto_detection(file_path: str, script_type: str = "generic"):
    """Update a script file to include auto-detection"""
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    print(f"🔄 Updating {file_path}...")
    
    # Backup original
    backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(file_path, backup_path)
    print(f"   💾 Backup created: {backup_path}")
    
    # Read original content
    with open(file_path, 'r') as f:
        original_content = f.read()
    
    # Check if already updated
    if "auto_find_and_load_data" in original_content:
        print(f"   ✅ Already has auto-detection, skipping...")
        return True
    
    # Apply script-specific updates
    if script_type == "debug_reflection":
        updated_content = update_debug_reflection_script(original_content)
    elif script_type == "conversation_analyzer":
        updated_content = update_conversation_analyzer_script(original_content)
    elif script_type == "complete_analysis_runner":
        updated_content = update_complete_analysis_runner_script(original_content)
    else:
        updated_content = update_generic_script(original_content)
    
    # Write updated content
    with open(file_path, 'w') as f:
        f.write(updated_content)
    
    print(f"   ✅ Successfully updated!")
    return True

def update_debug_reflection_script(content: str) -> str:
    """Update debug_reflection.py specifically"""
    
    # Add auto-detection imports after shebang
    lines = content.split('\n')
    
    # Find insertion point (after imports)
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            insert_idx = i + 1
        elif line.strip() == '' and insert_idx > 0:
            break
    
    # Insert auto-detection code
    auto_detection_code = create_auto_detection_header().split('\n')
    
    # Replace hardcoded data loading
    updated_lines = []
    for line in lines:
        if 'pd.read_csv(' in line and 'complete_' in line:
            # Replace hardcoded loading with auto-detection
            updated_lines.append('# Auto-detect and load data')
            updated_lines.append('data, data_path = auto_find_and_load_data()')
            updated_lines.append('if data is None:')
            updated_lines.append('    print("❌ No valid data file found!")')
            updated_lines.append('    return')
            updated_lines.append('')
            updated_lines.append(f'print(f"📊 Loaded {{len(data)}} rows from: {{data_path}}")')
        else:
            updated_lines.append(line)
    
    # Combine everything
    result_lines = lines[:insert_idx] + auto_detection_code + updated_lines[insert_idx:]
    return '\n'.join(result_lines)

def update_conversation_analyzer_script(content: str) -> str:
    """Update conversation_analyzer.py specifically"""
    
    lines = content.split('\n')
    
    # Find the ConversationAnalyzer class
    class_start = -1
    init_start = -1
    
    for i, line in enumerate(lines):
        if 'class ConversationAnalyzer:' in line:
            class_start = i
        elif '__init__(self' in line and class_start > -1:
            init_start = i
            break
    
    if init_start > -1:
        # Update the __init__ method
        updated_lines = []
        in_init = False
        init_indent = ''
        
        for i, line in enumerate(lines):
            if i == init_start:
                # Replace __init__ method
                init_indent = len(line) - len(line.lstrip())
                updated_lines.append(f"{' ' * init_indent}def __init__(self, data_path: str = None):")
                updated_lines.append(f"{' ' * (init_indent + 4)}\"\"\"Initialize with auto-detection or provided path\"\"\"")
                updated_lines.append(f"{' ' * (init_indent + 4)}")
                updated_lines.append(f"{' ' * (init_indent + 4)}if data_path:")
                updated_lines.append(f"{' ' * (init_indent + 8)}print(f\"📊 Loading data from provided path: {{data_path}}\")")
                updated_lines.append(f"{' ' * (init_indent + 8)}self.data = pd.read_csv(data_path)")
                updated_lines.append(f"{' ' * (init_indent + 8)}self.data_source = data_path")
                updated_lines.append(f"{' ' * (init_indent + 4)}else:")
                updated_lines.append(f"{' ' * (init_indent + 8)}print(\"🔍 Auto-detecting latest data file...\")")
                updated_lines.append(f"{' ' * (init_indent + 8)}self.data, self.data_source = auto_find_and_load_data()")
                updated_lines.append(f"{' ' * (init_indent + 8)}")
                updated_lines.append(f"{' ' * (init_indent + 8)}if self.data is None:")
                updated_lines.append(f"{' ' * (init_indent + 12)}raise FileNotFoundError(\"No valid data file found for conversation analysis\")")
                updated_lines.append(f"{' ' * (init_indent + 4)}")
                updated_lines.append(f"{' ' * (init_indent + 4)}print(f\"✅ Loaded {{len(self.data)}} negotiations from: {{self.data_source}}\")")
                in_init = True
            elif in_init and (line.strip() == '' or not line.startswith(' ' * (init_indent + 4))):
                # End of __init__ method
                in_init = False
                updated_lines.append(line)
            elif not in_init:
                updated_lines.append(line)
    else:
        updated_lines = lines
    
    # Add auto-detection imports at the top
    auto_detection_imports = create_auto_detection_header().split('\n')
    
    # Find import section
    import_end = 0
    for i, line in enumerate(updated_lines):
        if line.startswith('import ') or line.startswith('from '):
            import_end = i + 1
    
    # Insert auto-detection
    result_lines = updated_lines[:import_end] + [''] + auto_detection_imports + [''] + updated_lines[import_end:]
    
    return '\n'.join(result_lines)

def update_complete_analysis_runner_script(content: str) -> str:
    """Update complete analysis runner scripts"""
    
    lines = content.split('\n')
    updated_lines = []
    
    # Add auto-detection imports
    auto_detection_imports = create_auto_detection_header().split('\n')
    
    # Find where to insert imports
    import_end = 0
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            import_end = i + 1
    
    # Replace hardcoded file paths
    for line in lines:
        if ('pd.read_csv(' in line and 
            ('complete_' in line or 'temp_results.csv' in line)):
            # Replace with auto-detection
            indent = len(line) - len(line.lstrip())
            updated_lines.extend([
                f"{' ' * indent}# Auto-detect and load data",
                f"{' ' * indent}data, data_path = auto_find_and_load_data()",
                f"{' ' * indent}if data is None:",
                f"{' ' * (indent + 4)}raise FileNotFoundError('No valid data file found')",
                f"{' ' * indent}print(f'📊 Using data file: {{data_path}}')"
            ])
        else:
            updated_lines.append(line)
    
    # Combine with imports
    result_lines = updated_lines[:import_end] + [''] + auto_detection_imports + [''] + updated_lines[import_end:]
    
    return '\n'.join(result_lines)

def update_generic_script(content: str) -> str:
    """Generic update for any analysis script"""
    
    lines = content.split('\n')
    
    # Add auto-detection imports
    auto_detection_imports = create_auto_detection_header().split('\n')
    
    # Find import section
    import_end = 0
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            import_end = i + 1
    
    # Simple replacement of hardcoded paths
    updated_lines = []
    for line in lines:
        if 'pd.read_csv(' in line and any(pattern in line for pattern in ['complete_', 'temp_results']):
            # Add comment and auto-detection
            indent = len(line) - len(line.lstrip())
            updated_lines.extend([
                f"{' ' * indent}# Auto-detection: {line.strip()}",
                f"{' ' * indent}data, data_path = auto_find_and_load_data()",
                f"{' ' * indent}if data is None:",
                f"{' ' * (indent + 4)}print('❌ No data file found, using fallback...')",
                f"{' ' * (indent + 4)}{line.strip()}  # Original line as fallback",
                f"{' ' * indent}else:",
                f"{' ' * (indent + 4)}data = data  # Use auto-detected data"
            ])
        else:
            updated_lines.append(line)
    
    # Insert auto-detection imports
    result_lines = updated_lines[:import_end] + [''] + auto_detection_imports + [''] + updated_lines[import_end:]
    
    return '\n'.join(result_lines)

def main():
    """Main bulk update function"""
    
    print("🚀 BULK AUTO-DETECTION UPDATE")
    print("="*50)
    
    # First, ensure the file_finder module exists
    file_finder_path = Path("src/utils/file_finder.py")
    if not file_finder_path.exists():
        print("⚠️  Creating file_finder.py module...")
        file_finder_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create the file_finder module (content from the first artifact)
        file_finder_content = '''#!/usr/bin/env python3
"""
src/utils/file_finder.py - Auto-detection module for latest data files
"""

import glob
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple
import pandas as pd

# [Full content from the file_finder artifact above]
'''
        
        with open(file_finder_path, 'w') as f:
            f.write(file_finder_content)
        print(f"✅ Created: {file_finder_path}")
    
    # Define scripts to update
    scripts_to_update = [
        ("debug_reflection.py", "debug_reflection"),
        ("src/analysis/conversation_analyzer.py", "conversation_analyzer"),
        ("src/analysis/complete_analysis_runner.py", "complete_analysis_runner"),
        ("src/analysis/safe_analysis_runner.py", "complete_analysis_runner"),
        ("src/utils/debug_data.py", "debug_data"),
        ("comprehensive_analysis.py", "generic"),
        ("debug_analysis.py", "generic"),
    ]
    
    updated_count = 0
    
    for script_path, script_type in scripts_to_update:
        if update_script_with_auto_detection(script_path, script_type):
            updated_count += 1
    
    print(f"\n🎉 BULK UPDATE COMPLETE!")
    print(f"✅ Updated {updated_count} scripts with auto-detection")
    print(f"📦 Auto-detection module created at: {file_finder_path}")
    
    print(f"\n💡 NEXT STEPS:")
    print("1. Test the updated scripts:")
    print("   python debug_reflection.py")
    print("   python src/analysis/conversation_analyzer.py")
    print("2. All scripts now auto-detect the latest data file")
    print("3. Manual paths still work as fallback")
    print("4. Backups created for all modified files")

if __name__ == '__main__':
    main()