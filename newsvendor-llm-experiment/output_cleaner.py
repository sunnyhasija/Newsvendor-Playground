
#!/usr/bin/env python3
"""
Output Cleaner for Newsvendor LLM Experiment Analysis
Organizes generated analysis files into a structured folder system

This script:
1. Scans the root directory for analysis output files
2. Extracts timestamps from filenames
3. Creates organized folder structure: outputs/YYYY-MM-DD_HHMMSS/type/
4. Moves files to appropriate folders based on type and timestamp
5. Creates summary reports of what was moved
"""

import os
import shutil
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json

class OutputCleaner:
    """Organize analysis output files into structured folders"""
    
    def __init__(self, base_dir: str = ".", output_dir: str = "outputs"):
        """
        Initialize the output cleaner
        
        Args:
            base_dir: Directory to scan for files (default: current directory)
            output_dir: Base output directory (default: "outputs")
        """
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # File type mappings
        self.file_types = {
            '.png': 'images',
            '.jpg': 'images', 
            '.jpeg': 'images',
            '.svg': 'images',
            '.html': 'interactive',
            '.md': 'reports',
            '.txt': 'reports',
            '.json': 'data',
            '.csv': 'data',
            '.xlsx': 'data'
        }
        
        # Analysis type patterns (for better organization)
        self.analysis_patterns = {
            'comprehensive': r'comprehensive.*',
            'personality': r'personality.*',
            'tournament': r'tournament.*',
            'reflection': r'reflection.*',
            'conversation': r'conversation.*',
            'statistical': r'statistical.*',
            'figure': r'figure\d+.*',
            'dashboard': r'.*dashboard.*',
            'infographic': r'.*infographic.*'
        }
        
        # Timestamp pattern to extract from filenames
        self.timestamp_pattern = r'(\d{8}_\d{6})'
        
    def scan_files(self):
        """Scan directory for analysis output files"""
        
        print("🔍 Scanning for analysis output files...")
        
        # Patterns to identify analysis files
        analysis_file_patterns = [
            r'comprehensive.*\.(png|html|md|json|csv)',
            r'personality.*\.(png|html|md|json|csv)',
            r'tournament.*\.(png|html|md|json|csv)',
            r'reflection.*\.(png|html|md|json|csv)',
            r'conversation.*\.(png|html|md|json|csv)',
            r'statistical.*\.(png|html|md|json|csv)',
            r'figure\d+.*\.(png|html|md)',
            r'.*dashboard.*\.(png|html|md)',
            r'.*infographic.*\.(png|html|md)',
            r'.*_\d{8}_\d{6}\.(png|html|md|json|csv)',  # Any file with timestamp
        ]
        
        found_files = []
        
        for file_path in self.base_dir.iterdir():
            if file_path.is_file():
                filename = file_path.name
                
                # Check if file matches any analysis pattern
                for pattern in analysis_file_patterns:
                    if re.match(pattern, filename, re.IGNORECASE):
                        found_files.append(file_path)
                        break
        
        print(f"✅ Found {len(found_files)} analysis files")
        return found_files
    
    def extract_timestamp(self, filename: str):
        """Extract timestamp from filename"""
        
        match = re.search(self.timestamp_pattern, filename)
        if match:
            timestamp_str = match.group(1)
            try:
                # Parse timestamp: YYYYMMDD_HHMMSS
                timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                return timestamp, timestamp_str
            except ValueError:
                pass
        
        # If no timestamp found, use file modification time
        return None, None
    
    def get_file_category(self, file_path: Path):
        """Determine file category based on name patterns"""
        
        filename = file_path.name.lower()
        
        # Check analysis type patterns
        for analysis_type, pattern in self.analysis_patterns.items():
            if re.match(pattern, filename):
                return analysis_type
        
        # Default to 'misc' if no pattern matches
        return 'misc'
    
    def create_folder_structure(self, timestamp_str: str, categories: set):
        """Create organized folder structure"""
        
        # Create base folder with timestamp
        session_folder = self.output_dir / timestamp_str
        session_folder.mkdir(exist_ok=True)
        
        # Create subfolders for each file type and category
        created_folders = []
        
        for category in categories:
            category_folder = session_folder / category
            category_folder.mkdir(exist_ok=True)
            created_folders.append(category_folder)
            
            # Create type subfolders
            for file_type in self.file_types.values():
                type_folder = category_folder / file_type
                type_folder.mkdir(exist_ok=True)
                created_folders.append(type_folder)
        
        return session_folder, created_folders
    
    def organize_files(self, files: list):
        """Organize files into structured folders"""
        
        print("\n📁 Organizing files...")
        
        # Group files by timestamp
        timestamp_groups = defaultdict(list)
        no_timestamp_files = []
        
        for file_path in files:
            timestamp, timestamp_str = self.extract_timestamp(file_path.name)
            
            if timestamp_str:
                timestamp_groups[timestamp_str].append((file_path, timestamp))
            else:
                no_timestamp_files.append(file_path)
        
        # Handle files without timestamps (use current time)
        if no_timestamp_files:
            current_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            timestamp_groups[current_timestamp].extend([
                (file_path, datetime.now()) for file_path in no_timestamp_files
            ])
        
        organization_summary = {}
        
        # Process each timestamp group
        for timestamp_str, file_list in timestamp_groups.items():
            print(f"\n📅 Processing session: {timestamp_str}")
            
            # Determine categories needed
            categories = set()
            for file_path, _ in file_list:
                category = self.get_file_category(file_path)
                categories.add(category)
            
            # Create folder structure
            session_folder, created_folders = self.create_folder_structure(timestamp_str, categories)
            
            # Move files
            moved_files = []
            for file_path, timestamp in file_list:
                category = self.get_file_category(file_path)
                file_extension = file_path.suffix.lower()
                file_type = self.file_types.get(file_extension, 'misc')
                
                # Determine destination
                dest_folder = session_folder / category / file_type
                dest_path = dest_folder / file_path.name
                
                # Handle duplicate names
                counter = 1
                original_dest = dest_path
                while dest_path.exists():
                    stem = original_dest.stem
                    suffix = original_dest.suffix
                    dest_path = dest_folder / f"{stem}_{counter}{suffix}"
                    counter += 1
                
                # Move file
                try:
                    shutil.move(str(file_path), str(dest_path))
                    moved_files.append({
                        'original': str(file_path),
                        'destination': str(dest_path),
                        'category': category,
                        'type': file_type
                    })
                    print(f"  ✅ {file_path.name} → {category}/{file_type}/")
                except Exception as e:
                    print(f"  ❌ Error moving {file_path.name}: {e}")
            
            organization_summary[timestamp_str] = {
                'session_folder': str(session_folder),
                'timestamp': timestamp_str,
                'files_moved': len(moved_files),
                'categories': list(categories),
                'files': moved_files
            }
        
        return organization_summary
    
    def create_index_files(self, organization_summary: dict):
        """Create index files for easy navigation"""
        
        print("\n📋 Creating index files...")
        
        for timestamp_str, session_info in organization_summary.items():
            session_folder = Path(session_info['session_folder'])
            
            # Create session README
            readme_content = f"""# Analysis Session: {timestamp_str}

**Generated:** {datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M:%S')}  
**Files Organized:** {session_info['files_moved']}  
**Categories:** {', '.join(session_info['categories'])}

## File Organization

"""
            
            # Group files by category and type
            category_files = defaultdict(lambda: defaultdict(list))
            for file_info in session_info['files']:
                category = file_info['category']
                file_type = file_info['type']
                filename = Path(file_info['destination']).name
                category_files[category][file_type].append(filename)
            
            # Add category sections
            for category, types in category_files.items():
                readme_content += f"### {category.title()}\n\n"
                
                for file_type, filenames in types.items():
                    if filenames:
                        readme_content += f"**{file_type.title()}:**\n"
                        for filename in sorted(filenames):
                            readme_content += f"- [{filename}](./{category}/{file_type}/{filename})\n"
                        readme_content += "\n"
            
            # Add analysis descriptions
            readme_content += """
## Analysis Types

- **Comprehensive**: Complete statistical analysis and overview
- **Personality**: LLM negotiation personality fingerprints
- **Tournament**: March Madness style model competition
- **Reflection**: 3D analysis of reflection advantages
- **Conversation**: Flow diagrams and conversation analysis
- **Statistical**: Detailed statistical tests and visualizations
- **Figure**: Publication-ready figures
- **Dashboard**: Interactive comprehensive dashboards
- **Infographic**: Summary infographics

## File Types

- **Images**: PNG, JPG, SVG visualization files
- **Interactive**: HTML files with interactive visualizations
- **Reports**: Markdown and text analysis reports
- **Data**: JSON, CSV data files and results
"""
            
            # Write README
            readme_path = session_folder / "README.md"
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            
            print(f"  ✅ Created README.md for {timestamp_str}")
        
        # Create master index
        self.create_master_index(organization_summary)
    
    def create_master_index(self, organization_summary: dict):
        """Create master index file"""
        
        master_readme = f"""# Newsvendor LLM Experiment - Analysis Outputs

**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Total Sessions:** {len(organization_summary)}

## Analysis Sessions

"""
        
        # Sort sessions by timestamp (newest first)
        sorted_sessions = sorted(organization_summary.items(), key=lambda x: x[0], reverse=True)
        
        for timestamp_str, session_info in sorted_sessions:
            session_date = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
            master_readme += f"### [{session_date.strftime('%Y-%m-%d %H:%M:%S')}](./{timestamp_str}/README.md)\n\n"
            master_readme += f"- **Files:** {session_info['files_moved']}\n"
            master_readme += f"- **Categories:** {', '.join(session_info['categories'])}\n"
            master_readme += f"- **Folder:** `{timestamp_str}/`\n\n"
        
        master_readme += """
## Quick Navigation

### Latest Analysis
"""
        
        if sorted_sessions:
            latest_timestamp = sorted_sessions[0][0]
            latest_info = sorted_sessions[0][1]
            
            master_readme += f"**[{latest_timestamp}](./{latest_timestamp}/README.md)**\n\n"
            
            # Add quick links to common files
            master_readme += "**Quick Links:**\n"
            for file_info in latest_info['files']:
                filename = Path(file_info['destination']).name
                rel_path = f"./{latest_timestamp}/{file_info['category']}/{file_info['type']}/{filename}"
                
                if 'dashboard' in filename.lower():
                    master_readme += f"- [📊 Dashboard]({rel_path})\n"
                elif 'comprehensive' in filename.lower() and filename.endswith('.png'):
                    master_readme += f"- [📈 Comprehensive Analysis]({rel_path})\n"
                elif 'infographic' in filename.lower():
                    master_readme += f"- [📋 Summary Infographic]({rel_path})\n"
        
        master_readme += """

## Analysis Types Guide

- **📊 Comprehensive**: Complete statistical analysis with all key metrics
- **🧬 Personality**: Individual model negotiation "DNA" profiles  
- **🏆 Tournament**: March Madness style model competition brackets
- **🌟 Reflection**: 3D analysis showing reflection advantages
- **🌊 Conversation**: Flow diagrams of negotiation patterns
- **📈 Statistical**: Detailed hypothesis testing and statistics
- **📋 Dashboard**: Interactive overview with all key visualizations
- **🎨 Infographic**: Summary graphics for presentations

## File Organization

Each session folder contains:
```
YYYYMMDD_HHMMSS/
├── README.md                 # Session index
├── comprehensive/            # Complete analysis
│   ├── images/              # PNG visualizations
│   ├── interactive/         # HTML dashboards
│   └── reports/             # Markdown reports
├── personality/             # Model personalities
├── tournament/              # Model competitions
├── reflection/              # Reflection analysis
└── conversation/            # Flow analysis
```
"""
        
        # Write master index
        master_path = self.output_dir / "README.md"
        with open(master_path, 'w') as f:
            f.write(master_readme)
        
        print(f"  ✅ Created master README.md")
    
    def save_organization_log(self, organization_summary: dict):
        """Save detailed organization log"""
        
        log_path = self.output_dir / "organization_log.json"
        
        log_data = {
            'cleaned_at': datetime.now().isoformat(),
            'total_sessions': len(organization_summary),
            'total_files_moved': sum(info['files_moved'] for info in organization_summary.values()),
            'sessions': organization_summary
        }
        
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"  ✅ Saved organization log")
    
    def clean_and_organize(self):
        """Main function to clean and organize all files"""
        
        print("🧹 NEWSVENDOR ANALYSIS OUTPUT CLEANER")
        print("="*50)
        
        # Scan for files
        files = self.scan_files()
        
        if not files:
            print("✅ No analysis files found to organize")
            return
        
        # Organize files
        organization_summary = self.organize_files(files)
        
        # Create index files
        self.create_index_files(organization_summary)
        
        # Save log
        self.save_organization_log(organization_summary)
        
        # Summary
        total_files = sum(info['files_moved'] for info in organization_summary.values())
        total_sessions = len(organization_summary)
        
        print(f"\n🎉 ORGANIZATION COMPLETE!")
        print(f"="*30)
        print(f"📁 Total files moved: {total_files}")
        print(f"📅 Sessions created: {total_sessions}")
        print(f"📂 Base directory: {self.output_dir}")
        print(f"📋 Master index: {self.output_dir}/README.md")
        
        # Show session summary
        if organization_summary:
            print(f"\n📅 Sessions created:")
            for timestamp_str, info in sorted(organization_summary.items(), reverse=True):
                session_date = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                print(f"  • {session_date.strftime('%Y-%m-%d %H:%M:%S')}: {info['files_moved']} files")

def main():
    """Run the output cleaner"""
    
    # Initialize cleaner
    cleaner = OutputCleaner()
    
    # Clean and organize
    cleaner.clean_and_organize()

if __name__ == "__main__":
    main()