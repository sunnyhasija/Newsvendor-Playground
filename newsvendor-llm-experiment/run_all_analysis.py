#!/usr/bin/env python3
"""
Master Analysis Runner - Fixed to work with existing scripts as-is
Runs comprehensive analysis pipeline and uses your existing output_cleaner.py
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
import json
import os
from typing import List, Dict, Any

class MasterAnalysisRunner:
    """Orchestrates the complete analysis pipeline with your existing scripts"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.start_time = datetime.now()
        
        # Create timestamp for this analysis session
        self.session_timestamp = self.start_time.strftime('%Y%m%d_%H%M%S')
        
        self.results = {
            'pipeline_start': self.start_time.isoformat(),
            'session_timestamp': self.session_timestamp,
            'scripts_run': [],
            'errors': [],
            'files_generated': []
        }
        
        print("🚀 NEWSVENDOR LLM ANALYSIS PIPELINE")
        print("="*60)
        print(f"🕐 Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📅 Session ID: {self.session_timestamp}")
        print(f"📁 Working directory: {self.base_dir}")
    
    def run_script(self, script_path: str, script_name: str, args: List[str] = None, 
                   timeout: int = 300, required: bool = True) -> bool:
        """Run a single analysis script with error handling"""
        
        print(f"\n📊 Running {script_name}...")
        print(f"    Script: {script_path}")
        
        if not Path(script_path).exists():
            error_msg = f"Script not found: {script_path}"
            print(f"    ❌ {error_msg}")
            self.results['errors'].append({
                'script': script_name,
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            })
            return not required  # Continue if not required
        
        try:
            # Prepare command
            cmd = [sys.executable, script_path]
            if args:
                cmd.extend(args)
            
            # Run with timeout
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.base_dir
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"    ✅ Completed successfully ({duration:.1f}s)")
                self.results['scripts_run'].append({
                    'name': script_name,
                    'script': script_path,
                    'duration': duration,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'success'
                })
                return True
            else:
                error_msg = f"Exit code {result.returncode}"
                if result.stderr:
                    error_msg += f": {result.stderr[:200]}"
                
                print(f"    ⚠️  {error_msg}")
                self.results['errors'].append({
                    'script': script_name,
                    'error': error_msg,
                    'exit_code': result.returncode,
                    'stderr': result.stderr[:500] if result.stderr else "",
                    'timestamp': datetime.now().isoformat()
                })
                return not required
                
        except subprocess.TimeoutExpired:
            error_msg = f"Timeout after {timeout}s"
            print(f"    ⏰ {error_msg}")
            self.results['errors'].append({
                'script': script_name,
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            })
            return not required
            
        except Exception as e:
            error_msg = f"Exception: {str(e)}"
            print(f"    ❌ {error_msg}")
            self.results['errors'].append({
                'script': script_name,
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            })
            return not required
    
    def check_data_availability(self) -> bool:
        """Check if data files are available without running debug script"""
        print("\n🔍 DATA AVAILABILITY CHECK")
        print("-" * 30)
        
        # Look for data files using the same patterns as your auto-detection
        data_patterns = [
            "./full_results/processed/complete_*.csv*",
            "./complete_*.csv",
            "./temp_results.csv"
        ]
        
        import glob
        found_files = []
        
        for pattern in data_patterns:
            files = glob.glob(pattern)
            found_files.extend(files)
        
        if found_files:
            # Get the most recent file
            latest_file = max(found_files, key=lambda x: Path(x).stat().st_mtime)
            file_size = Path(latest_file).stat().st_size / (1024 * 1024)  # MB
            
            print(f"✅ Found data files:")
            print(f"    Latest: {latest_file}")
            print(f"    Size: {file_size:.1f} MB")
            print(f"    Total files: {len(found_files)}")
            
            return True
        else:
            print("❌ No data files found!")
            print("   Expected files matching patterns:")
            for pattern in data_patterns:
                print(f"     - {pattern}")
            return False
    
    def run_analysis_scripts(self) -> bool:
        """Run all analysis scripts"""
        print("\n📊 ANALYSIS EXECUTION PHASE")
        print("-" * 30)
        
        # Analysis scripts to run - using your existing scripts as-is
        analysis_scripts = [
            # Core comprehensive analyses (required)
            {
                'path': "final_comprehensive_analysis.py",
                'name': "Final Comprehensive Analysis", 
                'args': None,
                'timeout': 600,
                'required': True,
                'description': "Main comprehensive analysis with all visualizations"
            },
            
            # Secondary analyses (optional)
            {
                'path': "comprehensive_analysis_latest.py",
                'name': "Latest Comprehensive Analysis",
                'args': None,
                'timeout': 600,
                'required': False,
                'description': "Alternative comprehensive analysis"
            },
            
            # Specialized analyses (optional)
            {
                'path': "debug_reflection.py",
                'name': "Reflection Pattern Analysis",
                'args': None,
                'timeout': 120,
                'required': False,
                'description': "Detailed reflection pattern debugging"
            },
            
            {
                'path': "src/analysis/conversation_analyzer.py",
                'name': "Conversation Analysis",
                'args': None,
                'timeout': 300,
                'required': False,
                'description': "Turn-by-turn conversation analysis"
            },
            
            # Any other analysis scripts (optional)
            {
                'path': "comprehensive_analysis.py",
                'name': "General Comprehensive Analysis",
                'args': None,
                'timeout': 300,
                'required': False,
                'description': "General comprehensive analysis"
            }
        ]
        
        print(f"📋 Planning to run {len(analysis_scripts)} analysis scripts...")
        
        success_count = 0
        total_scripts = 0
        
        for script_config in analysis_scripts:
            total_scripts += 1
            
            # Check if script exists before attempting
            script_path = script_config['path']
            if not Path(script_path).exists():
                print(f"⚠️  Skipping {script_config['name']} - script not found: {script_path}")
                if script_config['required']:
                    print(f"❌ Required script missing - stopping analysis")
                    return False
                continue
            
            print(f"\n🎯 {script_config['description']}")
            
            success = self.run_script(
                script_path,
                script_config['name'],
                script_config['args'],
                script_config['timeout'],
                script_config['required']
            )
            
            if success:
                success_count += 1
            elif script_config['required']:
                print(f"❌ Required script {script_config['name']} failed - stopping analysis")
                return False
        
        print(f"\n✅ Analysis phase completed: {success_count}/{total_scripts} scripts successful")
        
        # Require at least one successful script
        if success_count == 0:
            print("❌ No analysis scripts completed successfully")
            return False
        
        return True
    
    def scan_generated_files(self) -> List[Path]:
        """Scan for files generated during this session"""
        print("\n📂 SCANNING FOR GENERATED FILES")
        print("-" * 30)
        
        # Look for files that match your output_cleaner patterns
        analysis_patterns = [
            r'comprehensive.*\.(png|html|md|json|csv)',
            r'personality.*\.(png|html|md|json|csv)',
            r'tournament.*\.(png|html|md|json|csv)',
            r'reflection.*\.(png|html|md|json|csv)',
            r'conversation.*\.(png|html|md|json|csv)',
            r'statistical.*\.(png|html|md|json|csv)',
            r'figure\d+.*\.(png|html|md)',
            r'.*dashboard.*\.(png|html|md)',
            r'.*infographic.*\.(png|html|md)',
            r'.*_\d{8}_\d{6}\.(png|html|md|json|csv)',  # Any file with timestamp pattern
        ]
        
        import re
        found_files = []
        
        for file_path in self.base_dir.iterdir():
            if file_path.is_file():
                filename = file_path.name
                
                # Check if file matches any analysis pattern
                for pattern in analysis_patterns:
                    if re.match(pattern, filename, re.IGNORECASE):
                        # Check if file was created/modified during our session
                        file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_mtime >= self.start_time:
                            found_files.append(file_path)
                            break
        
        self.results['files_generated'] = [str(f) for f in found_files]
        
        print(f"📄 Found {len(found_files)} files generated during this session:")
        for file_path in found_files[:10]:  # Show first 10
            file_size = file_path.stat().st_size / 1024  # KB
            print(f"    • {file_path.name} ({file_size:.1f} KB)")
        
        if len(found_files) > 10:
            print(f"    ... and {len(found_files) - 10} more files")
        
        return found_files
    
    def run_your_output_cleaner(self) -> bool:
        """Run your existing output_cleaner.py script"""
        print("\n🧹 OUTPUT CLEANING & ORGANIZATION PHASE")
        print("-" * 30)
        
        # Your output_cleaner.py should exist in the root directory
        cleaner_script = "output_cleaner.py"
        
        if not Path(cleaner_script).exists():
            print(f"❌ {cleaner_script} not found!")
            print("   This step organizes your analysis files by timestamp")
            print("   The files are still available in the current directory")
            return False
        
        print(f"🧹 Running your existing output cleaner...")
        print("   This will organize files into timestamp-based session folders")
        
        success = self.run_script(
            cleaner_script,
            "Output Cleaner & Organizer",
            [],  # Your cleaner doesn't need arguments
            timeout=120,
            required=False
        )
        
        if success:
            print("✅ Output cleaning and organization completed")
            print("📁 Files organized into timestamp-based sessions in outputs/ directory")
            print("📋 Check outputs/README.md for navigation")
        else:
            print("⚠️  Output cleaning had issues")
            print("   Files are still available in the current directory")
        
        return success
    
    def save_session_summary(self):
        """Save a summary of this analysis session"""
        print("\n📋 SAVING SESSION SUMMARY")
        print("-" * 30)
        
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        # Update results
        self.results.update({
            'pipeline_end': end_time.isoformat(),
            'total_duration_seconds': total_duration,
            'total_duration_minutes': round(total_duration / 60, 1),
            'total_scripts_attempted': len(self.results['scripts_run']) + len(self.results['errors']),
            'successful_scripts': len(self.results['scripts_run']),
            'failed_scripts': len(self.results['errors']),
            'files_generated_count': len(self.results['files_generated'])
        })
        
        # Save session log with timestamp (so your output_cleaner can organize it)
        session_log_file = f"analysis_session_{self.session_timestamp}.json"
        with open(session_log_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Create human-readable summary with timestamp
        summary_file = f"analysis_summary_{self.session_timestamp}.md"
        with open(summary_file, 'w') as f:
            f.write(f"# Newsvendor LLM Analysis Session\\n\\n")
            f.write(f"**Session ID:** {self.session_timestamp}\\n")
            f.write(f"**Start Time:** {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"**End Time:** {end_time.strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"**Duration:** {total_duration/60:.1f} minutes\\n")
            f.write(f"**Scripts Run:** {len(self.results['scripts_run'])}/{len(self.results['scripts_run']) + len(self.results['errors'])}\\n")
            f.write(f"**Files Generated:** {len(self.results['files_generated'])}\\n\\n")
            
            f.write("## Successful Scripts\\n\\n")
            for script in self.results['scripts_run']:
                f.write(f"- ✅ **{script['name']}** ({script['duration']:.1f}s)\\n")
            
            if self.results['errors']:
                f.write("\\n## Failed Scripts\\n\\n")
                for error in self.results['errors']:
                    f.write(f"- ❌ **{error['script']}**: {error['error']}\\n")
            
            f.write("\\n## Generated Files\\n\\n")
            for file_path in self.results['files_generated']:
                f.write(f"- 📄 `{file_path}`\\n")
            
            f.write(f"\\n## Notes\\n\\n")
            f.write(f"This analysis session generated files that should be organized by `output_cleaner.py` into the `outputs/{self.session_timestamp}/` directory structure.\\n")
        
        print(f"✅ Session summary saved: {summary_file}")
        print(f"📊 Detailed log saved: {session_log_file}")
        
        return session_log_file, summary_file
    
    def run_complete_pipeline(self):
        """Run the complete analysis pipeline"""
        
        try:
            # Phase 1: Check Data Availability (non-blocking)
            data_available = self.check_data_availability()
            if not data_available:
                print("⚠️  No data files found, but continuing...")
                print("   Some analysis scripts may fail without data")
            
            # Phase 2: Run All Analysis Scripts
            if not self.run_analysis_scripts():
                print("❌ Pipeline stopped - no analysis scripts completed successfully")
                return False
            
            # Phase 3: Scan for Generated Files
            generated_files = self.scan_generated_files()
            
            # Phase 4: Save Session Summary (before cleaning so it gets organized too)
            log_file, summary_file = self.save_session_summary()
            
            # Phase 5: Run Your Output Cleaner
            cleaner_success = self.run_your_output_cleaner()
            
            # Final summary
            end_time = datetime.now()
            total_duration = (end_time - self.start_time).total_seconds()
            
            print(f"\\n🎉 ANALYSIS PIPELINE COMPLETED!")
            print("="*60)
            print(f"🕐 Total time: {total_duration/60:.1f} minutes")
            print(f"📅 Session ID: {self.session_timestamp}")
            print(f"✅ Scripts run: {len(self.results['scripts_run'])}")
            print(f"📊 Files generated: {len(self.results['files_generated'])}")
            
            if cleaner_success:
                print(f"📁 Organized in: outputs/{self.session_timestamp}/")
                print(f"📋 Master index: outputs/README.md")
            else:
                print(f"📁 Files available in current directory")
                print(f"📋 Session summary: {summary_file}")
            
            if self.results['errors']:
                print(f"⚠️  {len(self.results['errors'])} scripts had issues")
            
            return True
            
        except KeyboardInterrupt:
            print("\\n⚠️  Pipeline interrupted by user")
            return False
        except Exception as e:
            print(f"\\n❌ Pipeline failed with unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run complete newsvendor LLM analysis pipeline")
    parser.add_argument('--base-dir', default='.', 
                       help='Base directory to run analysis from')
    parser.add_argument('--quick', action='store_true',
                       help='Run only essential analysis scripts')
    parser.add_argument('--skip-data-check', action='store_true',
                       help='Skip data availability check')
    
    args = parser.parse_args()
    
    # Create and run the pipeline
    runner = MasterAnalysisRunner(args.base_dir)
    
    if args.quick:
        print("🚀 Running QUICK analysis pipeline...")
        
        # Quick mode: just run the main comprehensive analysis
        success = True
        
        if not args.skip_data_check:
            success = runner.check_data_availability()
            if not success:
                print("⚠️  Continuing without data validation...")
                success = True  # Continue anyway in quick mode
        
        if success:
            success = runner.run_script("final_comprehensive_analysis.py", "Core Analysis", timeout=600, required=True)
            
        if success:
            runner.scan_generated_files()
            runner.save_session_summary()
            runner.run_your_output_cleaner()
    else:
        print("🚀 Running COMPLETE analysis pipeline...")
        success = runner.run_complete_pipeline()
    
    print(f"\\n{'✅ SUCCESS' if success else '❌ FAILURE'}")
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()