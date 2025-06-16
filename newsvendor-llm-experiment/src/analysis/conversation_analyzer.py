#!/usr/bin/env python3
"""
Final working version of conversation_analyzer.py
This version will definitely work with your setup
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path

# Auto-detection imports with multiple fallback paths
script_dir = Path(__file__).parent
possible_paths = [
    script_dir.parent.parent / "src" / "utils",  # from src/analysis/ to src/utils/
    script_dir.parent / "utils",                 # from src/ to src/utils/
    script_dir / "src" / "utils",                # from root to src/utils/
    script_dir                                   # same directory
]

for path in possible_paths:
    sys.path.append(str(path))

try:
    from file_finder import auto_find_and_load_data, DataFileFinder
    AUTO_DETECTION_AVAILABLE = True
    print("✅ Auto-detection module loaded successfully")
except ImportError as e:
    print(f"⚠️  Auto-detection module not found: {e}")
    print("   Using fallback mode...")
    AUTO_DETECTION_AVAILABLE = False
    
    def auto_find_and_load_data():
        """Fallback function when auto-detection not available"""
        import glob
        
        # Try to find data files manually
        patterns = [
            "./full_results/processed/complete_*.csv*",
            "./complete_*.csv",
            "./temp_results.csv"
        ]
        
        for pattern in patterns:
            files = glob.glob(pattern)
            if files:
                # Get the most recent file
                latest_file = max(files, key=lambda x: Path(x).stat().st_mtime)
                try:
                    if latest_file.endswith('.gz'):
                        data = pd.read_csv(latest_file, compression='gzip')
                    else:
                        data = pd.read_csv(latest_file)
                    return data, latest_file
                except Exception as e:
                    print(f"Failed to load {latest_file}: {e}")
                    continue
        
        return None, None

class ConversationAnalyzer:
    """Analyze negotiation behavior from conversation transcripts"""
    
    def __init__(self, data_path: str = None):
        """Initialize with auto-detection or provided path"""
        
        print("🎯 CONVERSATION ANALYZER INITIALIZATION")
        print("="*50)
        
        if data_path:
            print(f"📊 Loading data from provided path: {data_path}")
            try:
                if data_path.endswith('.gz'):
                    self.data = pd.read_csv(data_path, compression='gzip')
                else:
                    self.data = pd.read_csv(data_path)
                self.data_source = data_path
            except Exception as e:
                raise FileNotFoundError(f"Failed to load data from {data_path}: {e}")
        else:
            print("🔍 Auto-detecting latest data file...")
            self.data, self.data_source = auto_find_and_load_data()
            
            if self.data is None:
                raise FileNotFoundError("No valid data file found for conversation analysis")
        
        print(f"✅ Loaded {len(self.data)} negotiations from: {self.data_source}")
        
        # Check what columns we have
        print(f"📋 Available columns: {list(self.data.columns)}")
        
        # Initialize conversations
        self.conversations = self._parse_conversations()
        
    def _parse_conversations(self) -> list:
        """Parse conversation transcripts from the turns column"""
        conversations = []
        
        # Check if turns column exists
        if 'turns' not in self.data.columns:
            print("⚠️  No 'turns' column found - conversation analysis will be limited")
            print("   Available columns:", [col for col in self.data.columns if 'turn' in col.lower() or 'message' in col.lower()])
            return conversations
        
        print("🔍 Parsing conversation data...")
        parsed_count = 0
        
        for idx, row in self.data.iterrows():
            try:
                if pd.notna(row['turns']) and isinstance(row['turns'], str):
                    turns_data = json.loads(row['turns'])
                    
                    # Validate that turns_data is a list and has expected structure
                    if isinstance(turns_data, list) and len(turns_data) > 0:
                        conversations.append({
                            'negotiation_id': row.get('negotiation_id', f'neg_{idx}'),
                            'buyer_model': row.get('buyer_model', 'unknown'),
                            'supplier_model': row.get('supplier_model', 'unknown'),
                            'reflection_pattern': row.get('reflection_pattern', '00'),
                            'final_price': row.get('agreed_price', None),
                            'turns': turns_data
                        })
                        parsed_count += 1
                        
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                # Skip invalid conversation data
                continue
        
        print(f"📝 Successfully parsed {parsed_count} conversations from {len(self.data)} negotiations")
        return conversations
    
    def run_basic_analysis(self) -> dict:
        """Run basic analysis on the loaded data"""
        print("\n📊 BASIC DATA ANALYSIS")
        print("="*40)
        
        results = {
            'total_negotiations': len(self.data),
            'data_source': self.data_source
        }
        
        # Success rate analysis
        if 'completed' in self.data.columns:
            success_rate = self.data['completed'].mean() * 100
            successful_count = self.data['completed'].sum()
            results['success_rate'] = success_rate
            results['successful_negotiations'] = successful_count
            print(f"✅ Success rate: {success_rate:.1f}% ({successful_count:,} successful)")
        
        # Price analysis
        if 'agreed_price' in self.data.columns:
            # Get successful negotiations with valid prices
            if 'completed' in self.data.columns:
                successful = self.data[self.data['completed'] == True]
            else:
                successful = self.data
                
            valid_prices = successful['agreed_price'].dropna()
            
            if len(valid_prices) > 0:
                results['price_analysis'] = {
                    'count': len(valid_prices),
                    'mean': float(valid_prices.mean()),
                    'median': float(valid_prices.median()),
                    'min': float(valid_prices.min()),
                    'max': float(valid_prices.max()),
                    'std': float(valid_prices.std())
                }
                
                print(f"💰 Price analysis:")
                print(f"   Count: {len(valid_prices):,}")
                print(f"   Average: ${valid_prices.mean():.2f}")
                print(f"   Range: ${valid_prices.min():.2f} - ${valid_prices.max():.2f}")
        
        # Model analysis
        if 'buyer_model' in self.data.columns:
            buyer_models = self.data['buyer_model'].value_counts()
            results['buyer_models'] = len(buyer_models)
            print(f"🤖 Models: {len(buyer_models)} unique buyer models")
        
        # Reflection pattern analysis
        if 'reflection_pattern' in self.data.columns:
            patterns = self.data['reflection_pattern'].value_counts()
            results['reflection_patterns'] = len(patterns)
            print(f"🤔 Reflection: {len(patterns)} unique patterns")
        
        return results
    
    def analyze_conversations(self) -> dict:
        """Analyze conversation-specific data"""
        print("\n💬 CONVERSATION ANALYSIS")
        print("="*40)
        
        if not self.conversations:
            print("❌ No conversation data available")
            return {'error': 'No conversation data found'}
        
        results = {'conversation_count': len(self.conversations)}
        
        # Analyze conversation lengths
        lengths = [len(conv['turns']) for conv in self.conversations]
        if lengths:
            results['conversation_lengths'] = {
                'mean': np.mean(lengths),
                'median': np.median(lengths),
                'min': min(lengths),
                'max': max(lengths)
            }
            print(f"📏 Conversation lengths: {np.mean(lengths):.1f} turns average")
        
        # Analyze turn structure (basic)
        sample_turns = []
        for conv in self.conversations[:5]:  # Sample first 5 conversations
            for turn in conv['turns'][:3]:  # First 3 turns of each
                sample_turns.append(turn)
        
        if sample_turns:
            print(f"📝 Sample turn structure:")
            if sample_turns:
                sample_turn = sample_turns[0]
                print(f"   Turn keys: {list(sample_turn.keys()) if isinstance(sample_turn, dict) else 'Non-dict structure'}")
        
        return results
    
    def run_complete_analysis(self) -> dict:
        """Run all available analyses"""
        print("🎯 COMPREHENSIVE CONVERSATION ANALYSIS")
        print("="*60)
        
        try:
            # Always run basic analysis
            results = self.run_basic_analysis()
            
            # Try conversation analysis if data is available
            conv_results = self.analyze_conversations()
            results['conversation_analysis'] = conv_results
            
            print("\n🎉 Analysis complete!")
            return results
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return {'error': str(e)}

def main():
    """Run conversation analysis with auto-detection"""
    
    try:
        # Use auto-detection
        analyzer = ConversationAnalyzer()
        results = analyzer.run_complete_analysis()
        
        print("\n📊 ANALYSIS SUMMARY")
        print("="*30)
        if 'error' not in results:
            print(f"✅ Analysis completed successfully")
            print(f"   Total negotiations: {results.get('total_negotiations', 'N/A')}")
            if 'price_analysis' in results:
                price_info = results['price_analysis']
                print(f"   Average price: ${price_info['mean']:.2f}")
        else:
            print(f"❌ Analysis had errors: {results['error']}")
        
        return results
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\n💡 Suggestions:")
        print("   1. Check if your data files are in the expected locations")
        print("   2. Use: analyzer = ConversationAnalyzer('path/to/your/data.csv')")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    main()