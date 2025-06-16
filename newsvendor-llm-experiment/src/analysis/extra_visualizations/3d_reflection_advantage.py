#!/usr/bin/env python3
"""
3D Reflection Advantage Matrix
Shows the "sweet spot" where reflection provides maximum benefit
Auto-detects and analyzes the latest experiment results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from scipy.interpolate import griddata
from pathlib import Path
import json
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def find_latest_data_file():
    """Find the most recent data file from your experiment"""
    
    # Look for CSV files in the processed directory
    csv_patterns = [
        "./full_results/processed/complete_*.csv",
        "./full_results/processed/complete_*.csv.gz",
        "./complete_*.csv",
        "./temp_results.csv"
    ]
    
    latest_file = None
    latest_time = 0
    
    print("🔍 Searching for data files...")
    
    for pattern in csv_patterns:
        files = glob.glob(pattern)
        for file in files:
            file_path = Path(file)
            if file_path.exists():
                mtime = file_path.stat().st_mtime
                if mtime > latest_time:
                    latest_time = mtime
                    latest_file = file_path
                    
                print(f"  Found: {file} (modified: {datetime.fromtimestamp(mtime)})")
    
    if latest_file:
        print(f"✅ Using latest file: {latest_file}")
        return str(latest_file)
    else:
        print("❌ No data files found!")
        return None

def load_data_smart(file_path):
    """Smart data loading that handles both regular and compressed files"""
    
    print(f"📊 Loading data from: {file_path}")
    
    try:
        # Check if it's a compressed file
        if file_path.endswith('.gz'):
            print("  📦 Detected compressed file, using gzip decompression...")
            data = pd.read_csv(file_path, compression='gzip')
        else:
            print("  📄 Loading regular CSV file...")
            data = pd.read_csv(file_path)
        
        print(f"✅ Successfully loaded {len(data):,} rows with {len(data.columns)} columns")
        return data
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None

class ReflectionAdvantageAnalyzer:
    """Analyze and visualize reflection advantages across model sizes and conditions"""
    
    def __init__(self):
        """Initialize with auto-detected data"""
        # Auto-detect latest data file
        data_path = find_latest_data_file()
        if not data_path:
            raise FileNotFoundError("No data files found. Please check your data directory.")
        
        self.data = load_data_smart(data_path)
        if self.data is None:
            raise ValueError("Failed to load data file")
        
        self.successful = self.data[self.data['completed'] == True].copy()
        self.source_file = data_path
        
        # Model size mapping (in GB)
        self.model_sizes = {
            'tinyllama:latest': 0.6,
            'qwen2:1.5b': 0.9,
            'gemma2:2b': 1.6,
            'phi3:mini': 2.2,
            'llama3.2:latest': 2.0,
            'mistral:instruct': 4.1,
            'qwen:7b': 4.5,
            'qwen3:latest': 5.2
        }
        
        # Model tiers for analysis
        self.model_tiers = {
            'tinyllama:latest': 1,
            'qwen2:1.5b': 1,
            'gemma2:2b': 2,
            'phi3:mini': 2,
            'llama3.2:latest': 2,
            'mistral:instruct': 3,
            'qwen:7b': 3,
            'qwen3:latest': 4
        }
        
        # Reflection pattern names
        self.reflection_patterns = {
            '00': 'No Reflection',
            '01': 'Buyer Reflection',
            '10': 'Supplier Reflection', 
            '11': 'Both Reflection'
        }
        
    def calculate_reflection_advantages(self):
        """Calculate reflection advantages across all dimensions"""
        
        advantages = []
        
        # Debug: Check data structure
        print(f"Total successful negotiations: {len(self.successful)}")
        print(f"Available columns: {list(self.successful.columns)}")
        
        # Check reflection pattern format and normalize
        print("Original reflection pattern distribution:")
        print(self.successful['reflection_pattern'].value_counts())
        
        # Normalize reflection patterns to ensure consistent format
        self.successful['reflection_pattern'] = self.successful['reflection_pattern'].astype(str).str.zfill(2)
        
        print("Normalized reflection pattern distribution:")
        print(self.successful['reflection_pattern'].value_counts())
        
        for model in self.model_sizes.keys():
            model_data = self.successful[
                (self.successful['buyer_model'] == model) | 
                (self.successful['supplier_model'] == model)
            ]
            
            if len(model_data) == 0:
                print(f"No data for model: {model}")
                continue
            
            print(f"Processing {model}: {len(model_data)} negotiations")
            
            # Check if we have baseline data (no reflection)
            baseline_data = model_data[model_data['reflection_pattern'] == '00']
            if len(baseline_data) == 0:
                print(f"  ⚠️  No baseline (00) data for {model}, skipping...")
                continue
                
            baseline_price = baseline_data['agreed_price'].mean()
            baseline_rounds = baseline_data['total_rounds'].mean() 
            baseline_tokens = baseline_data['total_tokens'].mean()
            
            print(f"  Baseline for {model}: price=${baseline_price:.2f}, rounds={baseline_rounds:.1f}, tokens={baseline_tokens:.0f}")
            
            for pattern in ['01', '10', '11']:
                pattern_data = model_data[model_data['reflection_pattern'] == pattern]
                
                if len(pattern_data) > 0:
                    # Performance metrics
                    avg_price = pattern_data['agreed_price'].mean()
                    avg_rounds = pattern_data['total_rounds'].mean()
                    avg_tokens = pattern_data['total_tokens'].mean()
                    
                    # Calculate advantages (positive = improvement)
                    price_advantage = abs(65 - baseline_price) - abs(65 - avg_price)  # Closer to optimal = better
                    efficiency_advantage = baseline_rounds - avg_rounds  # Fewer rounds = better
                    token_efficiency = baseline_tokens - avg_tokens  # Fewer tokens = better
                    
                    # Combined advantage score
                    combined_advantage = (
                        price_advantage * 0.4 +  # 40% weight on price optimality
                        efficiency_advantage * 0.3 +  # 30% weight on round efficiency  
                        (token_efficiency / 100) * 0.3  # 30% weight on token efficiency
                    )
                    
                    print(f"  {pattern} ({self.reflection_patterns[pattern]}): price_adv={price_advantage:.2f}, eff_adv={efficiency_advantage:.2f}, combined={combined_advantage:.2f}")
                    
                    advantages.append({
                        'model': model,
                        'model_size_gb': self.model_sizes[model],
                        'model_tier': self.model_tiers[model],
                        'reflection_pattern': pattern,
                        'reflection_name': self.reflection_patterns[pattern],
                        'price_advantage': price_advantage,
                        'efficiency_advantage': efficiency_advantage,
                        'token_efficiency': token_efficiency,
                        'combined_advantage': combined_advantage,
                        'sample_size': len(pattern_data),
                        'avg_price': avg_price,
                        'avg_rounds': avg_rounds,
                        'avg_tokens': avg_tokens,
                        'baseline_price': baseline_price,
                        'baseline_rounds': baseline_rounds,
                        'baseline_tokens': baseline_tokens
                    })
                else:
                    print(f"  No data for pattern {pattern} ({self.reflection_patterns[pattern]})")
        
        # Return DataFrame
        if advantages:
            print(f"\n✅ Successfully calculated {len(advantages)} advantage measurements")
            return pd.DataFrame(advantages)
        else:
            print("❌ No advantages calculated - no models had complete data sets")
            return pd.DataFrame()  # Return empty DataFrame
    
    def create_3d_advantage_surface(self, advantages_df):
        """Create 3D surface plot showing reflection advantages"""
        
        # Prepare data for 3D plot
        fig = go.Figure()
        
        # Color mapping for reflection patterns
        pattern_colors = {
            '01': '#FF6B6B',  # Buyer Reflection - Red
            '10': '#4ECDC4',  # Supplier Reflection - Teal  
            '11': '#FFEAA7'   # Both Reflection - Yellow
        }
        
        for pattern in ['01', '10', '11']:
            pattern_data = advantages_df[advantages_df['reflection_pattern'] == pattern]
            
            if len(pattern_data) > 0:
                fig.add_trace(go.Scatter3d(
                    x=pattern_data['model_size_gb'],
                    y=pattern_data['model_tier'],
                    z=pattern_data['combined_advantage'],
                    mode='markers+text',
                    marker=dict(
                        size=pattern_data['sample_size'] / 10,  # Size by sample size
                        color=pattern_colors[pattern],
                        opacity=0.8,
                        line=dict(width=2, color='DarkSlateGrey')
                    ),
                    text=pattern_data['model'].str.replace(':latest', ''),
                    textposition="top center",
                    name=self.reflection_patterns[pattern],
                    hovertemplate=(
                        '<b>%{text}</b><br>' +
                        'Model Size: %{x:.1f} GB<br>' +
                        'Model Tier: %{y}<br>' +
                        'Advantage Score: %{z:.2f}<br>' +
                        '<extra></extra>'
                    )
                ))
        
        # Update layout
        fig.update_layout(
            title='3D Reflection Advantage Matrix<br>' +
                  '<sub>Bubble size = Sample size | Height = Performance gain from reflection</sub>',
            scene=dict(
                xaxis_title='Model Size (GB)',
                yaxis_title='Model Tier (1=Ultra, 4=Large)', 
                zaxis_title='Combined Advantage Score',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            width=900,
            height=700,
            showlegend=True
        )
        
        return fig
    
    def create_advantage_heatmap(self, advantages_df):
        """Create heatmap showing reflection advantages by model and pattern"""
        
        # Pivot data for heatmap
        heatmap_data = advantages_df.pivot_table(
            index='model',
            columns='reflection_pattern', 
            values='combined_advantage',
            fill_value=0
        )
        
        # Clean model names
        heatmap_data.index = heatmap_data.index.str.replace(':latest', '')
        
        # Rename columns
        heatmap_data.columns = [self.reflection_patterns[col] for col in heatmap_data.columns]
        
        # Create plotly heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='RdYlGn',
            text=np.round(heatmap_data.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Advantage Score<br>(Higher = Better)")
        ))
        
        fig.update_layout(
            title='Reflection Advantage Heatmap<br>' +
                  '<sub>Performance gain from reflection vs no reflection baseline</sub>',
            xaxis_title='Reflection Pattern',
            yaxis_title='LLM Model',
            width=700,
            height=500
        )
        
        return fig
    
    def create_sweet_spot_analysis(self, advantages_df):
        """Identify and visualize the reflection "sweet spot" """
        
        # Find the sweet spot (highest advantages)
        sweet_spot = advantages_df.loc[advantages_df['combined_advantage'].idxmax()]
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Price Advantage by Model Size',
                'Efficiency Advantage by Model Tier', 
                'Token Efficiency by Reflection Pattern',
                'Sweet Spot Analysis'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Price advantage vs model size
        for pattern in ['01', '10', '11']:
            pattern_data = advantages_df[advantages_df['reflection_pattern'] == pattern]
            fig.add_trace(
                go.Scatter(
                    x=pattern_data['model_size_gb'],
                    y=pattern_data['price_advantage'],
                    mode='markers+lines',
                    name=self.reflection_patterns[pattern],
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Plot 2: Efficiency advantage vs model tier
        tier_efficiency = advantages_df.groupby(['model_tier', 'reflection_pattern'])['efficiency_advantage'].mean().reset_index()
        for pattern in ['01', '10', '11']:
            pattern_data = tier_efficiency[tier_efficiency['reflection_pattern'] == pattern]
            fig.add_trace(
                go.Scatter(
                    x=pattern_data['model_tier'],
                    y=pattern_data['efficiency_advantage'],
                    mode='markers+lines',
                    name=self.reflection_patterns[pattern],
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Plot 3: Token efficiency by pattern
        pattern_tokens = advantages_df.groupby('reflection_pattern')['token_efficiency'].mean()
        fig.add_