#!/usr/bin/env python3
"""
LLM Negotiation Personality Fingerprints
Creates unique "negotiation DNA" radar charts for each model
Auto-detects and analyzes the latest experiment results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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

class LLMPersonalityAnalyzer:
    """Analyze and visualize LLM negotiation personalities"""
    
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
        
        # Model tier mapping
        self.model_tiers = {
            'tinyllama:latest': 'Ultra-Compact',
            'qwen2:1.5b': 'Ultra-Compact',
            'gemma2:2b': 'Compact',
            'phi3:mini': 'Compact',
            'llama3.2:latest': 'Compact',
            'mistral:instruct': 'Mid-Range',
            'qwen:7b': 'Mid-Range',
            'qwen3:latest': 'Large'
        }
        
        # Color scheme for models
        self.model_colors = {
            'tinyllama:latest': '#FF6B6B',
            'qwen2:1.5b': '#FF8E53',
            'gemma2:2b': '#4ECDC4',
            'phi3:mini': '#45B7D1',
            'llama3.2:latest': '#96CEB4',
            'mistral:instruct': '#FFEAA7',
            'qwen:7b': '#DDA0DD',
            'qwen3:latest': '#98D8C8'
        }
    
    def calculate_personality_metrics(self):
        """Calculate personality metrics for each model"""
        
        personalities = {}
        
        for model in self.data['buyer_model'].unique():
            # As buyer analysis
            buyer_data = self.successful[self.successful['buyer_model'] == model]
            buyer_prices = buyer_data['agreed_price'].dropna()
            
            # As supplier analysis  
            supplier_data = self.successful[self.successful['supplier_model'] == model]
            supplier_prices = supplier_data['agreed_price'].dropna()
            
            # Combined analysis
            all_model_data = self.successful[
                (self.successful['buyer_model'] == model) | 
                (self.successful['supplier_model'] == model)
            ]
            
            if len(all_model_data) > 0:
                personalities[model] = {
                    # 1. Aggressiveness (how far from fair $65)
                    'Aggressiveness': self._calculate_aggressiveness(buyer_prices, supplier_prices),
                    
                    # 2. Concession Rate (how much they move from opening)
                    'Concession_Rate': self._calculate_concession_rate(model),
                    
                    # 3. Price Anchoring (influence of opening bids)
                    'Price_Anchoring': self._calculate_anchoring_strength(model),
                    
                    # 4. Convergence Speed (rounds to agreement)
                    'Convergence_Speed': self._calculate_convergence_speed(all_model_data),
                    
                    # 5. Fairness Preference (how close to optimal $65)
                    'Fairness_Preference': self._calculate_fairness_preference(all_model_data),
                    
                    # 6. Consistency (low variance in outcomes)
                    'Consistency': self._calculate_consistency(all_model_data),
                    
                    # Metadata
                    'tier': self.model_tiers.get(model, 'Unknown'),
                    'total_negotiations': len(all_model_data),
                    'success_rate': len(all_model_data) / len(self.data[
                        (self.data['buyer_model'] == model) | 
                        (self.data['supplier_model'] == model)
                    ])
                }
        
        return personalities
    
    def _calculate_aggressiveness(self, buyer_prices, supplier_prices):
        """Calculate aggressiveness score (0-100)"""
        buyer_aggression = 0
        supplier_aggression = 0
        
        if len(buyer_prices) > 0:
            # Buyers are aggressive when they push for very low prices
            buyer_aggression = np.mean([(65 - price) / 35 for price in buyer_prices if price < 65])
            buyer_aggression = max(0, min(1, buyer_aggression)) if not np.isnan(buyer_aggression) else 0
        
        if len(supplier_prices) > 0:
            # Suppliers are aggressive when they push for very high prices  
            supplier_aggression = np.mean([(price - 65) / 35 for price in supplier_prices if price > 65])
            supplier_aggression = max(0, min(1, supplier_aggression)) if not np.isnan(supplier_aggression) else 0
        
        # Combined aggressiveness
        total_aggression = (buyer_aggression + supplier_aggression) / 2
        return total_aggression * 100
    
    def _calculate_concession_rate(self, model):
        """Calculate how much models concede during negotiations"""
        # This would require turn-by-turn data from conversation transcripts
        # For now, use a proxy based on final vs typical prices
        model_data = self.successful[
            (self.successful['buyer_model'] == model) | 
            (self.successful['supplier_model'] == model)
        ]
        
        if len(model_data) == 0:
            return 50
        
        # Proxy: how much variance in final prices (more variance = more concessions)
        price_variance = model_data['agreed_price'].std()
        normalized_variance = min(100, (price_variance / 20) * 100)  # Normalize to 0-100
        return normalized_variance
    
    def _calculate_anchoring_strength(self, model):
        """Calculate how much opening bids influence final prices"""
        # Proxy: models with strong anchoring show more clustered prices
        model_data = self.successful[
            (self.successful['buyer_model'] == model) | 
            (self.successful['supplier_model'] == model)
        ]
        
        if len(model_data) == 0:
            return 50
        
        # Strong anchoring = low price variance
        price_std = model_data['agreed_price'].std()
        anchoring_strength = max(0, 100 - (price_std * 5))  # Invert: low std = high anchoring
        return min(100, anchoring_strength)
    
    def _calculate_convergence_speed(self, model_data):
        """Calculate how quickly model reaches agreements"""
        if len(model_data) == 0:
            return 50
        
        avg_rounds = model_data['total_rounds'].mean()
        # Convert to 0-100 scale (fewer rounds = higher score)
        speed_score = max(0, 100 - (avg_rounds - 1) * 20)  # 1 round = 100, 6+ rounds = 0
        return min(100, speed_score)
    
    def _calculate_fairness_preference(self, model_data):
        """Calculate preference for fair/optimal outcomes"""
        if len(model_data) == 0:
            return 50
        
        prices = model_data['agreed_price'].dropna()
        if len(prices) == 0:
            return 50
        
        # How close to optimal $65
        distances = np.abs(prices - 65)
        avg_distance = distances.mean()
        
        # Convert to 0-100 scale (closer to 65 = higher fairness)
        fairness_score = max(0, 100 - (avg_distance * 3))  # Distance of 33 = 0 fairness
        return min(100, fairness_score)
    
    def _calculate_consistency(self, model_data):
        """Calculate consistency in negotiation outcomes"""
        if len(model_data) == 0:
            return 50
        
        prices = model_data['agreed_price'].dropna()
        if len(prices) <= 1:
            return 100
        
        # Low standard deviation = high consistency
        price_std = prices.std()
        consistency_score = max(0, 100 - (price_std * 4))
        return min(100, consistency_score)
    
    def create_radar_chart(self, personalities, model_name):
        """Create radar chart for single model"""
        
        if model_name not in personalities:
            return None
        
        personality = personalities[model_name]
        
        # Personality dimensions
        dimensions = [
            'Aggressiveness', 'Concession_Rate', 'Price_Anchoring',
            'Convergence_Speed', 'Fairness_Preference', 'Consistency'
        ]
        
        values = [personality[dim] for dim in dimensions]
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=dimensions,
            fill='toself',
            name=model_name.replace(':latest', ''),
            line_color=self.model_colors.get(model_name, '#FF6B6B'),
            fillcolor=self.model_colors.get(model_name, '#FF6B6B'),
            opacity=0.6
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title=f"{model_name.replace(':latest', '')} Negotiation Personality<br>"
                  f"<sub>Tier: {personality['tier']} | "
                  f"Negotiations: {personality['total_negotiations']} | "
                  f"Success Rate: {personality['success_rate']:.1%}</sub>",
            width=500,
            height=500
        )
        
        return fig
    
    def create_comparison_radar(self, personalities):
        """Create comparison radar chart with all models"""
        
        dimensions = [
            'Aggressiveness', 'Concession_Rate', 'Price_Anchoring',
            'Convergence_Speed', 'Fairness_Preference', 'Consistency'
        ]
        
        fig = go.Figure()
        
        for model_name, personality in personalities.items():
            values = [personality[dim] for dim in dimensions]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=dimensions,
                fill='toself',
                name=model_name.replace(':latest', ''),
                line_color=self.model_colors.get(model_name, '#FF6B6B'),
                opacity=0.4
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="LLM Negotiation Personality Comparison<br>"
                  "<sub>All Models Overlaid</sub>",
            width=800,
            height=600
        )
        
        return fig
    
    def create_personality_heatmap(self, personalities):
        """Create heatmap of personality traits"""
        
        # Prepare data
        models = list(personalities.keys())
        dimensions = [
            'Aggressiveness', 'Concession_Rate', 'Price_Anchoring',
            'Convergence_Speed', 'Fairness_Preference', 'Consistency'
        ]
        
        # Create matrix
        matrix = []
        model_labels = []
        
        for model in models:
            row = [personalities[model][dim] for dim in dimensions]
            matrix.append(row)
            model_labels.append(model.replace(':latest', ''))
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=dimensions,
            y=model_labels,
            colorscale='RdYlBu_r',
            text=np.round(matrix, 1),
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Personality Score (0-100)")
        ))
        
        fig.update_layout(
            title="LLM Negotiation Personality Heatmap<br>"
                  "<sub>Higher scores indicate stronger trait expression</sub>",
            xaxis_title="Personality Dimensions",
            yaxis_title="LLM Models",
            width=800,
            height=600
        )
        
        return fig
    
    def create_tier_comparison(self, personalities):
        """Create tier-based personality comparison"""
        
        # Group by tiers
        tier_data = {}
        for model, personality in personalities.items():
            tier = personality['tier']
            if tier not in tier_data:
                tier_data[tier] = []
            tier_data[tier].append(personality)
        
        # Calculate tier averages
        dimensions = [
            'Aggressiveness', 'Concession_Rate', 'Price_Anchoring',
            'Convergence_Speed', 'Fairness_Preference', 'Consistency'
        ]
        
        fig = go.Figure()
        
        tier_colors = {
            'Ultra-Compact': '#FF6B6B',
            'Compact': '#4ECDC4', 
            'Mid-Range': '#FFEAA7',
            'Large': '#98D8C8'
        }
        
        for tier, personalities_list in tier_data.items():
            if personalities_list:
                avg_values = []
                for dim in dimensions:
                    avg_val = np.mean([p[dim] for p in personalities_list])
                    avg_values.append(avg_val)
                
                fig.add_trace(go.Scatterpolar(
                    r=avg_values,
                    theta=dimensions,
                    fill='toself',
                    name=tier,
                    line_color=tier_colors.get(tier, '#666666'),
                    opacity=0.6
                ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Negotiation Personalities by Model Tier<br>"
                  "<sub>Average personality profiles for each tier</sub>",
            width=700,
            height=600
        )
        
        return fig
    
    def generate_personality_report(self, personalities):
        """Generate text report of personality insights"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        report = f"# LLM Negotiation Personality Analysis Report\n"
        report += f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"**Source File:** {self.source_file}\n\n"
        
        # Find extremes
        dimensions = [
            'Aggressiveness', 'Concession_Rate', 'Price_Anchoring',
            'Convergence_Speed', 'Fairness_Preference', 'Consistency'
        ]
        
        for dim in dimensions:
            scores = [(model, p[dim]) for model, p in personalities.items()]
            highest = max(scores, key=lambda x: x[1])
            lowest = min(scores, key=lambda x: x[1])
            
            report += f"## {dim}\n"
            report += f"**Highest:** {highest[0].replace(':latest', '')} ({highest[1]:.1f})\n"
            report += f"**Lowest:** {lowest[0].replace(':latest', '')} ({lowest[1]:.1f})\n\n"
        
        # Personality archetypes
        report += "## Personality Archetypes\n\n"
        
        for model, personality in personalities.items():
            model_clean = model.replace(':latest', '')
            
            # Determine archetype
            if personality['Aggressiveness'] > 70:
                archetype = "🗡️ **Aggressive Negotiator**"
            elif personality['Fairness_Preference'] > 80:
                archetype = "⚖️ **Fair Mediator**"
            elif personality['Convergence_Speed'] > 80:
                archetype = "⚡ **Quick Closer**"
            elif personality['Consistency'] > 80:
                archetype = "🎯 **Reliable Partner**"
            elif personality['Price_Anchoring'] > 80:
                archetype = "⚓ **Strong Anchor**"
            else:
                archetype = "🤝 **Balanced Negotiator**"
            
            report += f"### {model_clean}\n"
            report += f"{archetype}\n"
            report += f"- **Tier:** {personality['tier']}\n"
            report += f"- **Success Rate:** {personality['success_rate']:.1%}\n"
            report += f"- **Key Strengths:** "
            
            # Find top 2 traits
            trait_scores = [(dim, personality[dim]) for dim in dimensions]
            top_traits = sorted(trait_scores, key=lambda x: x[1], reverse=True)[:2]
            report += f"{top_traits[0][0]} ({top_traits[0][1]:.0f}), {top_traits[1][0]} ({top_traits[1][1]:.0f})\n\n"
        
        return report

def main():
    """Run personality fingerprint analysis"""
    print("🧬 Creating LLM Negotiation Personality Fingerprints...")
    
    # Initialize analyzer with auto-detection
    try:
        analyzer = LLMPersonalityAnalyzer()
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ {e}")
        return
    
    # Calculate personalities
    print("📊 Calculating personality metrics...")
    personalities = analyzer.calculate_personality_metrics()
    
    if not personalities:
        print("❌ No personality data calculated. Please check your data.")
        return
    
    print(f"✅ Calculated personalities for {len(personalities)} models")
    
    # Create timestamp for file naming
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create visualizations
    print("📊 Generating individual radar charts...")
    for model_name in personalities.keys():
        fig = analyzer.create_radar_chart(personalities, model_name)
        if fig:
            clean_name = model_name.replace(':', '_')
            fig.write_html(f"personality_radar_{clean_name}_{timestamp}.html")
            try:
                fig.write_image(f"personality_radar_{clean_name}_{timestamp}.png", 
                              width=500, height=500, scale=2)
            except Exception as e:
                print(f"⚠️  Skipping PNG export for {model_name}: {e}")
    
    print("📊 Generating comparison visualizations...")
    
    # Comparison radar
    comp_fig = analyzer.create_comparison_radar(personalities)
    comp_fig.write_html(f"personality_comparison_radar_{timestamp}.html")
    try:
        comp_fig.write_image(f"personality_comparison_radar_{timestamp}.png", width=800, height=600, scale=2)
    except Exception as e:
        print(f"⚠️  Skipping PNG export for comparison radar: {e}")
    
    # Heatmap
    heatmap_fig = analyzer.create_personality_heatmap(personalities)
    heatmap_fig.write_html(f"personality_heatmap_{timestamp}.html")
    try:
        heatmap_fig.write_image(f"personality_heatmap_{timestamp}.png", width=800, height=600, scale=2)
    except Exception as e:
        print(f"⚠️  Skipping PNG export for heatmap: {e}")
    
    # Tier comparison
    tier_fig = analyzer.create_tier_comparison(personalities)
    tier_fig.write_html(f"personality_tier_comparison_{timestamp}.html")
    try:
        tier_fig.write_image(f"personality_tier_comparison_{timestamp}.png", width=700, height=600, scale=2)
    except Exception as e:
        print(f"⚠️  Skipping PNG export for tier comparison: {e}")
    
    # Generate report
    print("📝 Generating personality report...")
    report = analyzer.generate_personality_report(personalities)
    report_path = f"personality_analysis_report_{timestamp}.md"
    with open(report_path, "w") as f:
        f.write(report)
    
    # Save personality data
    personality_data = []
    for model, data in personalities.items():
        row = {'model': model}
        row.update(data)
        personality_data.append(row)
    
    personality_df = pd.DataFrame(personality_data)
    data_path = f"personality_data_{timestamp}.csv"
    personality_df.to_csv(data_path, index=False)
    
    print("✅ LLM Personality Fingerprints complete!")
    print("📁 Files generated:")
    print(f"   - Individual radar charts: personality_radar_*_{timestamp}.html/png")
    print(f"   - Comparison radar: personality_comparison_radar_{timestamp}.html/png")
    print(f"   - Heatmap: personality_heatmap_{timestamp}.html/png") 
    print(f"   - Tier comparison: personality_tier_comparison_{timestamp}.html/png")
    print(f"   - Analysis report: {report_path}")
    print(f"   - Personality data: {data_path}")

if __name__ == "__main__":
    main()