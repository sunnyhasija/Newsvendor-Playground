#!/usr/bin/env python3
"""
LLM Negotiation Personality Fingerprints
Creates unique "negotiation DNA" radar charts for each model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class LLMPersonalityAnalyzer:
    """Analyze and visualize LLM negotiation personalities"""
    
    def __init__(self, data_path="./full_results/processed/complete_20250615_171248.csv"):
        """Initialize with negotiation data"""
        self.data = pd.read_csv(data_path)
        self.successful = self.data[self.data['completed'] == True].copy()
        
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
            buyer_aggression = max(0, min(1, buyer_aggression))
        
        if len(supplier_prices) > 0:
            # Suppliers are aggressive when they push for very high prices  
            supplier_aggression = np.mean([(price - 65) / 35 for price in supplier_prices if price > 65])
            supplier_aggression = max(0, min(1, supplier_aggression))
        
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
        
        report = "# LLM Negotiation Personality Analysis Report\n\n"
        
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
    
    # Initialize analyzer
    analyzer = LLMPersonalityAnalyzer()
    
    # Calculate personalities
    personalities = analyzer.calculate_personality_metrics()
    
    # Create visualizations
    print("📊 Generating individual radar charts...")
    for model_name in personalities.keys():
        fig = analyzer.create_radar_chart(personalities, model_name)
        if fig:
            fig.write_html(f"personality_radar_{model_name.replace(':', '_')}.html")
            fig.write_image(f"personality_radar_{model_name.replace(':', '_')}.png", 
                          width=500, height=500, scale=2)
    
    print("📊 Generating comparison visualizations...")
    
    # Comparison radar
    comp_fig = analyzer.create_comparison_radar(personalities)
    comp_fig.write_html("personality_comparison_radar.html")
    comp_fig.write_image("personality_comparison_radar.png", width=800, height=600, scale=2)
    
    # Heatmap
    heatmap_fig = analyzer.create_personality_heatmap(personalities)
    heatmap_fig.write_html("personality_heatmap.html")
    heatmap_fig.write_image("personality_heatmap.png", width=800, height=600, scale=2)
    
    # Tier comparison
    tier_fig = analyzer.create_tier_comparison(personalities)
    tier_fig.write_html("personality_tier_comparison.html")
    tier_fig.write_image("personality_tier_comparison.png", width=700, height=600, scale=2)
    
    # Generate report
    print("📝 Generating personality report...")
    report = analyzer.generate_personality_report(personalities)
    with open("personality_analysis_report.md", "w") as f:
        f.write(report)
    
    print("✅ LLM Personality Fingerprints complete!")
    print("📁 Files generated:")
    print("   - Individual radar charts: personality_radar_*.html/png")
    print("   - Comparison radar: personality_comparison_radar.html/png")
    print("   - Heatmap: personality_heatmap.html/png") 
    print("   - Tier comparison: personality_tier_comparison.html/png")
    print("   - Analysis report: personality_analysis_report.md")

if __name__ == "__main__":
    main()