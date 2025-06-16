#!/usr/bin/env python3
"""
3D Reflection Advantage Matrix
Shows the "sweet spot" where reflection provides maximum benefit
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')

class ReflectionAdvantageAnalyzer:
    """Analyze and visualize reflection advantages across model sizes and conditions"""
    
    def __init__(self, data_path="./full_results/processed/complete_20250615_171248.csv"):
        """Initialize with negotiation data"""
        self.data = pd.read_csv(data_path)
        self.successful = self.data[self.data['completed'] == True].copy()
        
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
        
        for model in self.model_sizes.keys():
            model_data = self.successful[
                (self.successful['buyer_model'] == model) | 
                (self.successful['supplier_model'] == model)
            ]
            
            if len(model_data) == 0:
                continue
            
            # Calculate baseline (no reflection)
            baseline_data = model_data[model_data['reflection_pattern'] == '00']
            baseline_price = baseline_data['agreed_price'].mean() if len(baseline_data) > 0 else 65
            baseline_rounds = baseline_data['total_rounds'].mean() if len(baseline_data) > 0 else 5
            baseline_tokens = baseline_data['total_tokens'].mean() if len(baseline_data) > 0 else 500
            
            for pattern in ['01', '10', '11']:
                pattern_data = model_data[model_data['reflection_pattern'] == pattern]
                
                if len(pattern_data) > 0:
                    # Performance metrics
                    avg_price = pattern_data['agreed_price'].mean()
                    avg_rounds = pattern_data['total_rounds'].mean()
                    avg_tokens = pattern_data['total_tokens'].mean()
                    
                    # Calculate advantages
                    price_advantage = abs(65 - baseline_price) - abs(65 - avg_price)  # Closer to optimal = better
                    efficiency_advantage = baseline_rounds - avg_rounds  # Fewer rounds = better
                    token_efficiency = baseline_tokens - avg_tokens  # Fewer tokens = better
                    
                    # Combined advantage score
                    combined_advantage = (
                        price_advantage * 0.4 +  # 40% weight on price optimality
                        efficiency_advantage * 0.3 +  # 30% weight on round efficiency  
                        (token_efficiency / 100) * 0.3  # 30% weight on token efficiency
                    )
                    
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
        
        return pd.DataFrame(advantages)
    
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
        fig.add_trace(
            go.Bar(
                x=[self.reflection_patterns[p] for p in pattern_tokens.index],
                y=pattern_tokens.values,
                name='Token Efficiency',
                showlegend=False,
                marker_color=['#FF6B6B', '#4ECDC4', '#FFEAA7']
            ),
            row=2, col=1
        )
        
        # Plot 4: Sweet spot highlight
        fig.add_trace(
            go.Scatter(
                x=advantages_df['model_size_gb'],
                y=advantages_df['combined_advantage'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=advantages_df['reflection_pattern'].map({
                        '01': '#FF6B6B', '10': '#4ECDC4', '11': '#FFEAA7'
                    }),
                    opacity=0.6
                ),
                text=advantages_df['model'].str.replace(':latest', ''),
                name='All Models',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Highlight sweet spot
        fig.add_trace(
            go.Scatter(
                x=[sweet_spot['model_size_gb']],
                y=[sweet_spot['combined_advantage']],
                mode='markers',
                marker=dict(
                    size=15,
                    color='red',
                    symbol='star',
                    line=dict(width=2, color='darkred')
                ),
                name=f'Sweet Spot: {sweet_spot["model"].replace(":latest", "")}',
                showlegend=True
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            width=1000,
            title_text="Reflection Advantage Sweet Spot Analysis<br>" +
                      f"<sub>🌟 Sweet Spot: {sweet_spot['model'].replace(':latest', '')} with " +
                      f"{sweet_spot['reflection_name']} (Score: {sweet_spot['combined_advantage']:.2f})</sub>"
        )
        
        return fig, sweet_spot
    
    def create_efficiency_frontier(self, advantages_df):
        """Create efficiency frontier plot"""
        
        fig = go.Figure()
        
        # Calculate efficiency metrics
        advantages_df['tokens_per_round'] = advantages_df['avg_tokens'] / advantages_df['avg_rounds']
        advantages_df['price_optimality'] = 100 - abs(advantages_df['avg_price'] - 65)  # Higher = better
        
        # Color by reflection pattern
        pattern_colors = {'01': '#FF6B6B', '10': '#4ECDC4', '11': '#FFEAA7'}
        
        for pattern in ['01', '10', '11']:
            pattern_data = advantages_df[advantages_df['reflection_pattern'] == pattern]
            
            fig.add_trace(go.Scatter(
                x=pattern_data['tokens_per_round'],
                y=pattern_data['price_optimality'],
                mode='markers+text',
                marker=dict(
                    size=pattern_data['model_size_gb'] * 3,  # Size by model size
                    color=pattern_colors[pattern],
                    opacity=0.7,
                    line=dict(width=1, color='black')
                ),
                text=pattern_data['model'].str.replace(':latest', ''),
                textposition="top center",
                name=self.reflection_patterns[pattern],
                hovertemplate=(
                    '<b>%{text}</b><br>' +
                    'Tokens/Round: %{x:.0f}<br>' +
                    'Price Optimality: %{y:.1f}<br>' +
                    '<extra></extra>'
                )
            ))
        
        # Add Pareto frontier line
        all_points = advantages_df[['tokens_per_round', 'price_optimality']].values
        
        # Simple Pareto frontier (top-right quadrant)
        sorted_points = all_points[np.argsort(all_points[:, 0])]
        pareto_points = []
        max_y = -np.inf
        
        for point in sorted_points:
            if point[1] > max_y:
                pareto_points.append(point)
                max_y = point[1]
        
        if len(pareto_points) > 1:
            pareto_points = np.array(pareto_points)
            fig.add_trace(go.Scatter(
                x=pareto_points[:, 0],
                y=pareto_points[:, 1],
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name='Efficiency Frontier',
                showlegend=True
            ))
        
        fig.update_layout(
            title='Reflection Efficiency Frontier<br>' +
                  '<sub>Bubble size = Model size | Top-right = Most efficient</sub>',
            xaxis_title='Tokens per Round (Lower = More Efficient)',
            yaxis_title='Price Optimality Score (Higher = Better)',
            width=800,
            height=600,
            showlegend=True
        )
        
        return fig
    
    def generate_advantage_report(self, advantages_df, sweet_spot):
        """Generate comprehensive advantage analysis report"""
        
        report = "# Reflection Advantage Matrix Analysis\n\n"
        
        # Sweet spot analysis
        report += f"## 🌟 Sweet Spot Discovery\n\n"
        report += f"**Optimal Configuration:** {sweet_spot['model'].replace(':latest', '')} with {sweet_spot['reflection_name']}\n"
        report += f"- **Combined Advantage Score:** {sweet_spot['combined_advantage']:.2f}\n"
        report += f"- **Price Advantage:** ${sweet_spot['price_advantage']:.2f} closer to optimal\n"
        report += f"- **Efficiency Advantage:** {sweet_spot['efficiency_advantage']:.1f} fewer rounds\n"
        report += f"- **Token Efficiency:** {sweet_spot['token_efficiency']:.0f} fewer tokens\n"
        report += f"- **Model Size:** {sweet_spot['model_size_gb']} GB\n\n"
        
        # Pattern analysis
        report += "## 📊 Reflection Pattern Analysis\n\n"
        
        pattern_summary = advantages_df.groupby('reflection_pattern').agg({
            'combined_advantage': ['mean', 'std', 'max'],
            'price_advantage': 'mean',
            'efficiency_advantage': 'mean',
            'token_efficiency': 'mean'
        }).round(2)
        
        for pattern in ['01', '10', '11']:
            if pattern in pattern_summary.index:
                stats = pattern_summary.loc[pattern]
                report += f"### {self.reflection_patterns[pattern]}\n"
                report += f"- **Average Advantage:** {stats[('combined_advantage', 'mean')]:.2f} ± {stats[('combined_advantage', 'std')]:.2f}\n"
                report += f"- **Best Performance:** {stats[('combined_advantage', 'max')]:.2f}\n"
                report += f"- **Price Improvement:** ${stats[('price_advantage', 'mean')]:.2f}\n"
                report += f"- **Efficiency Gain:** {stats[('efficiency_advantage', 'mean')]:.1f} rounds\n\n"
        
        # Model size analysis
        report += "## 📏 Model Size Effects\n\n"
        
        size_analysis = advantages_df.groupby('model_tier')['combined_advantage'].agg(['mean', 'count']).round(2)
        tier_names = {1: 'Ultra-Compact', 2: 'Compact', 3: 'Mid-Range', 4: 'Large'}
        
        for tier in sorted(size_analysis.index):
            tier_name = tier_names.get(tier, f'Tier {tier}')
            stats = size_analysis.loc[tier]
            report += f"### {tier_name} Models\n"
            report += f"- **Average Advantage:** {stats['mean']:.2f}\n"
            report += f"- **Sample Size:** {int(stats['count'])} configurations\n\n"
        
        # Key insights
        report += "## 💡 Key Insights\n\n"
        
        # Find best performers
        best_overall = advantages_df.loc[advantages_df['combined_advantage'].idxmax()]
        best_price = advantages_df.loc[advantages_df['price_advantage'].idxmax()]
        best_efficiency = advantages_df.loc[advantages_df['efficiency_advantage'].idxmax()]
        
        report += f"1. **Best Overall:** {best_overall['model'].replace(':latest', '')} with {best_overall['reflection_name']}\n"
        report += f"2. **Best Price Optimizer:** {best_price['model'].replace(':latest', '')} with {best_price['reflection_name']}\n"
        report += f"3. **Most Efficient:** {best_efficiency['model'].replace(':latest', '')} with {best_efficiency['reflection_name']}\n"
        
        # Reflection effectiveness
        avg_advantages = advantages_df.groupby('reflection_pattern')['combined_advantage'].mean()
        best_pattern = avg_advantages.idxmax()
        report += f"4. **Most Effective Reflection:** {self.reflection_patterns[best_pattern]} (avg score: {avg_advantages[best_pattern]:.2f})\n"
        
        # Size vs performance correlation
        size_corr = advantages_df['model_size_gb'].corr(advantages_df['combined_advantage'])
        report += f"5. **Size-Performance Correlation:** {size_corr:.2f} ({'Positive' if size_corr > 0 else 'Negative'} correlation)\n\n"
        
        return report

def main():
    """Run reflection advantage matrix analysis"""
    print("🎯 Creating 3D Reflection Advantage Matrix...")
    
    # Initialize analyzer
    analyzer = ReflectionAdvantageAnalyzer()
    
    # Calculate advantages
    print("📊 Calculating reflection advantages...")
    advantages_df = analyzer.calculate_reflection_advantages()
    
    # Create visualizations
    print("🎨 Generating 3D visualizations...")
    
    # 3D surface plot
    surface_fig = analyzer.create_3d_advantage_surface(advantages_df)
    surface_fig.write_html("reflection_advantage_3d.html")
    surface_fig.write_image("reflection_advantage_3d.png", width=900, height=700, scale=2)
    
    # Advantage heatmap
    heatmap_fig = analyzer.create_advantage_heatmap(advantages_df)
    heatmap_fig.write_html("reflection_advantage_heatmap.html")
    heatmap_fig.write_image("reflection_advantage_heatmap.png", width=700, height=500, scale=2)
    
    # Sweet spot analysis
    sweet_spot_fig, sweet_spot = analyzer.create_sweet_spot_analysis(advantages_df)
    sweet_spot_fig.write_html("reflection_sweet_spot_analysis.html")
    sweet_spot_fig.write_image("reflection_sweet_spot_analysis.png", width=1000, height=800, scale=2)
    
    # Efficiency frontier
    frontier_fig = analyzer.create_efficiency_frontier(advantages_df)
    frontier_fig.write_html("reflection_efficiency_frontier.html")
    frontier_fig.write_image("reflection_efficiency_frontier.png", width=800, height=600, scale=2)
    
    # Generate report
    print("📝 Generating advantage analysis report...")
    report = analyzer.generate_advantage_report(advantages_df, sweet_spot)
    with open("reflection_advantage_report.md", "w") as f:
        f.write(report)
    
    # Save data
    advantages_df.to_csv("reflection_advantages_data.csv", index=False)
    
    print("✅ 3D Reflection Advantage Matrix complete!")
    print("📁 Files generated:")
    print("   - 3D surface plot: reflection_advantage_3d.html/png")
    print("   - Advantage heatmap: reflection_advantage_heatmap.html/png")
    print("   - Sweet spot analysis: reflection_sweet_spot_analysis.html/png")
    print("   - Efficiency frontier: reflection_efficiency_frontier.html/png")
    print("   - Analysis report: reflection_advantage_report.md")
    print("   - Raw data: reflection_advantages_data.csv")
    
    print(f"\n🌟 Sweet Spot Found: {sweet_spot['model'].replace(':latest', '')} with {sweet_spot['reflection_name']}")
    print(f"   Combined Advantage Score: {sweet_spot['combined_advantage']:.2f}")

if __name__ == "__main__":
    main()